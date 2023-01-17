import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np

torch.manual_seed(1)
def argmax(vec):
    _, idx = torch.max(vec, 1) #returns max value of each row together with index
    return idx.item()
def prepare_seq(seq, word_to_ix):
    return torch.tensor( [word_to_ix[w] for w in seq], dtype= torch.long)
def log_sum_exp( vec ):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1] )
    V = vec - max_score_broadcast
    return max_score + torch.log( torch.sum( torch.exp(V) ) )
#this calculates the log sum of exponents of scores. The reason why we don't use torch.log(torch.sum(torch.exp(vec))) is that it can easily lead to value errors. exp(max_score) is very large (can reach INF), so we omit the error by carrying it out and performing the rest of computations on reasonable values

#The immediate output of the network is the tensor of emission scores
class BiLSTM_CRF( nn.Module ):
    def __init__(self, vocab_size, hidden_size, tag_to_ix, embedding_dim):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(self.tag_to_ix)
        
        self.word_embeds = nn.Embedding( vocab_size, embedding_dim)
        self.lstm = nn.LSTM( embedding_dim, hidden_size // 2, num_layers= 1, bidirectional= True )
        self.hidden2tag = nn.Linear( hidden_size, self.tagset_size)
        
        #entry i,j is trnsition score to i from j
        self.transitions = nn.Parameter( torch.randn(self.tagset_size, self.tagset_size) )
        
        #We never transition to the start tag and never transition from the stop tag
        self.transitions.data[ tag_to_ix[START_TAG], :] = -10000.
        self.transitions.data[ :, tag_to_ix[ STOP_TAG] ] = -10000.
        
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return ( torch.randn( 2, 1, self.hidden_size // 2), torch.randn(2, 1, self.hidden_size//2) )
    
    def _forward_alg( self, feats ):
        #Do the forward algorithm to compute the partition function (the denominator of softmax probability)
        #feats are emission scores, so that feats[i][t] is the emission score of the tag t at the i_th word
        
        init_alphas = torch.full( (1, self.tagset_size), -10000. ) #tensor of values equal to -10000
        #START_TAG has all of the score
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        
        #Wrap in the variable so that we get an automatic backpropagation
        forward_var = init_alphas #In forward_var store log of sum of all(already computed) possible scores, where each score is proportional to the exponents of their emissions and transitions
        #Iterate through the sentence
        for feat in feats:

            t = self.tagset_size
            log_forward = self.transitions + forward_var.view(1, -1).expand(t, t) + feat.view(-1, 1).expand(t, t)
            max_rows = torch.max(log_forward, 1)[0].view(-1, 1)
            max_broad = max_rows.expand(t, t)
            log_forward = log_forward - max_broad
            forward_var = max_rows.view(1, -1) + torch.log( torch.sum( torch.exp(log_forward), 1) ) 
            
            #ALTERNATIVE APPROACH (POORER COMPUATIONALLY, BUT EXPLAINS WELL WHAT'S GOING ON)
            '''
            alphas_t = []
            for next_tag in range( self.tagset_size):
                #Say, next_tag = y.t
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                #[ EMIT[y.t|x.f.t], EMIT[y.t|x.f.t], EMIT[y.t|x.f.t] , ..., EMIT[y.t|x.f.t] ]
                #the ith entry of trans_score is the transitions score from i to next_tag
                trans_score = self.transitions[next_tag].view(1, -1)
                #[ TR[y.t|y.0], TR[y.t|y.1], ..., TR[y.t|y.t] ]
                next_tag_var = forward_var + trans_score + emit_score #Each route ending (temporalily) at y.t has its sum multiplied by exp(trans_score+emit_score)
                #so now, in the ith index, we have the sum of log_exps of routes ending at i
                
                #[ EMIT[y.t|x.f.t]+TR[y.t|y.0] +ForVar[y.0], ..., EMIT[y.t|x.f.t]+TR[y.t|y.t] + ForVar[t]]
                alphas_t.append( log_sum_exp(next_tag_var).view(1)) #exponentiate to obtain routes and take logarithm to maintain the invariant
            forward_var =torch.cat(alphas_t).view(1, -1) #Now the old values of forward_var disappear. Forward_var is a new row tensor.
            '''
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp( terminal_var)
        return alpha #alpha is the logarithm of sums of exponentials of scores of all possible routes
    
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden( )
        embeds = self.word_embeds(sentence).view( len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm( embeds, self.hidden)
        lstm_out = lstm_out.view( len(sentence), self.hidden_size)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def _score_sentence(self, feats, tags):
        #Give a score of a provided tag sequence
        #Recall that len( feats ) = len( tags )
        score = torch.zeros(1)
        #Concatenate the start_tag, as all tags begin in the abstract start_tag
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[ tags[i+1], tags[i] ] + feat[ tags[i+1] ]  #i+1 as 0 is the start_tag
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1] ] #All tags end in an abtract stop_tag
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        #initialize the viterbi variables in log space
        init_vvars = torch.full( (1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG] ] = 0

        forward_var = init_vvars #Column/row tensor of the best paths ending in each node of a layer
        for feat in feats:
            bptrs_t = [] #holds the backpointers for this step
            viterbivars_t = [] #holds the viterbi variables for this step

            for next_tag in range( self.tagset_size ):
                next_tag_var = forward_var + self.transitions[next_tag] #column/row tensor of all potentailly the best routes ending in this node in this layer
                best_tag_id = argmax( next_tag_var) #choose the best one (don't care about emission, as it is the same for all of them ending in this node)
                bptrs_t.append(best_tag_id) #remember the tag which gives the best route
                viterbivars_t.append( next_tag_var[0][best_tag_id].view(1) ) #remember the score of the best path

            #Now add in the emission scores, and assign forward_var to the set of viterbi variable we just computed
            forward_var = ( torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append( bptrs_t)

        #Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG] ] #all of the paths end in the stop_tag which is connected to all of them with one particular edge value
        best_tag_id = argmax(terminal_var)#choose the shortest path
        path_score = terminal_var[0][best_tag_id]

        #Follow the backpointers to decode the best path
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id) #create a list of indexes of the tags of the best path in reversed order
            
        #Pop off the start tag
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG] #Sanity check
        best_path.reverse() #reverse the list, so that the indexes are now in correct order
        return path_score, best_path
    
    def neg_log_likelihood(self, sentence, tags): #A cost function we will use. 
        feats = self._get_lstm_features(sentence) #Compute the features
        denominator = self._forward_alg(feats)#The log of denominator of our probability, given that the sentence has features feats
        enumerator = self._score_sentence(feats, tags)#The log enumerator of our probability, -//-
        return denominator - enumerator # -log of our probability
                                          # want out probbility to be 1, so i f have 1-eps, the cost is -log(1-eps) > 0    
    def forward(self, sentence):
        #Get the emission score from BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        #Find the best path given the features
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq  
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4


# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tag in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
            
tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), HIDDEN_DIM, tag_to_ix, EMBEDDING_DIM)
optimizer = optim.SGD( model.parameters(), lr = 0.01, weight_decay = 1e-4)

#Check predictions before training 
with torch.no_grad():
    sent = training_data[0][0]
    precheck_sent = prepare_seq(sent, word_to_ix)
    tags = training_data[0][1]
    precheck_tags = torch.tensor( [tag_to_ix[t] for t in tags], dtype=torch.long)
    predictions = model(precheck_sent)
    print( predictions, [tag_to_ix[t] for t in training_data[0][1] ])
    
epochs = 400
losses = []
for epoch in range(epochs):
    model.zero_grad()
    for sentence, tags in training_data:
        model.zero_grad()
        sent = prepare_seq(sentence, word_to_ix)
        targets = torch.tensor( [tag_to_ix[t] for t in tags], dtype=torch.long)
        loss = model.neg_log_likelihood(sent, targets)
        
        loss.backward()
        optimizer.step()
        
        losses.append( loss.item() )
        #Surprised that there is no forward in the learning process? And where are we using that long _viterbi_decode function? Look at the explainations below the graph!
        
with torch.no_grad():
    predictions = model( prepare_seq(training_data[0][0], word_to_ix ) )
    print( predictions, [tag_to_ix[t] for t in training_data[0][1] ])

        
        
import matplotlib.pyplot as plt
plt.plot( losses)
#This work is an implementation based on the code from https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
#It contains my modificions and optimizations.
#My explainations provided in comments are my interpretation of reasoning provided on the website.
