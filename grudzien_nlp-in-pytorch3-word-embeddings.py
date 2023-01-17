#We want our model to learn to understand semantic correlations between words. Like for example, "Mathematician" and "Physicist" are both humans and scientists. 
#For a word can create a vecot of semantic attributes. q(mathematician) = ( "can run": 2.3, "likes coffee":9.1, ...)
#Similarity (mathematician, physicist) = normalized( q(mathematician)*q(physicist) )
#Let your neural nettwork learn the attributes itself. However, they will not be interpretable.
#Having vocabulary v, store embeddings in |v| x D matrix, where D is the dimensionality of the attributes. So the ith word has its attributes in ith row.
#We map the words to indices with a dictionary word_to_ix
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)
word_to_ix = {"hello":0, "word":1}
embeds = nn.Embedding(2, 5) #embedding is not a tensor; it is like a function which stores value; refer to rows with tensors
lookup_tensor = torch.tensor( [word_to_ix["hello"] ], dtype = torch.long )
print( embeds(lookup_tensor))
def relu(tensor):
    return tensor * torch.tensor( ten>0, dtype=torch.float)
#We want to compute P(w.i) given a sequence (w.i-n+1, w.i-n+2, ..., w.i-1)
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
#Build a list of tuples ([word.i-2, word.i-1], target_word)
trigrams = [ ( [test_sentence[i], test_sentence[i+1]], test_sentence[i+2]) for i in range( len(test_sentence) - 2) ]
print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
print(word_to_ix)

class NGramLanguageModeler(nn.Module):
    def __init__( self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size*embedding_dim, 128) #Receives context_size words, each having embedding_dim size representation
        self.linear2 = nn.Linear(128, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings( inputs).view(1, -1) #treat context as one word - that's why input is a row vector
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax( out, dim= 1)
        return log_probs
    
losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler( len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD( model.parameters(), lr=0.01)
    
for epoch in range(200):
    total_loss = 0.
    for context, target in trigrams:
        context_idxs = torch.tensor( [word_to_ix[w] for w in context], dtype=torch.long )
        target_id = torch.tensor( [word_to_ix[target] ], dtype=torch.long)
            
        model.zero_grad()
            
        log_probs = model( context_idxs )
        loss = loss_function( log_probs, target_id)
            
        loss.backward()
        optimizer.step()
            
        total_loss += loss.item()
        losses.append(loss.item())
plt.plot(losses)
#This work is based on the code from https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html and contains my interpretation of the knowledge given there.
