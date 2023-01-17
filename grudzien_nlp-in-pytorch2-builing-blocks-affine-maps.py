import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
#Pytorh maps rows of the input instead of the columns. In f(x) = Ax+b, ith row of the output is the map of the ith row of the input plus bias term
lin = nn.Linear(5, 3) #from R^5 to R^3
data = torch.randn(2, 5)
print( lin(data))
data = torch.randn(2, 2)
print( data )
print(F.relu(data))
data = torch.randn(5)
print( data )
print( F.softmax( data, dim=0))
print( F.softmax( data, dim=0).sum())

print( F.log_softmax(data, dim=0)) #softmax is a logarithm of each softmax
print( torch.exp( F.log_softmax(data, dim=0)) )
data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]
test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]
#word_to_ix maps each word onto an integer which will be the words index in the bag of words vector
word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2
class BoWClassifier(nn.Module): #inherits from nn.Module
    def __init__(self, num_labels, vocab_size): #it knows that it has to map input of size vocab_size (sentences) onto output of size num_labels (which are labels)
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)
    
    def forward(self, bow_vec):
        return F.log_softmax( self.linear(bow_vec), dim= 1 ) #does log_softmax along column vectors. Here obvious but in general not.
def make_bow_vector(sentence, word_to_ix):
        num_labels = len(word_to_ix)
        vec = torch.zeros(1, num_labels)
        for word in sentence: #Iterate over words in the sentence; for each new ord, incease the counter by one 1
            vec[0][ word_to_ix[word] ] += 1
        return vec
def make_target( label, label_to_ix ):
    return torch.LongTensor( [ label_to_ix[label] ] )
model = BoWClassifier( NUM_LABELS, VOCAB_SIZE)
#for param in model.parameters():
 #   print(param)
'''with torch.no_grad():
    sample = data[0]
    bow_vec = make_bow_vector(sample[0], word_to_ix)
    log_probs = model(bow_vec)
    print( log_probs )'''
label_to_ix = {"SPANISH":0, "ENGLISH":1}
with torch.no_grad(): #run on test data to see before-after
    for instance, label in test_data:
        bow_vec = make_bow_vector( instance, word_to_ix )
        log_probs = model(bow_vec)
        print( log_probs)
        

print( next(model.parameters())[:, word_to_ix["creo"]] )
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    for instance, label in data:
        model.zero_grad() #Clear gradients to prevent accumulation
        
        bow_vec = make_bow_vector( instance, word_to_ix )
        target = make_target( label, label_to_ix )
        
        log_probs = model( bow_vec )
        
        loss = loss_function( log_probs, target)
        loss.backward()
        optimizer.step()
        
with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)
print( next(model.parameters())[:,word_to_ix["creo"]] )
                
#This work is based on the code from the website https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html#sphx-glr-beginner-nlp-deep-learning-tutorial-py
#and contains my interpretation of the explainations provided on the website.
