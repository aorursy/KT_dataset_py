import re
import matplotlib.pyplot as plt
import numpy as np
sentences = """When forty winters shall besiege thy brow,
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
And see thy blood warm when thou feel'st it cold."""
# remove special characters
sentences = re.sub('[^A-Za-z0-9]+', ' ', sentences)

# remove 1 letter words
sentences = re.sub(r'(?:^| )\w(?:$| )', ' ', sentences).strip()

# lower all characters
sentences = sentences.lower()
# list of words
words = sentences.split()

# get vocabulary (set(words)) - unique words
vocab = set(words)
# word to id
word_to_ix = {w: i for i, w in enumerate(vocab)}

# id to word
ix_to_word = {i: w for i, w in enumerate(vocab)}
word_to_ix['the']
ix_to_word[83]
vocab_size = len(set(vocab))
embedding_dimension = 10
#embedding matrix = vocabulary_size x embedding_dimension
embeddings =  np.random.random_sample((vocab_size, embedding_dimension))
trigrams = [([words[i], words[i+1]], words[i+2]) for i in range(len(words) - 2)]
# inputs = ['word1', 'word2']
# targets = 'target'
# ( ['word1', 'word2'], 'target')
trigrams[0]
def relu(x):
    return x * (x > 0)

def drelu(x):
    return 1. * (x > 0)
def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())
def NLLLoss(logs, targets):
    out = logs[range(len(targets)), targets]
    return -out.sum()/len(out)
def linear1(x, theta):
    w1, w2 = theta
    return np.dot(x, w1.T)

def linear2(o, theta):
    w1, w2 = theta
    return np.dot(o, w2.T)
def forward(x, theta):
    m = embeddings[x].reshape(1, -1)
    n = linear1(m, theta)
    o = relu(n)
    p = linear2(o, theta)
    q = log_softmax(p)
    
    params = m, n, o, p, q
    return(params)
def log_softmax_crossentropy_with_logits(logits,target):

    out = np.zeros_like(logits)
    out[np.arange(len(logits)),target] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    
    return (- out + softmax) / logits.shape[0]
def backward(x, y, theta, params):
    m, n, o, p, q = params
    w1, w2 = theta
    
    dlog = log_softmax_crossentropy_with_logits(p, y)
    drelu = relu(n)
    
    # dw2 = dlog * o
    do = np.dot(dlog,w2)
    dw2 = np.dot(dlog.T, o)
    
    # dw1 = do * drelu * m
    dn = do * drelu
    dw1 = np.dot(dn.T, m)
    
    return dw1, dw2
def optimize(theta, grads, lr=0.03):
    w1, w2 = theta
    dw1, dw2 = grads
    
    w1 -= dw1 * lr
    w2 -= dw2 * lr
    
    return theta
theta = np.random.uniform(-1,1, (128, 20)), np.random.uniform(-1,1, (85, 128))
losses = {}

for epoch in range(1000):

    epoch_losses = []
    
    for context, target in trigrams:
        
        # convert context('word1', 'word2') into [#1, #2] word to ix
        context_ix = [word_to_ix[c] for c in context]

        # convert to numpy array and convert from (2,) to (1,2)
        context_ix = np.array(context_ix)
        context_ix = context_ix.reshape(1, 2)

        # forward propagation (predict)
        params = forward(context_ix, theta)

        # get the looses and append to epoch losses
        loss = NLLLoss(params[-1], [word_to_ix[target]])
        epoch_losses.append(loss)

        # get the gradients from back propagation
        grads = backward(context_ix, [word_to_ix[target]], theta, params)

        # optimize the weights Stochastic gradient descent (SGD)
        theta = optimize(theta, grads)
        
    losses[epoch] = epoch_losses
ix = np.arange(0,35)

fig = plt.figure()
fig.suptitle('Epoch/Losses', fontsize=20)
plt.plot(ix,[losses[i][0] for i in ix])
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Losses', fontsize=12)
def predict(words):
    context_ix = [word_to_ix[c] for c in words]
    params = forward(context_ix, theta)
    word = ix_to_word[np.argmax(params[-1])]
    
    return word
# 'And dig deep trenches in thy beauty's field'
# expected answer 'deep'
predict(["and", "dig"])
def accuracy():
    wrong = 0

    for context, target in trigrams:
        if(predict(context) != target):
            wrong += 1
            
    return (1 - (wrong / len(trigrams)))
accuracy()