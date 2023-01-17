import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(1)
CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):

    def __init__(self, embedding_dim, vocab_size, context_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size*embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embedding(inputs)
        out = F.relu( self.linear1( embeds.view(1, -1) ) )
        out = self.linear2( out.view(1, -1) )
        log_probs = F.log_softmax(out, dim= 1)
        return log_probs

# create your model and train.  here are some functions to help you make
# the data ready for use by your module

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

EMBEDDING_DIM = 10
model = CBOW( EMBEDDING_DIM, len(vocab), 2*CONTEXT_SIZE)
optimizer = optim.SGD( model.parameters(), lr=0.01)
loss_function = nn.NLLLoss()

epochs = 100
losses = []

for epoch in range(epochs):
    total_loss = 0
    for context, target in data:
        model.zero_grad()
        context = make_context_vector( context , word_to_ix)
        outs = torch.tensor( [word_to_ix[target]], dtype=torch.long)
        
        log_probs = model(context)
        loss = loss_function(log_probs, outs)
        loss.backward()
        optimizer.step()
        
        total_loss += loss
    print(total_loss)

make_context_vector(data[0][0], word_to_ix)  # example
#This is my solution of the exercise given on the website https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html. My work is based on the code provided there.

