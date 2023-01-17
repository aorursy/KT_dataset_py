import torch

import torch.nn as nn

import torch.nn.functional as F



import numpy as np

import matplotlib.pyplot as plt



import string



# Can we use a GPU?

if torch.cuda.is_available():

    dev = torch.device('cuda')

else:

    dev = torch.device('cpu')
from nltk.corpus import wordnet as wn



def get_nouns():

    nouns = []

    for synset in wn.all_synsets('n'):

        name = synset.name().split('.')[0]

        name = name.split('_')[0]

        nouns.append(name)

    return nouns
import unicodedata



all_letters = string.ascii_letters + " .,;'-'"

n_letters = len(all_letters) + 1 # For <EOS>





def unicodeToAscii(s):

    return ''.join(

        c for c in unicodedata.normalize('NFD', s)

        if unicodedata.category(c) != 'Mn'

        and c in all_letters

    )
# Filter with None to remove empty strings

nouns = list(filter(None, set(map(unicodeToAscii, get_nouns()))))



print(len(nouns))

# Look at the first ten words

print(nouns[:10])

# Create mappings



char_to_index = {j : i+1 for i, j in enumerate(all_letters)}

char_to_index['<EOS>'] = 0



index_to_char = {i+1 : j for i, j in enumerate(all_letters)}

index_to_char[0] = '<EOS>'



print('c', '->', char_to_index['c'])



print('3', '->', index_to_char[3])

# Get tensor for a word

def get_tensor(word):

    tensors = torch.zeros(len(word), 1, n_letters)

    for i in range(len(word)):

        tensors[i][0][char_to_index[word[i]]] = 1

    return tensors.to(dev)



print(get_tensor('hello').shape)
class Net(nn.Module):

    def __init__(self, input_size, hidden_size):

        super(Net, self).__init__()

        

        self.input_size = input_size

        self.hidden_size = hidden_size

        

        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size)

        

        # Linear layer to predict output from the previous letter

        # and the current hidden state

        self.i2o = nn.Linear(input_size + hidden_size, input_size)

        

        # A dropout layer to allow for generalization

        self.dropout = nn.Dropout(p=0.2)

        

    def forward(self, inp):

        # inp is a tensor of shape (len(word), 1, input_size)

        output, _ = self.lstm(inp.view(-1, 1, self.input_size))

        # output is a tensor of shape (len(word), 1, hidden_size)

        

        # Pass everything through the i2o layer to get predictions

        output = torch.cat((output, inp), dim=2)

        output = self.i2o(output)

        output = F.log_softmax(output, dim=-1)

        output = self.dropout(output)

        return output

    

        

    
learning_rate = 0.001

hidden_size = 32





net = Net(n_letters, hidden_size).to(dev)

loss_fn = nn.NLLLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

def get_target_tensor(word):

    # We want to predict the target characters and EOS

    target_word = list(word[1:])

    target_word.append('<EOS>')

    

    return torch.tensor([char_to_index[c] for c in target_word]).to(dev)

# Training

num_epochs = 50





running_losses = []



# Go into training mode

net.train()

for i in range(num_epochs):

    running_loss = 0

    for noun in nouns:

        net.zero_grad()

        # Get input and target tensors for the noun

        noun_tensor = get_tensor(noun)

        target_noun_tensor = get_target_tensor(noun)

        

        # Get predictions and losses

        pred = net(noun_tensor)

        loss = loss_fn(pred.view(-1, n_letters), target_noun_tensor)

        

        running_loss += loss.item()

        

        # Back-propagate and update parameters

        loss.backward()

        optimizer.step()

    # Early stop if loss starts increasing

    if running_losses and running_losses[-1] < running_loss:

        break

    else:

        running_losses.append(running_loss)

    if i % 2 == 0:

        # Occasionally print out loss values

        print(f'Epoch {i+1} / {num_epochs}: {running_loss}')

        

plt.plot(running_losses)

plt.xlabel('Epochs')

plt.ylabel('NLL Loss')



plt.title('Training error of the model')
def predict_letter(word, sample=True):

    # Go into evaluation mode

    net.eval()

    preds = net(get_tensor(word))

    if not sample:

        # Choose the letter with highest priority

        return index_to_char[torch.argmax(preds[-1]).item()]

    else:

        # Get probabilities and sample the next letter

        probabilities = preds[-1].exp().cpu().detach().numpy().ravel()

        return index_to_char[np.random.choice(range(n_letters), p=probabilities)]

    
def generate_word(letter=None):

    if letter is None:

        letter = np.random.choice(list(string.ascii_lowercase))

        

    word = letter

    net.eval()

    while True:

        next_letter = predict_letter(word, sample=True)

        if next_letter == '<EOS>':

            return word

        else:

            word += next_letter
for i in range(10):

    print(generate_word())