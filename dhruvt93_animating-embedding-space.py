%matplotlib notebook
%load_ext autoreload
%autoreload 2
import torch
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import plotly.offline as py
import plotly.graph_objs as go

# put plotly in Jupyter mode
py.init_notebook_mode(connected=True)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.numInputs = 7     # max length of input string
        self.numOutputs = 1    # single number out
        self.numVocab = 13     # number of unique characters
        self.numDimensions = 3 # each embedding has three numbers
        self.numHidden = 10    # hidden fully connected layer
                
        self.embedding = nn.Embedding(self.numVocab, self.numDimensions)
        self.lin1 = nn.Linear(self.numInputs * self.numDimensions, self.numHidden)
        self.lin2 = nn.Linear(self.numHidden, self.numOutputs)

    def forward(self, input):
        x = self.embedding(input).view(-1, self.numInputs * self.numDimensions)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return torch.squeeze(x)

def generateData(num):
    train = []
    for i in range(0, num):
        a = random.randint(0, 100)
        b = random.randint(-100-a, 100-a)
        train.append((f'{a}{b:+}'.ljust(7), a+b))
    return train
generateData(1)
vocab = list('0123456789-+ ')
char2index = {char: i for i, char in enumerate(vocab)}

train = generateData(100000)
trainloader = torchdata.DataLoader(train, batch_size=100)
net = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=5e-4, momentum=0.9)
embeddingFrames = []
vocab
torch.tensor(range(0, len(vocab)))
net.embedding(torch.tensor(range(0, len(vocab))))
net.embedding.weight
net.embedding.weight.size()
# next(iter(trainloader))
# loop over the dataset multiple times
for epoch in range(2): 
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        # save emdebbings for creating the animation later
        embeddingFrames.append(net.embedding(torch.tensor(range(0, len(vocab)))))
        
        # prepare batch
        sums, actuals = data
        input = torch.tensor([[char2index[character] for character in sum] for sum in sums], dtype=torch.long)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(input)
        actualsTensor = actuals.type(torch.FloatTensor)
        loss = criterion(outputs, actualsTensor)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 200}')
            running_loss = 0.0

print('Finished Training')
frames = []
for embeddings in embeddingFrames[0:300:5]:
    x, y, z = torch.transpose(embeddings, 0, 1).tolist()
    data = [go.Scatter3d(x=x,y=y,z=z, mode='markers+text', text=vocab)]
    frame = dict(data=data)
    frames.append(frame)
    
fig = dict(data=frames[0]['data'], frames=frames)
py.iplot(fig)