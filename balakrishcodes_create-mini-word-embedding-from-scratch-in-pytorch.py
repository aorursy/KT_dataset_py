import numpy as np

import matplotlib.pylab as plt

import pandas as pd

import torch 

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models

from torch.utils.data import Dataset, DataLoader

import random

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import OneHotEncoder

!pip install torchsummaryX  --quiet

from torchsummaryX import summary
# Sample Document # Recreated from the tom and jerry cartoon

        

docs = ["cat and mice are buddies",

        'mice lives in hole',

        'cat lives in house',

        'cat chases mice',

        'cat catches mice',

        'cat eats mice',

        'mice runs into hole',

        'cat says bad words',

        'cat and mice are pals',

        'cat and mice are chums',

        'mice stores food in hole',

        'cat stores food in house',

        'mice sleeps in hole',

        'cat sleeps in house']
idx_2_word = {}

word_2_idx = {}

temp = []

i = 1

for doc in docs:

    for word in doc.split():

        if word not in temp:

            temp.append(word)

            idx_2_word[i] = word

            word_2_idx[word] = i

            i += 1



print(idx_2_word)

print(word_2_idx)
vocab_size = 25



def one_hot_map(doc):

    x = []

    for word in doc.split():

        x.append(word_2_idx[word])

    return x

  

encoded_docs = [one_hot_map(d) for d in docs]

encoded_docs
max_len = 10

padded_docs = pad_sequences(encoded_docs, maxlen=max_len, padding='post')

padded_docs
training_data = np.empty((0,2))



window = 2

for sentence in padded_docs:

    sent_len = len(sentence)

    

    for i, word in enumerate(sentence):

        w_context = []

        if sentence[i] != 0:

            w_target = sentence[i]

            for j in range(i-window, i + window + 1):

                if j != i and j <= sent_len -1 and j >=0 and sentence[j]!=0:

                    w_context = sentence[j]

                    training_data = np.append(training_data, [[w_target, w_context]], axis=0)



print(len(training_data))

print(training_data.shape)

training_data
enc = OneHotEncoder()

enc.fit(np.array(range(30)).reshape(-1,1))

onehot_label_x = enc.transform(training_data[:,0].reshape(-1,1)).toarray()

onehot_label_x
enc = OneHotEncoder()

enc.fit(np.array(range(30)).reshape(-1,1))

onehot_label_y = enc.transform(training_data[:,1].reshape(-1,1)).toarray()

onehot_label_y
print(onehot_label_x[0])

print(onehot_label_y[0])



# From Numpy to Torch



onehot_label_x = torch.from_numpy(onehot_label_x)

onehot_label_y = torch.from_numpy(onehot_label_y)

print(onehot_label_x.shape, onehot_label_y.shape)
class WEMB(nn.Module):

    def __init__(self, input_size, hidden_size):

        super(WEMB, self).__init__()

        self.input_size = input_size

        self.hidden_size = hidden_size

        self.softmax = nn.Softmax(dim=1)

    

        self.l1 = nn.Linear(self.input_size, self.hidden_size, bias=False)

        self.l2 = nn.Linear(self.hidden_size, self.input_size, bias=False)

   

    def forward(self, x):

        out_bn = self.l1(x) # bn - bottle_neck

        out = self.l2(out_bn)

        out = self.softmax(out)

        return out, out_bn



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''

m = nn.Softmax() #nn.Sigmoid()

loss = nn.BCELoss()

input = torch.tensor([0.9,0.0,0.0]) #torch.randn(3, requires_grad=True)

target = torch.tensor([1.0,0.0,0.0])

output = loss(m(input), target)

print(input, m(input), target) tensor([0.9000, 0.0000, 0.0000]) tensor([0.5515, 0.2242, 0.2242]) tensor([1., 0., 0.])

output -> 0.36

'''



'''

m = nn.Sigmoid()

loss = nn.BCELoss()

input = torch.tensor([0.9,0.0,0.0])

target = torch.tensor([1.0,0.0,0.0])

output = loss(m(input), target)

print(input, m(input), target) tensor([0.9000, 0.0000, 0.0000]) tensor([0.7109, 0.5000, 0.5000]) tensor([1., 0., 0.])

output -> 0.5758

'''
input_size = 30

hidden_size = 2

learning_rate = 0.01

num_epochs = 5000



untrained_model = WEMB(input_size, hidden_size).to(device)

model = WEMB(input_size, hidden_size).to(device)

model.train(True)

print(model)

print()



# Loss and optimizer

criterion = nn.BCELoss()

#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0, dampening=0, weight_decay=0, nesterov=False)

summary(model, torch.ones((1,30)).to(device)) 
loss_val = []

onehot_label_x = onehot_label_x.to(device)

onehot_label_y = onehot_label_y.to(device)



for epoch in range(num_epochs):

    for i in range(onehot_label_y.shape[0]):

        inputs = onehot_label_x[i].float()

        labels = onehot_label_y[i].float()

        inputs = inputs.unsqueeze(0)

        labels = labels.unsqueeze(0)



      # Forward pass

        output, wemb = model(inputs)

        loss = criterion(output, labels)



      # Backward and optimize

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    loss_val.append(loss.item())



    if (epoch+1) % 100 == 0:

        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



plt.plot(loss_val)
docs = ['cat and mice are buddies hole lives in house chases catches runs into says bad words pals chums stores sleeps']

encoded_docs = [one_hot_map(d) for d in docs]



test_arr = np.array([[ 1.,  2., 3., 4., 5., 8., 6., 7., 9., 10., 11., 13., 14., 15., 16., 17., 18., 19., 20., 22.]])

test = enc.transform(test_arr.reshape(-1,1)).toarray()



output = []

for i in range(test.shape[0]):

    _, wemb2 = model(torch.from_numpy(test[i]).unsqueeze(0).float().to(device))

    wemb2 = wemb2[0].detach().cpu().numpy()

    output.append(wemb2)

print(len(output))

print(output)
docs = ['cat', 'and', 'mice', 'are', 'buddies', 'hole', 'lives', 'in', 'house', 'chases', 'catches', 'runs', 'into', 'says', 'bad', 'words', 'pals', 'chums', 'stores', 'sleeps']

for i in range(0, len(docs)):

    print("Word - {} - It's Word Embeddings : {:.3} & {:.3} -".format(docs[i], output[i][0], output[i][0]))
xs = []

ys = []

for i in range(len(output)):

    xs.append(output[i][0])

    ys.append(output[i][1])

print(xs, ys)



docs = ['cat', 'and', 'mice', 'are', 'buddies', 'hole', 'lives', 'in', 'house', 'chases', 'catches', 'runs', 'into', 'says', 'bad', 'words', 'pals', \

        'chums', 'stores', 'sleeps']



plt.clf()

plt.figure(figsize=(12,12))

plt.scatter(xs,ys)

label = docs



for i,(x,y) in enumerate(zip(xs,ys)):

    plt.annotate(label[i], (x,y), textcoords="offset points", xytext=(0,10), fontsize=20, ha = random.choice(['left', 'right']))

    plt.title("Trained Model")

plt.show()
import plotly

import plotly.express as px



import plotly.express as px

fig = px.scatter(x=xs, y=ys, text=docs, size_max=100)

fig.update_traces(textposition= random.choice(['top center', 'bottom center','bottom left']))

fig.update_layout(height=800,title_text='Custom Word Embeddings')

fig.show()
output = []

for i in range(test.shape[0]):

    _, wemb2 = untrained_model(torch.from_numpy(test[i]).unsqueeze(0).float().to(device)) # Here I am loading the untrained model

    wemb2 = wemb2[0].detach().cpu().numpy()

    output.append(wemb2)

print(len(output))

print(output)



xs = []

ys = []

for i in range(len(output)):

    xs.append(output[i][0])

    ys.append(output[i][1])

print(xs, ys)



docs = ['cat', 'and', 'mice', 'are', 'buddies', 'hole', 'lives', 'in', 'house', 'chases', 'catches', 'runs', 'into', 'says', 'bad', 'words', 'pals', \

        'chums', 'stores', 'sleeps']



plt.clf()

plt.figure(figsize=(12,12))

plt.scatter(xs,ys)

label = docs



for i,(x,y) in enumerate(zip(xs,ys)):

    plt.annotate(label[i], (x,y), textcoords="offset points", xytext=(0,10), fontsize=20, ha = random.choice(['left', 'right']))

    plt.title("Un-Trained Model")

plt.show()
import plotly

import plotly.express as px



import plotly.express as px

fig = px.scatter(x=xs, y=ys, text=docs, size_max=100)

fig.update_traces(textposition= random.choice(['top center', 'bottom center','bottom left']))

fig.update_layout(height=800,title_text='Custom Word Embeddings')

fig.show()