import pandas as pd
import numpy as np
#import codecs
import matplotlib.pyplot as plt
import itertools

# Import Data

input_file = open("../input/nlp-starter-test/socialmedia_relevant_cols.csv", "r",encoding='utf-8', errors='replace')

# read_csv will turn CSV files into dataframes
questions = pd.read_csv(input_file)
# to clean data
def normalise_text (text):
    text = text.str.lower()
    text = text.str.replace(r"\#","")
    text = text.str.replace(r"http\S+","URL")
    text = text.str.replace(r"@"," ")
    text = text.str.replace(r"[^A-Za-z0-9(),!?\'\`\"]", " ")
    text = text.str.replace("\s{2,}", " ")
    return text
questions["text"]=normalise_text(questions["text"])
#could save to another file
import spacy
nlp = spacy.load("en")
doc = questions["text"].apply(nlp)
questions.choose_one.value_counts()
max_sent_len=max(len(doc[i]) for i in range(0,len(doc)))
print(max_sent_len)

vector_len=len(doc[0][0].vector)
print(vector_len)
tweet_matrix=np.zeros((len(doc),max_sent_len,vector_len))
print(tweet_matrix[0:2,0:3,0:4]) #test print
for i in range(0,len(doc)):
    for j in range(0,len(doc[i])):
        tweet_matrix[i][j]=doc[i][j].vector
list_labels = np.array(questions["class_label"])
print(list_labels.shape[0])
print(tweet_matrix.shape[0])
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
#if you need to convert numpy ndarray to tensor explicitely
#tweet_matrix = torch.from_numpy(tweet_matrix)
#for GPU - CUDA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
len_for_split=[int(tweet_matrix.shape[0]/4),int(tweet_matrix.shape[0]*(3/4))]
print(len_for_split)
test, train=random_split(tweet_matrix,len_for_split)
test.dataset.shape
# Hyperparameters
num_epochs = 10
num_classes = 3
learning_rate = 0.001
batch_size=100
# to transform the data and labels
class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
#load labels #truncating total data to keep batch size 100
labels_train=list_labels[train.indices[0:8100]]
labels_test=list_labels[test.indices[0:2700]]

#load train data
training_data=train.dataset[train.indices[0:8100]].astype(float)
#training_data=training_data.unsqueeze(1)

#load test data
test_data=test.dataset[test.indices[0:2700]].astype(float)
#test_data=test_data.unsqueeze(1)

dataset_train = MyDataset(training_data, labels_train)
dataset_test = MyDataset(test_data, labels_test)


#loading data batchwise
train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

## setting up the CNN network

#arguments(input channel, output channel, kernel size, strides, padding)
            
            #layer 1 : 
            # height_out=(h_in-F_h)/S+1=(72-5)/1+1=68
            # width_out=(w_in-F_w)/S+1=(384-35)/1+1=350
            # no padding given
            # height_out=68/2=34 
            # width_out=350/5=70
            
            #layer 2:
            # height_out=(h_in-F_h)/S+1=(34-5)/1+1=30
            # width_out=(w_in-F_w)/S+1=(70-5)/1+1=66
            # height_out=30/2=15 
            # width_out=66/2=33


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 15, kernel_size=(5,35), stride=1,padding=0), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,5), stride=(2,5)))
        self.layer2 = nn.Sequential(
            nn.Conv2d(15, 30, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(15 * 33 * 30, 5000)
        self.fc2 = nn.Linear(5000, 100)
        self.fc3 = nn.Linear(100,3)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return(out)
#creating instance of our ConvNet class
model = ConvNet()
model.to(device) #CNN to GPU


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
#CrossEntropyLoss function combines both a SoftMax activation and a cross entropy loss function in the same function

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
total_step = 8100/batch_size
loss_list = []
acc_list = []


for epoch in range(num_epochs):
    for i, (data_t, labels) in enumerate(train_loader):
        data_t=data_t.unsqueeze(1)
        data_t, labels = data_t.to(device), labels.to(device)
        
        # Run the forward pass
        outputs = model(data_t)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        #print("==========forward pass finished==========")
            
        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print("==========backward pass finished==========")
        
        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)
    
        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),(correct / total) * 100))
        
        
        
## evaluating model

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data_t, labels in test_loader:
        data_t=data_t.unsqueeze(1)
        data_t, labels = data_t.to(device), labels.to(device)
        
        outputs = model(data_t)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model: {} %'.format((correct / total) * 100))


import matplotlib.pyplot as plt
plt.xlabel("runs")
x_len=list(range(len(acc_list)))
plt.axis([0, max(x_len), 0, 1])
plt.title('result of convNet')
loss=np.asarray(loss_list)/max(loss_list)
plt.plot(x_len, loss, 'r.', x_len, acc_list, 'b.')
plt.show
