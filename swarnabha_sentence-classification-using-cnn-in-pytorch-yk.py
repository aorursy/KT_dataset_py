import pandas as pd

import numpy as np

#import codecs

import matplotlib.pyplot as plt
# Import Data



input_file = open("../input/nlp-starter-test/socialmedia_relevant_cols.csv", "r",encoding='utf-8', errors='replace')



# read_csv will turn CSV files into dataframes

questions = pd.read_csv(input_file)
questions.head()
questions.choose_one.value_counts()
# to clean data

def normalise_text (text):

    text = text.str.lower()

    text = text.str.replace(r"\#","")

    text = text.str.replace(r"http\S+","URL")

    text = text.str.replace(r"@"," ")

    text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")

    text = text.str.replace("\s{2,}", " ")

    return text
questions["text"]=normalise_text(questions["text"])

#could save to another file
questions.head()
import spacy

nlp = spacy.load("en_core_web_lg")
#apply the spaCy nlp pipeline

doc = questions["text"].apply(nlp) 



#you could extract a lot more information once you pass the data through the nlp pipeline, such as POS tagging, recognising important entities, etc. 
max_sent_len=max(len(doc[i]) for i in range(0,len(doc)))

print("length of longest sentence: ", max_sent_len)

#point to be noted this is the number of tokens in the sentence, NOT words





vector_len=len(doc[0][0].vector)

print("length of each word vector: ", vector_len)
#creating the 3D array

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

num_epochs = 25

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

            

            #layer 1x : 

            # height_out=(h_in-F_h)/S+1=(72-x)/1+1=73-x

            # width_out=(w_in-F_w)/S+1=(384-384)/1+1=1

            # no padding given

            # height_out=(70-x)/(70-x)=1 

            # width_out=1/1=1

            





class ConvNet(nn.Module):

    def __init__(self):

        super(ConvNet, self).__init__()

        self.layer13 = nn.Sequential(

            nn.Conv2d(1, 100, kernel_size=(3,vector_len), stride=1,padding=0), 

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(70,1), stride=1))

        self.layer14 = nn.Sequential(

            nn.Conv2d(1, 100, kernel_size=(4,vector_len), stride=1,padding=0), 

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(69,1), stride=1))

        self.layer15 = nn.Sequential(

            nn.Conv2d(1, 100, kernel_size=(5,vector_len), stride=1,padding=0), 

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(68,1), stride=1))

        #self.layer2 = nn.Sequential(

            #nn.Conv2d(15, 30, kernel_size=5, stride=1, padding=0),

            #nn.ReLU(),

            #nn.MaxPool2d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout()

        #concat operation

        self.fc1 = nn.Linear(1 * 1 * 100 * 3, 30)

        self.fc2 = nn.Linear(30, 3)

        #self.fc3 = nn.Linear(100,3)

        

    def forward(self, x):

        x3 = self.layer13(x)

        x4 = self.layer14(x)

        x5 = self.layer15(x)

        x3 = x3.reshape(x3.size(0), -1)

        x4 = x4.reshape(x4.size(0), -1)

        x5 = x5.reshape(x5.size(0), -1)

        x3 = self.drop_out(x3)

        x4 = self.drop_out(x4)

        x5 = self.drop_out(x5)

        out = torch.cat((x3,x4,x5),1)

        out = self.fc1(out)

        out = self.fc2(out)

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

val_acc_list = []



for epoch in range(num_epochs):

    loss_list_element = 0

    acc_list_element = 0

    for i, (data_t, labels) in enumerate(train_loader):

        data_t=data_t.unsqueeze(1)

        data_t, labels = data_t.to(device), labels.to(device)

        

        # Run the forward pass

        outputs = model(data_t)

        loss = criterion(outputs, labels)

        loss_list_element += loss.item()

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

        acc_list_element += correct

        

        

    loss_list_element = loss_list_element/np.shape(labels_train)[0]

    acc_list_element = acc_list_element/np.shape(labels_train)[0]

    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'

          .format(epoch + 1, num_epochs, loss_list_element,acc_list_element * 100))  

    loss_list.append(loss_list_element)

    acc_list.append(acc_list_element)

    

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

        val_acc_list.append(correct / total)



    print('Test Accuracy of the model: {} %'.format((correct / total)*100))

    print()

        

        
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

plt.ylabel("normalised measure of loss/accuracy")

x_len=list(range(len(acc_list)))

plt.axis([0, max(x_len), 0, 1])

plt.title('result of convNet')

loss=np.asarray(loss_list)/max(loss_list)

plt.plot(x_len, loss, 'r.',label="loss")

plt.plot(x_len, acc_list, 'b.', label="accuracy")

plt.plot(x_len, val_acc_list, 'g.', label="val_accuracy")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show
