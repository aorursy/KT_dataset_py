import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import networkx as nx

import math

import torch

import os

import warnings

import dateutil

from torch.utils.data import Dataset

import torchvision

from sklearn.metrics import accuracy_score , confusion_matrix , r2_score , classification_report

import torchvision.transforms as transforms

from tqdm import tqdm

import spacy

nlp = spacy.load('en')

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

plt.style.use('ggplot')

warnings.filterwarnings('ignore')
def create_neural_network(inputs = 4 , activations = 4 , outputs = 3 , node_col = 'g' , edge_col = 'y'):

    # max number of nodes in Graph are 50

    dense  = nx.Graph()

    # input layer  , activation layer , output layer in neural network 

    inputs = {i: (0,i) for i in range(0,inputs)}

    activations = {i+50 : (1,i) for i in range(0,activations)}

    outputs = {i+100 : (2,i) for i in range(0,outputs)}

    all = {**inputs , **activations , **outputs}

    for input in inputs:

        for activation in activations:

            dense.add_edge(input , activation)

    for activation in activations:

        for output in outputs:

            dense.add_edge(activation, output)

    nx.draw_networkx_nodes(dense , all , nodelist = all.keys() , node_color = node_col)

    nx.draw_networkx_edges(dense , all  , edge_color = edge_col)

    axes = plt.axis('off')

    

    

create_neural_network()

# This is Simple View of Neural Network
#Creating Simple Neural Netwok 

inputs = torch.rand(1,1,64,64)

print(inputs.shape ,'\n',inputs)

outputs = torch.rand(1,2)

print(outputs.shape , '\n' , outputs)

# Creating Linear Sequential Model 

model = torch.nn.Sequential(

            torch.nn.Linear(64,256),

            torch.nn.Linear(256,256),

            torch.nn.Linear(256,2)

)

result = model(inputs)

# print(result , '\n' ,result.shape )

print('Prediction shape is ' , result.shape )
x = torch.range(-1,1,0.1) # X values for input

graph_x = 2 ; graph_y = 3 # Parameter for Graph

plt.figure(figsize=(16,10)) # Figure Size

plt.subplot(graph_x,graph_y,1);y = torch.nn.functional.softplus(x);plt.title('Softplus');plt.plot(x.numpy() , y.numpy())

plt.subplot(graph_x,graph_y,2);y = torch.nn.functional.relu(x);plt.title('Relu');plt.plot(x.numpy() , y.numpy())

plt.subplot(graph_x,graph_y,3);y = torch.nn.functional.elu(x);plt.title('Elu');plt.plot(x.numpy() , y.numpy())

plt.subplot(graph_x,graph_y,4);y = torch.nn.functional.tanh(x);plt.title('Tanh');plt.plot(x.numpy() , y.numpy())

plt.subplot(graph_x,graph_y,5);y = torch.nn.functional.sigmoid(x);plt.title('Sigmoid');plt.plot(x.numpy() , y.numpy())

plt.subplot(graph_x,graph_y,6);y = torch.nn.functional.gelu(x);plt.title('Gelu');plt.plot(x.numpy() , y.numpy())
#Creating More Complex Neural Netwok 

inputs = torch.rand(1,1,64,64)

print(inputs.shape ,'\n',inputs)

outputs = torch.rand(1,2)

print(outputs.shape , '\n' , outputs)

# Creating Linear Sequential Model 

model = torch.nn.Sequential(

            torch.nn.Linear(64,256),

            torch.nn.ReLU(),

            torch.nn.Linear(256,256),

            torch.nn.ReLU(),

            torch.nn.Linear(256,2)

)

result = model(inputs)

# print(result , '\n' ,result.shape )

print('Prediction shape is ' , result.shape )

loss = torch.nn.MSELoss()(result,outputs)

print("Mean Square Error is ", loss )

model.zero_grad()

loss.backward()

learning_rate = 0.01

for parameter in model.parameters():

    parameter.data -= parameter.grad.data * learning_rate

after_result = model(inputs)

print("Mean Square Error is ", torch.nn.MSELoss()(after_result,outputs))
class Model(torch.nn.Module):

    def __init__(self):

        super().__init__()# for multiple inheritence super is used

        self.layer_one = torch.nn.Linear(64,256)

        self.activation_one = torch.nn.ReLU()

        self.layer_two = torch.nn.Linear(256,256)

        self.activation_two = torch.nn.ReLU()

        self.shape_outputs = torch.nn.Linear(64*256 , 2)



    def forward(self, inputs):

        buffer = self.layer_one(inputs)

        buffer = self.activation_one(buffer)

        buffer = self.layer_two(buffer)

        buffer = self.activation_two(buffer)

        buffer = buffer.flatten(start_dim = 1)

        return self.shape_outputs(buffer)

    
model = Model()

test_results = model(inputs)

print('Test Results are -> ',test_results)

print('Outputs are -> ',outputs)
loss_function = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters() , lr=0.01)

for i in range(10000):

    optimizer.zero_grad()

    results = model(inputs)

    loss = loss_function(results , outputs)

    loss.backward()

    optimizer.step()

    gradients = 0.0

    # Looking for vanishing point

    for parameter in model.parameters():

        gradients += parameter.grad.data.sum()

    if abs(gradients) <= 0.0001:

        print(gradients)

        print('Gradient Vanished at iterations {0}'.format(i))

        break

        
test_results = model(inputs)

print(test_results)

print(outputs)
class MushRoomDataset(Dataset):

    def __init__(self):

        self.data = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

    def __len__():

        return len(self.data)

    def __getitem__(self , idx):

        if type(idx) is torch.Tensor:

            idx = idx.item()

        return self.data.iloc[idx][1:] , self.data.iloc[idx][0:1]

    def sample(self , samples = 5 , transverse = False):

        if transverse:

            display(self.data.sample(samples)).T

        else:

            display(self.data.sample(samples)) 

    def test_data(self , per_cen = 0.05 , shuffle = True):

        number_of_testing = int(len(self.data) * per_cen)

        number_of_training = len(self.data) - number_of_testing

        train,test = torch.utils.data.random_split(self.data , [number_of_training , number_of_testing])

        return train,test

mrooms = MushRoomDataset()

train , test = mrooms.test_data()
class OneHotEncoder():

    def __init__(self,series):

        unique_values = series.unique()

        self.ordinals = {

            val: i for i,val in enumerate(unique_values)

        }

        self.encoder = torch.eye(len(unique_values) , len(unique_values))

        

    def __getitem__(self,value):

        return self.encoder[self.ordinals[value]]

    

class CategoricalCSV(Dataset):

    def __init__(self , datafile , output_series_name):

        self.dataset = pd.read_csv(datafile)

        self.output_series_name = output_series_name

        self.encoders = {}

        for series_name , series in self.dataset.items():

            self.encoders[series_name] = OneHotEncoder(series)

    

    def __len__(self):

        return len(self.dataset)

    

    def __getitem__(self , index):

        if type(index) is torch.Tensor:

            index = index.item()

        sample = self.dataset.iloc[index]

        output = self.encoders[self.output_series_name][sample[self.output_series_name]]

        input_components= []

        for name, value in sample.items():

            if name != self.output_series_name:

                input_components.append(self.encoders[name][value])

        inputs = torch.cat(input_components)

        return inputs , output



cmrooms = CategoricalCSV('/kaggle/input/mushroom-classification/mushrooms.csv' , 'class')
cmrooms[0]
# Creating Predictions With Developed Data

class Model(torch.nn.Module):

    def __init__(self , input_dimesions , output_dimesions , size = 128):

        super().__init__()

        self.layer_one = torch.nn.Linear(input_dimesions , size)

        self.activation_one = torch.nn.ReLU()

        self.layer_two = torch.nn.Linear(size , size)

        self.activation_two = torch.nn.ReLU()

        self.shape_outputs = torch.nn.Linear(size , output_dimesions)

    def forward(self, inputs):

        buffer = self.layer_one(inputs)

        buffer = self.activation_one(buffer)

        buffer = self.layer_two(buffer)

        buffer = self.activation_two(buffer)

        buffer = self.shape_outputs(buffer)

        return torch.nn.functional.softmax(buffer , dim = -1)

    

model = Model(cmrooms[0][0].shape[0] , cmrooms[0][1].shape[0])

optimizer = torch.optim.Adam(model.parameters())

loss_functions = torch.nn.BCELoss()

# train , test = cmrooms.test_data()

number_of_testing = int(len(cmrooms) * 0.05)

number_of_training = len(cmrooms) - number_of_testing

train,test = torch.utils.data.random_split(cmrooms, [number_of_training , number_of_testing])



training = torch.utils.data.DataLoader(train , batch_size = 16 , shuffle = True)

testing = torch.utils.data.DataLoader(test , batch_size = len(test) , shuffle = True)



for epoch in range(3):

    for inputs,outputs in training:

        optimizer.zero_grad()

        results = model(inputs)

        loss = loss_function(results , outputs)

        loss.backward()

        optimizer.step()

    print("Loss : {0} ".format(loss))

    

for inputs , outputs in testing:

    results = model(inputs).argmax(dim = 1).numpy()

    actual = outputs.argmax(dim = 1).numpy()

    accuracy = accuracy_score(actual , results)

    print("Test Accuracy is -> ",accuracy)

    

sns.heatmap(confusion_matrix(actual , results), annot=True, annot_kws={"size": 16}) # font size

plt.show()
sample_1 = pd.read_csv('/kaggle/input/kc-housesales-data/kc_house_data.csv')
class DateEncoder():

    def __getitem__(self, datestring):

        parsed = dateutil.parser.parse(datestring)

        return torch.Tensor([parsed.year, parsed.month , parsed.day])

    

class MixedCSV(Dataset):

    def __init__(self , datafile , output_series_name , date_series_name , categorical_series_name , ignore_series_name):

        self.dataset = pd.read_csv(datafile)

        self.output_series_name = output_series_name

        self.encoders = {}

        for series_name in date_series_name:

            self.encoders[series_name] = DateEncoder()

        for series_name in categorical_series_name:

            self.encoders[series_name] = OneHotEncoder(self.dataset[series_name])

        self.ignore = ignore_series_name

        

    def __len__(self):

        return len(self.dataset)

    

    def __getitem__(self , index):

        if type(index) is torch.Tensor:

            index = index.item()

        sample = self.dataset.iloc[index]

        output = torch.Tensor([sample[self.output_series_name]])

        input_components = []

        for name , value in sample.items():

            if name in self.ignore:

                continue

            elif name in self.encoders:

                input_components.append(self.encoders[name][value])

            else:

                input_components.append(torch.Tensor([value]))

        

        input = torch.cat(input_components)

        return input , output

date_type = ['date']

categorical_data =  ['zipcode' , 'waterfront' , 'condition' , 'grade']

discard = ['id']

output_column = 'price'

houses = MixedCSV('/kaggle/input/kc-housesales-data/kc_house_data.csv' , 

                    output_column,

                    date_type , 

                    categorical_data , 

                    discard)
class Model(torch.nn.Module):

    def __init__(self , input_dimesions , size = 128):

        super().__init__()

        self.layer_one = torch.nn.Linear(input_dimesions , size)

        self.activation_one = torch.nn.ReLU()

        self.layer_two = torch.nn.Linear(size , size)

        self.activation_two = torch.nn.ReLU()

        self.shape_outputs = torch.nn.Linear(size , 1)

    

    def forward(self , inputs):

        buffer = self.layer_one(inputs)

        buffer = self.activation_one(buffer)

        buffer = self.layer_two(buffer)

        buffer = self.activation_two(buffer)

        buffer = self.shape_outputs(buffer)

        return buffer

    

model = Model(houses[0][0].shape[0])

optimizer = torch.optim.Adam(model.parameters())

loss_function = torch.nn.MSELoss()



number_for_testing = int(len(houses)*0.05)

number_for_training = len(houses) - number_for_testing

train , test = torch.utils.data.random_split(houses , [number_for_training , number_for_testing])

training = torch.utils.data.DataLoader(train , batch_size = 32 , shuffle = True)

for epoch in range(3):

    for inputs , outputs in training:

        optimizer.zero_grad()

        results = model(inputs)

        loss = loss_function(results , outputs)

        loss.backward()

        optimizer.step()

    print('Loss {0}'.format(loss))

    

print('Printing R-Square Score ....\n')

testing = torch.utils.data.DataLoader(test , batch_size = len(test) , shuffle= False)

for inputs , outputs in testing:

    predicted = model(inputs).detach().numpy()

    actual = outputs.numpy()

    print(r2_score(predicted , actual))
# Loading Dataset

mnist = torchvision.datasets.MNIST('./var' , download = True)

# transform = transforms.Compose([transforms.ToTensor() , transforms.Normalize(mean = (0.5,0.5,0.5) , std = (0.5,0.5,0.5))]) # Normalize take Mean and Standard Deviation for each channel

# transform = transforms.Compose([transforms.ToTensor() , transforms.Normalize(mean = (0.5,) , std = (0.5,))]) # Normalize take Mean and Standard Deviation for each channel

transform = transforms.Compose([transforms.ToTensor()]) # Normalize take Mean and Standard Deviation for each channel



train = torchvision.datasets.MNIST('./var', train = True , transform = transform)

trainloader = torch.utils.data.DataLoader(train , batch_size = 32 , shuffle = True)

test = torchvision.datasets.MNIST('./var', train = False , transform = transform)

testloader = torch.utils.data.DataLoader(test , batch_size = len(test) , shuffle = True)



for inputs , outputs  in trainloader:

    image = inputs[0][0]

    plt.imshow(image.numpy() , cmap = plt.get_cmap('binary'))

    break

class Net(torch.nn.Module):

    def __init__(self):

        super(Net , self).__init__()

        self.conv1 = torch.nn.Conv2d(1,3,3) # Input Channel ,Output Channel , Kernel Size

        self.pool = torch.nn.MaxPool2d(2,2) # Kernel Size , Stride

        self.conv2 = torch.nn.Conv2d(3,6,3)

        self.fc1 = torch.nn.Linear(150,128)

        self.fc2 = torch.nn.Linear(128,128)

        self.fc3 = torch.nn.Linear(128,10)

    

    def forward(self , x):

        x = self.pool(torch.nn.functional.relu(self.conv1(x)))

        self.after_con1 = x

        x = self.pool(torch.nn.functional.relu(self.conv2(x)))

        self.after_con2 = x

        x = x.flatten(start_dim = 1)

        x = torch.nn.functional.relu(self.fc1(x))

        x = torch.nn.functional.relu(self.fc2(x))

        x = self.fc3(x)

        return x

net_cnn = Net()

loss_functional = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net_cnn.parameters())



for epoch in range(3):

    for inputs, outputs in trainloader:

        optimizer.zero_grad()

        results = net_cnn(inputs)

        loss = loss_functional(results, outputs)

        loss.backward()

        optimizer.step()

    print("Loss for epoch {} is {}".format(epoch , loss))

    

for inputs , actual in testloader:

    results = net_cnn(inputs).argmax(dim = 1).numpy()

    accuracy = accuracy_score(actual , results)

    print(accuracy)

print(classification_report(actual , results))

    
for inputs , outputs  in trainloader:

    figure = plt.figure()

    image = inputs[0][0]

    plt.subplot(3,6,1)

    plt.imshow(image.numpy() , cmap = plt.get_cmap('binary'))

    output = net_cnn(inputs)

    

    filter_one = net_cnn.after_con1[0].detach()

    for i in range(3):

        plt.subplot(3,6,6+1+i)

        plt.imshow(filter_one[i].numpy() , cmap = plt.get_cmap('binary'))

    filter_two = net_cnn.after_con2[0].detach()

    # detach remove all gradients and return as simple numpy 

    for i in range(6):

        plt.subplot(3,6,12+1+i)

        plt.imshow(filter_two[i].numpy() , cmap = plt.get_cmap('binary'))

    plt.show()

    break
class SlimAlexNet(torch.nn.Module):

    def __init__(self , num_classes = 10):

        super().__init__()

        self.features = torch.nn.Sequential(

            torch.nn.Conv2d(1 , 32 , kernel_size = 3 , stride = 1),

            torch.nn.ReLU(inplace = True),

            torch.nn.MaxPool2d(kernel_size = 3, stride = 2), 

            torch.nn.Conv2d(32 , 64 , kernel_size = 3 ),

            torch.nn.ReLU(inplace = True),

            torch.nn.MaxPool2d(kernel_size = 3, stride = 2), 

            torch.nn.Conv2d(64 , 128 , kernel_size = 3 , padding = 1),

            torch.nn.ReLU(inplace = True),

            torch.nn.Conv2d(128 , 256 , kernel_size = 3 , padding = 1),

            torch.nn.ReLU(inplace = True),

            torch.nn.Conv2d(256 , 128 , kernel_size = 3 , padding = 1),

            torch.nn.ReLU(inplace = True),

            torch.nn.MaxPool2d(kernel_size = 3, stride = 2),

        )

        self.classifier = torch.nn.Sequential(

            torch.nn.Dropout(),

            torch.nn.Linear(128 , 1024),

            torch.nn.ReLU(inplace = True),

            torch.nn.Dropout(),

            torch.nn.Linear(1024 , 1024),

            torch.nn.ReLU(inplace = True),

            torch.nn.Linear(1024 , num_classes),

        )

        

    def forward(self, x):

        x = self.features(x)

        x = x.flatten(start_dim = 1)

        x = self.classifier(x)

        return x



workers = int(os.cpu_count())



transform  = transforms.Compose([transforms.ToTensor()])

mnist = torchvision.datasets.MNIST('./var' , download = True)

train = torchvision.datasets.MNIST('./var' , train = True , transform = transform)

trainloader = torch.utils.data.DataLoader(train , batch_size = 32 , shuffle = True)

test = torchvision.datasets.MNIST('./var' , train = False , transform = transform)

testloader = torch.utils.data.DataLoader(test , batch_size = len(test) , shuffle = True )



net_alex = SlimAlexNet(num_classes = 10)

loss_function = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net_alex.parameters())



if torch.cuda.is_available():

    device = torch.device('cuda')

    not_device = torch.device('cpu')

else:

    device = torch.device('cpu')

    not_device = torch.device('cuda')

    

net_alex.to(device)

for epoch in range(3):

    total_loss = 0

    for inputs, outputs in trainloader:

        inputs = inputs.to(device)

        outputs = outputs.to(device)

        optimizer.zero_grad()

        results = net_alex(inputs)

        loss = loss_functional(results, outputs)

        total_loss +=loss.item()

        loss.backward()

        optimizer.step()

    print("Loss for epoch {} is {}".format(epoch , total_loss/len(trainloader)))

    

for inputs , actual in testloader:

    inputs = inputs.to(device)

    results = net_alex(inputs).argmax(dim = 1).to(not_device).numpy()

    accuracy = accuracy_score(actual , results)

    print(accuracy)

print(classification_report(actual , results))
vgg = torchvision.models.vgg11_bn(pretrained = True)

transform  = transforms.Compose([transforms.Grayscale(3) , transforms.CenterCrop(224) , transforms.ToTensor()])

mnist = torchvision.datasets.MNIST('./var' , download = True)

workers = int(os.cpu_count())



train = torchvision.datasets.MNIST('./var' , train = True , transform = transform)

trainloader = torch.utils.data.DataLoader(train , batch_size = 32 , shuffle = True , num_workers = workers)



test = torchvision.datasets.MNIST('./var' , train = False , transform = transform)

testloader = torch.utils.data.DataLoader(test , batch_size = 32 , shuffle = True , num_workers = workers)



for inputs , outputs in trainloader:

    image = inputs[0][0]

    plt.imshow(image.numpy() , cmap=  plt.get_cmap('binary'))

    break

    

# Changing the output from 1000 to 10

vgg.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=10, bias=True)

# display(vgg.classifier) # uncomment to Visualize VGG CLassifiers layers

if torch.cuda.is_available():

    device = torch.device('cuda')

    not_device = torch.device('cpu')

else:

    device = torch.device('cpu')

    not_device = torch.device('cuda')

    

loss_function = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(vgg.parameters())



vgg.to(device)

vgg.train()



for epoch in range(3):

    for inputs , outputs in trainloader:

        inputs = inputs.to(device , non_blocking = True)

        outputs = outputs.to(device , non_blocking = True)

        optimizer.zero_grad()

        results = vgg(inputs)

        loss = loss_function(results , outputs)

        loss.backward()

        optimizer.step()

    print("Last Loss : {0}".format(loss))
results_buffer = []

actual_buffer = []

with torch.no_grad(): # no_grad will set all the Gradients to Zero

    vgg.eval()

    for inputs, outputs in testloader:

        inputs = inputs.to(device , non_blocking = True)

        results = vgg(inputs).argmax(dim = 1).to('cpu')

        results_buffer.append(results)

        actual_buffer.append(outputs)

print(classification_report(np.concatenate(actual_buffer) , np.concatenate(results_buffer)))
model = torchvision.models.resnet18(pretrained = True)

print(model.fc)



model.fc = torch.nn.Linear(model.fc.in_features , 10)

loss_function = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters())



model.to(device)

model.train() # To reactivate all the layes to training mode..

for epoch in range(3):

    for inputs , outputs in trainloader:

        inputs = inputs.to(device , non_blocking = True)

        outputs = outputs.to(device , non_blocking = True)

        optimizer.zero_grad()

        results = model(inputs)

        loss = loss_function(results , outputs)

        loss.backward()

        optimizer.step()

    print("Last Loss : {0}".format(loss))
results_buffer = []

actual_buffer = []

with torch.no_grad(): # no_grad will set all the Gradients to Zero

    model.eval()

    for inputs, outputs in testloader:

        inputs = inputs.to(device , non_blocking = True)

        results = model(inputs).argmax(dim = 1).to('cpu')

        results_buffer.append(results)

        actual_buffer.append(outputs)

print(classification_report(np.concatenate(actual_buffer) , np.concatenate(results_buffer)))
class SentimentDataset(Dataset):

    def __init__(self):

        self.data = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip' , sep= '\t' , header = 0).groupby('SentenceId').first()

        self.ordinals = {}

        for sample in tqdm(self.data.Phrase):

            for token in nlp(sample.lower() , disable = ['parser' , 'tagger' , 'ner']): # https://spacy.io/usage/processing-pipelines

                if token.text not in self.ordinals:

                    self.ordinals[token.text] = len(self.ordinals) # Tokenizing each word with a number

    def __len__(self):

        return len(self.data)

    def __getitem__(self , idx):

        if type(idx) is torch.Tensor:

            idx = idx.item()

        sample = self.data.iloc[idx]

        bag_of_words = torch.zeros(len(self.ordinals))

        for token in nlp(sample.Phrase.lower() , disable = ['parser' , 'tagger' , 'ner']):

            bag_of_words[self.ordinals[token.text]] += 1

        return bag_of_words , torch.tensor(sample.Sentiment)



class Model(torch.nn.Module):

    def __init__(self, input_dimensions , size = 128 , output_shape = 5):

        super().__init__()

        self.layer_one = torch.nn.Linear(input_dimensions , size)

        self.activation_one = torch.nn.ReLU()

        self.layer_two = torch.nn.Linear(size , size)

        self.activation_two = torch.nn.ReLU()

        self.shape_outputs = torch.nn.Linear(size , output_shape)

    

    def forward(self, inputs):

        buffer = self.layer_one(inputs)

        buffer = self.activation_one(buffer)

        buffer = self.layer_two(buffer)

        buffer = self.activation_two(buffer)

        buffer = self.shape_outputs(buffer)

        return buffer
sentiment = SentimentDataset() # Initalizing the Dataset



number_of_testing = int(len(sentiment)*0.05) 

number_of_training = len(sentiment) - number_of_testing

train , test = torch.utils.data.random_split(sentiment , [number_of_training , number_of_testing])



# Creating the DataLoader

trainloader = torch.utils.data.DataLoader(train , batch_size = 32 , shuffle = True)

testloader = torch.utils.data.DataLoader(test , batch_size = 32 , shuffle = True)



# Creating the Model

model = Model(len(sentiment.ordinals)) # 15354 Unique Words for Model

# Training the Model

optimizer = torch.optim.Adam(model.parameters())

loss_functions = torch.nn.CrossEntropyLoss()



model.train() # Setting the model to Train



for epoch in range(15):

    losses = []

    for inputs , outputs in tqdm(trainloader):

        optimizer.zero_grad()

        results = model(inputs)

        loss = loss_functions(results , outputs)

        losses.append(loss.item())

        loss.backward()

        optimizer.step()

    print("Loss {0}".format(torch.tensor(losses).mean()))

    
results_buffer = []

actual_buffer = []

with torch.no_grad():

    model.eval()

    for inputs, outputs in testloader:

        results = model(inputs).argmax(dim = 1).numpy()

        results_buffer.append(results)

        actual_buffer.append(outputs)



print(classification_report(np.concatenate(actual_buffer) , np.concatenate(results_buffer)))
nlp_embedding = spacy.load('en_core_web_lg') # 300 Vector of Words
class SentimentDataset(Dataset):

    def __init__(self):

        self.data = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip' , sep= '\t' , header = 0).groupby('SentenceId').first()

    def __len__(self):

        return len(self.data)

    def __getitem__(self , idx):

        if type(idx) is torch.Tensor:

            idx = idx.item()

        sample = self.data.iloc[idx]

        token_vector = []

        for token in nlp(sample.Phrase.lower(), disable = ['ner']):

            token_vector.append(token.vector)

        return (torch.tensor(token_vector) , torch.tensor(len(token_vector)) , torch.tensor(sample.Sentiment))

class Model(torch.nn.Module):

    def __init__(self,input_dimensions , size = 128 , layer = 1 , output_shape = 5):

        super().__init__()

        self.seq = torch.nn.LSTM(input_dimensions , size , layer)

        self.layer_one = torch.nn.Linear(size*layer , size)

        self.activation_one = torch.nn.ReLU()

        self.layer_two = torch.nn.Linear(size , size)

        self.activation_two = torch.nn.ReLU()

        self.shape_outputs = torch.nn.Linear(size , output_shape)

        

    def forward(self, inputs , lenghts):

        number_of_batches = lenghts.shape[0]

        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs , lenghts , batch_first=True)

        buffer , (hidden , cell) = self.seq(packed_inputs)

        buffer = hidden.permute(1,0,2) # Reshaping

        buffer = buffer.contiguous().view(number_of_batches , -1)

        buffer = self.layer_one(buffer)

        buffer = self.activation_one(buffer)

        buffer = self.layer_two(buffer)

        buffer = self.activation_two(buffer)

        buffer = self.shape_outputs(buffer)

        return buffer

    

def collate(batch):

    batch.sort(key = lambda x:x[1] , reverse = True) # sort by decresing lenths

    sequences , lengths , sentiment = zip(*batch)

    sequences = torch.nn.utils.rnn.pad_sequence(sequences , batch_first = True) # Padding the Sequence with Max Length

    sentiment = torch.stack(sentiment)

    lengths = torch.stack(lengths)

    return sequences , lengths , sentiment

sentiment = SentimentDataset()



number_of_testing = int(len(sentiment)*0.05) 

number_of_training = len(sentiment) - number_of_testing

train , test = torch.utils.data.random_split(sentiment , [number_of_training , number_of_testing])



# Creating the DataLoader

trainloader = torch.utils.data.DataLoader(train , batch_size = 32 , shuffle = True , collate_fn = collate)

testloader = torch.utils.data.DataLoader(test , batch_size = 32 , shuffle = True , collate_fn = collate)

for batch in trainloader:

    print(batch[0].shape , batch[1].shape , batch[2].shape)

    print(batch[1][0])

    break

    

model = Model(sentiment[0][0].shape[1])

optimizer = torch.optim.Adam(model.parameters())

loss_functions = torch.nn.CrossEntropyLoss()



model.train() # Setting the model to Train



for epoch in range(15):

    losses = []

    for sequences , lengths , outputs in tqdm(trainloader):

        optimizer.zero_grad()

        results = model(sequences , lengths)

        loss = loss_functions(results , outputs)

        losses.append(loss.item())

        loss.backward()

        optimizer.step()

    print("Loss {0}".format(torch.tensor(losses).mean()))
results_buffer = []

actual_buffer = []

with torch.no_grad():

    model.eval()

    for inputs , lenghts, outputs in testloader:

        results = model(inputs,lenghts).argmax(dim = 1).numpy()

        results_buffer.append(results)

        actual_buffer.append(outputs)



print(classification_report(np.concatenate(actual_buffer) , np.concatenate(results_buffer)))
class Generator(torch.nn.Module):

    def __init__(self , context , features , channels):

        super().__init__()

        self.main = torch.nn.Sequential(

            torch.nn.ConvTranspose2d(in_channels = context, out_channels=features*8, kernel_size = 4, stride=1, padding=0 ,bias=False), #https://pytorch.org/docs/stable/nn.html#convtranspose2d

            torch.nn.BatchNorm2d(features*8),

            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(in_channels = features *8 , out_channels = features*4 , kernel_size = 4 , stride = 2 , padding = 1 , bias = True),

            torch.nn.BatchNorm2d(features*4),

            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(in_channels = features *4 , out_channels = features*2 , kernel_size = 4 , stride = 2 , padding = 1 , bias = True),

            torch.nn.BatchNorm2d(features*2),

            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(in_channels = features *2 , out_channels = features , kernel_size = 4 , stride = 2 , padding = 1 , bias = True),

            torch.nn.BatchNorm2d(features),

            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(in_channels = features , out_channels = channels , kernel_size = 4 , stride = 2 , padding = 1 , bias = True),

            torch.nn.Tanh()

        )

    def forward(self , input):

        return self.main(input)



class Discriminator(torch.nn.Module):

    def __init__(self , features , channels):

        super().__init__()

        self.main = torch.nn.Sequential(

            torch.nn.Conv2d(channels , features , 4, 2, 1 , bias = False),

            torch.nn.LeakyReLU(0.2 , inplace = True),

            torch.nn.Conv2d(features , features*2 , 4, 2, 1 , bias = False),

            torch.nn.BatchNorm2d(features*2),

            torch.nn.LeakyReLU(0.2 , inplace = True),

            torch.nn.Conv2d(features*2 , features*4 , 4, 2, 1 , bias = False),

            torch.nn.BatchNorm2d(features*4),

            torch.nn.LeakyReLU(0.2 , inplace = True),

            torch.nn.Conv2d(features*4 , features*8 , 4, 2, 1 , bias = False),

            torch.nn.BatchNorm2d(features*8),

            torch.nn.LeakyReLU(0.2 , inplace = True),

            torch.nn.Conv2d(features*8 , 1 , 4, 1, 0 , bias = False), # It will give output as 0 or 1

            torch.nn.Sigmoid()

        )

    def forward(self , inputs):

        return self.main(inputs)



def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        torch.nn.init.normal_(m.weights.data , 0.0 , 0.2)

    elif classname.find('Conv') != -1:

        torch.nn.init.normal_(m.weights.data , 0.0 , 0.2)

        torch.nn.init.constant_(m.bias.data , 0)



class GeneratorCustome(Generator):

    def __init__(self , context , features , channels):

        super.__init__(context , features , channels)

        weights_init(self)

        

class DiscriminatorCustome(Discriminator):

    def __init__(self , features , channels):

        super.__init__(features , channels)

        weights_init(self)
batch_size = 128

transform = transforms.Compose([transforms.CenterCrop(64) , transforms.ToTensor(), transforms.Normalize(mean = (0.5,) , std = (0.5,))])

mnist = torchvision.datasets.MNIST('./var' , download = True)

real = torchvision.datasets.MNIST('./var' , train = True , transform = transform)

realloader = torch.utils.data.DataLoader(real , batch_size = batch_size , shuffle = True)



if torch.cuda.is_available():

    device = torch.device('cuda')

else:

    device = torch.device('cpu')

    

epochs = 16

# Real or Fake

real_label = 1

fake_label = 0



context_size = 10

features = 32

channels = 1

# Learning rate and beat1 for Optimizers

lr = 0.0002

beta1 = 0.5



# Bianry Loss Such as Real or Fake

criteria = torch.nn.BCELoss()



# Random Number Generator

fixed_noise = torch.randn(features , context_size , 1,1 , device = device)



# List to Track Progress

img_list = []

G_losses = []

D_losses = []



netD = Discriminator(features , channels).to(device)

netG = Generator(context_size , features, channels).to(device)

# Adam Optmiziers

optimizerD = torch.optim.Adam(netD.parameters() , lr = lr , betas = (beta1 , 0.999))

optimizerG = torch.optim.Adam(netG.parameters() , lr = lr , betas = (beta1 , 0.999))

fake = netG(fixed_noise).detach().cpu()

samples = torchvision.utils.make_grid(fake , padding = 2 , normalize = True)

plt.axes().imshow(samples.permute(1,2,0))
for epoch in range(epochs):

    with tqdm(realloader , unit = 'batches') as progress:

        for i , (data , _) in enumerate(realloader):

            netD.zero_grad()

            # Preparing Data

            batch_size = data.shape[0]

            real_data = data.to(device)

            real_labels = torch.full((batch_size ,) , real_label , device  = device)

            output = netD(real_data).view(-1)

            # Loss Function for Real or Fake

            errD_real = criteria(output , real_labels)

            errD_real.backward()

            # Creating Noise and Fake Labels 

            noise = torch.randn(batch_size , context_size , 1,1,device = device)

            fake_labels = torch.full((batch_size ,) , fake_label ,device = device)

            fake_data = netG(noise)

            output = netD(fake_data).view(-1)

            errD_fake = criteria(output , fake_labels)

            errD_fake.backward(retain_graph = True)

            errD = errD_fake + errD_real

            optimizerD.step()

            netG.zero_grad()

            # How well is Discriminator is working

            fake_labels.fill_(real_label)

            output = netD(fake_data).view(-1)

            errG = criteria(output , real_labels)

            errG.backward()

            # Update Optimizer

            optimizerG.step()

            # Saving Losses

            G_losses.append(errG.item())

            D_losses.append(errD.item())

            progress.set_postfix(G_loss = torch.tensor(G_losses).mean(),

                                D_loss = torch.tensor(D_losses).mean(),

                                refresh = False)

            progress.update()

            G_losses.append(errG.item())

            D_losses.append(errD.item())

        with torch.no_grad():

            fake= netG(fixed_noise).detach().cpu()

        samples = torchvision.utils.make_grid(fake , padding = 2 , normalize = True)

        img_list.append(samples)

        

ims = plt.axes().imshow(img_list[0].permute(1,2,0))
ims = plt.axes().imshow(img_list[-1].permute(1,2,0))