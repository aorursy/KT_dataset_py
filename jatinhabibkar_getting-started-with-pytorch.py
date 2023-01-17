import torch 

import numpy as np

import pandas as pd

from torchvision import datasets

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import torch.nn as nn

from torch.autograd import Variable

from sklearn.model_selection import train_test_split

%matplotlib inline
dataset=pd.read_csv("../input/digit-recognizer/train.csv")
dataset
test_data=pd.read_csv("../input/digit-recognizer/test.csv")
targets_numpy = dataset.label.values
feature_numpy=dataset.loc[:,dataset.columns != "label"].values/255
X_train, X_test, y_train, y_test = train_test_split(feature_numpy, targets_numpy, test_size=0.2, random_state=42)
featuresTrain=torch.from_numpy(X_train)

targetsTrain=torch.from_numpy(y_train).type(torch.LongTensor)
featuresTest=torch.from_numpy(X_test)

targetsTest=torch.from_numpy(y_test).type(torch.LongTensor)
# batch_size, epoch and iteration

batch_size = 100

n_iters = 10000

num_epochs = n_iters / (len(X_train) / batch_size)

num_epochs = int(num_epochs)
train=torch.utils.data.TensorDataset(featuresTrain,targetsTrain)

test=torch.utils.data.TensorDataset(featuresTest,targetsTest)





#data loader

trainloader=torch.utils.data.DataLoader(train,batch_size=batch_size,shuffle=False)

testloader=torch.utils.data.DataLoader(test,batch_size=batch_size,shuffle=False)
plt.imshow(feature_numpy[10].reshape(28,28))

plt.axis("off")

plt.title(targets_numpy[10])

plt.show()
import torch.nn as nn

import torch.nn.functional as F



# Create Logistic Regression Model

class Net(nn.Module):

    def __init__(self):

        super(Net,self).__init__()

        hidden_1=512

        hidden_2=512

        

        self.fc1=nn.Linear(28*28,hidden_1)

        self.fc2=nn.Linear(hidden_1,hidden_2)

        self.fc3=nn.Linear(hidden_2,10)

        self.dropout=nn.Dropout(0.2)

        

    def forward(self, x):

        x = F.relu(self.fc1(x))

        

        x = self.dropout(x)

        

        x = F.relu(self.fc2(x))

        

        x = self.dropout(x)

        

        x = self.fc3(x)

        

        return x

# Instantiate Model Class

input_dim = 28*28 # size of image px*px

output_dim = 10  # labels 0,1,2,3,4,5,6,7,8,9



# create logistic regression model

model = Net()
if torch.cuda.is_available():

    model.cuda()

    print("cuda is available")
error =nn.CrossEntropyLoss()



optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
for i,(images,label) in enumerate(trainloader):

    train=Variable(images.view(-1,28*28))

    print(train.float())

    print(train.shape)

    break
n_epochs = 50
count = 0

step=0

loss_list = []

iteration_list = []



for epoch in range(n_epochs):

    train_loss=0.0

    model.train()

    for i,(images,labels) in enumerate(trainloader):

        if torch.cuda.is_available():

            images,labels=images.cuda(),labels.cuda()

        

        train=Variable(images.view(-1,28*28))

        labels=Variable(labels)

        

        

        optimizer.zero_grad()

        

        output=model(train.float())

        

        loss=error(output,labels)

        loss.backward()

        

        optimizer.step()

        

        count+=1

            

        train_loss +=loss.item()*images.size(0)



        if count% 50==0:

            #calculate accuracy

            correct=0

            total=0

            for images,labels in testloader:

                if torch.cuda.is_available():

                    images,labels=images.cuda(),labels.cuda()

                test=Variable(images.view(-1,28*28))

                outputs = model(test.float())

                

                predicted = torch.max(outputs.data, 1)[1]

                

                total +=len(labels)

                

                correct +=(predicted==labels).sum()

            accuracy =100 *correct/float(total)

            loss_list.append(loss.data)

            iteration_list.append(count)

        if count % 500 == 0:

            print('Iteration: {}  Loss: {}  Accuracy: {}%'.format(count, loss.data, accuracy))

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))

#loss:4414.322540 acc:95%

for i,(images,labels) in enumerate(trainloader):

    images=images.numpy()

    print(images.shape)

    break
# visualization

plt.plot(iteration_list,loss_list)

plt.xlabel("Number of iteration")

plt.ylabel("Loss")

plt.title("Logistic Regression: Loss vs Number of iteration")

plt.show()
torch.save(model.state_dict(),"../working/model.pt")
model1 = Net()

model1.load_state_dict(torch.load("../working/model.pt"))
batch_size=100

features_numpy_final= test_data.values/255



finaltest=torch.from_numpy(features_numpy_final)



final_test=torch.utils.data.TensorDataset(finaltest)

final_loader=torch.utils.data.DataLoader(finaltest,batch_size=batch_size,shuffle=False)
plt.imshow(features_numpy_final[3].reshape(28,28))

plt.axis("off")

plt.savefig('../working/graph.png')

plt.show()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

list_predict_to_append=[]

list_of_number=[]

for i,(images) in enumerate(final_loader,start=1):

    images = images.to(device)

    testoutput=Variable(images.view(-1,28*28))

    if torch.cuda.is_available():

        outputs = model1(testoutput.cuda().float().cpu())

    else:

        outputs = model1(testoutput.float())

    predicted =torch.max(outputs.data,1)[1]

    list_predict_to_append.extend(predicted.tolist())

    list_of_number.append(i)
len(list_predict_to_append)
list_of_number=np.arange(1,28001).tolist()

print(len(list_of_number))
data={'ImageId':list_of_number,'Label':list_predict_to_append}
new_prediction=pd.DataFrame(data,columns=['ImageId','Label'])
new_prediction