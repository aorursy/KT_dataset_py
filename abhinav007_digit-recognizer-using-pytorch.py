from torchvision.models.resnet import ResNet, BasicBlock
from tqdm.autonotebook import tqdm
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch.autograd import Variable
import seaborn as sns
import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv", dtype = np.float32)
pd_train.tail(10)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(pd_train["label"])
y_train = pd_train.label.values
x_train = pd_train.loc[:,pd_train.columns != "label"].values/255

#splitting the Data 80/20
x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                    y_train,
                                                    test_size = 0.2,
                                                    random_state = 42) 
# converting to tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

#Binding the tests together
train = torch.utils.data.TensorDataset(x_train,y_train)
test = torch.utils.data.TensorDataset(x_test,y_test)

batch_size = 100
train_loader = DataLoader(train, batch_size=batch_size , shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size , shuffle=True)

plt.imshow(x_train[80].reshape(28,28))
x_train.reshape(-1,1,28,28).shape
class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, 
                            kernel_size=(7, 7), 
                            stride=(2, 2), 
                            padding=(3, 3), bias=False)
# For Training on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MnistResNet()
model = model.to(device)
optimizer = optim.Adagrad(model.parameters(),lr=0.01)

loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
loss_list = []
iterate_list = []
accuracy_list = []
epochs = 40
count = 0 

for a in tqdm(range(epochs)):
    for i , (images, labels) in enumerate(train_loader):
        
        X, y = Variable(images.view(100,1,28,28)).to(device) , Variable(labels).to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_function(outputs,y)
        loss.backward()
        optimizer.step()
        
        count +=1
        if count %50 == 0:
            correct = 0
            total = 0
            with torch.no_grad():
                for images ,labels in test_loader:
                    test, labels = Variable(images.view(100,1,28,28)).to(device) , labels.to(device) 

                    outputs = model(test)
                    predicted = torch.max(outputs.data, 1)[1]
                    total += len(labels)
                    correct += (predicted == labels).sum()
                accuracy = 100 * correct / float(total)
            
            loss_list.append(loss.data)
            iterate_list.append(count)
            accuracy_list.append(accuracy)
        if count % 200 == 0:
            print("Epoch: {} ".format(a))
            print('Iteration: {} ||  Loss: {} || Accuracy: {}%'.format(count, loss.data, accuracy))

# visualization loss 
plt.figure(figsize=[20,11])
plt.plot(iterate_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("CNN: Loss vs Number of iteration")
plt.show()
# visualization accuracy 
plt.figure(figsize=[20,11])
plt.plot(iterate_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of iteration")
plt.show()
pd_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv",dtype = np.float32)
pd_sample_submission = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

testing = pd_test.loc[:,:].values/255
test = torch.from_numpy(testing)
test = test.to(device)

submit = model(test.view(-1,1,28,28))
pd_sample_submission["Label"] = torch.max(submit.data.cpu(),1).indices
pd_sample_submission.to_csv("submission.csv", index=False)