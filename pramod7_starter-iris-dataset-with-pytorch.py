import torch as tr
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as f
from sklearn.model_selection import train_test_split
import pandas as pd

#load the iris data
df=pd.read_csv('../input/iris.csv')
# read the columns
print (df.columns)

# separate them into features and labels
x=df.drop('target',axis=1).values
y=df['target'].values

# split them into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

#convert x_train and x_test into float tensors.  y_train and y_test into int
x_train=tr.FloatTensor(x_train)
x_test=tr.FloatTensor(x_test)
y_train=tr.LongTensor(y_train)
y_test=tr.LongTensor(y_test)


# build NN tensors model using python class objects, three hidden layers and fully connected by using linear model
class model (nn.Module):
    def __init__(self,in_feat=4,h1=12,h2=12,h3=9,out_feat=3):
        super().__init__()
        self.layr1=nn.Linear(in_feat,h1)
        self.layr2=nn.Linear(h1,h2)
        self.layr3=nn.Linear(h2,h3)
        self.outlayr=nn.Linear(h3,out_feat)
        
    def forward_pro(self,inpts):
        out1=f.relu(self.layr1(inpts)) #output of 1st layer
        out2=f.relu(self.layr2(out1))
        out3=f.relu(self.layr3(out2))
        outpt=self.outlayr(out3)
        return outpt
# data is ready, model is ready. Now lets prepare it to be trained with Epochs
epochs=150
loss_list=[]
algo=model()

#creating criterion for finding the loss values between predicted values and y train values
criterion = nn.CrossEntropyLoss()
optimizer = tr.optim.Adam(algo.parameters(), lr=0.1)

for e in range(epochs):
    e=e+1
    training_pred=algo.forward_pro(x_train)
    loss=criterion(training_pred,y_train)
    loss_list.append(loss)
    
    #below is to print the loss values for each epochs
    if e%5==1:
        print(f'epoch: {e:2}  loss: {loss.item():10.4f}')        
    
    #reset the optimizer    
    optimizer.zero_grad() # reset grad
    loss.backward()
    optimizer.step()
#plot the loss list
plt.plot(range(epochs), loss_list)
plt.ylabel('Loss')
plt.xlabel('epoch')
# we are testing the model to check the loss against test data, Here we do not require back propagation
with tr.no_grad():
    y_pred_loss=algo.forward_pro(x_test)
    val_loss=criterion(y_pred_loss, y_test)
    print (f'{val_loss:.4f}')
    
#let us validate the predicated classes against the actual class
correct=0
with tr.no_grad():
    for i,cls in enumerate (x_test):
        y_pred=algo.forward_pro(cls)
        print(f'{i+1:2}  {str(y_pred):38} {y_test[i]}')
        if y_pred.argmax().item()==y_test[i]:
            correct=correct+1
    print(correct)
tr.save(algo.state_dict(), 'IrisDatasetModel.pt')
new_model = model()
new_model.load_state_dict(tr.load('IrisDatasetModel.pt'))
new_model.eval()

with tr.no_grad():
    y = new_model.forward_pro(x_test)
    loss = criterion(y, y_test)
print(f'{loss:.8f}')
