import numpy as np 

import pandas as pd 

import time 

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.nn.functional as F 

import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

import sklearn.model_selection as skl





# from clean import *

# from control import *
data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
Y_j = "diagnosis"
dtype = torch.float

# device = torch.device("cpu")

device = torch.device("cuda:0")
data = data.drop(["Unnamed: 32", "id"], axis=1)



data.loc[data.diagnosis == "M", "diagnosis"] = 1

data.loc[data.diagnosis == "B", "diagnosis"] = 0
# Control.check_na_all_cols(data)
def norm(xij, mean_j, min_j, max_j):

    result1 = xij- mean_j

    result2  =   max_j - min_j

    return result1/result2
columns_x =list(data.columns)

columns_x.remove(Y_j )



for j in data.columns :

    data[j] = data[j].astype(np.float64)

    

for j in columns_x:

    j_max = data[j].max()

    j_min = data[j].min()

    j_mean = data[j].mean()



    data[j] = data[j].apply(lambda xi :norm(xi , j_mean, j_min , j_max))
for j in data.columns :

    data[j] = data[j].astype(np.float64)



X = np.array(data.drop(Y_j, axis=1) )

y = np.array(data[Y_j])



X = torch.from_numpy(X).float()

y = torch.from_numpy(y).float()



X= X.to(device)

y = y.to(device)

X_train, X_test, y_train, y_test = skl.train_test_split(X, y, test_size=0.3, random_state=30)





X_train.is_cuda
class Module(torch.nn.Module):

    def __init__(self):

        super( Module,  self).__init__()

        self.linear1 = torch.nn.Linear(30, 10)

        self.linear2 = torch.nn.Linear(10, 1)





    def forward(self, x):

        x = F.prelu(self.linear1(x), weight=torch.Tensor([0.2]).to(device))

        y_pred = F.prelu(self.linear2(x), weight=torch.Tensor([0.2]).to(device))

        return y_pred



%time 

# model instance

model = Module()

model.to(device)



# var

learning_rate = 1e-6

regularization_num = 2

m = y_train.size()[0]

D_in, H, D_out = X_train.shape[0], 10 , 1



# func

loss_fn = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)





start = time.time()

loss_hold = []

for epoch in range (261560):

    for i in range(m):

        optimizer.zero_grad()

        y_pre = model(X_train[i].view(-1,30))

        loss_ = loss_fn(y_pre, y_train[i].view(1,1))

        

        r = 0 

        for param in model.parameters():

            e= torch.sum(abs(param))

            r +=e



        loss= loss_+ r*(regularization_num/m)



        loss.backward()

        optimizer.step()

    loss_hold.append(loss_)

    print(loss_, epoch)

    

end = time.time()   

plt.plot(np.array(loss_hold))
true_neg = 0

true_pos = 0

false_pos = 0

false_neg = 0 

total = 0

print("Accuracy scores")

with torch.no_grad():

    for i  in range(len(X_test)):

        # Calculate Accuracy



        # Load images to a Torch Variable

        images = X_test[i].view(-1, 30)



        # Forward pass only to get logits/output

        outputs = model(images)



        # Get predictions from the maximum value

        predicted = [1 if outputs[0]> 0.42 else 0]



        

        

        # Total correct predictions

        if 0 == int(predicted[0]) == y_test[i]:

            true_neg +=1



        elif 1 == int(predicted[0]) == y_test[i]:

            true_pos +=1



        elif 1== int(predicted[0]):

            false_pos +=1



        elif 0 == int(predicted[0]):

            false_neg +=1

 



    accuracy =  (true_pos + true_neg)/  len(X_test)

    precision = true_pos/(true_pos + false_pos)

    recall = true_pos/(true_pos + false_neg)

    F1score = precision*recall*2 /(recall+precision)

    print("accuracy ", str(accuracy))

    print("precision ",str(precision))

    print("recall ",str(recall))

    print("F1score ", str(F1score) )
save_model = True

if save_model is True:

    # Saves only parameters

    torch.save(model.state_dict(), '3_model_Breast_Cancer.pkl')