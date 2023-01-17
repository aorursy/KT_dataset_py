
import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input/processed-creditdata"))

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
# pd.read_csv("../input/creditcard.csv")
# print(data.tail())
# data["Time+!"] = data["Time"].shift(1)
# data = data.fillna(0)
# import numpy as np
# data["label_new"] = ""
# data["Time_fixed"] = ""
# for index in range(data.shape[0]):
#     data["Time_fixed"][index] = data["Time"][index] - data["Time+!"][index]
#     if data["Class"][index]==0:
#         data["label_new"][index] = np.array([1,0])
#     else:
#         data["label_new"][index] = np.array([0,1])
# data.to_csv("processed_creditdata.csv")
data = pd.read_csv("../input/processed-creditdata/processed_creditdata.csv")
del data["Time"], data["Time+!"], data["Class"], data["Unnamed: 0"]
print(data.tail())
# data.to_csv("processed_creditdata.csv")
label = data["label_new"]
del data["label_new"]
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
data = pd.DataFrame(data_scaled)
train_x, test_x, train_y, test_y = train_test_split(data, label, test_size = 0.2, shuffle=False)
print(train_x.shape)
a = {"0": [], "1": []}
for i in range(train_y.size):
    if train_y[i] == '[1 0]':
        a["0"].append(1)
        a["1"].append(0)
    else:
        a["0"].append(0)
        a["1"].append(1)
label_df = pd.DataFrame(a)
print(label_df.shape)
torch_tensor = torch.tensor(train_x.values)
torch_tensor_label = torch.tensor(label_df.values)
torch_tensor = torch_tensor.float()
torch_tensor_label = torch_tensor_label.float()
class Model(nn.Module):
    def __init__(self, in_dim):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = Model(30)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
def acc(actual, pred):    
    count = 0
    for i in range(pred.shape[0]):
        if pred[i][0] > 0.7 and actual[i][0] > 0.7:
            count += 1
        elif pred[i][1] > 0.7 and actual[i][1] > 0.7:
            count +=1
        else:
            pass
    return count/pred.shape[0]
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(torch_tensor)
    loss = criterion(outputs, torch_tensor_label )
    loss.backward()
    optimizer.step()
    
    accuracy = acc(torch_tensor_label, outputs)
    if epoch%5 == 0:
        print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch+1,50, loss.data[0], accuracy))
test_tensor = torch.tensor(test_x.values).float()
test_y = test_y.reset_index(drop=True)

a = {"0": [], "1": []}
for i in range(test_y.size):
    if test_y[i] == '[1 0]':
        a["0"].append(1)
        a["1"].append(0)
    else:
        a["0"].append(0)
        a["1"].append(1)
test_y = pd.DataFrame(a)
print(test_y.shape)
tensor_test_y = torch.tensor(test_y.values).float()
pred = model(test_tensor)
accuracy = acc(tensor_test_y, pred)
print(accuracy)