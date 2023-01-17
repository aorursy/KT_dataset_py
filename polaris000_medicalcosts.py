import pandas as pd

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

from torch import optim

import seaborn as sns
df = pd.read_csv('../input/insurance/insurance.csv')
print("shape: ", df.shape)

print('--------')

print(df.info())
df.head()
sns.pairplot(df);
sns.heatmap(df.corr());
sns.distplot(df['charges']);
sns.scatterplot(x='age', y='charges',data=df, )
df.isnull().sum()
# smoker encoding

df['smoker'] = [1 if a == 'yes' else 0 for a in df['smoker']]



# region encoding

regions = {

    'northeast': [0, 0],

    'northwest': [0, 1],

    'southeast': [1, 0],

    'southwest': [1, 1]

    }



df['dir1'] = 0

df['dir2'] = 0

df[['dir1', 'dir2']] = [regions[dir] for dir in df['region']]

df = df.drop('region', axis=1)



# sex encoding

df['sex'] = [1 if sex == 'male' else 0 for sex in df['sex']]

    

df.head()
# normalization

target = df['charges'].copy()



for i in df.columns:

    df[i] = (df[i] - min(df[i])) / (max(df[i] - min(df[i])))

df.head()

print(target.head())
# training data

X_train = torch.tensor((df.drop('charges', axis=1).iloc[:1100]).values.astype(pd.np.float32))

y_train = torch.tensor((df['charges'].iloc[:1100]).values.astype(pd.np.float32))



y_train = y_train.reshape(-1, 1)

# model (using pytorch abstractions like Autograd)



train_tensor = torch.utils.data.TensorDataset(X_train, y_train)

trainloader = torch.utils.data.DataLoader(train_tensor, batch_size=10)

model = nn.Sequential(nn.Linear(7, 1))



criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)



losses = []



epochs = 1000

for e in range(epochs):

    running_loss = 0

    main_outs = []

    for data, targets in trainloader:

   

        optimizer.zero_grad()

        outs = model.forward(data)



        main_outs.extend(outs)

        loss = criterion(outs, targets)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

    else:

        print(f"Training loss: {running_loss/len(trainloader)}")

        losses.append(running_loss/len(trainloader))



plt.style.use('seaborn')

plt.plot(losses);
preds = model(X_train) * (max(target) - min(target)) + min(target)

sns.regplot(y_test.squeeze(),denorm.squeeze());
# testing

X_test = torch.tensor((df.drop('charges', axis=1).iloc[1100:]).values.astype(pd.np.float32))

y_test = torch.tensor((df['charges'].iloc[1100:]).values.astype(pd.np.float32))



y_test = y_test.reshape(-1, 1)
# denormalize test predictions

preds = model(X_test).clone().detach()



denorm = preds * (max(target) - min(target)) + min(target)
sns.regplot(y_test.squeeze(),denorm.squeeze());
target_tensor = torch.tensor(target.iloc[1100:].values.astype(pd.np.float32))

target_tensor = target_tensor.reshape(-1, 1)

SSE = pd.np.sqrt(sum((target_tensor - denorm)**2))

SST = pd.np.sqrt(sum((denorm - denorm.mean())**2))
(1- SSE/SST)
# for i in range(25):

#   nn_.losses.append(torch.mean((y - nn_(X))**2).detach().item())

#   print("#" + str(i) + " Loss: " + str(torch.mean((y - nn_(X))**2).detach().item()))



#   nn_.train(X, y)

#   print(nn_.W1, nn_.W2)
# model



# class NN (nn.Module):

#   def __init__(self):

#     super(NN, self).__init__()



#     self.inpsize = 7

#     self.otpsize = 1

#     self.hdnsize = 5



#     self.losses = []

#     self.outputs = []



#     self.W1 = torch.Tensor(self.inpsize, self.hdnsize) # 7 x 5

#     self.W2 = torch.Tensor(self.hdnsize, self.otpsize) # 5 x 1

#     self.W1.fill_(1000)

#     self.W2.fill_(1000)

#     self.b1 = torch.Tensor()



#   def train(self, X, y):

#     # forward + backward pass for training

#     o = self.forward(X)

#     self.outputs = o

#     print("o: ", o)

#     self.backward(X, y, o)



#   def forward(self, X):

#     self.z = torch.matmul(X, self.W1)

#     self.z2 = torch.Tensor(self.z)



#     # relu on hidden layer

#     p, q = self.z.shape

#     for i in range(p):

#       for j in range(q):

#         self.z2[i][j] = self.relu(self.z[i][j])



#     self.z3 = torch.matmul(self.z2, self.W2)



#     # o = torch.Tensor(self.z3)

#     # r, s = self.z3.shape

#     # for i in range(r):

#     #   for j in range(s):

#     #     o[i][j] = self.relu(self.z3[i][j])



#     # print(o, y)

#     return self.z3



#   def sigmoid(self, s):

#     return 1 / (1 + torch.exp(-s))



#   def dsigmoid_dx(self, s):

#     return s * (1 - s)



#   def backward(self, X, y, o):

#       self.o_error = 0.5*(y - o)**2 # error in output

#       # 30000 x 1



#       # print(self.o_error.shape, o.shape, y.shape)

#       self.o_delta = torch.t(self.o_error) @ self.dsigmoid_dx(o) 

#       self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))



    

#       # .... -> 30000 x 3

#       self.z2_delta = self.z2_error * self.dsigmoid_dx(self.z2)



#       # 9 x 30000 * 30000 x 3 -> 9 x 3

#       self.W1 += ((torch.matmul(torch.t(X), self.z2_delta)) * 0.00001)

#       self.W2 += (torch.matmul(torch.t(self.z2), self.o_delta) * 0.00001)



#   def relu(self, x):

#     return max(torch.tensor(0), x)