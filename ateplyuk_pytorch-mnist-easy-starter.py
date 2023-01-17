import torch

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv", dtype=np.float32)

test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv", dtype=np.float32)

sample_sub = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
y = train_df.label.values

X = train_df.loc[:, train_df.columns != 'label'].values/255



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train.shape, X_test.shape
class Model(torch.nn.Module):

    def __init__(self, n_neurons):

        super(Model, self).__init__()

        self.fc1 = torch.nn.Linear(28 * 28, n_neurons)

        self.ac1 = torch.nn.Sigmoid()

        self.fc2 = torch.nn.Linear(n_neurons, 10) 

    

    def forward(self, x):

        x = self.fc1(x)

        x = self.ac1(x)

        x = self.fc2(x)

        return x

    

model = Model(100)
torch.cuda.is_available()
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
XTrain = torch.from_numpy(X_train)

yTrain = torch.from_numpy(y_train).type(torch.LongTensor)



XTest = torch.from_numpy(X_test)

yTest = torch.from_numpy(y_test).type(torch.LongTensor)
bs = 100



# X_test = X_test.to(device) # for GPU

# y_test = y_test.to(device) # for GPU



for epoch in range(10):

    order = np.random.permutation(len(XTrain))

    

    for i in range(0, len(XTrain), bs):

        optimizer.zero_grad()

        

        bi = order[i:i+bs]

        

        X_batch = XTrain[bi] #.to(device) # for GPU

        y_batch = yTrain[bi] #.to(device) # for GPU

        

        preds = model.forward(X_batch) 

        

        loss_value = loss(preds, y_batch)

        loss_value.backward()

        

        optimizer.step()



    test_preds = model.forward(XTest)        

    accuracy = (test_preds.argmax(dim=1) == yTest).float().mean()    

    

    print(accuracy)
pred = torch.from_numpy(test_df.values)
sample_sub['Label'] = model(pred).argmax(dim=1)

sample_sub.head()
sample_sub.to_csv("submission.csv", index=False)