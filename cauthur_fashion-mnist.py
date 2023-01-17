# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch

import matplotlib.pyplot as plt



from torch import nn

from torch import optim

import torch.nn.functional as F



from sklearn.model_selection import train_test_split
df = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv",dtype=np.float32)
df.shape
X_np = df.loc[:,df.columns!="label"].values/255

y_np = df.loc[:,"label"].values
X_tensor = torch.from_numpy(X_np)

y_tensor = torch.from_numpy(y_np).type(torch.LongTensor)
X_train,X_test,y_train,y_test = train_test_split(X_tensor,y_tensor,test_size=0.2,shuffle=True)
trainset = torch.utils.data.TensorDataset(X_train,y_train)

testset = torch.utils.data.TensorDataset(X_test,y_test)
train_loader = torch.utils.data.DataLoader(trainset,batch_size=256,shuffle=True)

test_loader = torch.utils.data.DataLoader(testset,batch_size=16,shuffle=False)
val = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]



img, label = next(iter(train_loader))





plt.figure(figsize=(6,6))



plt.imshow(img[0].reshape(28,28))

plt.title("{}".format(val[label[0].numpy().tolist()]),color="Green")



        

class Classifier(nn.Module):

    

    def __init__(self):

        super().__init__()

        

        self.fc1 = nn.Linear(784,512)

        self.fc2 = nn.Linear(512,256)

        self.fc3 = nn.Linear(256,128)

        self.fc4 = nn.Linear(128,64)

        self.fc5 = nn.Linear(64,10)

        

        self.dropout = nn.Dropout(p=0.2)

        

        self.log_softmax = F.log_softmax

        

    def forward(self,x):

        

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))

        x = self.dropout(F.relu(self.fc3(x)))

        x = self.dropout(F.relu(self.fc4(x)))

        x = self.log_softmax(self.fc5(x),dim=1)

        

        return x
model = Classifier()



criterion = nn.NLLLoss()



optimizer = optim.Adam(model.parameters(),lr=0.003)



epoch = 25



print_every = 50



step = 0



train_losses, test_losses = [],[]



for e in range(epoch):

    

    running_loss = 0

    

    for img, label in train_loader:

        

        step += 1

        

        optimizer.zero_grad()

        logps = model(img)

        loss = criterion(logps,label)

        

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        

        if step % print_every == 0:

            

            test_loss = 0

            accuracy = 0

            

            with torch.no_grad():

                

                model.eval()

                

                for img, label in test_loader:

                    

                    logps = model(img)

                    loss = criterion(logps,label)

                    test_loss += loss.item()

                    

                    ps = torch.exp(logps)

                    

                    top_k,top_class = ps.topk(1,dim=1)

                    

                    equal = top_class == label.view(*top_class.shape)

                    

                    accuracy += torch.mean(equal.type(torch.FloatTensor))

                    

                

                model.train()

            

            train_losses.append(running_loss/len(train_loader))

            test_losses.append(test_loss/len(test_loader))

            

            print("{} / {}".format(e+1,epoch),

                 "train loss : {:.3f}".format(train_losses[-1]),

                 "test loss : {:.3f}".format(test_losses[-1]),

                 "accruacy : {:.3f}".format(accuracy/len(test_loader)))

            

            

                    

            

            

        
img, label = next(iter(test_loader))





for img,label in test_loader:

    

    with torch.no_grad():

        

        model.eval()

        

       

        logps = model(img)

        

        ps = torch.exp(logps)

        

        top_k, top_class = ps.topk(1,dim=1)

        

        

result_predict = top_class.numpy().tolist()     

n = 0



fig, ax = plt.subplots(4,4,figsize=(32,32))



for i in range(0,4):

    

    for j in range(0,4):

        

        ax[i][j].imshow(img[n].reshape(28,28))

        ax[i][j].set_title("Result : {}".format(val[result_predict[n][0]]),fontsize=15,color="Green",fontweight="bold")

        ax[i][j].axis("off")

        n += 1

        
df_test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv",dtype = np.float32)
df_test_X = torch.from_numpy(df_test.loc[:,df_test.columns!="label"].values)



fianl_loader = torch.utils.data.DataLoader(df_test_X,batch_size=10000,shuffle=False)
for img in fianl_loader:

    

    with torch.no_grad():

        

        model.eval()

        

        logps = model(img)

        

        ps = torch.exp(logps)

        

        top_k,top_class = ps.topk(1,dim=1)

        

result = pd.DataFrame(top_class.numpy(),columns=["predict_label"])



test_label = df_test.loc[:,"label"].astype("int32").to_frame(name="test_label")
submission = result.join(test_label)
accuracy = []



for i in range(len(submission)):

    

    equal = submission.predict_label[i] == submission.test_label[i]

                                                                 

    accuracy.append(int(equal))

np.mean(accuracy)
submission.to_csv("fashion_mnist_submission.csv",index=False)