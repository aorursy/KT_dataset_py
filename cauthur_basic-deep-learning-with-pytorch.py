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

from torch import nn

from torch import optim

import torch.nn.functional as F



from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt
df = pd.read_csv("../input/digit-recognizer/train.csv",dtype=np.float32)

print(df.shape)
X_np = df.loc[:,df.columns != "label"].values / 255

y_np = df.loc[:,"label"].values
feature_train,feature_test,target_train,target_test = train_test_split(X_np,y_np,test_size=0.2,random_state=62)
X_train = torch.from_numpy(feature_train)

y_train = torch.from_numpy(target_train).type(torch.LongTensor)

X_test = torch.from_numpy(feature_test)

y_test = torch.from_numpy(target_test).type(torch.LongTensor)
train = torch.utils.data.TensorDataset(X_train,y_train)

test = torch.utils.data.TensorDataset(X_test,y_test)



train_loader = torch.utils.data.DataLoader(train,batch_size=256,shuffle=True)

test_loader= torch.utils.data.DataLoader(test,batch_size=32,shuffle=True)
plt.imshow(X_np[0].reshape(28,28))
class Classifier(nn.Module):

    

    def __init__(self):

        super().__init__()

        

        self.fc1 = nn.Linear(784, 512)

        self.fc2 = nn.Linear(512, 256)

        self.fc3 = nn.Linear(256, 128)

        self.fc4 = nn.Linear(128, 64)

        self.fc5 = nn.Linear(64, 10)

        

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



optimizer = optim.Adam(model.parameters(),lr=0.0015)



epoch = 25

steps = 0

print_every = 50

train_losses, test_losses = [],[]



for e in range(epoch):

    running_loss = 0

    

    for images,labels in train_loader:

        

        steps += 1

        

        optimizer.zero_grad()

        

        log_ps = model(images)

        loss = criterion(log_ps,labels)

        

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        

        

        

        if steps % print_every == 0:

            test_loss = 0

            accuracy = 0

            

            with torch.no_grad():

                

                model.eval()

                

                for images, labels in test_loader:

                    

                    log_ps = model(images)

                    test_loss += criterion(log_ps,labels)

                    ps = torch.exp(log_ps)

                    

                    top_p, top_class = ps.topk(1,dim=1)

                    

                    equal = top_class == labels.view(*top_class.shape)

                    

                    accuracy += torch.mean(equal.type(torch.FloatTensor))

                

                model.train()

            

            train_losses.append(running_loss/len(train_loader))

            test_losses.append(test_loss/len(test_loader))

            

            

            print("Epoch : {} / {}".format(e+1,epoch),

                 "train_loss : {:.3f}".format(train_losses[-1]),

                 "test_loss : {:.3f}".format(test_losses[-1]),

                 "test accuracy : {:.3f}".format(accuracy / len(test_loader)))               
for img, label in train_loader:

    

    with torch.no_grad():

        logps = model(img)

    

    ps = torch.exp(logps)

    

top_p,top_class = ps.topk(1,dim=1)   





fig, ax = plt.subplots(4,4,figsize=(32,32))





arg = ps[0].argmax()



n1 = 0



for i in range(0,4):

    

    for j in range(0,4):

        

        ax[i][j].imshow(img[n1].reshape(28,28))

        ax[i][j].axis("off")

        ax[i][j].set_title("{}".format(top_class[n1]),fontsize=15,color="Green")

        n1 += 1
result_np = pd.read_csv("../input/digit-recognizer/test.csv",dtype=np.float32).values / 255

submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

result = torch.from_numpy(result_np)



result_loader = torch.utils.data.DataLoader(result,batch_size=100,shuffle=False)
print(result_np.shape)
results = []

with torch.no_grad():

    

    for img in result_loader:

        

        logps = model(img)

        ps = torch.exp(logps)

        top_k,top_class = ps.topk(1,dim=1)

      

        

        results += top_class.numpy().tolist()
submission.drop(labels=["Label"],axis=1,inplace=True)
results_frame = pd.DataFrame(results,columns=["Label"])

final = submission.join(results_frame)
final
final.to_csv('mnist_submission.csv', index = False)