import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import torch

import torch.nn as nn

import torchvision

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

# from scipy.ndimage.interpolation import rotate

# from sklearn.model_selection import KFold
plt.rcParams['image.cmap']='gray'
train=pd.read_csv('../input/digit-recognizer/train.csv')

test=pd.read_csv('../input/digit-recognizer/test.csv')
print("Train data size:",train.shape,"\nTest data size:",test.shape)
train.head()
test.head()
train.describe()
test.describe()
# L1,L2=[],[]

# for i in range(len(train)):

#     if len(train.iloc[i][1:][train.iloc[i][1:]>255])>0 or len(train.iloc[i][1:][train.iloc[i][1:]<0])>0:

#         L1.append(train.iloc[i][1:][train.iloc[i][1:]>255])



# for i in range(len(test)):

#     if len(test.iloc[i][1:][test.iloc[i][1:]>255])>0 or len(test.iloc[i][1:][test.iloc[i][1:]<0])>0:

#         L2.append(train.iloc[i][1:][train.iloc[i][1:]>255])

    

# print("Number of data points in train data having invlaid pixel value = %d\

# \nNumber of data points in test data having invalid pixel value = %d"%(len(L1),len(L2)))
x=np.array(train[train.columns[1::]])

x=x.reshape(x.shape[0],28,28)

y=np.array(train[train.columns[:1:]])
digits=[str(i) for i in range(10)]

plt.figure(figsize=(7,7))

num_digits=8

for i,digit in enumerate(digits):

    selection=np.flatnonzero(y==i)

    selection=np.random.choice(selection,num_digits,replace=False)

    for j,selected in enumerate(selection):

        idx=num_digits*i+j+1

        plt.subplot(len(digits),num_digits,idx)

        plt.imshow(x[selected])

        plt.axis('off')
# def crop_img(array,original_width=32,crop_to_width=28,original_height=32,crop_to_height=28):

#     return array[(original_width-crop_to_width)//2:original_width-((original_width-crop_to_width)//2),(original_height-crop_to_height)//2:original_height-((original_height-crop_to_height)//2)]
# plt.figure(figsize=(5,10))

# for i,digit in enumerate(digits):

#     selection=np.flatnonzero(y==i)

#     selection=np.random.choice(selection,2,replace=False)

#     idx=2*i+1

#     plt.subplot(len(digits),2,idx)

#     plt.imshow(x[selection[0]])

#     plt.axis('off')

#     idx=2*i+2

#     plt.subplot(len(digits),2,idx)

#     plt.imshow(crop_img(rotate(x[selection[0]],20)))

#     plt.axis('off')
# plt.figure(figsize=(5,10))

# for i,digit in enumerate(digits):

#     selection=np.flatnonzero(y==i)

#     selection=np.random.choice(selection,2,replace=False)

#     idx=2*i+1

#     plt.subplot(len(digits),2,idx)

#     plt.imshow(x[selection[0]])

#     plt.axis('off')

#     idx=2*i+2

#     plt.subplot(len(digits),2,idx)

#     plt.imshow(crop_img(rotate(x[selection[0]],-20)))

#     plt.axis('off')
# x1=x

# L=[]

# for i in range(len(x)):

#     L.append(crop_img(rotate(x[i],20)))

# x=np.append(x,L,axis=0)
# L=[]

# for i in range(len(x1)):

#     L.append(crop_img(rotate(x[i],-20)))
# x=np.append(x,L,axis=0)

# y1=y

# y=np.append(y,y,axis=0)

# y=np.append(y,y1,axis=0)
# Normalize from 0 to 1

x=x/255
x.shape,y.shape
x=x.reshape(x.shape[0],1,28,28)
x.shape,y.shape
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=42)
# Hyperparameters

num_epochs = 25

num_classes = 10
# Device configuration

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') # used to run on GPU

device
# class Flatten(nn.Module):

#     def forward(self, input):

#         return input.view(input.size(0), -1)
class ConvNet(nn.Module):

    def __init__(self):

        super(ConvNet, self).__init__()

        self.passconv=nn.Sequential(

        nn.Conv2d(1, 32, kernel_size=3, stride=1),

        nn.ReLU(inplace=True),

        nn.BatchNorm2d(32),

        nn.Conv2d(32, 32, kernel_size=3, stride=1),

        nn.ReLU(inplace=True),

        nn.BatchNorm2d(32),

#         nn.Dropout(),

        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(32,64,kernel_size=3,stride=1),

        nn.ReLU(inplace=True),

        nn.BatchNorm2d(64),

        nn.Conv2d(64,64,kernel_size=3,stride=1),

        nn.ReLU(inplace=True),

        nn.BatchNorm2d(64),

#         nn.Dropout(),

        nn.MaxPool2d(kernel_size=2,stride=2)

        )

        self.fcpass=nn.Sequential(

        nn.Linear(4*4*64, 256),

        nn.Linear(256,10)

        )





    def forward(self,x):

        out=self.passconv(x)

        out=out.reshape(-1,out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3])

        out=self.fcpass(out)

        return out
model=ConvNet().to(device)



# Loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters())
# X_train.shape[0]
# X_train=X_train.reshape(X_train.shape[0],1,28,28)

# X_train.shape
model
# y_train.shape
# cv = KFold(n_splits=5, random_state=42, shuffle=False)
# for train_index,test_index in cv.split(x):

#     print("Train Index: ", train_index)

#     print("Test Index: ", test_index)

#     X_train, X_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]

    # Train the model

total_step = len(X_train)

loss_list = []

acc_list = []

labels=torch.tensor(y_train,dtype=torch.long)

X_train=torch.tensor(X_train,dtype=torch.float)

for epoch in range(num_epochs):

    total_ans=0

    correct_ans=0

    for i,images in enumerate(X_train):

        # Run the forward pass

        images=images.unsqueeze(0)

        image,label=images.to(device),labels[i].to(device)

        outputs = model(image)

        loss = criterion(outputs, label)

        loss_list.append(loss.item())



        # Backprop and perform Adam optimisation

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Track the accuracy

        total_ans+=1

        _, predicted = torch.max(outputs.data, 1)

        correct = (predicted == label).sum().item()

        correct_ans+=correct

        acc_list.append(correct_ans/total_ans)

        if (i + 1) % 100 == 0:

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'

                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),

                            (correct_ans / total_ans) * 100))
total_ans
# Test the model

total_step = len(X_test)

labels=torch.tensor(y_test,dtype=torch.long)

X_test=torch.tensor(X_test,dtype=torch.float)

with torch.no_grad():

    total_ans=0

    correct_ans=0

    for i,images in enumerate(X_test):

        # Run the forward pass

        images=images.unsqueeze(0)

        image,label=images.to(device),labels[i].to(device)

        outputs = model(image)





        # Track the accuracy

        total_ans+=1

        _, predicted = torch.max(outputs.data, 1)

        correct = (predicted == label).sum().item()

        correct_ans+=correct



        if (i + 1) % 100 == 0:

            print('Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'

                  .format(i + 1, total_step, loss.item(),

                          (correct_ans / total_ans) * 100))
test.shape
x=test

x=x/255
x=x.values.reshape(x.shape[0],1,28,28)
# Test the model

total_step = len(x)

x=torch.tensor(x,dtype=torch.float)

with torch.no_grad():

    predictions=[]

    for i,images in enumerate(x):

        # Run the forward pass

        image=images.unsqueeze(0)

        image=image.to(device)

        outputs = model(image)





        # Track the accuracy

        _, predicted = torch.max(outputs.data, 1)

        predictions.append(predicted.item())



        if (i + 1) % 100 == 0:

            print('Step [{}/{}]'

                  .format(i + 1, total_step))
pd.read_csv('../input/digit-recognizer/sample_submission.csv').head()
test.shape
index=[i for i in range(1,test.shape[0]+1)]

index[-1]
df=pd.DataFrame({'ImageId':index,'Label':predictions})
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "cnn_ans.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)





# create a link to download the dataframe

create_download_link(df)



# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓
df.to_csv('cnn_ans.csv',index=False)