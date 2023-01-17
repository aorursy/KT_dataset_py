# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# load data

root_dir = "../input/"

train = pd.read_csv(root_dir + "mnist_train.csv")
test = pd.read_csv(root_dir + "mnist_test.csv")
train
# test data 확인

test
train_columns = list(train.columns)
train_columns.remove('label')

train_x = train[train_columns].copy()
train_y = train['label'].copy()
# 'label'이라는 column이 나누어진 train_x에 포함되어 있는지 확인

print('label' in train_x.columns)
# 나누어진 train_x (features)

train_x
# 나누어진 train_y(label)

train_y
# image와 image에 해당하는 label을 보여주는 함수를 만든다.
    
    # ! Cautions: image vector가 784개의 pixel로 되어 있으므로, 28x28 dimension을 갖는 image로 convert 후 보여주기

def show_mnist_image_and_label(x, y, index):
    train_image = train_x.iloc[index, :].values
    train_image = train_image.reshape(28, 28)
    imshow(X=train_image)
    
    print("train image label is ", y[index])
show_mnist_image_and_label(x=train_x, y=train_y, index=0)
g = sns.countplot(train_y)
train_y.value_counts()

# check whether label class is biased or not
def draw_correlation_matrix(data, mode):
    corr = data.iloc[:,2:].corr()
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    plt.figure(figsize=(14,14))
    sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 8},
    cmap = colormap, linewidths=0.1, linecolor='white')
    plt.title('Correlation of ' + mode + ' Features', y=1.05, size=15)
# draw_correlation_matrix(data=train_x, mode="train")
# draw_correlation_matrix(data=test, mode="test")
def convert_label_to_one_hot(label):
    one_hot_labels = np.zeros((train_y.shape[0], 10))
    
    one_hot_labels[np.arange(one_hot_labels.shape[0]), label] = 1
    
    return one_hot_labels
one_hot_encoded_train_y = convert_label_to_one_hot(label=train_y)
one_hot_encoded_train_y
train_x, valid_x, train_y, valid_y = train_test_split(train_x, one_hot_encoded_train_y, test_size=0.1)

# 10%만 validation data로 활용
import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.conv1=nn.Conv2d(1,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
        
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))

        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(-1,self.num_flat_features(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
    
    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        return num_features
net=Net()
print(net)
        
        
        
params=list(net.parameters())
print(len(params))
params[0].size() #conv1's weight


# model이 정의된 함수

def MLP():
    model = Sequential()

    # 건드리세요!
    # layer를 더 쌓아 보시거나 layer에 있는 node 수를 바꾸면서 실험 해보세요 ㅎㅎ
    model.add(Dense(32,  activation='relu'))
    model.add(Dense(32,  activation='relu'))
    
    model.add(Dense(10, init='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # optimizer 건드리셔도 됩니다!

    return model
model = MLP()

model.fit(train_x.values, train_y, validation_data=(valid_x.values, valid_y), epochs=10, batch_size=10, verbose=2)
# epochs, batch_size 건드리셔도 됩니다ㅌ
probabilities=model.predict(test)
predicted_result = np.argmax(probabilities, axis=1)
image_ids = np.arange(1, test.shape[0]+1)
data = {'ImageId':image_ids, 'Label': predicted_result}
submission = pd.DataFrame(data = data)
submission.to_csv('submission.csv', index=False)
submission
