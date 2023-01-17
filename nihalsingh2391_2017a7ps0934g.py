import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")
train
train.isna().sum()
f3_mean=train.feature3.mean()

f4_mean=train.feature4.mean()

f5_mean=train.feature5.mean()

f8_mean=train.feature8.mean()

f9_mean=train.feature9.mean()

f10_mean=train.feature10.mean()

f11_mean=train.feature11.mean()
train['feature3']=train.feature3.fillna(f3_mean)

train['feature4']=train.feature4.fillna(f4_mean)

train['feature5']=train.feature5.fillna(f5_mean)

train['feature8']=train.feature8.fillna(f8_mean)

train['feature9']=train.feature9.fillna(f9_mean)

train['feature10']=train.feature10.fillna(f10_mean)

train['feature11']=train.feature11.fillna(f11_mean)
train.isna().sum()
import seaborn as sns

plt.figure(figsize=(12,10))

cor = train.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
from sklearn.linear_model import LogisticRegression

labels=train['rating']
features=train.drop(['id','rating'],axis=1)
features.head()
labels.head()
types=[1 if values=='new' else 0 for values in train.type]
features['type']=types
features.head()
test=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")
test
f3_meant=train.feature3.mean()

f4_meant=train.feature4.mean()

f5_meant=train.feature5.mean()

f8_meant=train.feature8.mean()

f9_meant=train.feature9.mean()

f10_meant=train.feature10.mean()

f11_meant=train.feature11.mean()



test['feature3']=test.feature3.fillna(f3_meant)

test['feature4']=test.feature4.fillna(f4_meant)

test['feature5']=test.feature5.fillna(f5_meant)

test['feature8']=test.feature8.fillna(f8_meant)

test['feature9']=test.feature9.fillna(f9_meant)

test['feature10']=test.feature10.fillna(f10_meant)

test['feature11']=test.feature11.fillna(f11_meant)



test_feat=test.drop(['id'],axis=1)

types=[1 if values=='new' else 0 for values in test.type]

test_feat['type']=types
test.isna().sum()
# fans=[]

# for i in ans:

#     if(i<1.5):

#         fans.append(1)

#     elif(i<2.5):

#         fans.append(2)

#     elif(i<3.5):

#         fans.append(3)

#     elif(i<4.5):

#         fans.append(4)

#     elif(i<5.5):

#         fans.append(5)

#     elif(i<6.5):

#         fans.append(6)

#     elif(i<7.5):

#         fans.append(7)
# fans
# a.score(test_feat,fans)
# abc=pd.DataFrame(fans)
# abc['id']=test.id
# abc
# abc=abc.rename(columns={'0':'rating'})
# abc
# pd.DataFrame(abc).to_csv("/home/nihal/Desktop/fans3.csv")
from sklearn import ensemble

featuresd=features.drop('feature6',axis=1)

featuresd
a1=ensemble.GradientBoostingRegressor(n_estimators=50000,max_depth=10,min_samples_split=2,learning_rate=0.05,loss='ls')
a1.fit(features,labels)
a1.score(features,labels)
# test_featd=test_feat.drop('feature6',axis=1)

ans123=a1.predict(test_feat)
ans123
fans2=[]

for i in ans123:

    if(i<1.5):

        fans2.append(1)

    elif(i<2.5):

        fans2.append(2)

    elif(i<3.5):

        fans2.append(3)

    elif(i<4.5):

        fans2.append(4)

    elif(i<5.5):

        fans2.append(5)

    elif(i<6.5):

        fans2.append(6)

    elif(i<7.5):

        fans2.append(7)
test_featd=test_feat.drop('feature6',axis=1)
abc2=pd.DataFrame(fans2)
abc2
abc2['id']=test.id
abc2
# pd.DataFrame(abc2).to_csv("/home/nihal/Desktop/fans24.csv")
a2=ensemble.GradientBoostingRegressor(n_estimators=50000,max_depth=10,min_samples_split=2,learning_rate=0.05,loss='ls')
a2.fit(featuresd,labels)
a2.score(featuresd,labels)
ans1234=a2.predict(test_featd)

ans1234
fans3=[]

for i in ans1234:

    if(i<1.5):

        fans3.append(1)

    elif(i<2.5):

        fans3.append(2)

    elif(i<3.5):

        fans3.append(3)

    elif(i<4.5):

        fans3.append(4)

    elif(i<5.5):

        fans3.append(5)

    elif(i<6.5):

        fans3.append(6)

    elif(i<7.5):

        fans3.append(7)
abc3=pd.DataFrame(fans3)
abc3['id']=test.id

abc3
# from sklearn.svm import SVC
# svm_model_linear = SVC(kernel = 'poly', C = 1, degree=3).fit(features, labels)
# svm_predictions=svm_model_linear.predict(test_feat)
# svm_predictions
# svm_model_linear.score(features,labels)
# fans4=pd.DataFrame(svm_predictions)
# fans4['id']=test.id
# fans4
# pd.DataFrame(fans4).to_csv("/home/nihal/Desktop/fans5.csv")
# import torch.nn as nn

# import torch.nn.functional as F



# # define the CNN architecture

# class Net(nn.Module):

#     def __init__(self):

#         super(Net, self).__init__()

#         # convolutional layer

# #         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)

# #         self.conv2 = nn.Conv2d(16, 32, 3,padding=1)

# #         self.conv3 = nn.Conv2d(32, 64, 3,padding=1)

#         # max pooling layer

# #         self.pool = nn.MaxPool2d(2, 2)

#         self.fc1 = nn.Linear(12, 50)

#         # linear layer (500 -> 10)

#         self.fc2 = nn.Linear(50, 20)

#         self.fc3 = nn.Linear(20, 10)

#         # dropout layer (p=0.25)

#         self.dropout = nn.Dropout(0.10)

#     def forward(self, x):

#         # add sequence of convolutional and max pooling layers

# #         x = self.pool(F.relu(self.conv3(self.pool(F.relu(self.conv2(self.pool(F.relu(self.conv1(x)))))))))

# #         x=x.view(-1,4*4*64)

# #         x=self.dropout(x)

#         x=F.relu(self.fc1(x))

# #         x=self.dropout(x)

#         x=F.relu(self.fc2(x))

# #         x=self.dropout(x)

#         x=self.fc3(x)

#         return x



# # create a complete CNN

# model = Net()

# print(model)
# import torch.optim as optim



# # specify loss function

# criterion = nn.CrossEntropyLoss()



# # specify optimizer

# optimizer = optim.SGD(model.parameters(), lr=0.01)
# # number of epochs to train the model

# n_epochs = 20



# valid_loss_min = np.Inf # track change in validation loss



# for epoch in range(1, n_epochs+1):



#     # keep track of training and validation loss

#     train_loss = 0.0

#     valid_loss = 0.0

    

#     ###################

#     # train the model #

#     ###################

#     model.train()

#     for data, target in zip(features,labels):

#         # move tensors to GPU if CUDA is available

# #         if train_on_gpu:

# #             data, target = data.cuda(), target.cuda()

#         # clear the gradients of all optimized variables

#         optimizer.zero_grad()

#         # forward pass: compute predicted outputs by passing inputs to the model

#         print(data)

#         print(target)

#         output = model(data)

#         # calculate the batch loss

#         loss = criterion(output, target)

#         # backward pass: compute gradient of the loss with respect to model parameters

#         loss.backward()

#         # perform a single optimization step (parameter update)

#         optimizer.step()

#         # update training loss

#         train_loss += loss.item()*data.size(0)

#         print(output.shape)

        

#     ######################    

#     # validate the model #

#     ######################

#     model.eval()

#     for data, target in valid_loader:

#         # move tensors to GPU if CUDA is available

# #         if train_on_gpu:

# #             data, target = data.cuda(), target.cuda()

#         # forward pass: compute predicted outputs by passing inputs to the model

#         output = model(data)

#         # calculate the batch loss

#         loss = criterion(output, target)

#         # update average validation loss 

#         valid_loss += loss.item()*data.size(0)

    

#     # calculate average losses

#     train_loss = train_loss/len(train_loader.dataset)

#     valid_loss = valid_loss/len(valid_loader.dataset)

        

#     # print training/validation statistics 

#     print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

#         epoch, train_loss, valid_loss))

    

#     # save model if validation loss has decreased

#     if valid_loss <= valid_loss_min:

#         print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

#         valid_loss_min,

#         valid_loss))

#         torch.save(model.state_dict(), 'model_cifar.pt')

#         valid_loss_min = valid_loss
# import pandas as pd

# import numpy as np

# from matplotlib import pyplot as plt

# import seaborn as sns

# %matplotlib inline
# import operator

# # Calculate euclidean distance between x1 and x2. You can assume both x1 and x2 are numpy arrays

# def euclideanDistance(x1, x2):

#     euclideanDistance = 0

#     for i in range(3):

#         euclideanDistance += pow((x1[i] - x2[i]), 2)

#     return (euclideanDistance)**0.5



# # Implement knn algorithm. Return majority label for given test_sample and k

# def knn(X_train, y_train, test_sample, k):

#     distances = []

#     length = len(test_sample)-1

#     for x in range(len(X_train)):

# #         print(test_sample.shape)

# #         print(X_train[x])

#         dist = euclideanDistance(test_sample, X_train[x])

#         distances.append((dist, y_train[x]))

#     distances.sort()

# #     print(distances)

#     neighbors = []

#     for x in range(k):

#         neighbors.append(distances[1][1])

#     c0=0

#     c1=0

#     c2=0

#     c3=0

#     c4=0

#     c5=0

#     for i in neighbors:

#         if i==1:

#             c0+=1;

#         if i==2:

#             c1+=1;

#         if i==3:

#             c2+=1

#         if i==4:

#             c3+=1;

#         if i==5:

#             c4+=1;

#         if i==6:

#             c5+=1;

#     if max(c1,c2,c0,c3,c4,c5)==c0:

#         return 1

#     elif max(c1,c2,c0,c3,c4,c5)==c1:

#         return 2

#     elif max(c1,c2,c0,c3,c4,c5)==c2:

#         return 3

#     elif max(c1,c2,c0,c3,c4,c5)==c3:

#         return 4

#     elif max(c1,c2,c0,c3,c4,c5)==c4:

#         return 5

#     else:

#         return 6

# # Return class of each test sample predicted by knn 

# def predict(X_train, y_train, X_test, k):

#     ans=[]

#     for i in X_test:

# #         print(X_test)

#         ans.append(knn(X_train,y_train,i,k));

#     return ans
# # y_pred = predict(features,labels,test_feat,3)

# # y_pred

# featurest=pd.DataFrame(features).values

# featurest.shape

# test_feat.shape

# labels.shape

# test_feat_t=pd.DataFrame(test_feat).values

# test_feat_t
# # print(test_feat)

# y_pred = predict(featurest,labels,test_feat_t,3)

# y_pred
# abc6=pd.DataFrame(y_pred)
# abc6['id']=test.id
# abc6
# pd.DataFrame(abc6).to_csv("/home/nihal/Desktop/fans7.csv")
# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators = 50000, random_state = 0)
# regressor.fit(features, labels)
# regressor.score(features,labels)
# regressor.predict(test_feat)
# fans2=[]

# for i in ans123:

#     if(i<1.5):

#         fans2.append(1)

#     elif(i<2.5):

#         fans2.append(2)

#     elif(i<3.5):

#         fans2.append(3)

#     elif(i<4.5):

#         fans2.append(4)

#     elif(i<5.5):

#         fans2.append(5)

#     elif(i<6.5):

#         fans2.append(6)

#     elif(i<7.5):

#         fans2.append(7)
# fans2
# regressor.score(test_feat,fans2)