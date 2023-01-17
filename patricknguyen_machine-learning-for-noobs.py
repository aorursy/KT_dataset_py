import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import os
# Load .csv to variables
breeds = pd.read_csv('../input/breed_labels.csv')
colors = pd.read_csv('../input/color_labels.csv')

train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')

train['dstype']='train'
test['dstype']='test'
all=pd.concat([train,test])
all=all.reset_index(drop=True)
train.drop('Description',axis=1,inplace=True)
test.drop('Description',axis=1,inplace=True)
all.drop('Description',axis=1,inplace=True)
# Fetch train dataset's infomation
all.info()
train.head()
%matplotlib inline
data = train['AdoptionSpeed'].value_counts()
data.plot('barh')
for i,v in enumerate(data.values):
    plt.gca().text(v+50,i-0.1,str(v),color='teal',fontweight='bold')
# Change type 1 to dog, 2 to cat
all['Type']=all['Type'].apply(lambda x:'Dog' if x==1 else 'Cat')
all[all.Name=="Brisco"]
plt.figure(figsize=(5,3))
sns.countplot(x='dstype',data=all,hue='Type')
plt.title('Amount of cats and dogs in test set and train set')
# See the age of dogs & cats
train.Age.value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Type", data=train)
plt.title('AdoptionSpeed by Type and age')

# Set all cells in column 'Name' to 1 if there is 'no name' or blank (null)
for i,value in enumerate(all.Name):
    if(str(value).lower().find('no name')==0 ):
        all.at[i,'Name']=1
for i in all[all['Name'].isnull()].index:
    all.at[i,'Name']=1
# Set the remaining cells to 0
for i,value in enumerate(all.Name):
    if(str(value)!= '1' ):
        all.at[i,'Name']=0
# Rename Name to No-Name
all.rename(columns={'Name':'NoName'},inplace=True)


ax = sns.barplot(x="AdoptionSpeed", y="AdoptionSpeed", data=all[all.NoName==1], estimator=lambda x: len(x) / len(all[all.NoName==1]) * 100)
ax.set(ylabel="Percent")
ax.set_title('Adoption Speed for no names')
ax = sns.barplot(x="AdoptionSpeed", y="AdoptionSpeed", data=all[all.NoName==0], estimator=lambda x: len(x) / len(all[all.NoName==0]) * 100)
ax.set(ylabel="Percent")
ax.set_title('Adoption Speed for named pets')
all.groupby('Quantity').agg(['count','mean'])['AdoptionSpeed']
breeds[breeds.BreedName=='Mixed Breed']
train['Pure']=0
train.loc[train['Breed2']==0,'Pure']=1
train.loc[train['Breed1']==307,'Pure']=0
test['Pure']=0
test.loc[test['Breed2']==0,'Pure']=1
test.loc[test['Breed1']==307,'Pure']=0
print('-Train:')
print('There are',len(train[train.Pure==1]), 'Pure Breed',len(train[train.Pure==1])/len(train)*100,"%")
print('There are',len(train[train.Pure==0]), 'Mixed Breed',len(train[train.Pure==0])/len(train)*100,"%")
print('-Test:')
print('There are',len(test[test.Pure==1]), 'Pure Breed',len(test[test.Pure==1])/len(test)*100,"%")
print('There are',len(test[test.Pure==0]), 'Mixed Breed',len(test[test.Pure==0])/len(test)*100,"%")
all['Pure']=0
all.loc[all['Breed2']==0,'Pure']=1
all.loc[all['Breed1']==307,'Pure']=0
all
train[train.Pure==1]['AdoptionSpeed'].mean()
train[train.Pure==0]['AdoptionSpeed'].mean()
sns.factorplot('Type', col='Gender', data=all, kind='count', hue='dstype');
plt.subplots_adjust(top=0.8)
plt.suptitle('Count of cats and dogs in train and test set by gender');
sns.countplot(x='AdoptionSpeed',data=all,hue='Gender')
#One Hot Encoder
from sklearn.preprocessing import LabelBinarizer

LaBi = LabelBinarizer()

Breed1_lb=LaBi.fit_transform(all.Breed1)
Breed2_lb=LaBi.fit_transform(all.Breed2)
Type_lb=LaBi.fit_transform(all.Type)
Gender_lb=LaBi.fit_transform(all.Gender)
Vaccinated_lb=LaBi.fit_transform(all.Vaccinated)
Dewormed_lb = LaBi.fit_transform(all.Dewormed)
FurLength_lb = LaBi.fit_transform(all.FurLength)
Sterilized_lb = LaBi.fit_transform(all.Sterilized)
Health_lb = LaBi.fit_transform(all.Health)
Color1_lb = LaBi.fit_transform(all.Color1)
Color2_lb = LaBi.fit_transform(all.Color2)
Color3_lb = LaBi.fit_transform(all.Color3)
allLB=np.append(Breed1_lb,Breed2_lb,axis=1)
allLB=np.append(allLB,Type_lb,axis=1)
allLB=np.append(allLB,Gender_lb,axis=1)
allLB=np.append(allLB,Vaccinated_lb,axis=1)
allLB=np.append(allLB,Dewormed_lb,axis=1)
#allLB=np.append(allLB,FurLength_lb,axis=1)
allLB=np.append(allLB,Sterilized_lb,axis=1)
allLB=np.append(allLB,Health_lb,axis=1)
allLB=np.append(allLB,Color1_lb,axis=1)
allLB=np.append(allLB,Color2_lb,axis=1)
#allLB=np.append(allLB,Color3_lb,axis=1)
allLB.shape
all_mat=np.append(allLB,all[['Age','NoName','Pure','Quantity']].values.reshape(18941,4),axis=1)
all_mat=np.append(all_mat,all['AdoptionSpeed'].values.reshape(18941,1),axis=1)
train_x=all_mat[:14993,:-1]
train_y=all_mat[:14993,-1]
test_x=all_mat[14993:,:-1]
test_y=all_mat[14993:,-1]
all_mat.shape
submission=pd.read_csv('../input/test/sample_submission.csv')
submission.head()
from sklearn.model_selection import train_test_split
train_x,cv_x,train_y,cv_y=train_test_split(train_x,train_y,test_size=0.2)

import sklearn
from sklearn.preprocessing import scale 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
train.values[:,:-2]
train.columns
LogReg = LogisticRegression()
LogReg.fit(train_x, list(train_y))

y_pred = LogReg.predict(cv_x)
y_pred
# Metrics
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(list(cv_y),y_pred)
accuracy
#y_test_pred=LogReg.predict(test_x)

# Get predicted array into dataframe
#for i,value in enumerate(y_test_pred):
#    submission.set_value(i,'AdoptionSpeed',value)
#submission.tail()
# Import into CSV
#submission.to_csv('submission.csv', index=False)
from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(train_x, train_y)
# Predict
prediction = xgb_model.predict(cv_x)
# Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(list(cv_y),list(prediction))
accuracy
#y_pred = xgb_model.predict(test_x)
#for i,value in enumerate(y_pred):
#    submission.set_value(i,'AdoptionSpeed',value)
#submission.to_csv('submission.csv', index=False)
#y_pred[-1]
#y_pred
#submission.head()
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.layers import Dropout
model = Sequential()
model.add(Dense(600, activation='relu',  kernel_initializer='normal',input_dim=train_x.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))
mcp = keras.callbacks.ModelCheckpoint("model.h5", monitor="val_acc",  save_best_only=True, save_weights_only=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_y_lb = LaBi.fit_transform(list(train_y))
cv_y_lb = LaBi.fit_transform(list(cv_y))
model.fit(train_x, train_y_lb, epochs=10, validation_data=(cv_x,cv_y_lb), callbacks=[mcp], batch_size=16)
from keras.models import load_model
best_model = load_model("model.h5")

#Evaluate
score = best_model.evaluate(cv_x, cv_y_lb, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
y_pred=best_model.predict(test_x)
y_pred[0:20]
import operator
# Y_list: from one-hot y_pred to choosing output based index. e.g : if [0 0 1] then 2 or if [0 1 0] then 1
y_list=[]
for i in y_pred:
    index, value = max(enumerate(i), key=operator.itemgetter(1))
    y_list.append(index)
y_pred=best_model.predict(test_x)
y_list
for i,value in enumerate(y_list):
        submission.set_value(i,'AdoptionSpeed',value)
    
submission.to_csv('submission.csv', index=False)
submission.head()
train_x.shape
train_y_lb.shape
