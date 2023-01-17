%matplotlib inline



import pandas as pd

import numpy as np



dat = pd.read_csv('../input/data.csv',header=0)



Class = np.where(dat['diagnosis']=='M',1,0) ## isolate target



## drop variables

dat.drop(['id','diagnosis','Unnamed: 32'], axis=1, inplace=True)
import matplotlib.pyplot as plt



fig, axes = plt.subplots(nrows=6, ncols=5,figsize=(12,12))

axes = axes.flatten()

columns = dat.columns



for i in range(30):

  axes[i].hist(dat[columns[i]], normed=1,facecolor='b',alpha=0.75)

  axes[i].set_title(columns[i])

  plt.setp(axes[i].get_xticklabels(), visible=False) 

  plt.setp(axes[i].get_yticklabels(), visible=False) 
from sklearn import preprocessing



dat2 = dat



for column in dat2.columns:

  dat2[column] = np.log(dat2[column]+0.001)



X = dat2.values

X = preprocessing.scale(X)
import random



random.seed(1234)



allIndices = np.arange(len(Class))



numTrain = int(round(0.50*len(Class)))

numValid = int(round(0.30*len(Class)))

numTest = len(Class)-numTrain-numValid



inTrain = sorted(np.random.choice(allIndices, size=numTrain, replace=False))

inValidTest = [i for i in allIndices if i not in inTrain]

inValid= sorted(np.random.choice(inValidTest, size=numValid, replace=False))

inTest = [i for i in inValidTest if i not in inValid]



train = X[inTrain,:]

valid= X[inValid,:]

test =  X[inTest,:]



trainY = Class[inTrain]

validY = Class[inValid]

testY = Class[inTest]
mtry=[2,3,4,5] ##max_features

nodesize=[1,5,20,25] ## min_samples_leaf



a1,a2=np.meshgrid(mtry, nodesize,indexing='ij')

a1=a1.flatten()

a2=a2.flatten()



params = pd.DataFrame(data={'mtry':a1,'nodesize':a2})

params
from sklearn.ensemble import RandomForestClassifier



def train_model(train,trainY,test,testY,mtry,nodesize):

    clf = RandomForestClassifier(n_estimators=1000,

                                 max_features=mtry,

                                 min_samples_leaf=nodesize,

                                 class_weight='balanced')

    clf=clf.fit(train,trainY)

    preds = clf.predict(test)

    acc=(preds == testY).mean()



    return(clf,preds,acc)







best_params = (-1,-1)

best_acc= -1

for i in range(len(params)):

    mtry,nodesize=params.loc[i]

    print('\nmtry= {0:3d} nodesize= {1:3d}'.format(mtry,nodesize))



    clf,preds,acc=train_model(train,trainY,valid,validY,mtry,nodesize)



    if acc > best_acc:

        best_acc=acc

        best_params=(mtry,nodesize)

    print('accuracy: {0:>6.4f} best so far: {1:>6.4f}'.format(acc,best_acc))   

    

mtry,nodesize=best_params    
clf,preds,acc=train_model(train,trainY,test,testY,mtry,nodesize)

print('\nfinal accuracy on the test set: {0:>6.4f}'.format(acc))