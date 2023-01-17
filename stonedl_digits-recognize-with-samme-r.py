# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import zero_one_loss

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

import time

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.decomposition import PCA

%matplotlib inline
train_df=pd.read_csv('../input/train.csv')

test_df=pd.read_csv('../input/test.csv')



y_train=train_df['label']

train_df=train_df.drop('label',axis=1)

image_origin=train_df.ix[99,:].values.reshape((28,28))
#set the pixel values greater than (mean-std) within every single image to 1,others to 0

train_df['threshold']=train_df[train_df>0].mean(axis=1)-train_df[train_df>0].std(axis=1)

threshold=train_df['threshold']

train_df=train_df.T

train_df=train_df-threshold

train_df=train_df.T

train_df.drop('threshold',axis=1,inplace=True)

train_df[train_df<=0]=0

train_df[train_df>0]=1



test_df['threshold']=test_df[test_df>0].mean(axis=1)-test_df[test_df>0].std(axis=1)

threshold=test_df['threshold']

test_df=test_df.T

test_df=test_df-threshold

test_df=test_df.T

test_df.drop('threshold',axis=1,inplace=True)

test_df[test_df<=0]=0

test_df[test_df>0]=1
#compare the image before normalization and after

with sns.axes_style('dark'):

    image_new=train_df.ix[99,:].values.reshape((28,28))

    plt.subplot(121)

    plt.imshow(image_origin,cmap=plt.get_cmap('gray'))

    plt.subplot(122)

    plt.imshow(image_new,cmap=plt.get_cmap('gray'))
#remove those columns all values are 0 both in train and test data

std_mask=((test_df.std()!=0)|(train_df.std()!=0))

train_df=train_df.ix[:,std_mask]

test_df=test_df.ix[:,std_mask]
#split the train data into train and validation partions

X_train=train_df.values

y_train=y_train.values



X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,train_size=0.85,random_state=9)



X_test=test_df.values



train_df=None

test_df=None

test_label=None
#check the dimensions

X_train.shape,y_train.shape,X_val.shape,y_val.shape
n_estimators=200

learning_rate=0.25
base_estimator=DecisionTreeClassifier(max_depth=1,min_samples_leaf=1,

                                      min_samples_split=2,random_state=9)

samme_r=AdaBoostClassifier(base_estimator=base_estimator,n_estimators=n_estimators,

                          learning_rate=learning_rate,algorithm='SAMME.R',random_state=9)

start=time.clock()

samme_r.fit(X_train,y_train)

end=time.clock()

print('Time:',end-start,'s')
validation_err=np.zeros(n_estimators)

for i,y_pred in enumerate(samme_r.staged_predict(X_val)):

    validation_err[i]=zero_one_loss(y_val,y_pred)



train_err=np.zeros(n_estimators)

for i,y_pred in enumerate(samme_r.staged_predict(X_train)):

    train_err[i]=zero_one_loss(y_train,y_pred)



fig=plt.figure()

ax=fig.add_subplot(111)

ax.plot(np.arange(n_estimators)+1,validation_err,label='Validation Error',color='darkorange')

ax.plot(np.arange(n_estimators)+1,train_err,label='Train Error',color='k')

ax.set_title('AdaBoost SAMME.R')

ax.set_xlabel('n_estimators')

ax.set_ylabel('error')

plt.legend()
samme_r.fit(np.concatenate((X_train,X_val),axis=0),np.concatenate((y_train,y_val)))
predict=samme_r.predict(X_test)

result=pd.DataFrame({'ImageId':np.arange(len(predict))+1,

                    'Label':predict})

result.to_csv('result-20170328.csv',index=False,header=True)