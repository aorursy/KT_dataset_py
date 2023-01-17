# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.





import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn import preprocessing

from sklearn.model_selection import KFold
parkinson_df = pd.read_csv('../input/parkinsons2.csv')
parkinson_df.head().transpose()
parkinson_df.columns
#Since column names are big it will be easy to do plots and calculations if the column names are small

parkinson_df.columns = ['Fo','Fhi','Flo','Jitter(%)','Jitter(Abs)','RAP','PPQ','DDP','Shimmer','Shimmer(dB)','APQ3','APQ5','APQ','DDA','NHR','HNR','RPDE','DFA','spread1','spread2','D2','PPE','status']
parkinson_df.info()
parkinson_df.describe().transpose()
parkinson_df[parkinson_df.isnull().any(axis=1)]
parkinson_df.boxplot(figsize=(24,8))
parkinson_df.corr()
parkinson_df['status'].value_counts().sort_index()
X = parkinson_df.drop(['Fhi','NHR','status'],axis=1)

Y = parkinson_df['status']
#Splitting the data into train and test in 70/30 ratio with random state as 2.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)
LR = LogisticRegression()

LR.fit(X_train, Y_train)
Y1_predict = LR.predict(X_test)

Y1_predict
Y_acc = metrics.accuracy_score(Y_test,Y1_predict)

print("Accuracy of the model is {0:2f}".format(Y_acc*100))

Y_cm=metrics.confusion_matrix(Y_test,Y1_predict)

print(Y_cm)
#Sensitivity

TPR=Y_cm[1,1]/(Y_cm[1,0]+Y_cm[1,1])

print("Sensitivity of the model is {0:2f}".format(TPR))
#Specificity

TNR=Y_cm[0,0]/(Y_cm[0,0]+Y_cm[0,1])

print("Specificity of the model is {0:2f}".format(TNR))
Y_CR=metrics.classification_report(Y_test,Y1_predict)

print(Y_CR)
fpr,tpr, _ = roc_curve(Y_test, Y1_predict)

roc_auc = auc(fpr, tpr)



print("Area under the curve for the given model is {0:2f}".format(roc_auc))

plt.figure()

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.show()
X = parkinson_df.drop(['Fhi','NHR','status'],axis=1)

Y = parkinson_df['status']
# K-fold cross validation for the given model:

#Since the dataset contains 197 rows, we are taking the number of splits as 3

kf=KFold(n_splits=3,shuffle=True,random_state=2)

acc=[]

for train,test in kf.split(X,Y):

    M=LogisticRegression()

    Xtrain,Xtest=X.iloc[train,:],X.iloc[test,:]

    Ytrain,Ytest=Y[train],Y[test]

    M.fit(Xtrain,Ytrain)

    Y_predict=M.predict(Xtest)

    acc.append(metrics.accuracy_score(Ytest,Y_predict))

    print(metrics.confusion_matrix(Ytest,Y_predict))

    print(metrics.classification_report(Ytest,Y_predict))

print("Cross-validated Score:{0:2f} ".format(np.mean(acc)))
#Accuracy for each fold

acc
#Error

error=1-np.array(acc)

error
# Variance Error of the model

np.var(error,ddof=1)
fpr,tpr, _ = roc_curve(Ytest, Y_predict)

roc_auc = auc(fpr, tpr)



print("Area under the curve for the given model is {0:2f}".format(roc_auc))

plt.figure()

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.show()