# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import pandas as pd
diabetes = pd.read_csv("../input/diabetes.csv")
import pandas as pd
diabetes['Outcome'].value_counts() # here we can see the data set is imbalance so we need to apply confusion matrix
import matplotlib.pyplot as plt
plt.scatter(diabetes['Outcome'],diabetes['BMI'])


plt.xlabel('Outcome')
plt.ylabel('BMI')
plt.title("BMI Vs Outcome")
plt.show()
diabetes.hist(bins=50,figsize=(20,20))
plt.show
diabetes.info()
diabetes.shape
diabetes.isnull().sum()
corre=diabetes.corr()
corre

import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))
sns.heatmap(corre,annot=True,linewidths=.05,fmt='.2f')
traindata=diabetes.drop(['Outcome'],axis=1)
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()

traindata[traindata.columns]=scale.fit_transform(traindata[traindata.columns])
traindata
testdata=diabetes['Outcome']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(traindata,testdata,test_size=.20)
x_train.shape,x_test.shape
y_train.shape,y_test.shape
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(x_train,y_train)
log.score(x_test,y_test)*100
from sklearn.metrics import confusion_matrix
y_predict=log.predict(x_test)
results=confusion_matrix(y_test,y_predict)
print("confusion matrix:")
print(results)
proba=log.predict_proba(x_test[0:5]) #this shows the probability of any result being No or Yes
print(proba)

from sklearn.metrics import roc_curve,auc,roc_auc_score
fpr,tpr,threshold=roc_curve(y_test,log.predict_proba(x_test)[:,1])
fpr[0:5]
tpr[0:5]
threshold[0:5]
roc_auc=roc_auc_score(y_test,log.predict(x_test))
roc_auc
import matplotlib.pyplot as plt # matplotlib is used for plotting the curve
plt.figure()
lw=2
plt.plot(fpr,tpr,color='blue',lw=lw,label='ROC curve (area=%.5f100)'% roc_auc)
plt.plot([0,1],[0,1],color="navy",lw=lw,linestyle='--')
plt.xlim([0.0,1.1])
plt.ylim([0.0,1.1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower_right")
plt.show()

y_predict=log.predict_proba(x_test)
y_predict[0:5]

from sklearn.preprocessing import binarize 
#important for converting the probability values in binary form
y_pred=binarize(y_predict,0.495)
y_pred[0:5]
y_pred1=y_pred[:,1]
y_pred1 # the output is in float we need to convert this output into integer type

y_pred2=y_pred1.astype(int)
y_pred2 # now the output is in integer type
tuned_result=confusion_matrix(y_test,y_pred2)
print("confusion martics after tuning:")
print(tuned_result)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred2)

