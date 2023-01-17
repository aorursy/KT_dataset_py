#Loading libraries 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from collections import Counter
import time
import datetime as dt
from datetime import datetime
import collections
import os # accessing directory structure
from matplotlib.pyplot import rcParams
%matplotlib inline
rcParams['figure.figsize'] = 10,8
sns.set(style='whitegrid', palette='muted',
        rc={'figure.figsize': (15,10)})
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#loading data
data = pd.read_csv('../input/voice.csv')
data.head()
data.shape 
data.info()
data.dtypes
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.corr()
data.label.value_counts()
sns.countplot(x=data['label'])
data['meanfreq'].value_counts(dropna=False)
data['sd'].value_counts(dropna=False)
data['median'].value_counts(dropna=False)
data['Q25'].value_counts(dropna=False)
data['Q75'].value_counts(dropna=False)
data['IQR'].value_counts(dropna=False)
data['skew'].value_counts(dropna=False)
data['kurt'].value_counts(dropna=False)
data['sp.ent'].value_counts(dropna=False)
data['sfm'].value_counts(dropna=False)
data['mode'].value_counts(dropna=False)
data['centroid'].value_counts(dropna=False)
data['meanfun'].value_counts(dropna=False)
data['minfun'].value_counts(dropna=False)
data['maxfun'].value_counts(dropna=False)
data['meandom'].value_counts(dropna=False)
data['mindom'].value_counts(dropna=False)
data['maxdom'].value_counts(dropna=False)
data['dfrange'].value_counts(dropna=False)
data['modindx'].value_counts(dropna=False)
data['label'].value_counts(dropna=False)
data.isnull().sum()
data.duplicated()
sns.lmplot( x="sfm", y="meanfreq", data=data, fit_reg=False, hue='label', legend=False)
plt.show()
sns.lmplot( x="meanfun", y="meanfreq", data=data, fit_reg=False, hue='label', legend=False)
plt.show()
#Lets check the distribution of meanfreq (mean vs women)
plt.figure()
sns.kdeplot(data['meanfreq'][data['label']=='male'], shade=True);
sns.kdeplot(data['meanfreq'][data['label']=='female'], shade=True);
plt.xlabel('meanfreq value')
plt.show()
plt.figure(figsize=(8,7))
sns.boxplot(x="label", y="dfrange", data=data)
plt.show()
data.label=[  1 if i=="male" else 0 for i in data.label]
data.head(5)
data.info()
y=data.label.values
x_data=data.drop(["label"],axis=1)
y
x_data.head()
# normalization =(a-min(a))/(max(a)-min(a))

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
x.head()
# create x_train, y_train, x_test, y_test arrays
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# our features must be row in our matrix.

X_train=X_train.T
X_test=X_test.T
y_train=y_train.T
y_test=y_test.T

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)
#Lets use logistic Regression:
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#Producing X and y
X = np.array(data.drop(['label'], 1))
y = np.array(data['label'])

#Dividing the data randomly into training and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#Now we’ll fit the model on the training data:
model=LogisticRegression()
model.fit(X_train,y_train)

print('Accuracy1 :',model.score(X_train,y_train))
print('Accuracy2 :',model.score(X_test,y_test))
model.predict(X_test)
from sklearn import svm
from sklearn.metrics import roc_auc_score
svc = svm.SVC(kernel='linear', C=1,gamma='auto').fit(X_train, y_train)
y_pred=svc.predict(X_test)
accuracy=roc_auc_score(y_test,y_pred)
print('Accuracy :',accuracy)
