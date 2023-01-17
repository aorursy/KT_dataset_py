# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import load_breast_cancer

cancerdata = load_breast_cancer()

cancerdata.keys()
# Lets see the description of the dataset 

print(cancerdata['DESCR'])
cancerdata['data'].shape # Tells number of example and features
# Creating dataframe from our data 

df = pd.DataFrame(np.c_[cancerdata['data'],cancerdata['target']], columns = np.append(cancerdata['feature_names'],['target']))

df.head()
import seaborn as sns

distinctfeature = cancerdata['feature_names'][:10] # We know first 10 features are mean features

#sns.pairplot(df,hue='target',vars=['mean radius','mean texture','mean perimeter','mean area','mean smoothness','mean compactness'])

sns.pairplot(df,hue='target',vars=distinctfeature)
plt.figure(figsize=(15,6))

sns.heatmap(df.corr()) # Checking the correlation in the data 
X = df.drop(['target'],axis=1)

y = df['target']
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

clf_svc = SVC()

clf_svc.fit(X_train,y_train)
baseline_ypred = clf_svc.predict(X_test)

cm = confusion_matrix(y_test,baseline_ypred)

print(cm)
print(classification_report(y_test,baseline_ypred))
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_trainScaled = sc.fit_transform(X_train)

X_testScaled = sc.transform(X_test)



# standard scaler X = (X - mean ) / StandardDeviation
clf_svc.fit(X_trainScaled,y_train)

scaled_ypred = clf_svc.predict(X_testScaled)

print(classification_report(y_test,scaled_ypred))

print(confusion_matrix(y_test,scaled_ypred))
# Lets just change the split and rerun the test. 

XScaled = sc.transform(X)

# using the same Xtrain y train nomenclature 

X_train,X_test,y_train,y_test = train_test_split(XScaled,y,test_size=0.4)

clf_svc.fit(X_train,y_train)

scaled_ypred = clf_svc.predict(X_test)

print(classification_report(y_test,scaled_ypred))

print(confusion_matrix(y_test,scaled_ypred))
sns.heatmap(confusion_matrix(y_test,scaled_ypred), annot=True)