# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sn 
data=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
data.head()
data.describe()
data.shape
data.isna().sum()
## No Null values found, So we can proceed to EDA part
# distribution of target variable 

data.target.value_counts()

print("Patients with Heart Disease : ",round((165/303)*100,2))

print("Patients Not with Heart Disease : ",round((138/303)*100,2))





sn.boxplot(data.trestbps)
sn.distplot(data.chol)
data[data.trestbps==200.00]
sn.boxplot(data.thalach)
data.sex.value_counts()
pd.crosstab(data.cp,data.target).plot(kind="bar",figsize=(20,6))
plt.title('CP and No of Patient with heart disease')
plt.xlabel('CP')
plt.ylabel('Frequency')
plt.show()
pd.crosstab(data.sex,data.target).plot(kind="bar",figsize=(20,6))
plt.title('Sex and No of Patient with heart disease')
plt.xlabel('Sex')
plt.ylabel('Frequency')
plt.show()
sn.regplot(data.age,data.trestbps)
sn.barplot(x='ca',y='target',data=data)
data[data.ca==4]
data.ca=data.ca.replace(4,0)
pd.crosstab(data.ca,data.target).plot(kind="bar",figsize=(20,6))
plt.title('CA and No of Patient with heart disease')
plt.xlabel('CA')
plt.ylabel('Frequency')
plt.show()
pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(20,6))
plt.title('Age and No of Patient with heart disease')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
sn.regplot(data.age,data.thalach)
corr_matrix=data.corr()
plt.figure(figsize=(15,6))
sn.heatmap(corr_matrix ,
           annot = True,
            linewidth = 0.5,
            fmt = ".2f",
            cmap = "YlGnBu")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import metrics
variables=['sex','cp','fbs','restecg','exang','thal','ca','slope']

for i in variables:
    data[i]=data[i].astype('category')
data.head()
data=pd.get_dummies(data)
x = data.drop(['target'], axis = 1)
y = data['target']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 5)
feature_scaler = MinMaxScaler()
x_train = feature_scaler.fit_transform(x_train)
x_test = feature_scaler.transform(x_test)
radm_clf = RandomForestClassifier()

parameters = {'n_estimators': [150,175,200,225,250,300,325,350,375,400],'criterion': ['gini','entropy'],'max_features':['auto','sqrt','log2']}

clf = GridSearchCV(radm_clf, parameters, scoring='roc_auc' ,cv =5)
clf.fit(x_train, y_train)

clf.best_score_
clf.best_params_
model=RandomForestClassifier(criterion = 'gini',max_features = 'log2',n_estimators = 150)
model.fit(x_train,y_train)
accuracy = model.score(x_test,y_test)
print(accuracy)
import scikitplot as skplt
pred=model.predict(x_test)
matrix6 = (y_test,pred)
skplt.metrics.plot_confusion_matrix(y_test ,pred ,figsize=(10,5))
from sklearn.metrics import classification_report

print("Testing Accuracy :", model.score(x_test, y_test))
cr = classification_report(y_test, pred)
print(cr)
