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
dataset=pd.read_csv('/kaggle/input/hackerearth/train.csv',index_col='ID')

test1=pd.read_csv('/kaggle/input/hackerearth/train.csv')
dataset.columns[dataset.isnull().any()]
alldata=dataset.append(test1)
dataset.shape

alldata.shape
dataset.head()
test1.isnull().any()

test1=test1.fillna(method='ffill')
dataset.describe()
import matplotlib.pyplot as plt

import seaborn as sns

import warnings

%matplotlib inline

warnings.filterwarnings('ignore')
plt.figure(figsize=(10,8))

sns.countplot(dataset['Result'],label='Count')
dataset.isnull().any()

dataset = dataset.fillna(method='ffill')

dataset.hist(bins=10,figsize=(20,15))

plt.show()
plt.figure(figsize=(50,25))

sns.heatmap(data=dataset.corr(),annot=True)

plt.title('Co-Relation Mattrix')

plt.tight_layout()

plt.show()
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix,classification_report,r2_score,accuracy_score

X=dataset.drop('Result',axis=1)

Y=dataset['Result']

model1=DecisionTreeClassifier()
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
model1.fit(X_train,y_train)
pred=model1.predict(X_test)
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))
print(classification_report(y_test,pred))
r2_score(y_test,pred)
df=pd.DataFrame({'Actual Pred':y_test,'Predicted ':pred})

df1=df.head(25)

print(df1)


df1.plot(kind='bar',figsize=(20,5))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
print('Mean absolute Error',metrics.mean_absolute_error(y_test,pred))

print('Mean squared Error',metrics.mean_squared_error(y_test,pred))

print('Mean squared Error',np.sqrt(metrics.mean_absolute_error(y_test,pred)))
from sklearn.linear_model import LogisticRegression
model2=LogisticRegression()
model2.fit(X_train,y_train)
model2_pred=model2.predict(X_test)
print(classification_report(y_test,model2_pred))
print(accuracy_score(y_test,model2_pred))
r2_score(y_test,model2_pred)
print(confusion_matrix(y_test,model2_pred))
from sklearn.svm import LinearSVC
model3=LinearSVC(C=1000)
model3.fit(X_train,y_train)
model3_pred=model3.predict(X_test)
print(accuracy_score(y_test,model3_pred))
print(classification_report(y_test,model3_pred))
print(confusion_matrix(y_test,model3_pred))
from sklearn.neighbors import KNeighborsClassifier

KNNclassifier = KNeighborsClassifier(n_neighbors=5)

KNNclassifier.fit(X_train, y_train)
KNN_pred = KNNclassifier.predict(X_test)
print(confusion_matrix(y_test, KNN_pred))
print(accuracy_score(y_test,KNN_pred))
print(classification_report(y_test,KNN_pred))


error = []

# Calculating error for K values between 1 and 40

for i in range(1, 40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error.append(np.mean(pred_i != y_test))


plt.figure(figsize=(12, 6))

plt.plot(range(1, 40), error, color='black', linestyle='dashed', marker='.',

         markerfacecolor='black', markersize=10)

plt.title('Error Rate K Value')

plt.xlabel('K Value')

plt.ylabel('Mean Error')

from xgboost import XGBRegressor
model4=XGBRegressor(n_estimators=1000,learning_rate=0.05)

model4.fit(X_train,y_train,early_stopping_rounds=50,eval_set=[(X_test,y_test)],verbose=False)
model4_pred=model4.predict(X_test)
print(r2_score(y_test,model4_pred))
df4=pd.DataFrame({'Actual prediction ': y_test, 'Model Prediction': model4_pred})

df5=df.head(25)

print(df5)