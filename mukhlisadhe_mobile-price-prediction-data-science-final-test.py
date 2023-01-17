import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
dataset=pd.read_csv('../input/train.csv')
dataset.head()
dataset.info()
dataset.describe()
sns.jointplot(x='ram',y='price_range',data=dataset,color='red',kind='kde');
sns.pointplot(y="int_memory", x="price_range", data=dataset)
labels = ["3G-supported",'Not supported']

values=dataset['three_g'].value_counts().values
fig1, ax1 = plt.subplots()

ax1.pie(values, labels=labels, autopct='%1.1f%%',shadow=True,startangle=90)

plt.show()

labels4g = ["4G-supported",'Not supported']

values4g = dataset['four_g'].value_counts().values

fig1, ax1 = plt.subplots()

ax1.pie(values4g, labels=labels4g, autopct='%1.1f%%',shadow=True,startangle=90)

plt.show()
sns.boxplot(x="price_range", y="battery_power", data=dataset)
plt.figure(figsize=(10,6))

dataset['fc'].hist(alpha=0.5,color='blue',label='Front camera')

dataset['pc'].hist(alpha=0.5,color='red',label='Primary camera')

plt.legend()

plt.xlabel('MegaPixels')
sns.jointplot(x='mobile_wt',y='price_range',data=dataset,kind='kde');
sns.pointplot(y="talk_time", x="price_range", data=dataset)
X=dataset.drop('price_range',axis=1)
y=dataset['price_range']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
from sklearn import svm

svm_model = svm.SVC()
svm_model.fit(X_train,y_train)
svm_model.score(X_test,y_test)
# from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train,y_train)
gnb.score(X_test,y_test)
plt.plot(y_test,y_pred)
from sklearn.metrics import classification_report,confusion_matrix
pred = gnb.predict(X_test)
print(classification_report(y_test,pred))
matrix=confusion_matrix(y_test,pred)

print(matrix)
plt.figure(figsize = (10,7))

sns.heatmap(matrix,annot=True)

data_test=pd.read_csv('../input/test.csv')
data_test.head()
data_test=data_test.drop('id',axis=1)
data_test.head()
predicted_price=gnb.predict(data_test)
predicted_price
data_test['price_range']=predicted_price
data_test