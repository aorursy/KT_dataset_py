# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')

df_test = pd.read_csv('/kaggle/input/mobile-price-classification/test.csv')
df_train.head()
df_test.head()
df_train.describe()
df_train.info()
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df_train.corr()
fig = plt.figure(figsize=(15,12))

sns.heatmap(df_train.corr())
df_train['price_range'].unique()
sns.pairplot(df_train,hue='price_range')
plt.hist(df_train['battery_power'])

plt.show()
plt.hist(df_train['ram'])

plt.show()
sns.countplot(df_train['price_range'])

plt.show()
sns.boxplot(df_train['price_range'],df_train['talk_time'])
sns.countplot(df_train['dual_sim'])

plt.show()
sns.boxplot(df_train['dual_sim'],df_train['price_range'])
plt.hist(df_train['clock_speed'])
sns.boxplot(df_train['price_range'],df_train['clock_speed'])
sns.boxplot(df_train['fc'],df_train['price_range'])

plt.show()
df_train['n_cores'].unique()
sns.boxplot(df_train['wifi'],df_train['price_range'])

plt.show()
labels = ["3G-supported",'Not supported']

values = df_train['three_g'].value_counts().values
fig1, ax1 = plt.subplots()

colors = ['gold', 'lightskyblue']

ax1.pie(values, labels=labels, autopct='%1.1f%%',shadow=True,startangle=90,colors=colors)

plt.show()
labels = ["4G-supported",'Not supported']

values = df_train['four_g'].value_counts().values

fig1, ax1 = plt.subplots()

colors = ['gold', 'lightskyblue']

ax1.pie(values, labels=labels, autopct='%1.1f%%',shadow=True,startangle=90,colors=colors)

plt.show()
plt.figure(figsize=(10,6))

df_train['fc'].hist(alpha=0.5,color='blue',label='Front camera')

df_train['pc'].hist(alpha=0.5,color='red',label='Primary camera')

plt.legend()

plt.xlabel('MegaPixels')
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



scaler = StandardScaler()

x = df_train.drop('price_range',axis=1)

y = df_train['price_range']



scaler.fit(x)

x_transformed = scaler.transform(x)



x_train,x_test,y_train,y_test = train_test_split(x_transformed,y,test_size=0.3)
#Linear Regression

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(x_train,y_train)
lm.score(x_train,y_train)
#Logistic Regression



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix,classification_report





model = LogisticRegression()

model.fit(x_train,y_train)

y_train_pred = model.predict(x_train)

y_test_pred = model.predict(x_test)





print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))

print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))

print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))

print("\nClassification Report:\n%s"%classification_report(y_test_pred,y_test))

#KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(x_train,y_train)

knn.score(x_test,y_test)
pred = knn.predict(x_test)
error_rate = []

for i in range(1,20):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_i = knn.predict(x_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,20),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=5)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
print(classification_report(y_test,pred))
matrix=confusion_matrix(y_test,pred)

print(matrix)
#Decision Tree

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(x_train,y_train)
dtree.score(x_test,y_test)
#Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(x_train, y_train)
rfc.score(x_test,y_test)
#SVM

from sklearn.svm import SVC

model = SVC()

model.fit(x_train,y_train)

y_train_pred = model.predict(x_train)

y_test_pred = model.predict(x_test)





print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))

print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))

print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))

print("\nClassificationReport:\n%s"%classification_report(y_test_pred,y_test))
#Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()

model.fit(x_train,y_train)

y_train_pred = model.predict(x_train)

y_test_pred = model.predict(x_test)





print("Train Set Accuracy:"+str(accuracy_score(y_train_pred,y_train)*100))

print("Test Set Accuracy:"+str(accuracy_score(y_test_pred,y_test)*100))

print("\nConfusion Matrix:\n%s"%confusion_matrix(y_test_pred,y_test))

print("\nClassificationReport:\n%s"%classification_report(y_test_pred,y_test))