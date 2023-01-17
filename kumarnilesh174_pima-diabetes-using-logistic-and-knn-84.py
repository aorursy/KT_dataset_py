

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





import os

print(os.listdir("../input"))

df= pd.read_csv('../input/diabetes.csv')
df
# shape

print(df.shape)
#columns*rows

df.size
df.isnull().sum()
print(df.info())
df.head(5)
df.tail()
df.sample(5)
df.describe()
df.isnull().sum()
df.columns
# histograms

df.hist(figsize=(16,48))

plt.figure()
df.hist(figsize=(8,8))

plt.show()
# Using seaborn pairplot to see the bivariate relation between each pair of features

sns.pairplot(df)
sns.heatmap(df.corr())
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.distplot(df['Pregnancies'])
plt.figure(figsize=(8,6))

sns.countplot(x='Outcome',data=df)

plt.title('Positive Outcome to Diabetes in Dataset')

plt.ylabel('Number of People')

plt.show()
plt.figure(figsize=(10,6))

sns.barplot(data=df,x='Outcome',y='Pregnancies')

plt.title('Pregnancies Among Diabetes Outcomes.')

plt.show()
plt.figure(figsize=(10,6))

sns.countplot(x='Pregnancies',data=df,hue='Outcome')

plt.title('Diabetes Outcome to Pregnancies')

plt.show()
plt.figure(figsize=(13,6))

sns.countplot(x='Age',data=df,hue='Outcome')

plt.title('Diabetes Outcome to Age')

plt.show()
plt.figure(figsize=(13,6))

sns.countplot(x='SkinThickness',data=df,hue='Outcome')

plt.title('Diabetes Outcome to SkinThickness')

plt.show()
X=df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age']]

y=df['Outcome']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Outcome',axis=1), 

                                                    df['Outcome'], test_size=0.2, 

                                                    random_state=201)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

#** Train/fit lm on the training data.**

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
#Standardize the Variables

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('Outcome',axis=1))
scaled_features = scaler.transform(df.drop('Outcome',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])

df_feat.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['Outcome'],

                                                    test_size=0.10,random_state=200)
#Using KNN

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
#Predictions and Evaluations

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
#Choosing a K Value

error_rate = []



# Will take some time

for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1

knn = KNeighborsClassifier(n_neighbors=1)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=1')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
# NOW WITH K=19

knn = KNeighborsClassifier(n_neighbors=16)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



#print('WITH K=19')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))