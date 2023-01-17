# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



df_income = pd.read_csv('/kaggle/input/adult-income-dataset/adult.csv')
df_income.head()
df_income.shape
df_income.info()
df_income.replace('?',np.nan,inplace=True)

df_income.info()
#Dropping all NULL values

df_income.dropna(inplace=True)

df_income.info()
df_income.describe()
def impute_age(age):

    if((age>15) & (age<=30)):

        return 1

    elif((age>30) & (age<=45)):

        return 2

    elif((age>45) & (age<=60)):

        return 3

    elif((age>60) & (age<=75)):

        return 4

    else:

        return 5



df_income.age = df_income.age.apply(impute_age)



df_income.age.value_counts()
sns.countplot(data=df_income,x='income',hue='age')
plt.figure(figsize=(12,20))

sns.countplot(data=df_income,y='native-country',hue='income')
plt.figure(figsize=(12,6))

sns.countplot(data=df_income,x='workclass',hue='income')
plt.figure(figsize=(20,10))

sns.countplot(data=df_income,x='education',hue='income')
plt.figure(figsize=(15,8))

sns.countplot(data=df_income,x='marital-status',hue='income')
plt.figure(figsize=(25,10))

sns.countplot(data=df_income,x='occupation',hue='income')
plt.figure(figsize=(10,6))

sns.countplot(data=df_income,x='race',hue='income')
sns.countplot(data=df_income,x='gender',hue='income')
#Encoding the Categorical values to Numericals using LabelEncoder

from sklearn.preprocessing import LabelEncoder



Labelenc_workclass = LabelEncoder()

df_income['workclass'] = Labelenc_workclass.fit_transform(df_income['workclass'])



Labelenc_education = LabelEncoder()

df_income['education'] = Labelenc_education.fit_transform(df_income['education'])



Labelenc_marital_status = LabelEncoder()

df_income['marital-status'] = Labelenc_marital_status.fit_transform(df_income['marital-status'])



Labelenc_occupation = LabelEncoder()

df_income['occupation'] = Labelenc_occupation.fit_transform(df_income['occupation'])



Labelenc_relationship = LabelEncoder()

df_income['relationship'] = Labelenc_relationship.fit_transform(df_income['relationship'])



Labelenc_race = LabelEncoder()

df_income['race'] = Labelenc_race.fit_transform(df_income['race'])



Labelenc_gender = LabelEncoder()

df_income['gender'] = Labelenc_gender.fit_transform(df_income['gender'])



Labelenc_native_country = LabelEncoder()

df_income['native-country'] = Labelenc_native_country.fit_transform(df_income['native-country'])



Labelenc_income = LabelEncoder()

df_income['income'] = Labelenc_income.fit_transform(df_income['income'])



df_income.info()
#Creating a HeatMap



plt.figure(figsize=(10,8))

sns.heatmap(df_income.corr())
#Scaling the values using StandardScaler



from sklearn.preprocessing import StandardScaler

st_scaler = StandardScaler()



st_scaler.fit(df_income.drop('income',axis=1))



scaled_features = st_scaler.transform(df_income.drop('income',axis=1))
#Creating X & y for train test split & Splitting them for the model

X = pd.DataFrame(scaled_features,columns=df_income.columns[:-1])

y = df_income['income']



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
#Creating the KNN Model, running it for 1 Neighbour and predicting values 



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
#Creating Classification Report and Confusion Matrix

from sklearn.metrics import confusion_matrix,classification_report



print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
error_rate=[]

for i in range(1,30):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    y_pred_i = knn.predict(X_test)

    error_rate.append(np.mean(y_pred_i!=y_test))
#Plotting the error rate

plt.figure(figsize=(10,6))

plt.plot(range(1,30),error_rate,color='blue')

plt.xlabel("K values")

plt.ylabel("Error Rate")
knn_new = KNeighborsClassifier(n_neighbors=10)

knn_new.fit(X_train,y_train)

y_pred_new = knn_new.predict(X_test)
#Classification Report wth k=10



print(classification_report(y_test,y_pred_new))
#Confusion Matrix with k=10



print(confusion_matrix(y_test,y_pred_new))