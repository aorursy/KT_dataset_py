# import the required packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization

import seaborn as sns # visualization

from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression, Ridge

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.preprocessing import scale, LabelEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# read the train dataset

df1 = pd.read_csv("../input/train.csv")

print(df1.shape)

df1.head()

# create a new dataset that we will use for further computations and recreate the same when required from df1.

titanic = df1.copy()
# check datatype of features

titanic.describe().T
# check the datatype of each feature

titanic.info()
print(titanic.isna().sum())

print(f"percentage of missing values:\nAge: {titanic['Age'].isna().sum()/len(titanic) :.2f}, \nCabin: {titanic['Cabin'].isna().sum()/len(titanic) :.2f}, \nEmbarked: {titanic['Embarked'].isna().sum()/len(titanic) :.2f}")
titanic.Embarked.fillna(titanic.Embarked.mode()[0], inplace=True)
# Convert all features into required datatype

# titanic.Survived = titanic.Survived.astype('object')

# titanic.Pclass = titanic.Pclass.astype('object')

# titanic.SibSp = titanic.SibSp.astype('object')

# titanic.Parch = titanic.Parch.astype('object')
titanic.drop('Cabin', axis=1, inplace=True)
survived_count = round(titanic.Survived.value_counts(normalize=True)*100, 2)

survived_count.plot.bar(title='Proportion of Survived and non-Survived passengers in dataset')

plt.xticks(rotation=0, fontsize=12)

plt.yticks(fontsize=12)

for x,y in zip([0,1],survived_count):

    plt.text(x,y,y,fontsize=12)

plt.show()
plt.subplots(figsize=(15,15))



# We check the number of passengers survived in each class.

ind = sorted(titanic.Pclass.unique())

sur_0 = titanic.Pclass[titanic['Survived'] == 0].value_counts().sort_index()

sur_1 = titanic.Pclass[titanic['Survived'] == 1].value_counts().sort_index()

total = sur_0.values+sur_1.values

sur_0_prop = np.true_divide(sur_0, total)*100

sur_1_prop = np.true_divide(sur_1, total)*100

plt.subplot(321)

plt.bar(ind, sur_1_prop.values, bottom=sur_0_prop.values, label='1')

plt.bar(ind, sur_0_prop.values, label='0')

plt.title("Number of Passengers survived in each class", fontsize=15)

for x,y,z in zip(ind,[100]*3,sur_1):

    plt.text(x,y,z,fontsize=12)

for x,y,z in zip(ind,sur_0_prop,sur_0):

    plt.text(x,y,z,fontsize=12)

plt.xticks(ind)





# plot survival proportion in Age, for it we create age bins

bins = [0,10,20,30,40,50,60,70,80]

names = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80']

df_temp = titanic.dropna()

df_temp['Age_bins'] = pd.cut(x=titanic.Age, bins=bins, labels=names, right=False)



ind = sorted(df_temp.Age_bins.unique()[:8])

age_0 = df_temp.Age_bins[df_temp['Survived'] == 0].value_counts().sort_index()

age_1 = df_temp.Age_bins[df_temp['Survived'] == 1].value_counts().sort_index()

total = age_0.values+age_1.values

age_0_prop = np.true_divide(age_0, total)*100

age_1_prop = np.true_divide(age_1, total)*100

plt.subplot(322)

plt.bar(ind, age_1_prop.values, bottom=age_0_prop.values, label='1')

plt.bar(ind, age_0_prop.values, label='0')

plt.title("Number of Passengers survived in each age group", fontsize=15)

for x,y,z in zip(ind,[100]*8,age_1):

    plt.text(x,y,z,fontsize=12)

for x,y,z in zip(ind,age_0_prop,age_0):

    plt.text(x,y,z,fontsize=12)



plt.legend(loc='upper right')

    

# check the proportion of passengers survived as per gender

ind = sorted(titanic.Sex.unique())

sex_0 = titanic.Sex[titanic['Survived'] == 0].value_counts().sort_index()

sex_1 = titanic.Sex[titanic['Survived'] == 1].value_counts().sort_index()

total = sex_0.values+sex_1.values

sex_0_prop = np.true_divide(sex_0, total)*100

sex_1_prop = np.true_divide(sex_1, total)*100

plt.subplot(323)

plt.bar(ind, sex_1_prop.values, bottom=sex_0_prop.values, label='1')

plt.bar(ind, sex_0_prop.values, label='0')

plt.title("Number of Passengers survived genderwise", fontsize=15)

for x,y,z in zip(ind,[100]*2,sex_1):

    plt.text(x,y,z,fontsize=12)

for x,y,z in zip(ind,sex_0_prop,sex_0):

    plt.text(x,y,z,fontsize=12)





# check the proportion of passengers survived from port embarked

ind = sorted(titanic.Embarked.unique())

emb_0 = titanic.Embarked[titanic['Survived'] == 0].value_counts().sort_index()

emb_1 = titanic.Embarked[titanic['Survived'] == 1].value_counts().sort_index()

total = emb_0.values+emb_1.values

emb_0_prop = np.true_divide(emb_0, total)*100

emb_1_prop = np.true_divide(emb_1, total)*100

plt.subplot(324)

plt.bar(ind, emb_1_prop.values, bottom=emb_0_prop.values, label='1')

plt.bar(ind, emb_0_prop.values, label='0')

plt.title("Number of Passengers survived from port embarked", fontsize=15)

for x,y,z in zip(ind,[100]*3,emb_1):

    plt.text(x,y,z,fontsize=12)

for x,y,z in zip(ind,emb_0_prop,emb_0):

    plt.text(x,y,z,fontsize=12)





# check the proportion of passengers survived with Siblings and Spouse

ind = sorted(titanic.SibSp.unique())

sib_0 = titanic.SibSp[titanic['Survived'] == 0].value_counts().sort_index()

sib_1 = titanic.SibSp[titanic['Survived'] == 1].value_counts().sort_index()

sib_1 = titanic.SibSp[titanic['Survived'] == 1].value_counts().sort_index()

for i in sib_0.index:

    if i not in sib_1.index:

        sib_1[i]=0

total = sib_0.values+sib_1.values

sib_0_prop = np.true_divide(sib_0, total)*100

sib_1_prop = np.true_divide(sib_1, total)*100

plt.subplot(325)

plt.bar(ind, sib_1_prop.values, bottom=sib_0_prop.values, label='1')

plt.bar(ind, sib_0_prop.values, label='0')

plt.title("Number of Passengers survived with Siblings and Spouse onboard", fontsize=15, loc='center')

for x,y,z in zip(ind,[100]*9,sib_1):

    plt.text(x,y,z,fontsize=12)

for x,y,z in zip(ind,sib_0_prop,sib_0):

    plt.text(x,y,z,fontsize=12)

plt.xticks(ind)





ind = sorted(titanic.Parch.unique())

par_0 = titanic.Parch[titanic['Survived'] == 0].value_counts().sort_index()

par_1 = titanic.Parch[titanic['Survived'] == 1].value_counts().sort_index()

for i in par_0.index:

    if i not in par_1.index:

        par_1[i]=0

total = par_0.values+par_1.values

par_0_prop = np.true_divide(par_0, total)*100

par_1_prop = np.true_divide(par_1, total)*100

plt.subplot(326)

plt.bar(ind, par_1_prop.values, bottom=par_0_prop.values, label='1')

plt.bar(ind, par_0_prop.values, label='0')

plt.title("Number of Passengers survived with Parents and Children onboard", fontsize=15, loc='left')

for x,y,z in zip(ind,[100]*7,par_1):

    plt.text(x,y,z,fontsize=12)

for x,y,z in zip(ind,par_0_prop,par_0):

    plt.text(x,y,z,fontsize=12)

plt.xticks(ind)



plt.show()
# check the distribution of Fare in data. For better prediction results we need the continous data to be nearly closely normally distributed

plt.subplots(figsize=(15,6))

plt.subplot(121)

sns.distplot(titanic.Fare)

# We can see the data is positively skewed. We remove the skewness by taking natlog of the Fare column values

# we have 0s in Fare column which we replace with 0.0001 so as to have the record in data but not change it's meaning

titanic.Fare[titanic['Fare'] == 0] = 0.0001

Fare_ln = np.log(titanic.Fare)

plt.subplot(122)

sns.distplot(Fare_ln)

plt.show()
# separate the missing value records from titanic data for imputation

age_X = titanic[titanic['Age'].notna()].drop(['Age','PassengerId','Name','Ticket','Fare','Survived'], axis=1)

age_y = titanic.Age[titanic['Age'].notna()]

age_test = titanic[titanic['Age'].isna()].drop(['Age','PassengerId','Name','Ticket','Fare','Survived'], axis=1)
age_X[['SibSp','Parch']] = MinMaxScaler().fit_transform(age_X[['SibSp','Parch']])

age_test[['SibSp','Parch']] = MinMaxScaler().fit_transform(age_test[['SibSp','Parch']])



age_X = pd.get_dummies(age_X)

age_test = pd.get_dummies(age_test)
linreg = LinearRegression()

linreg.fit(age_X, age_y)

age_pred = linreg.predict(age_test)
titanic.Age[titanic['Age'].isna()] = age_pred
titanic.isna().sum()
X_train = titanic.drop(['Survived','PassengerId','Name','Ticket','Fare'], axis=1)

col = X_train.columns

y_train = titanic.Survived

y_train = y_train.astype('int')
X_train[['SibSp','Parch','Age']] = MinMaxScaler().fit_transform(X_train[['SibSp','Parch','Age']])

X_train = pd.get_dummies(X_train)
logreg = LogisticRegression().fit(X_train,y_train)
# Read the test data

test = pd.read_csv('../input/test.csv')

test = test[col]

test.isna().sum()
# separate the missing value records from test for imputation

age_test = test[test['Age'].isna()].drop('Age', axis=1)



age_test[['SibSp','Parch']] = MinMaxScaler().fit_transform(age_test[['SibSp','Parch']])



age_test = pd.get_dummies(age_test)



test.Age[test['Age'].isna()] = linreg.predict(age_test)

test.isna().sum()
test[['SibSp','Parch','Age']] = MinMaxScaler().fit_transform(test[['SibSp','Parch','Age']])

test = pd.get_dummies(test)

y_pred = logreg.predict(test)
submit = pd.read_csv('../input/gender_submission.csv')

submit.head()
submit['Survived'] = y_pred

# submit.to_csv('gender_submission.csv', index=False)
import keras

from keras.layers import Dense

from keras.models import Sequential

from keras.callbacks import EarlyStopping

earl = EarlyStopping(patience=3) # early stopping

#Setting up the model

model1 = Sequential()

model2 = Sequential()

#Add first layer

model1.add(Dense(50,activation='relu',input_shape=(titanic.shape[1],)))

model2.add(Dense(100,activation='relu',input_shape=(titanic.shape[1],)))

#Add second layer

model1.add(Dense(32,activation='relu'))

model2.add(Dense(50,activation='relu'))

#Add output layer

model1.add(Dense(2,activation='sigmoid'))

model2.add(Dense(2,activation='sigmoid'))

#Compile the model

model1.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])

model2.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
X_train = titanic.drop(['Survived','PassengerId','Name','Ticket','Fare'], axis=1).values

y_train = titanic.Survived.values

y_train = y_train.astype('int')