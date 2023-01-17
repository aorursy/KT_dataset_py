# import libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.model_selection import train_test_split
# Read input data

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
# A glance of the data - Train

df_train.head()
# A glance of the data - test

df_test.head()
# Concat train set and test test to do data cleaning

df_all = pd.concat([df_train, df_test], ignore_index=True)

df_all
# Data Description

df_all.info()
# Check null data

df_all.isnull().sum()
# Fill value for null data (except cabin)

df_all['Embarked'].fillna(df_all['Embarked'].mode()[0], inplace = True)

df_all['Fare'].fillna(df_all['Fare'].median(), inplace = True)

df_all['Age'].fillna(df_all['Age'].median(), inplace = True)

df_all['Survived'].fillna(9, inplace = True)
# convert object values to category using LabelEncoder()

label = LabelEncoder()

df_all['Sex_Code'] = label.fit_transform(df_all['Sex'])

df_all['Ticket_Code'] = label.fit_transform(df_all['Ticket'])

df_all['Embarked_Code'] = label.fit_transform(df_all['Embarked'])
# Add "Family Size" column = Siblings/Spouse + Parents/Children + Self

df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1
# Use Name column to determine title

df_all['Title'] = df_all['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

# Look for Title appears less than 10 times 

stat_min = 10

title_names = (df_all['Title'].value_counts() < stat_min)

df_all['Title'] = df_all['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
df_all['Title'].value_counts()
df_all['Title_Code'] = label.fit_transform(df_all['Title'])
# Check null data again

df_all.isnull().sum()
df_all
# Get back our original train set

df_train = df_all.loc[df_all['Survived'].isin([0,1])]
plt.figure(figsize=[16,12])



# Pclass

plt.subplot(231)

plt.hist(x = [df_train[df_train['Survived']==1]['Pclass'], df_train[df_train['Survived']==0]['Pclass']], 

         stacked=True, 

         color = ['g','r'],

         label = ['Survived','Dead'])

plt.title('Pclass Histogram by Survival')

plt.xlabel('Pclass ($)')

plt.ylabel('# of Passengers')

plt.legend()



# Sex

plt.subplot(232)

plt.hist(x = [df_train[df_train['Survived']==1]['Sex'], df_train[df_train['Survived']==0]['Sex']], 

         stacked=True, 

         color = ['g','r'],

         label = ['Survived','Dead'])

plt.title('Sex Histogram by Survival')

plt.xlabel('Sex')

plt.ylabel('# of Passengers')

plt.legend()



# Age

plt.subplot(233)

plt.hist(x = [df_train[df_train['Survived']==1]['Age'], df_train[df_train['Survived']==0]['Age']], 

         stacked=True, 

         color = ['g','r'],

         label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age')

plt.ylabel('# of Passengers')

plt.legend()



# Fare

plt.subplot(234)

plt.hist(x = [df_train[df_train['Survived']==1]['Fare'], df_train[df_train['Survived']==0]['Fare']], 

         stacked=True, 

         color = ['g','r'],

         label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()



# Embarked

plt.subplot(235)

plt.hist(x = [df_train[df_train['Survived']==1]['Embarked'], df_train[df_train['Survived']==0]['Embarked']], 

         stacked=True, 

         color = ['g','r'],

         label = ['Survived','Dead'])

plt.title('Embarked Histogram by Survival')

plt.xlabel('Embarked')

plt.ylabel('# of Passengers')

plt.legend()



# Title

plt.subplot(236)

plt.hist(x = [df_train[df_train['Survived']==1]['Title'], df_train[df_train['Survived']==0]['Title']], 

         stacked=True, 

         color = ['g','r'],

         label = ['Survived','Dead'])

plt.title('Title Histogram by Survival')

plt.xlabel('Title')

plt.ylabel('# of Passengers')

plt.legend()



plt.show()
# Features (columns) will be used

X_columns = ['Pclass', 'Sex_Code', 'Age', 'Family_Size', 'Fare', 'Ticket_Code', 'Embarked_Code', 'Title_Code']



# Convert to Array

X = df_train[X_columns].values

Y = df_train['Survived'].values
# Scale the data

scaler = preprocessing.StandardScaler()

scaler.fit(X)

X = scaler.transform(X.astype(float))

X 
# Split into Train and Cross Validation

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=3)

print ('Train set:', X_train.shape,  Y_train.shape)

print ('Test set:', X_test.shape,  Y_test.shape)
# I will use a bigger number in order to get a better K

Ks = 100

mean_acc = np.zeros((Ks-1))

std_acc = np.zeros((Ks-1))

ConfustionMx = [];

for n in range(1,Ks):

    # Train Model 

    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,Y_train)

    # Predict

    yhat=neigh.predict(X_test)

    # Accuracy

    mean_acc[n-1] = metrics.accuracy_score(Y_test, yhat)

    std_acc[n-1]=np.std(yhat==Y_test)/np.sqrt(yhat.shape[0])



mean_acc
# visualize the K

plt.plot(range(1,Ks),mean_acc,'g')

plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)

plt.legend(('Accuracy ', '+/- 3xstd'))

plt.ylabel('Accuracy ')

plt.xlabel('Number of Neighbors (K)')

plt.tight_layout()

plt.show()
# looks like 29 is the better option for KNN (81%)

neigh = KNeighborsClassifier(n_neighbors = 29).fit(X, Y)
# Get the original test set

df_test = df_all.loc[df_all['Survived'].isin([9])]

X_test_real = df_test[X_columns].values

# Scale the data using the same Scaler 

X_test_real = scaler.transform(X_test_real)

# Predict 

yhat=neigh.predict(X_test_real)

yhat.astype(int)
df_result = pd.DataFrame({'PassengerId':  df_test['PassengerId'], 'Survived': yhat.astype(int)})

df_result
df_result.to_csv('submission.csv', index=False)