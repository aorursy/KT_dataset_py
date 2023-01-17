import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn

import re

from sklearn import model_selection 

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_test.head()
df_train.shape
df_train.info()
df_train.describe()
df_train['Survived'].value_counts()
df_train['Embarked'].value_counts()
df_train['Died'] = 1 - df_train['Survived']
df_train.groupby('Pclass').agg('sum')[['Survived','Died']].plot(kind='bar', figsize=(25, 7), stacked=True, color=['c', 'm']);

plt.show()
df_train.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7), stacked=True, color=['b', 'y']);

plt.show()
plt.hist([df_train[df_train['Survived'] == 1]['Age'],df_train[df_train['Survived']== 0]['Age']], bins = 10, stacked = True, color = ['g','r'], label = ['Survived','Died'] )

plt.xlabel('Age')

plt.ylabel('Number of passengers')

plt.legend()

plt.show()
plt.hist([df_train[df_train['Survived'] == 1]['Fare'],df_train[df_train['Survived']== 0]['Fare']], bins = 10, stacked = True, color = ['g','r'], label = ['Survived','Died'] )

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend()

plt.show()
plt.figure(figsize = (30,10))

ax = plt.subplot()

ax.scatter(df_train[df_train['Survived'] == 1]['Age'], df_train[df_train['Survived'] == 1]['Fare'], c = 'm', s = df_train[df_train['Survived'] == 1]['Fare'])

ax.scatter(df_train[df_train['Survived'] == 0]['Age'], df_train[df_train['Survived'] == 0]['Fare'], c = 'y', s = df_train[df_train['Survived'] == 0]['Fare'])

plt.show()
# adults with largest ticket fare survived (magenta) and adults with low ticket fare died(yellow). 

#childern between 0-10 with magenta dots survived.  
label = df_train['Survived']

df_train = df_train.drop(['Survived','Died'], axis = 1)

df = df_train.append(df_test)

df.reset_index(inplace=True)

df = df.drop(['index','PassengerId'], axis = 1)

df.shape
df['Age'] = df['Age'].fillna(df['Age'].median()) # using median bcz more robust with outliers 
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df.head()
# Processing the column 'Cabin'. 

# Step 1. Replacing all the NaN with N 

# Step 2. Replacing other Cabin Numbers with their first letter.

# Step 3. Counting all the unique values and numbering them from 0 to the total number of unique values and replacing the Cabin column with those numbers. 
# Step 1

df['Cabin'].fillna('N',inplace = True)
# Step 2

for x,y in zip(df['Cabin'],range(df.shape[0])):

    df.loc[y,'Cabin'] = str(x)[0]
df.head()
# Step 3

cabin_list = list(df['Cabin'].unique())

for x,y in zip(df['Cabin'],range(df.shape[0])):

               df.loc[y,'Cabin'] = cabin_list.index(x)

    
df.head(10)
# Processing the column 'Embarked'. 

# Step 1. Replacing all the NaN with S which is the highest count in this column.  

# Step 2. Counting all the unique values and numbering them from 0 to the total number of unique values and replacing the Cabin column with those` numbers. 
df['Embarked'].value_counts()
# Step 1

df['Embarked'].fillna('S',inplace = True)
# Step 2

embarked_list = list(df['Embarked'].unique())

for x,y in zip(df['Embarked'],range(df.shape[0])):

               df.loc[y,'Embarked'] = embarked_list.index(x)
df.head(10)
# Processing the column Sex

# Step 1. Replacing 'male' as 0

# Step 2. Replacing 'female' as 1
# Step 1

df['Sex'].replace('male',0,inplace = True)
# Step 2

df['Sex'].replace('female',1,inplace = True)
df.head(10)
for name,y in zip(df['Name'],range(df.shape[0])):

    df.loc[y,'Name'] = name.split(',')[1].split('.')[0]
name_list = list(df['Name'].unique())

for x,y in zip(df['Name'],range(df.shape[0])):

    df.loc[y,'Name'] = name_list.index(x)
df.head(10)
for x,y in zip(df['Ticket'],range(df.shape[0])):

    x = x.replace('/','')

    x = x.replace('.','')

    p = re.findall('[a-zA-Z0-9]',x)

    for i in range(1):

        if p[i].isdigit():

            df.loc[y,'Ticket'] = 'x'

            continue

        else:

            df.loc[y,'Ticket'] = p[0] + p[1]    
df.head(10)
Ticket_list = list(df['Ticket'].unique())

for x,y in zip(df['Ticket'],range(df.shape[0])):

    df.loc[y,'Ticket'] = Ticket_list.index(x)
df.head()
x = np.array(df[:891].astype(float))

x = preprocessing.scale(x)

y = np.array(label)

x_train,x_test,y_train,y_test = model_selection.train_test_split(x, y, test_size = 0.2,random_state=7)
model = LogisticRegression(solver='liblinear', multi_class='ovr')

kfold = model_selection.KFold(n_splits=10, random_state=7)

cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

print(cv_results.mean())
models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('QDA', QuadraticDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('RF', RandomForestClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))
results = []

names = []

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=7)

    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))
RF = RandomForestClassifier(bootstrap = True, n_estimators = 32, n_jobs = -1, max_depth = 16,  min_samples_split = 4, min_samples_leaf = 1)

RF.fit(x_train, y_train)

predictions = RF.predict(x_test)

pred_result = accuracy_score(y_test, predictions)

print(pred_result)



RF.fit(x_train, y_train)

predictions = RF.predict(x_test)

print(RF, ':', accuracy_score(y_test, predictions))

new_pred = RF.predict(np.array(df[891:].astype(float)))

new_pred_df = pd.DataFrame(new_pred, columns = ['Survived'])

new_pred_df['PassengerId'] = df_test['PassengerId']

new_pred_df = new_pred_df[['PassengerId','Survived']]

new_pred_df.to_csv('submission3.csv',index = False)