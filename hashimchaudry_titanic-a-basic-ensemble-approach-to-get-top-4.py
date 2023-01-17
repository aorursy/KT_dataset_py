# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
gs = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
train = pd.read_csv("/kaggle/input/titanic/train.csv")

train.head()
test = pd.read_csv("/kaggle/input/titanic/test.csv")

test.head()
train.describe()
train.isnull().sum()
test.isnull().sum()
pd.pivot_table(train, index = 'Survived', values = ['Age','SibSp','Parch','Fare'])
#distributions for all numeric variables 

for i in train[['Age','SibSp','Parch','Fare']].columns:

    plt.hist(train[i])

    plt.title(i)

    plt.show()
sns.barplot(train['Survived'].value_counts().index,train['Survived'].value_counts()).set_title('Survived')

plt.show()
pd.pivot_table(train, index = 'Survived', columns = 'Pclass', values = 'Ticket' ,aggfunc ='count')
sns.barplot(train['Pclass'].value_counts().index,train['Pclass'].value_counts()).set_title('Pclass')

plt.show()
pd.pivot_table(train, index = 'Survived', columns = 'Sex', values = 'Ticket' ,aggfunc ='count')
sns.barplot(train['Sex'].value_counts().index,train['Sex'].value_counts()).set_title('Sex')

plt.show()
pd.pivot_table(train, index = 'Survived', columns = 'Embarked', values = 'Ticket' ,aggfunc ='count')
sns.barplot(train['Embarked'].value_counts().index,train['Embarked'].value_counts()).set_title('Embarked')

plt.show()
sns.barplot(train['Cabin'].value_counts().index,train['Cabin'].value_counts()).set_title('Cabin')

plt.show()
sns.barplot(train['Ticket'].value_counts().index,train['Ticket'].value_counts()).set_title('Ticket')

plt.show()
#Check correlation between our features.

print(train.corr())

sns.heatmap(train.corr())
#Creating variable for total family.

train['Fam'] = train['SibSp'] + train['Parch']

train.head()
#Get cabin Letters from the cabin column.



train['cabin_adv'] = train.Cabin.apply(lambda x: str(x)[0])



#comparing surivial rate by cabin

print(train.cabin_adv.value_counts())

pd.pivot_table(train,index='Survived',columns='cabin_adv', values = 'Name', aggfunc='count')

#make things simple drop T since theres only 1 row.



train[train['cabin_adv'] == 'T'].index

train.drop(index = 339, inplace = True)

train.cabin_adv.value_counts()
#Getting the numbers and letters out of the tickets column

train['numeric_ticket'] = train.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)

train['ticket_letters'] = train.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)



#lets us view all rows in dataframe through scrolling. This is for convenience 

pd.set_option("max_rows", None)

train['ticket_letters'].value_counts()
#survival rate across different tyicket types 

pd.pivot_table(train,index='Survived',columns='ticket_letters', values = 'Ticket', aggfunc='count')
#feature engineering on person's title 

train.Name.head(50)

train['name_title'] = train.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())



#mr., ms., master. etc



train['name_title'].value_counts()
pd.pivot_table(train,index='Survived',columns='name_title', values = 'Ticket', aggfunc='count')
test['Fam'] = test['SibSp'] + test['Parch']

test['cabin_adv'] = test.Cabin.apply(lambda x: str(x)[0])

test['numeric_ticket'] = test.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)

test['ticket_letters'] = test.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)

test.Name.head(50)

test['name_title'] = test.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

# test.head()

test['cabin_adv'].value_counts()
#impute nulls for continuous data 

train.Age = train.Age.fillna(train.Age.median())

train.Fare = train.Fare.fillna(train.Fare.median())

#drop null 'embarked' rows. Only 2 instances of this in training and 0 in test 

train.dropna(subset=['Embarked'],inplace = True)

train.drop(columns=['Name', 'PassengerId', 'Cabin'], inplace = True)

train.head()
#impute nulls for continuous data 

test.Age = test.Age.fillna(train.Age.median())

test.Fare = test.Fare.fillna(train.Fare.median())



#drop null 'embarked' rows. Only 2 instances of this in training and 0 in test 

test.dropna(subset=['Embarked'],inplace = True)

passid = test['PassengerId'].copy()

test.drop(columns=['Name', 'PassengerId', 'Cabin'], inplace = True)

test.head()
df = pd.concat([train.assign(ind=1), test.assign(ind=0)])

df_hot = pd.get_dummies(df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked', 'Fam', 'cabin_adv', 'numeric_ticket', 'ticket_letters', 'name_title', 'ind']])

#notice that I left out our target column.

df_hot.head()
test_hot, train_hot = df_hot[df_hot["ind"].eq(0)], df_hot[df_hot["ind"].eq(1)]

# Scale data 

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

train_hot[['Age','SibSp','Parch','Fare', 'Fam']]= scale.fit_transform(train_hot[['Age','SibSp','Parch','Fare', 'Fam']])

train_hot.head()
scale = StandardScaler()

test_hot[['Age','SibSp','Parch','Fare', 'Fam']]= scale.fit_transform(test_hot[['Age','SibSp','Parch','Fare', 'Fam']])

test_hot.head()
Y_train = train['Survived']

Y_train.head()
print(train_hot.shape)

print(test_hot.shape)

print(Y_train.shape)

train_hot.head()
test_hot.head()
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from xgboost import XGBClassifier

xgb = XGBClassifier(random_state =1, learning_rate = 0.5, min_child_weight = 0.03)

cv = cross_val_score(xgb,train_hot,Y_train,cv=5)

print(cv)

print(cv.mean())
rfc = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=2)

cv = cross_val_score(rfc,train_hot,Y_train,cv=5)

print(cv)

print(cv.mean())
svc = SVC(probability = True)

cv = cross_val_score(svc,train_hot,Y_train,cv=5)

print(cv)

print(cv.mean())
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(

    train_hot, Y_train, test_size=0.33, random_state=42)
svc = SVC(probability = True)

svc.fit(X_train, y_train)
xgb = XGBClassifier(random_state =1, learning_rate = 0.5, min_child_weight = 0.03)

xgb.fit(X_train, y_train)
from keras.models import Sequential

from keras.layers import Dense

dl1 = Sequential()

dl1.add(Dense(1002, activation='relu'))

dl1.add(Dense(512, activation='relu'))

dl1.add(Dense(1, activation='sigmoid'))

# compile the keras model

dl1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset

dl1.fit(X_train, y_train, epochs=150, batch_size=10)

# evaluate the keras model

_, accuracy = dl1.evaluate(X_train, y_train)

print('Accuracy: %.2f' % (accuracy*100))
dl2 = Sequential()

dl2.add(Dense(1002, activation='relu'))

dl2.add(Dense(512, activation='relu'))

dl2.add(Dense(256, activation='relu'))

# dl2.add(Dense(128, activation='relu'))

dl2.add(Dense(1, activation='sigmoid'))

# compile the keras model

dl2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset

dl2.fit(X_train, y_train, epochs=150, batch_size=10)

# evaluate the keras model

_, accuracy = dl2.evaluate(X_train, y_train)

print('Accuracy: %.2f' % (accuracy*100))
knn = KNeighborsClassifier(algorithm= 'auto', n_neighbors = 7,weights= 'uniform')

knn.fit(X_train, y_train)
lr = LogisticRegression(max_iter = 2000,

              solver = 'liblinear')

lr.fit(X_train, y_train)
rf = RandomForestClassifier(random_state = 1)

rf.fit(X_train, y_train)
#Valid preds

valid_preds1 = svc.predict(X_valid)

valid_preds2 = xgb.predict(X_valid)

valid_preds_ur3 = dl1.predict(X_valid)

valid_preds3 = [round(x[0]) for x in valid_preds_ur3]

valid_preds_ur4 = dl2.predict(X_valid)

valid_preds4 = [round(x[0]) for x in valid_preds_ur4]

valid_preds5 = knn.predict(X_valid)

valid_preds6 = lr.predict(X_valid)

valid_preds7 = rf.predict(X_valid)
valid_preds6
valid_preds7
from sklearn.metrics import accuracy_score

print("SVC:", accuracy_score(y_valid, valid_preds1))

print("XGB:", accuracy_score(y_valid, valid_preds2))

print("DL1:", accuracy_score(y_valid, valid_preds3))

print("DL2:", accuracy_score(y_valid, valid_preds4))

print("KNN:", accuracy_score(y_valid, valid_preds5))

print("LR:", accuracy_score(y_valid, valid_preds5))

print("RF:", accuracy_score(y_valid, valid_preds5))
#Test preds

test_preds1 = svc.predict(test_hot)

test_preds2 = xgb.predict(test_hot)

test_preds_ur3 = dl1.predict(test_hot)

test_preds3 = [round(x[0]) for x in test_preds_ur3]

test_preds_ur4 = dl2.predict(test_hot)

test_preds4 = [round(x[0]) for x in test_preds_ur4]

test_preds5 = knn.predict(test_hot)

test_preds6 = lr.predict(test_hot)

test_preds7 = rf.predict(test_hot)
test_preds6
test_preds7
stacked_predictions = np.column_stack([valid_preds1,valid_preds2,valid_preds3,valid_preds4, valid_preds5, valid_preds6, valid_preds7])

stacked_test_predictions = np.column_stack([test_preds1,test_preds2,test_preds3,test_preds4, test_preds5, test_preds6, test_preds7])
stacked_predictions
stacked_test_predictions
from sklearn.linear_model import LinearRegression

#Making our Final predictions of ensemble.

meta_model = LinearRegression()

meta_model.fit(stacked_predictions, y_valid)

meta_predictions_ur = meta_model.predict(stacked_test_predictions)

preds = [round(x) for x in meta_predictions_ur]

preds
#Checking out the sample submission

gs.head()
df = pd.DataFrame({'PassengerId': passid,

                  'Survived' : preds})

df.Survived = df.Survived.astype(int)

print(df.shape)

df.head()
df['Survived'].sum()
df.to_csv('submission_final.csv', index =False)