import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split#splits arrays or matrices into a random train and test subsets

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

# Neural Network

import keras 

from keras.models import Sequential 

from keras.layers import Dense, Dropout

from keras.optimizers import Adam

from keras import regularizers

from keras.callbacks import EarlyStopping
#Load the dataset into Pandas dataframe 

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.describe()
train.isnull().sum(axis=0)
test.isnull().sum(axis=0)
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
train[['Pclass', 'Age']].groupby(['Pclass'], as_index=False).median()
test[['Pclass', 'Age']].groupby(['Pclass'], as_index=False).median()
test[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean()
#dealing with the ‘NaN’ values in the train dataset.As the NaN values are in only in Age,Cabin&Embarked

def impute_age_train(cols):

    Age = cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return 37 # Since Pclass[1] has a median 37

        elif Pclass == 2:

            return 29 # Since Pclass[2] has a median 29

        else:

            return 24 # Since Pclass[3] has a median 24

    else:

        return Age

train['Age'] = train[['Age', 'Pclass']].apply(impute_age_train, axis = 1)
#dealing with the ‘NaN’ values in the train dataset.As the NaN values are in only in Age,Cabin&Embarked

def impute_Embarked_train(train):

    freq_port_train = train.Embarked.dropna().mode()[0]

    train['Embarked'] = train['Embarked'].fillna(freq_port_train)

    train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    return train

train = impute_Embarked_train(train)
#dealing with the ‘NaN’ values in the test dataset.As the NaN values are in only in Age,Cabin&Fare

def impute_age_test(cols):

    Age = cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return 42 # Since Pclass[1] has a median 42

        elif Pclass == 2:

            return 26.5 # Since Pclass[2] has a median 26.5

        else:

            return 24 # Since Pclass[3] has a median 24

    else:

        return Age

test['Age'] = test[['Age', 'Pclass']].apply(impute_age_test, axis = 1)
#dealing with the ‘NaN’ values in the test dataset.As the NaN values are in only in Age,Cabin&Fare

def impute_fare_test(cols):

    Pclass = cols[0]

    Fare = cols[1]

    if pd.isnull(Fare):

        if Pclass == 1:

            return 60 # Since Pclass[1] has a median 60

        elif Pclass == 2:

            return 15.75 # Since Pclass[2] has a median 15.75

        else:

            return 7.896 # Since Pclass[3] has a median 7.896

    else:

        return Fare

test['Fare'] = test[['Pclass','Fare']].apply(impute_fare_test, axis = 1)
#Converting Categorial data into numerical

test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
from sklearn.preprocessing import LabelEncoder

#labeling genders

def gender(data):

    le = LabelEncoder()

    le.fit(["male", "female"])

    data["Sex"] = le.transform(data["Sex"])

    return data

from sklearn.preprocessing import MinMaxScaler

#Narmalizing age and fare

def normalize_data(data):

    mms = MinMaxScaler()

    data["Age"] = mms.fit_transform(data["Age"].values.reshape(-1, 1))

    data["Fare"] = mms.fit_transform(data["Fare"].values.reshape(-1, 1))

    return data
gender(train)

gender(test)

normalize_data(train)

normalize_data(test)
X_train = train.drop(["PassengerId", "Name", "Ticket", "Cabin","Survived"], axis=1).copy()

Y_train = train["Survived"].copy()

X_test  = test.drop(["PassengerId", "Name", "Ticket", "Cabin"],axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
X_test.head()
#Train Test Split

#X_train, X_test, Y_train, Y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'], test_size=0.25,random_state=101)
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

#acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

print(round(decision_tree.score(X_train, Y_train) * 100, 2))

#print(accuracy_score(Y_test,Y_pred)* 100)
Y_final = (Y_pred > 0.5).astype(int).reshape(X_test.shape[0])



output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': Y_final})

output.to_csv('pred_decision_tree.csv', index=False)

output.head()
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

#acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(round(random_forest.score(X_train, Y_train) * 100, 2))

#print(accuracy_score(Y_test,Y_pred)* 100)
Y_final = (Y_pred > 0.5).astype(int).reshape(X_test.shape[0])



output1 = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': Y_final})

output1.to_csv('pred_random_forest.csv', index=False)

output1.head()
model = Sequential()

# layers

model.add(Dense(32, kernel_initializer = 'uniform', activation = 'relu', kernel_regularizer=regularizers.l2(0.003),input_dim = X_train.shape[1]))

#model.add(Dropout(rate=0.2))

model.add(Dense(16, kernel_initializer = 'uniform', activation = 'relu',kernel_regularizer=regularizers.l2(0.002)))

#model.add(Dropout(rate=0.2))

model.add(Dense(5, kernel_initializer = 'uniform', activation = 'relu',kernel_regularizer=regularizers.l2(0.001)))

#model.add(Dropout(rate=0.1))

model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compile the model

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size = 16, epochs = 1000)

sc = model.evaluate(X_train,Y_train)

#score = model.evaluate(X_test,Y_test)

print("")

print("Train Accuracy:{0}".format(sc[1]))

#print("Test accuracy:{0}".format(score[1]))
Y_pred = model.predict(X_test)

Y_final = (Y_pred > 0.5).astype(int).reshape(X_test.shape[0])



output2 = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': Y_final})

output2.to_csv('pred_neural_nets_l2.csv', index=False)

output2.head()