import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

import tensorflow as tf

import tensorflow.keras as keras
# save train and test data paths into a variable



train_path = '/kaggle/input/titanic/train.csv'

test_path = '/kaggle/input/titanic/test.csv'

submission_file = 'submission.csv'
# read train and test data from paths



train_data = pd.read_csv(train_path)

test_data = pd.read_csv(test_path)
# let's observe the train_data shape and head



print(train_data.shape)

print(train_data.head())
# split features and labels

train_labels = train_data.loc[:,'Survived']



# get test passengers id for the submission later

test_passenger_id = test_data.loc[:,'PassengerId']



# drop PassengerId, Name and Ticket columns as each row has an unique value for these features

train_data = train_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], errors='ignore')

test_data = test_data.drop(columns=['PassengerId', 'Name', 'Ticket'], errors='ignore')



print(train_data.head())
# generating a new feature Family by summing  SibSp and Parch columns

train_data['Family'] = train_data.loc[:,['SibSp', 'Parch']].sum(axis=1)

test_data['Family'] = test_data.loc[:,['SibSp', 'Parch']].sum(axis=1)



# dropping SibSp and Parch columns from data

train_data = train_data.drop(columns=['SibSp', 'Parch'], errors='ignore')

test_data = test_data.drop(columns=['SibSp', 'Parch'], errors='ignore')
# generating a new feature FareLog which is log(Fare + 10)

train_data['FareLog'] = np.log(train_data.loc[:,'Fare'] + 10)

test_data['FareLog'] = np.log(test_data.loc[:,'Fare'] + 10)



# dropping Fare column

train_data = train_data.drop(columns=['Fare'], errors='ignore')

test_data = test_data.drop(columns=['Fare'], errors='ignore')
train_data['HasCabin'] = pd.notnull(train_data['Cabin']) + 0

test_data['HasCabin'] = pd.notnull(test_data['Cabin']) + 0



# dropping Cabin column

train_data = train_data.drop(columns=['Cabin'], errors='ignore')

test_data = test_data.drop(columns=['Cabin'], errors='ignore')
# Sex and Embarked features label encoding



for feature in ['Sex', 'Embarked']:

    codes, _ = pd.factorize(train_data.loc[:,feature])

    train_data.loc[:,feature] = codes

    codes, _ = pd.factorize(test_data.loc[:,feature])

    test_data.loc[:, feature] = codes
train_data['HasFare'] = pd.notnull(train_data['FareLog']) + 0

test_data['HasFare'] = pd.notnull(test_data['FareLog']) + 0
# train missing fare values

rfr = RandomForestRegressor()

where = pd.notnull(train_data['FareLog'].append(test_data['FareLog']))

rfr.fit(train_data.append(test_data).drop(columns=['FareLog', 'Age'])[where], train_data.append(test_data)['FareLog'][where])



# predict missing values

train_where = pd.isnull(train_data['FareLog'])

test_where = pd.isnull(test_data['FareLog'])

test_data.loc[test_where,'FareLog'] = rfr.predict(test_data.drop(columns=['FareLog', 'Age'])[test_where])
train_data['HasAge'] = pd.notnull(train_data['Age']) + 0

test_data['HasAge'] = pd.notnull(test_data['Age']) + 0
# train missing age values

rfr = RandomForestRegressor()

where = pd.notnull(train_data['Age'].append(test_data['Age']))

rfr.fit(train_data.append(test_data).drop(columns=['Age'])[where], train_data.append(test_data)['Age'][where])



# predict missing values

train_where = pd.isnull(train_data['Age'])

test_where = pd.isnull(test_data['Age'])

train_data.loc[train_where,'Age'] = rfr.predict(train_data.drop(columns=['Age'])[train_where])

test_data.loc[test_where,'Age'] = rfr.predict(test_data.drop(columns=['Age'])[test_where])
stsc = StandardScaler()

stsc.fit(test_data.append(train_data).loc[:,['Age', 'FareLog', 'Family']])

train_data.loc[:,['Age', 'FareLog', 'Family']] = stsc.transform(train_data.loc[:,['Age', 'FareLog', 'Family']])

test_data.loc[:,['Age', 'FareLog', 'Family']] = stsc.transform(test_data.loc[:,['Age', 'FareLog', 'Family']])
skf = StratifiedKFold(shuffle=True)



acc = []

c_val = []

for c in range(400, 500, 20):

    model = keras.Sequential([

        keras.layers.Dense(units=c, activation='relu'),

        keras.layers.Dense(units=c, activation='relu'),

        keras.layers.Dense(units=c, activation='relu'),

        keras.layers.Dense(units=1, activation='sigmoid')

    ])



    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    acc_val = []

    for t_idx, v_idx in skf.split(train_data, train_labels):

        t_data = train_data.iloc[t_idx,:]

        t_labels = train_labels[t_idx]

        v_data = train_data.iloc[v_idx,:]

        v_labels = train_labels[v_idx]

        history = model.fit(t_data, t_labels, validation_data=(v_data, v_labels), epochs=100, verbose=0)

        acc_val.append(history.history['val_acc'])

    

    acc.append(np.mean(acc_val))

    c_val.append(c)

    

plt.plot(c_val, acc)
model1 = keras.Sequential([

        keras.layers.Dense(units=c_val[np.argmax(acc)], activation='relu'),

        keras.layers.Dense(units=c_val[np.argmax(acc)], activation='relu'),

        keras.layers.Dense(units=c_val[np.argmax(acc)], activation='relu'),

        keras.layers.Dense(units=1, activation='sigmoid')

    ])



model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])



history = model1.fit(train_data, train_labels, epochs=200, verbose=0)

print(history.history['acc'][-1])



model2 = keras.Sequential([

        keras.layers.Dense(units=c_val[np.argmax(acc)], activation='relu'),

        keras.layers.Dense(units=c_val[np.argmax(acc)], activation='relu'),

        keras.layers.Dense(units=c_val[np.argmax(acc)], activation='relu'),

        keras.layers.Dense(units=1, activation='sigmoid')

    ])



model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])



history = model2.fit(train_data, train_labels, epochs=200, verbose=0)

print(history.history['acc'][-1])



model3 = keras.Sequential([

        keras.layers.Dense(units=c_val[np.argmax(acc)], activation='relu'),

        keras.layers.Dense(units=c_val[np.argmax(acc)], activation='relu'),

        keras.layers.Dense(units=c_val[np.argmax(acc)], activation='relu'),

        keras.layers.Dense(units=1, activation='sigmoid')

    ])



model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])



history = model3.fit(train_data, train_labels, epochs=200, verbose=0)

print(history.history['acc'][-1])



predicted_labels1 = model1.predict(test_data)

predicted_labels2 = model2.predict(test_data)

predicted_labels3 = model3.predict(test_data)

predicted_labels = (((predicted_labels1 + predicted_labels2 + predicted_labels3) / 3) >= 0.5) + 0

submission = pd.DataFrame({'PassengerId': test_passenger_id, 'Survived': predicted_labels.squeeze()})

submission.to_csv(submission_file, index=False)