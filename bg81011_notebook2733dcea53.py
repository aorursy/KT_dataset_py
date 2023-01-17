import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

%matplotlib inline
data = pd.read_csv('../input/train.csv')

testdata = pd.read_csv('../input/test.csv')



print(data.columns[1:])

print(testdata.columns[1:])
#To check NaN values

print(data.isnull().sum())

print(data.count())
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(1, 1, 1)

ax.set_title('Pie chart')

survived = len(data[data.Survived == 1])

died  = len(data[data.Survived == 0])

ax.pie(list(data.Survived.value_counts()), labels=['died', 'survived'])

print('Survival rate ', np.ceil(survived/(survived+died) * 100))
copy_data = data.groupby(['Sex', 'Survived'])

print((copy_data['Sex'].count()))

arr = list(copy_data['Sex'].count())

print(arr)



fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(2,2,1)

ax2 = fig.add_subplot(2,2,2)

ax3 = fig.add_subplot(2,2,(3,4))



ax1.bar(['died', 'survived'], arr[0:2] , label='female')

ax1.set_title('female')

ax2.bar(['died', 'survived'], arr[2:4], label='male')

ax2.set_title('male')



ind = np.arange(2) 

width = 0.35

ax3.bar(ind, arr[0::2] , width, label='died', color='r')

ax3.bar(ind + width, arr[1::2] , width, label='survived', color='b')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xticks(ind + width / 2, ('woman', 'man'))

ax3.set_title('sex')



plt.show()

pd.crosstab(data.Pclass,data.Survived, margins=True).style.background_gradient(cmap='summer_r')
arr = np.asarray(pd.crosstab(data.Pclass, data.Survived))



fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)



print(data['Pclass'].value_counts())

print(data['Pclass'].count())



ax1.bar(['1','2','3'], list(data['Pclass'].value_counts()), label='people')

ax1.legend()

ax1.set_title('the number of people by class')



n = np.arange(3) 

width = 0.35

ax2.bar(n, arr[:,0], width, label='died', color='r')

ax2.bar(n+width, arr[:,1], width, label='survived', color='g')

ax2.legend()

ax2.set_title('survived vs death')

plt.xticks(n + width / 3, ('1', '2', '3'))

sns.factorplot(x='Pclass', y='Survived', hue='Sex', data=data, fit_reg=False)
print('Oldest Passenger was of :', data['Age'].max(), 'Years')

print('Youngest Passenger was of :', data['Age'].min(), 'Years')

print('Average Passenger was of :', data['Age'].mean(), 'Years')
fig = plt.figure(figsize=(18,8))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)



sns.violinplot('Pclass', 'Age', hue='Survived', split=True, data=data, ax=ax1)

sns.violinplot('Sex', 'Age', hue='Survived', split=True, data=data, ax=ax2)



data['initial'] = 0

#ata['initial']

data['initial'] = data.Name.str.extract('([A-Za-z]+)\.')

data['initial'].value_counts()



testdata['initial'] = 0

#ata['initial']

testdata['initial'] = testdata.Name.str.extract('([A-Za-z]+)\.')

testdata['initial'].value_counts()



pd.crosstab(data.initial, data.Sex).style.background_gradient(cmap='summer_r')
data['initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady',

                         'Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs',

                         'Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)



testdata['initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady',

                         'Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs',

                         'Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
data.groupby('initial')['Age'].mean()
data.loc[(data.Age.isnull()) & (data.initial=='Mr'), 'Age'] = 33

data.loc[(data.Age.isnull()) & (data.initial=='Mrs'), 'Age'] = 36

data.loc[(data.Age.isnull()) & (data.initial=='Master'), 'Age'] = 5

data.loc[(data.Age.isnull()) & (data.initial=='Miss'), 'Age'] = 22

data.loc[(data.Age.isnull()) & (data.initial=='Other'), 'Age'] = 46
data.loc[data.Age.isnull()]



print(data.columns[1:])

print(testdata.columns[1:])
fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(1, 2, 1)

ax2 = fig.add_subplot(1, 2, 2)



data[data['Survived'] == 0].Age.plot.hist(ax=ax1, bins=20, color='r', edgecolor='b')

data[data['Survived'] == 1].Age.plot.hist(ax=ax2, bins=20, color='b', edgecolor='g')
sns.factorplot('Pclass', 'Survived', col='initial', data=data)
fig = plt.figure(figsize=(20, 15))

ax1 = fig.add_subplot(2, 2, 1)

ax2 = fig.add_subplot(2, 2, 2)

ax3 = fig.add_subplot(2, 2, 3)

ax4 = fig.add_subplot(2, 2, 4)



sns.countplot('Embarked', data=data, ax=ax1)

sns.countplot('Embarked', hue='Sex', data=data, ax=ax2)

sns.countplot('Embarked', hue='Survived', data=data, ax=ax3)

sns.countplot('Embarked', hue='Pclass', data=data, ax=ax4)



sns.factorplot('Pclass', 'Survived', hue='Sex', col='Embarked', data=data)
data['Embarked'].fillna('S', inplace=True)

data['Embarked'].isnull().any()
pd.crosstab(data.SibSp, data.Survived).style.background_gradient(cmap='summer_r')
sns.countplot('SibSp', hue='Survived', data=data)

plt.legend(bbox_to_anchor=(1,1), loc=2)
print('Highest Fare was:', data['Fare'].max())

print('Lowest Fare was:', data['Fare'].min())

print('Average Fare was', data['Fare'].mean())
fig = plt.figure(figsize=(20,45))

ax1 = fig.add_subplot(3, 2, 1)

ax2 = fig.add_subplot(3, 2, 2)

ax3 = fig.add_subplot(3, 2, 3)

ax4 = fig.add_subplot(3, 2, 4)

ax5 = fig.add_subplot(3, 2, 3)

ax6 = fig.add_subplot(3, 2, 4)



sns.countplot('Fare', data=data, ax = ax1)

sns.countplot('Fare', hue='Pclass', data=data, ax = ax2)



sns.distplot(data[data['Pclass'] == 1].Fare, ax = ax3)

sns.distplot(data[data['Pclass'] == 2].Fare, ax = ax4)



pd.crosstab(data.Pclass, data.Fare).style.background_gradient(cmap='summer_r')

fig = plt.figure(figsize=(10, 10))

sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)
data['Age_band'] = 0

data.loc[data['Age']<=16, 'Age_band'] = 0

data.loc[ (data['Age']> 16) & (data['Age']<= 32) , 'Age_band'] = 1

data.loc[ (data['Age']> 32) & (data['Age']<= 48) , 'Age_band'] = 2

data.loc[ (data['Age']> 48) & (data['Age']<= 64) , 'Age_band'] = 3

data.loc[data['Age']> 64, 'Age_band'] = 4

data.head()



testdata['Age_band'] = 0

testdata.loc[testdata['Age']<=16, 'Age_band'] = 0

testdata.loc[ (testdata['Age']> 16) & (testdata['Age']<= 32) , 'Age_band'] = 1

testdata.loc[ (testdata['Age']> 32) & (testdata['Age']<= 48) , 'Age_band'] = 2

testdata.loc[ (testdata['Age']> 48) & (testdata['Age']<= 64) , 'Age_band'] = 3

testdata.loc[testdata['Age']> 64, 'Age_band'] = 4

testdata.head()
data['Age_band'].value_counts()
sns.factorplot('Age_band', 'Survived', data=data, hue='Pclass')

sns.factorplot('Age_band', 'Survived', data=data)

plt.show()
data['Family_size'] = 0

data['Family_size'] = data['Parch'] + data['SibSp']

data['Alone'] = 0

data.loc[data['Family_size'] == 0, 'Alone'] = 1



fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(1, 2, 1)

ax2 = fig.add_subplot(1, 2, 2)



sns.factorplot('Family_size', 'Survived', data=data, ax=ax1)

sns.factorplot('Alone', 'Survived', data=data, ax=ax2)





testdata['Family_size'] = 0

testdata['Family_size'] = testdata['Parch'] + testdata['SibSp']

testdata['Alone'] = 0

testdata.loc[testdata['Family_size'] == 0, 'Alone'] = 1
sns.factorplot('Alone', 'Survived', data=data, hue='Sex', col='Pclass')
data['Fare_Range'] = pd.qcut(data['Fare'], 4)

testdata['Fare_Range'] = pd.qcut(data['Fare'], 4)



pd.crosstab(data.Fare_Range, data.Survived).style.background_gradient(cmap='summer_r')
data['Fare_cat'] = 0

data.loc[data['Fare']<=7.91, 'Fare_cat'] = 0

data.loc[ (data['Fare']>7.91) & (data['Fare']<=14.454), 'Fare_cat'] = 1

data.loc[ (data['Fare']>14.454) & (data['Fare']<=31), 'Fare_cat'] = 2

data.loc[ (data['Fare']>31) & (data['Fare']<=513), 'Fare_cat'] = 3



testdata['Fare_cat'] = 0

testdata.loc[testdata['Fare']<=7.91, 'Fare_cat'] = 0

testdata.loc[ (testdata['Fare']>7.91) & (testdata['Fare']<=14.454), 'Fare_cat'] = 1

testdata.loc[ (testdata['Fare']>14.454) & (testdata['Fare']<=31), 'Fare_cat'] = 2

testdata.loc[ (testdata['Fare']>31) & (testdata['Fare']<=513), 'Fare_cat'] = 3
sns.factorplot('Fare_cat', 'Survived', data=data, hue='Sex')





def get_series_ids(x):

    '''Function returns a pandas series consisting of ids, 

       corresponding to objects in input pandas series x

       Example: 

       get_series_ids(pd.Series(['a','a','b','b','c'])) 

       returns Series([0,0,1,1,2], dtype=int)'''



    values = np.unique(x)

    values2nums = dict(zip(values,range(len(values))))

    return x.replace(values2nums)



data['Sex'] = get_series_ids(data['Sex'])

data['Embarked'] = get_series_ids(data['Embarked'])

data['initial'] = get_series_ids(data['initial'])



testdata['Sex'] = get_series_ids(testdata['Sex'])

testdata['Embarked'] = get_series_ids(testdata['Embarked'])

testdata['initial'] = get_series_ids(testdata['initial'])



print(data['initial'].value_counts())





fig = plt.figure(figsize=(15, 15))

data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'], axis=1,inplace=True)

testdata.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'], axis=1, inplace=True)

print(data.columns)

print(testdata.columns)



sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})

#importing all the required ML packages

from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn import svm #support vector Machine

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.model_selection import train_test_split #training and testing data split

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix
train, test = train_test_split(data, test_size = 0.2)



X_train = train[train.columns[1:]]

Y_train = train[train.columns[:1]]



X_test = test[test.columns[1:]]

Y_test = test[test.columns[:1]]



X = data[data.columns[1:]]

Y = data['Survived']



print(X_train)

print(Y_train)

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Dropout

from keras.utils.np_utils import to_categorical

from keras.callbacks import ModelCheckpoint



X_train = data[data.columns[1:]]

Y_train = data[data.columns[:1]]



X_train = X_train.as_matrix()

Y_train_one_hot = to_categorical(Y_train, 2)



print(data.columns[:])

print(testdata.columns[1:])



X_test = testdata[testdata.columns[:]]

X_test = X_test.as_matrix()



#print(X_train)

checkpoint = ModelCheckpoint(filepath='my_model.h5', verbose=1, save_best_only=True)



#print(X_train.shape)

model = Sequential()

model.add(Dense(512, input_shape=(10,), activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1024, activation='relu'))

model.add(Dense(2, activation='softmax'))

model.summary()



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



hist = model.fit(X_train, Y_train_one_hot, epochs=200, verbose=1, callbacks=[checkpoint], validation_split=0.2)



#score = model.evaluate(X_test, Y_test_one_hot, verbose=1)

#print('accuracy :', score)



pred = model.predict(X_test)

print(pred)



fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(1, 2, 1)

ax2 = fig.add_subplot(1, 2, 2)



ax1.plot(hist.history['loss'], color='r')

ax1.plot(hist.history['val_loss'], color='b')



ax2.plot(hist.history['acc'], color='r')

ax2.plot(hist.history['val_loss'], color='b')



plt.show()

# from keras.models import load_model



# model = load_model('my_model.h5')

# score = model.evaluate(X_test, Y_test_one_hot, verbose=1)

# print('accuracy :', score)
print(pred.shape)

expect_survival = np.argmax(pred, axis=1)

print(pred)
print(testdata.columns)
reload = pd.read_csv('../input/test.csv')

print(reload.columns)

reload.drop(['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',

       'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)



reload['Survived'] = expect_survival



reload.head




