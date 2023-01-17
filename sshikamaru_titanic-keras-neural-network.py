

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

#make a copy so your original data is not touched

train = train_data.copy()

test = test_data.copy()

train.shape

y_train = train['Survived']



#We won't need passenger ID or ticket price for the model! They do not provide much insight on the training.

Id = pd.DataFrame(test['PassengerId'])

train.drop(['PassengerId'], axis = 1, inplace=True)

test.drop(['PassengerId'], axis = 1, inplace=True)

train.drop(['Survived'], axis = 1, inplace=True)

train.drop(['Ticket'], axis = 1, inplace=True)

test.drop(['Ticket'], axis = 1, inplace=True)
train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar='BuPu')

train.isnull().sum().sort_values(ascending=False)[0:20]

# we can see that cabin is midding a lot of values, and age is tooi!




#clean the train data

for i in list(train.columns):

    dtype = train[i].dtype

    values = 0

    if(dtype == float or dtype == int):

        method = 'mean'

    else:

        method = 'mode'

    if(train[i].notnull().sum() / 891 <= .5):

        train.drop(i, axis = 1, inplace=True)

    elif method == 'mean':

        train[i]=train[i].fillna(train[i].mean())



    else:

        train[i]=train[i].fillna(train[i].mode()[0])



# WE CAN DO THIS FOR THE TEST SET TOO!



#clean the test data

for i in list(test.columns):

    dtype = test[i].dtype

    values = 0

    if(dtype == float or dtype == int):

        method = 'mean'

    else:

        method = 'mode'

    if(test[i].notnull().sum() / 418 <= .5):

        test.drop(i, axis = 1, inplace=True)

    elif method == 'mean':

        test[i]=test[i].fillna(test[i].mean())



    else:

        test[i]=test[i].fillna(test[i].mode()[0])





sns.heatmap(train.isnull(),yticklabels=False,cbar='BuPu')

#TITLE



train_test_data = [train, test] # combining train and test dataset



for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)





title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 1, 

                 "Master": 0, "Dr": 1, "Rev": 0, "Col": 0, "Major": 0, "Mlle": 1,"Countess": 1,

                 "Ms": 1, "Lady": 1, "Jonkheer": 1, "Don": 0, "Dona" : 1, "Mme": 0,"Capt": 0,"Sir": 0 }

for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    

    

sex_mapping = {"male": 0, "female":1}

for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

Pclass1 = train_data[train_data['Pclass'] == 1]['Embarked'].value_counts()

Pclass2 = train_data[train_data['Pclass'] == 2]['Embarked'].value_counts()

Pclass3 = train_data[train_data['Pclass'] == 3]['Embarked'].value_counts()



df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class','2nd class', '3rd class']

df.plot(kind='bar',stacked=True, figsize=(10,5))
for data in train_test_data:

    data['Embarked'] = data['Embarked'].fillna("S")

    

embarked_mapping = {"S": 0, "C": 1, "Q": 2}

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
train["FamilySize"] = train['SibSp'] + train['Parch'] + 1

test["FamilySize"] = test['SibSp'] + test['Parch'] + 1
sns.heatmap(train.corr(),cbar='plasma')

train.drop(['Name'], axis = 1, inplace=True)

test.drop(['Name'], axis = 1, inplace=True)
train.head()
test.head()
#imports

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D,MaxPool2D



import keras
continuous = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'FamilySize']



scaler = StandardScaler()



for var in continuous:

    train[var] = train[var].astype('float64')

    train[var] = scaler.fit_transform(train[var].values.reshape(-1, 1))

for var in continuous:

    test[var] = test[var].astype('float64')

    test[var] = scaler.fit_transform(test[var].values.reshape(-1, 1))
train.describe(include='all').T

import tensorflow as tf

tf.keras.optimizers.Adam(

    learning_rate=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,

    name='Adam', 

)

#creating a model with sequential columns

model = Sequential()



#flattens the data into a 1d array

model.add(Flatten())

#creates the first latyer with the input dimanetion. 



model.add(Dense(32, input_dim=train.shape[1],kernel_initializer = 'uniform', activation='relu'))

#next layer with 32 dense nodes

model.add(Dense(32, kernel_initializer = 'uniform', activation = 'relu'))

#drops 0.4 of the values from the next layer, so it does not over fit!



model.add(Dropout(0.4))

#last layer is initiated here

model.add(Dense(32,kernel_initializer = 'uniform', activation = 'relu'))



# create output layer

    # Feel free to experiment with the activation functions and the optimizers

model.add(Dense(1, activation='sigmoid'))  # output layer

    

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train, y_train, epochs=20, batch_size=50, validation_split = 0.2)



#val_acc = np.mean(training.history['val_acc'])

#print("\n%s: %.2f%%" % ('val_acc', val_acc*100))
print(model.summary())

scores = model.evaluate(train, y_train, batch_size=32)
y_pred = model.predict(test)



y_final = (y_pred > 0.5).astype(int).reshape(test.shape[0])



output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_final})

output.to_csv('prediction-ann.csv', index=False)
loss_train = history.history['loss']

loss_val = history.history['val_loss']

epochs = range(1,21)

plt.plot(epochs, loss_train, 'g', label='Training loss')

plt.plot(epochs, loss_val, 'b', label='validation loss')

plt.title('Training and Validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
loss_train = history.history['accuracy']

loss_val = history.history['val_accuracy']

epochs = range(1,21)

plt.plot(epochs, loss_train, 'g', label='Training accuracy')

plt.plot(epochs, loss_val, 'b', label='validation accuracy')

plt.title('Training and Validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()