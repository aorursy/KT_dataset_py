import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math



import matplotlib.pyplot as plt

from matplotlib.pyplot import rcParams

%matplotlib inline



import os

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout



from numpy.random import seed

import tensorflow as tf
def convertToAgeGroup(age):

    if(math.isnan(age)):

        return 0

    if(age < 13):

        return 1

    elif(age < 45):

        return 2

    else:

        return 3
#Adding both to normalize both at the same time

#We will separate it using the survival column

df_orig_train = pd.read_csv('dataset/train.csv')

df_orig_test = pd.read_csv('dataset/test.csv')

df_orig =pd.concat([df_orig_train, df_orig_test], axis=0, sort=True)
df = pd.DataFrame()

df['Survived'] = df_orig['Survived']

df['Class'] = df_orig['Pclass']

df['Simblings'] = df_orig['SibSp']

df['Sex'] = pd.factorize(df_orig['Sex'])[0]

df['AgeGroup'] = df_orig['Age'].apply(lambda age: convertToAgeGroup(age))

df['Cabingroup'] = pd.factorize(df_orig['Cabin'].str[0])[0]

df['Embarked'] = pd.factorize(df_orig['Embarked'])[0]



df
X_train = df[pd.notnull(df['Survived'])].drop(['Survived'], axis=1)

y_train = df[pd.notnull(df['Survived'])]['Survived']

X_test = df[pd.isnull(df['Survived'])].drop(['Survived'], axis=1)
def create_model(lyrs=[8], act='linear', opt='Adam', dr=0.0):

    

    # set random seed for reproducibility

    seed(42)

    tf.random.set_seed(42)

    

    model = Sequential()

    

    # create first hidden layer

    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))

    

    # create additional hidden layers

    for i in range(1,len(lyrs)):

        model.add(Dense(lyrs[i], activation=act))

    

    # add dropout, default is none

    model.add(Dropout(dr))

    

    # create output layer

    model.add(Dense(1, activation='sigmoid'))  # output layer

    

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    

    return model
model = create_model()

print(model.summary())
training = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.33, verbose=0)

val_acc = np.mean(training.history['val_accuracy'])

print("\n%s: %.2f%%" % ('val_accuracy', val_acc*100))
history_dict = training.history

print(history_dict.keys())
plt.plot(training.history['accuracy'])

plt.plot(training.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
# calculate predictions

df_orig_test['Survived'] = model.predict(X_test)

df_orig_test['Survived'] = df_orig_test['Survived'].apply(lambda x: round(x,0)).astype('int')

solution = df_orig_test[['PassengerId', 'Survived']]
solution.to_csv("final_solution.csv", index=False)
#To send to kaggle, after installed the kaggle command tool:

#kaggle competitions submit -f final_solution.csv -m "testing sending" titanic