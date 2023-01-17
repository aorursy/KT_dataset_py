# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.wrappers.scikit_learn import KerasClassifier

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.optimizers import SGD





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv', index_col='PassengerId')



train_df.head()
train_df.isnull().sum()
def prep_dataset(df):

    # Name, Ticket are useless fields, better remove them.

    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)



    # Removing NaN

    df[['Age']] = df[['Age']].fillna(df[['Age']].mean())

    df[['Fare']] = df[['Fare']].fillna(df[['Fare']].mean())

    print(df['Embarked'].value_counts())

    df[['Embarked']] = df[['Embarked']].fillna(value='S')



    # Mapping male = 1 female = 0

    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0}).astype(int)



    # Converting embarked to one_hot

    embarked_one_hot = pd.get_dummies(df[['Embarked']], prefix='Embarked')

    df = df.drop('Embarked', axis=1)

    df = df.join(embarked_one_hot)

    return df



train_df = prep_dataset(train_df)

train_df.head()
X = train_df.drop(['Survived'], axis=1).values.astype(float)

scale = StandardScaler()

X = scale.fit_transform(X)

Y = train_df['Survived'].values
def create_model(optimizer='adam', init='uniform'):

    # create model

    model = Sequential()

    model.add(Dense(500, input_dim=X.shape[1], kernel_initializer=init, activation='relu'))

    model.add(Dense(300, kernel_initializer=init, activation='relu'))

    model.add(Dense(200, kernel_initializer=init, activation='relu'))

    model.add(Dense(10, kernel_initializer=init, activation='relu'))

    model.add(Dense(2, kernel_initializer=init, activation='softmax'))

    # Compile model

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
model = create_model()
model.fit(X, Y, epochs=300)
test_df = pd.read_csv('../input/test.csv', index_col='PassengerId')

test_df = prep_dataset(test_df)

test_df.head()

X_test = test_df.values.astype(float)

X_test = scale.transform(X_test)

prediction = model.predict(X_test)

print(prediction[0])

finalPred = []

# print(np.argmax(prediction[0]))

for pred in prediction:

    if pred[0] > pred[1]:

        finalPred.append(0)

    else:

        finalPred.append(1)



print(finalPred)
submission = pd.DataFrame({

    'PassengerId': test_df.index,

    'Survived': finalPred,

})



print(submission)
submission.sort_values('PassengerId', inplace=True)    

submission.to_csv('submission.csv', index=False)
d = pd.read_csv('submission.csv', index_col='PassengerId')