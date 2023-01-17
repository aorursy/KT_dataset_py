__author__ = 'Albert: https://www.kaggle.com/albertholmes'

# File nn_valid.py

# Use the K-Fold corss validation to do experiment based on titanic data



# Import modules

import numpy as np

import pandas as pd

import gc

from sklearn.base import TransformerMixin

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from matplotlib import pyplot as plt



%matplotlib inline



# Mean Square Error function

mse = lambda actual, pred: np.mean((actual - pred) ** 2)
class DataFrameImputer(TransformerMixin):

    """

    TransformerMixin is an interface that you can create your own

    transformer or models.

    The .fit_transform method that calls .fit and .transform methods,

    you should define the two methods by yourself.

    """

    def fit(self, X, y=None):

        """

        The pandas.Series.value_counts method returns the object

        containing counts of unique values.

        The resulting object will be in descending order so that

        the first element is the most frequently-occurring element.



        np.dtype('O'): The 'O' means the Python objects

        """

        d = [X[c].value_counts().index[0] if X[c].dtype == np.dtype('O')

             else X[c].median() for c in X]



        self.fill = pd.Series(d, index=X.columns)

        return self



    def transform(self, X, y=None):

        return X.fillna(self.fill)



# Garbage collection

gc.enable()
# Read the data

train = pd.read_csv('../input/train.csv')



# Get target value

target = train['Survived'].values

del train['Survived']



# Prepare the k-fold cross validation

skf = StratifiedKFold(n_splits=10)

"""

print(skf.get_n_splits(train, target))

"""

# Add new features

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1



feature_names = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']

nonnumeric_fea = ['Sex', 'Embarked']

categorical_fea = ['Pclass', 'Sex', 'Embarked']



# Impute the missing values

imputed = DataFrameImputer().fit_transform(train[feature_names])
"""

Preprocessing the nonnumeric feature

Encode labels with value between 0 and (n_classes - 1)

"""

le = LabelEncoder()

for feature in nonnumeric_fea:

    imputed[feature] = le.fit_transform(imputed[feature])



"""

Use one hot encoder to get new feature from categorical feature

"""

enc = OneHotEncoder()

chosen_features = imputed[categorical_fea]

new_fea = enc.fit_transform(chosen_features).toarray()

for feature in categorical_fea:

    del imputed[feature]



train = imputed.values

train = np.concatenate((train, new_fea), axis=1)
"""

Normalize the features

"""

for col in range(train.shape[1]):

    max_value = max(train[:, col])

    train[:, col] /= max_value



import keras 

from keras.models import Sequential # intitialize the ANN

from keras.layers import Dense      # create layers



nn_errors = []



for train_idx, valid_idx in skf.split(train, target):

    # print('TRAIN:', train_idx, 'VALID:', valid_idx)

    tra, val = train[train_idx], train[valid_idx]

    target_tra, target_val = target[train_idx], target[valid_idx]



    # Initialising the NN

    model = Sequential()



    # layers

    model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

    model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

    model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



    # Compiling the ANN

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



    # Train the ANN

    # model.fit(tra, target_tra, batch_size = 32, epochs = 200)

    model.fit(tra, target_tra, batch_size=64, epochs=5)

    

    pred = model.predict(val)

    error = mse(target_val, pred)

    

    print('NN MSE: %.3f' % (error))

    

    nn_errors.append(error)



nn_final_error = sum(nn_errors) / len(nn_errors)

print('NN FINAL MSE: %.3f' % (nn_final_error))