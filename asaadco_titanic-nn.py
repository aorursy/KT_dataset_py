# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Import libraries



# data anlysis

import numpy as np

import pandas as pd



# data visualization

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



# data preparation for modelling

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

# model optimization

from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, learning_curve

from sklearn.feature_selection import SelectFromModel

from scipy.stats import randint

import itertools

from sklearn.metrics import confusion_matrix, roc_curve

from sklearn.metrics import precision_score



# Artificial Neural Network

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Input, Dense, Dropout, AlphaDropout

from tensorflow.keras.optimizers import SGD, RMSprop, Adamax, Adagrad, Adam, Nadam, SGD

import eli5

from eli5.sklearn import PermutationImportance



# Random Forest and Gradient Boosting (Appendix)

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier



# ensure comparability of different runs

np.random.seed(42)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Import data



# load train and test data

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



data = [train, test]

data[0].describe() # Let's look at the first 21 rows 
data[0].isnull().sum().sort_values()
data[0].dtypes
# preprocessing

# imputing values

for df in data:

    df['Age'] = df['Age'].fillna(df['Age'].mean())

    df['Embarked'] = df['Embarked'].fillna('S') # most common value 75%

# drop nonsense

data[0] = data[0].drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'])

data[1] = data[1].drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'])

data[0].isnull().sum().sort_values()
# Following good practice, it's good to have age categories

 

for df in data:

    df.loc[df['Age'] <= 10, 'Age'] = 0

    df.loc[(df['Age'] > 10) & (df['Age'] <= 18), 'Age'] = 1 # Teen = 1

    df.loc[(df['Age'] > 18) & (df['Age'] <= 22), 'Age'] = 2 # Young Adult = 2

    df.loc[(df['Age'] > 22) & (df['Age'] <= 27), 'Age'] = 3 # Adult = 3

    df.loc[(df['Age'] > 27) & (df['Age'] <= 33), 'Age'] = 4 # M Adult = 4

    df.loc[(df['Age'] > 33) & (df['Age'] <= 40), 'Age'] = 5 # Old Adult = 5

    df.loc[(df['Age'] > 40) & (df['Age'] <= 60), 'Age'] = 6 # Middle Age = 6

    df.loc[(df['Age'] > 60), 'Age'] = 6 # Old

    

data[0]['Age'].value_counts()

# The fare is all over the place too, so let's categorize that 

# (look up qcut. pretty nice)

data[0]['Fare'] = pd.qcut(data[0]['Fare'].values, q=4, labels=[0, 1, 2, 3]).astype(float)

data[1]['Fare'] = pd.qcut(data[1]['Fare'].values, q=4, labels=[0, 1, 2, 3]).astype(float)
data[0]['Fare'].value_counts()

# We can see the distribution is fair. 100 values for each category.

# 0 being the first quartile, lowest fare, could mean low class

# 1 being the last quartile, highest fare, could mean high class
# Looked through literature and found many have recommend 

# Relatives & Age * Class & Fare per Person features so lets do that

#train

embark_ports = {"S": 0, "C": 1, "Q": 2}

gender = {"male": 0, "female": 1}

data[0]["relatives"] = data[0]['SibSp'] + data[0]['Parch']

data[0]['Age_Class'] = data[0]['Age'] * data[0]['Pclass']

data[0].loc[data[0]["relatives"] > 0, 'alone'] = 0 

data[0].loc[data[0]["relatives"] == 0, 'alone'] = 1 

data[0]['Sex'] = data[0]['Sex'].map(gender)

data[0]['Embarked'] = data[0]['Embarked'].map(embark_ports)



#test

data[1]["relatives"] = data[1]['SibSp'] + data[1]['Parch']

data[1]['Age_Class'] = data[1]['Age'] * data[1]['Pclass']

data[1].loc[data[1]["relatives"] > 0, 'alone'] = 0 

data[1].loc[data[1]["relatives"] == 0, 'alone'] = 1

data[1]['Sex'] = data[1]['Sex'].map(gender)

data[1]['Embarked'] = data[1]['Embarked'].map(embark_ports)

# FPP = Fare Per Person



# let's drop Sex now

data[0].drop(columns=["Sex"])

data[1].drop(columns=["Sex"])

data[0]['relatives'].value_counts()

data[0]['Age_Class'].value_counts()
data[0]['alone'].value_counts()
data[0]['Fare'].value_counts()

# High Age (6), High PClass (1) = Low Age_Class (6)

# Low Age (1), High PClass (1) =  Low Age_Class (1)

# High Age (6), Low PClass (3) =  High Age_Class (18)



data[1].dtypes

X_train = data[0].drop(columns=['Survived'])

Y_train = data[0]['Survived']

X_test = data[1]

def create_model(input_shape=X_train.shape[1:],

                number_hidden=2, ## 

                neurons_per_hidden=10, # make it higher

                hidden_drop_rate= 0.2,      # too high, make it 0.1

                hidden_activation = 'selu', # Make it RELU

                hidden_initializer="lecun_normal",

                output_activation ='sigmoid', # ok

                loss='binary_crossentropy',  # ok

                optimizer = Nadam(lr=0.0005), # stokes-gradient, 

                #lr=0.0005,

                ):

    

    #create model

    model = Sequential()

    model.add(Input(shape=input_shape)),

    for layer in range(number_hidden):

        model.add(Dense(neurons_per_hidden, activation = hidden_activation ,kernel_initializer=hidden_initializer))

        #model.add(Dropout(hidden_drop_rate))

    model.add(Dense(1, activation = output_activation))



    # Compile model

    model.compile(loss=loss, 

                  #optimizer = Nadam(lr=lr), 

                  optimizer = Nadam(lr=0.0005),

                  metrics = ['accuracy'])

    return model
keras.backend.clear_session()

np.random.seed(42)

tf.random.set_seed(42)



dnn_clf = create_model()



history = dnn_clf.fit(np.asarray(X_train).astype(np.float32), np.asarray(Y_train).astype(np.float32), epochs=30, batch_size=30, verbose=1)
# score of Ann classifier on training data

print('Training score: ' + str((pd.DataFrame(history.history)['accuracy'].max()*100)) + '%')
y_train_pred_dnn = dnn_clf.predict_classes(X_test)
predictions = pd.DataFrame(y_train_pred_dnn)

submission = pd.DataFrame(test['PassengerId'])

submission["Survived"] = y_train_pred_dnn

submission.head(20)

submission.to_csv("submission.csv", index=False)