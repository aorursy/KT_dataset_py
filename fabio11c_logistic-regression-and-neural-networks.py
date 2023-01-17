# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

import tensorflow as tf

import seaborn as sns



plt.style.use("seaborn")

print(tf.__version__)

print(pd.__version__)

print(np.__version__)
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
train_df.head()
test_df.head()
# the label here is the column called 'Survived' and it is a binary variable



print('Train df columns')

print(train_df.columns)

print()

print('Test df columns')

print(test_df.columns)
def clean_df(input_df):



    df_to_use = pd.DataFrame(index=input_df.index)



    aux_df = input_df.Pclass.astype(str)

    aux_df = pd.get_dummies(aux_df)

    aux_df.columns = ['Class_'+i for i in aux_df.columns]

    df_to_use = pd.concat([df_to_use, aux_df], axis=1)



    aux_df = input_df.SibSp.astype(str)

    aux_df = pd.get_dummies(aux_df)

    aux_df.columns = ['SibSp_'+i for i in aux_df.columns]

    df_to_use = pd.concat([df_to_use, aux_df], axis=1)



    aux_df = pd.get_dummies(input_df.Sex)

    df_to_use = pd.concat([df_to_use, aux_df], axis=1)



    aux_df = pd.get_dummies(input_df.Embarked)

    df_to_use = pd.concat([df_to_use, aux_df], axis=1)



    df_to_use['Age'] = input_df['Age']

    df_to_use['Fare'] = input_df['Fare']



    return df_to_use.dropna()
# clean the train_df

train_df = train_df.loc[~np.isnan(train_df.Age)]



# create the new variable just to keep the things clean

label_train = train_df['Survived']



# drop the label from the feature dataframe

train_df = train_df.drop('Survived', axis=1)
# transform the feature dataframe

train_df = clean_df(train_df)



# transform the label_train which has the label series

test_df = clean_df(test_df)
# Estimation of the logistic regression and store the results in a variable

clf = LogisticRegression(random_state=0).fit(train_df, label_train)
# print the accuracy of the logistic regression using all the features that we extracted

print('The accuracy of the Logistic Regression is %.2f%%' % (clf.score(train_df, label_train)*100))
# just to remember the dimensions of the train_df

D = train_df.shape[1]

train_df.shape
# create the model variable

model = tf.keras.models.Sequential([

    tf.keras.layers.Input(shape=(D,)),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



# compile the model

model.compile(optimizer='adam',

             loss='binary_crossentropy',

             metrics=['accuracy'])



# train the neural network

r = model.fit(train_df, label_train, epochs=100)
# print the accuracy of the ANN

print('The accuracy of the Artifial Neural Network is %.2f%%' % (r.history['accuracy'][-1]*100))
# store both estimated parameters from the logistic regression and the ANN

weights = np.append(model.layers[0].get_weights()[0].flatten(), model.layers[0].get_weights()[1].flatten())

betas = np.append(np.array(clf.coef_[0]), np.array(clf.intercept_))
# compare them

pd.DataFrame([weights, betas], index=['Weights', 'Betas']).T