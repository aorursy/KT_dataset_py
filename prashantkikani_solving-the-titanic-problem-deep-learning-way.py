# import necessary libraries first



# pandas to open data files & processing it.

import pandas as pd

# to see all columns

pd.set_option('display.max_columns', None)



# numpy for numeric data processing

import numpy as np



# sklearn to do preprocessing & ML models

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# keras for deep learning model creation

from keras.models import Model

from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation

from keras.optimizers import Adam

from keras.utils import plot_model



# Matplotlob & seaborn to plot graphs & visulisation

import matplotlib.pyplot as plt 

import seaborn as sns



# for fixing the random seed

import random

import os, tensorflow as tf

import torch



# ignore warnings

import warnings

warnings.simplefilter(action='ignore')
titanic_data = pd.read_csv("../input/titanic/train.csv")

titanic_data.shape
titanic_data.head()
titanic_data.describe()
# Survival

titanic_data['Survived'].value_counts()
# Ticket class

titanic_data['Pclass'].value_counts()
# Gender

titanic_data['Sex'].value_counts()
# Siblings

titanic_data['SibSp'].value_counts()
# Parent or Childs

titanic_data['Parch'].value_counts()
# Embarked station

titanic_data['Embarked'].value_counts()
sns.countplot(titanic_data['Sex']);
sns.barplot(titanic_data['Survived'], titanic_data['Sex']);
sns.barplot(titanic_data['Survived'], titanic_data['Fare'], titanic_data['Pclass']);
sns.boxplot(x=titanic_data["Fare"])

plt.show()
# Only take rows which have "Fare" value less than 250.

titanic_data = titanic_data[titanic_data['Fare'] < 250]

titanic_data.shape
sns.boxplot(x=titanic_data["Age"])

plt.show()
titanic_data.isna().sum()
titanic_data.drop("Cabin", axis=1, inplace=True)

titanic_data.shape
titanic_data.columns
age_mean = titanic_data['Age'].mean()

print(age_mean)
titanic_data['Age'].fillna(age_mean, inplace=True)
titanic_data.isna().sum()
titanic_data['Embarked'].value_counts()
titanic_data['Embarked'].fillna("S", inplace=True)
titanic_data.isna().sum()
titanic_data.head(10)
titanic_data['total_family_members'] = titanic_data['Parch'] + titanic_data['SibSp'] + 1



# if total family size is 1, person is alone.

titanic_data['is_alone'] = titanic_data['total_family_members'].apply(lambda x: 0 if x > 1 else 1)



titanic_data.head(10)
sns.barplot(titanic_data['total_family_members'], titanic_data['Survived'])
sns.barplot(titanic_data['is_alone'], titanic_data['Survived'])
def age_to_group(age):

    if 0 < age < 12:

        # children

        return 0

    elif 12 <= age < 50:

        # adult

        return 1

    elif age >= 50:

        # elderly people

        return 2

    

titanic_data['age_group'] = titanic_data['Age'].apply(age_to_group)

titanic_data.head(10)
sns.barplot(titanic_data['age_group'], titanic_data['Survived']);
titanic_data['name_title'] = titanic_data['Name'].str.extract('([A-Za-z]+)\.', expand=False)

titanic_data.head()
titanic_data['name_title'].value_counts()
def clean_name_title(val):

    if val in ['Rev', 'Col', 'Mlle', 'Mme', 'Ms', 'Sir', 'Lady', 'Don', 'Jonkheer', 'Countess', 'Capt']:

        return 'RARE'

    else:

        return val



titanic_data['name_title'] = titanic_data['name_title'].apply(clean_name_title)

titanic_data['name_title'].value_counts()
sns.barplot(titanic_data['name_title'], titanic_data['Survived']);
titanic_data.head(10)
# save the target column 

target = titanic_data['Survived'].tolist()



titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1, inplace=True)
titanic_data.head()
le = preprocessing.LabelEncoder()

titanic_data['Sex'] = le.fit_transform(titanic_data['Sex'])

titanic_data['Embarked'] = le.fit_transform(titanic_data['Embarked'])

titanic_data['name_title'] = le.fit_transform(titanic_data['name_title'])

titanic_data.head()
titanic_data = pd.get_dummies(titanic_data, columns=["Pclass", "Sex", "SibSp", "Parch", "Embarked", "total_family_members", "is_alone", "age_group", "name_title"])
titanic_data.head()
mm = preprocessing.MinMaxScaler(feature_range=(-1, 1))

titanic_data['Age'] = mm.fit_transform(titanic_data['Age'].to_numpy().reshape(-1, 1))

titanic_data['Fare'] = mm.fit_transform(titanic_data['Fare'].to_numpy().reshape(-1, 1))

titanic_data.head(10)
train_data, val_data, train_target, val_target = train_test_split(titanic_data, target, test_size=0.2)

train_data.shape, val_data.shape, len(train_target), len(val_target)
def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    tf.random.set_seed(seed)



# We fix all the random seed so that, we can reproduce the results.

seed_everything(2020)
input_layer = Input(shape=(titanic_data.shape[1],),name='input_layer')

hidden_layer_1 = Dense(32, activation = 'relu')(input_layer)

hidden_layer_2 = Dense(16, activation = 'relu')(hidden_layer_1)

output_layer = Dense(1, activation = 'sigmoid')(hidden_layer_2)



model = Model(input=input_layer, output=output_layer)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
plot_model(model, show_shapes=True)
# We will give training 10 times with the same data.

EPOCHS = 10



# We will process 64 rows at a time.

BATCH_SIZE = 32



model.fit(

        train_data, train_target,

        nb_epoch=EPOCHS,

        batch_size=BATCH_SIZE,

        validation_data=(val_data, val_target),

        verbose = 1,

)
# Predict labels on Validation data which model have never seen before.



val_predictions = model.predict(val_data)

len(val_predictions)
# first 10 values of validation_predictions

val_predictions[:10]
val_predictions1 = [1 if x >= 0.5 else 0 for x in val_predictions]

val_predictions1[:10]
# Calculate the accuracy score on validation data.

# We already have correct target information for them.



accuracy = accuracy_score(val_target, val_predictions1)

accuracy
print("We got %.3f percent accuracy on our validation unseen data !!"%(accuracy*100))