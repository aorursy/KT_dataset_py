import os

#must do before importing tensorflow

os.environ['PYTHONHASHSEED']=str(2)



import tensorflow as tf

from tensorflow.keras import layers, Input, Model, Sequential

from sklearn.model_selection import train_test_split



import sys

import random

import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns



print(tf.__version__)



def reset_random_seeds():

   os.environ['PYTHONHASHSEED']=str(2)

   tf.random.set_seed(2)

   np.random.seed(2)

   random.seed(2)

train = pd.read_csv("../input/titanic/train.csv")

y_test = pd.read_csv("../input/titanic/gender_submission.csv")

x_test = pd.read_csv("../input/titanic/test.csv")
reset_random_seeds()



df = train



#Drop

df = df.drop(["PassengerId","Cabin","Name","Ticket"],1)

        

#Fill NaN

age_mean = df["Age"].mean()

age_std =  df["Age"].std()

df["Age"] = df["Age"].fillna(np.random.normal(loc=age_mean, scale = age_std))

df['Embarked'] = df['Embarked'].fillna(np.random.choice(df['Embarked'].unique()))





#Create Features

df["Family"] = ''

df["Family"].loc[df['SibSp'] + df['Parch'] <=2] = "small"

df["Family"].loc[df['SibSp'] + df['Parch'] >5] = "large"

df["Family"].loc[(df['SibSp'] + df['Parch'] > 2) & \

                 (df['SibSp'] + df['Parch'] <=5)] = "medium"







df['IsAlone'] = 1

df['IsAlone'].loc[df['SibSp'] + df['Parch'] > 0] = 0





df["SexCat"] = ''

df['SexCat'].loc[(df['Sex'] == 'male') & (df['Age'] <= 21)] = 'youngmale'

df['SexCat'].loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age']) < 50)] = 'maturemale'

df['SexCat'].loc[(df['Sex'] == 'male') & (df['Age'] > 50)] = 'seniormale'

df['SexCat'].loc[(df['Sex'] == 'female') & (df['Age'] <= 21)] = 'youngfemale'

df['SexCat'].loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age']) < 50)] = 'maturefemale'

df['SexCat'].loc[(df['Sex'] == 'female') & (df['Age'] > 50)] = 'seniorfemale'





y_train = df["Survived"].to_frame()

x_train = df.drop(["Survived"], 1)



print(x_train.info())

print(y_train.info())

reset_random_seeds()



df = x_test



#Drop

df = df.drop(["PassengerId","Cabin","Name","Ticket"],1)

        

#Fill NaN

df["Age"] = df["Age"].fillna(np.random.normal(loc=age_mean, scale = age_std))

df['Embarked'] = df['Embarked'].fillna(np.random.choice(df['Embarked'].unique()))

df["Fare"] = df["Fare"].fillna(df["Fare"][df["Pclass"]==3].mean())



#Create Features

df["Family"] = ''

df["Family"].loc[df['SibSp'] + df['Parch'] <=2] = "small"



df["Family"].loc[(df['SibSp'] + df['Parch'] > 2) & \

                 (df['SibSp'] + df['Parch'] <=5)] = "medium"



df["Family"].loc[df['SibSp'] + df['Parch'] >5] = "large"





df['IsAlone'] = 1

df['IsAlone'].loc[df['SibSp'] + df['Parch'] > 0] = 0





df["SexCat"] = ''

df['SexCat'].loc[(df['Sex'] == 'male') & (df['Age'] <= 21)] = 'youngmale'

df['SexCat'].loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age']) < 50)] = 'maturemale'

df['SexCat'].loc[(df['Sex'] == 'male') & (df['Age'] > 50)] = 'seniormale'

df['SexCat'].loc[(df['Sex'] == 'female') & (df['Age'] <= 21)] = 'youngfemale'

df['SexCat'].loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age']) < 50)] = 'maturefemale'

df['SexCat'].loc[(df['Sex'] == 'female') & (df['Age'] > 50)] = 'seniorfemale'





x_test = df

y_test = y_test.drop(["PassengerId"],1)



print(x_test.info())

print(y_test.info())

#categorical_cols = ["Sex", "SexCat", "Embarked", "Pclass", "IsAlone", "Family"]

# numerical_cols = ["SibSp", "Parch","Family"]

#numerical_cols = []

#bucketized_cols = ["Age", "Fare"]



categorical_cols = ["Sex", "Pclass", "Embarked"]

numerical_cols = []

bucketized_cols = []







feature_columns = []

for feature_name in categorical_cols:

    vocabulary = x_train[feature_name].unique()

    cat_c = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)

    feature_columns.append(tf.feature_column.indicator_column(cat_c))



for feature_name in numerical_cols:

    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))









# age = tf.feature_column.numeric_column("Age", dtype=tf.float32)

# age = tf.feature_column.bucketized_column(age, list(range(10,85,10)))



# fare = tf.feature_column.numeric_column("Fare", dtype=tf.float32)

# fare = tf.feature_column.bucketized_column(fare, list(range(20,110,20)))



# feature_columns.append(age)

#feature_columns.append(fare)





feature_layer = layers.DenseFeatures(feature_columns,trainable=False)



def show_col(cols, dummy_data): #cols is list

    print(layers.DenseFeatures(cols,trainable=False)(dummy_data))



# f = {"Family":["small", "medium", "large"]}

# show_col([feature_columns[5]],f)
reset_random_seeds()



#Applying feature_layers in Dataset instead of model

def make_dataset(df_x, df_y, apply_fc = True, shuffle = True, batch_size=10):

    ds = tf.data.Dataset.from_tensor_slices((dict(df_x), df_y))

    if shuffle:

        ds = ds.shuffle(10000, reshuffle_each_iteration = True)

    ds = ds.batch(batch_size)

    if apply_fc:

        map_fn = lambda x,y : (feature_layer(x), y)

        ds = ds.map(map_fn)

    return ds



train_ds = make_dataset(x_train, y_train, batch_size =10)

test_ds = make_dataset(x_test, y_test, shuffle =False, batch_size = len(x_test))



def train_input_fn():

    return make_dataset(x_train, y_train, apply_fc = False, batch_size =10)



def test_input_fn():

    return make_dataset(x_test, y_test, apply_fc = False, \

                        shuffle=False, batch_size =len(x_test))



model = tf.estimator.LinearClassifier(feature_columns=feature_columns,

    n_classes=2)



model = model.train(input_fn=train_input_fn, steps=200)
result = model.evaluate(test_input_fn, steps=10)

print(result)
ids = []

for pred in model.predict(test_input_fn):

    ids.append(pred['class_ids'][0])

    

sub = pd.read_csv("../input/titanic/gender_submission.csv")

sub = sub.drop("Survived", axis=1)

sub["Survived"] = ids

sub.to_csv("sub3.csv", index = False)
_, y = next(iter(test_ds))

print(y.numpy().shape)

print(len(ids))

count = 0

for i,j in zip(y.numpy(),ids):

    if i[0]== j:

        count+=1



print("{}/{}   {}".format(count,len(ids), count/len(ids)))