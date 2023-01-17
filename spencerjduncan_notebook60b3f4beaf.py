import numpy as np

import pandas as pd

import tensorflow as tf

import tempfile



df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
#Let's get rid of these:

def process_data(df):

    df = df.drop(['PassengerId','Name', 'Ticket'], axis=1)

    df['Room'] = df['Cabin'].apply(getRoom)

    df['Cabin'] = df['Cabin'].apply(getCabin)

    df['Pclass'] = df['Pclass'].apply(str)

    df['Embarked'] = df['Embarked'].fillna('S')

    age_median = df['Age'].median()

    df['Age'] = df['Age'].fillna(age_median)

    room_median = df['Room'].median()

    df['Room'] = df['Room'].fillna(room_median)

    return df





def getCabin(x):

    try:

        return x[0]

    except:

        return ''



def getRoom(x):

    try:

        return float(x[1:])

    except:

        return float('NaN')



df_train = process_data(df_train)

df_test  = process_data(df_test)



df_train.info()
#Define column types



LABEL_COLUMN = "Survived"

CATEGORICAL_COLUMNS = [ "Pclass", "Sex", "Cabin", "Embarked" ]

CONTINUOUS_COLUMNS = [ "Age", "SibSp", "Parch", "Fare", "Room" ]



def input_fn(df,predict=False):

    # Create mapping dict from continuous feature names to values in a constant Tensor

    continuous_cols = {k: tf.constant(df[k].values)

                      for k in CONTINUOUS_COLUMNS}

    #Same thing, but for categoricals to a sparse tensor

    categorical_cols = {k:

                        tf.SparseTensor(

                            indices=[[i,0] for i in range (df[k].size)],

                        values=df[k].values,

                        shape=[df[k].size, 1])

                       for k in CATEGORICAL_COLUMNS}

    # Merge the dicts

    feature_cols = {**continuous_cols, **categorical_cols}

    # Convert the label column into a constant Tensor

    if predict:

        return feature_cols

    label = tf.constant(df[LABEL_COLUMN].values)

    return feature_cols, label

    



def train_input_fn():

  return input_fn(df_train)



def train_test_fn():

  return input_fn(df_test,predict=True)



gender = tf.contrib.layers.sparse_column_with_keys(

  column_name="Sex", keys=["female", "male"])



port = tf.contrib.layers.sparse_column_with_keys(

  column_name="Embarked", keys=["S", "Q", "C"])



Pclass =  tf.contrib.layers.sparse_column_with_keys(column_name="Pclass", keys=['1', '2', '3'])



Cabin = tf.contrib.layers.sparse_column_with_hash_bucket("Cabin", 

                                                         hash_bucket_size=32)



age = tf.contrib.layers.real_valued_column("Age")

sibs = tf.contrib.layers.real_valued_column("SibSp")

parch = tf.contrib.layers.real_valued_column("Parch")

fare = tf.contrib.layers.real_valued_column("Fare")

room = tf.contrib.layers.real_valued_column("Room")



model_dir = tempfile.mkdtemp()

m = tf.contrib.learn.LinearClassifier(feature_columns=[gender, port, Pclass, Cabin,

                                                       age, sibs, parch, fare, room],

                                      optimizer=tf.train.FtrlOptimizer(

                                          learning_rate=0.1,

                                          l1_regularization_strength=1.0,

                                          l2_regularization_strength=1.0),

                                      model_dir=model_dir)
m.fit(input_fn=train_input_fn, steps=200)
predictions = m.predict(input_fn=train_test_fn)
output=[]

for i in range (0,418):

    output.append([(i+892),predictions[i]])

    

df_result = pd.DataFrame(output, columns=['PassengerId', 'Survived'])



df_result.head()



df_result.to_csv('titanic_1-1.csv', index=False)