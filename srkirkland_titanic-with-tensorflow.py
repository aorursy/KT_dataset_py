import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf # good 'ol tensorflow for machine learning



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')
# get train csv, we don't care about test for now

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")



df_train.head()
df_train.groupby('Sex')['PassengerId'].nunique()
df_train.groupby('SibSp')['PassengerId'].nunique()
df_train.count()
df_train.groupby('Embarked')['PassengerId'].nunique()
# most common embarked is S, so let's fill those few N/A rows with S

df_train["Embarked"] = df_train["Embarked"].fillna("S")
df_train['Age'].describe()
# let's just replace each null age value with the mean age of 30 yrs old.

# unoriginal, but it'll get the job done

df_train['Age'].fillna(30, inplace=True)



# let's drop cabin since it's our only null column and we probably won't use it

df_train.drop("Cabin",axis=1,inplace=True)

df_train.count()
# catgegorical features

sex = tf.feature_column.categorical_column_with_vocabulary_list(

    "Sex", ["female", "male"])

embarked = tf.feature_column.categorical_column_with_vocabulary_list(

    "Embarked", ["C", "Q", "S"])



# let's try using pclass as a distinct category instead of numeric values

pclass_category = tf.feature_column.categorical_column_with_vocabulary_list(

    "Pclass", [1, 2, 3])



# continual numeric features

age = tf.feature_column.numeric_column("Age")

pclass = tf.feature_column.numeric_column("Pclass")

sibsp = tf.feature_column.numeric_column("SibSp")

parch = tf.feature_column.numeric_column("Parch")

fare = tf.feature_column.numeric_column("Fare")



# lets bucket the ages instead of treating it as continuous

age_buckets = tf.feature_column.bucketized_column(

    age, boundaries=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65])



# let's group columns together as base columns and crossed columns

base_columns = [

    sex, embarked, sibsp, parch, fare, age_buckets

]



# in case certains features make sense to be combined

# let's combine siblings and parents/children, since maybe just having family is a good indicator

# also let's combine age and pclass, maybe young 1st class children have a strong correlation

# also let's combine Pclass and where they embarked from

crossed_columns = [

    tf.feature_column.crossed_column(

        ["SibSp", "Parch"], hash_bucket_size=1000),

    tf.feature_column.crossed_column(

        [age_buckets, pclass_category], hash_bucket_size=1000),

    tf.feature_column.crossed_column(

        [embarked, pclass_category], hash_bucket_size=1000)

]
import tempfile



model_dir = tempfile.mkdtemp() # base temp directory for running models



# our Y value labels, i.e. the thing we are classifying

labels = df_train["Survived"].astype(int)



# let's make a training function we can use with our estimators

train_fn = tf.estimator.inputs.pandas_input_fn(

    x=df_train,

    y=labels,

    batch_size=100,

    num_epochs=None, # unlimited

    shuffle=True, # shuffle the training data around

    num_threads=5)



# let's try a simple linear classifier

linear_model = tf.estimator.LinearClassifier(

    model_dir=model_dir, feature_columns=base_columns + crossed_columns)

train_steps = 2000



# now let's train that model!

linear_model.train(input_fn=train_fn, steps=train_steps)
# unconventional but let's see how well we did on our training set

train_test_fn = tf.estimator.inputs.pandas_input_fn(

    x=df_train,

    y=labels,

    batch_size=100,

    num_epochs=1, # just one run

    shuffle=False, # don't shuffle test here

    num_threads=5)



linear_model.evaluate(input_fn=train_test_fn)["accuracy"]