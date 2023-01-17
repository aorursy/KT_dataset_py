# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def fill_ages_fare(df):
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    return df

def fill_embarked_cabin(df):
    return df.fillna({'Embarked': 'X','Cabin':'Dummy'})

def normalize_numeric_columns(df):
    df[['Age','Fare','Parch','SibSp']] = (df[['Age','Fare','Parch','SibSp']] - df[['Age','Fare','Parch','SibSp']].mean(axis=0))/df[['Age','Fare','Parch','SibSp']].std(axis=0)
    return df

def cabin_feature_columns():
    cabin_feature_columns = tf.feature_column.categorical_column_with_hash_bucket(
         key = 'Cabin',
         hash_bucket_size = 10)
    return  tf.feature_column.indicator_column(cabin_feature_columns)

def embarked_feature_columns():
    embarked_feature_columns = tf.feature_column.categorical_column_with_vocabulary_list( 
        key='Embarked',
        vocabulary_list = ('C','Q','S') )
    return tf.feature_column.indicator_column(embarked_feature_columns)

def sex_feature_columns():
    sex_feature_columns = tf.feature_column.categorical_column_with_vocabulary_list( 
        key='Sex',
        vocabulary_list = ['male','female'] )
    return tf.feature_column.indicator_column(sex_feature_columns)

def pclass_feature_columns():
    pclass_feature_columns = tf.feature_column.categorical_column_with_identity(
        key='Pclass',
        num_buckets = 4)
    return tf.feature_column.indicator_column(pclass_feature_columns)

def name_feature_columns():
    name_feature_columns = tf.feature_column.categorical_column_with_hash_bucket(
         key = 'Name',
         hash_bucket_size = 10)
    return tf.feature_column.indicator_column(name_feature_columns)

def ticket_feature_columns():
    ticket_feature_columns = tf.feature_column.categorical_column_with_hash_bucket(
        key = 'Ticket',
        hash_bucket_size = 5)
    return tf.feature_column.indicator_column(ticket_feature_columns)
# read datasets , fill NaNs and normalize feature on numeric features
train_file = "../input/train.csv"
test_file = "../input/test.csv"
df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)
df_train = fill_ages_fare(df_train)
df_test = fill_ages_fare(df_test)
df_train = fill_embarked_cabin(df_train)
df_test = fill_embarked_cabin(df_test)
df_train = normalize_numeric_columns(df_train)
df_test = normalize_numeric_columns(df_test)
#define feature columns
my_feature_columns = []
my_feature_columns.append(pclass_feature_columns())
my_feature_columns.append(name_feature_columns())
my_feature_columns.append(sex_feature_columns())
my_feature_columns.append(tf.feature_column.numeric_column(key='Age'))
my_feature_columns.append(tf.feature_column.numeric_column(key='SibSp'))
my_feature_columns.append(tf.feature_column.numeric_column(key='Parch'))
my_feature_columns.append(ticket_feature_columns())
my_feature_columns.append(tf.feature_column.numeric_column(key='Fare'))
my_feature_columns.append(cabin_feature_columns())
my_feature_columns.append(embarked_feature_columns())
print(my_feature_columns)
# Instantiate Estimator 
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    optimizer=tf.train.ProximalAdagradOptimizer(learning_rate = 0.1,l2_regularization_strength=3.),
    hidden_units=[40,20,5],
    model_dir='titanic/model_2',
    n_classes=2)

# define input functions
def my_train_input_fn(features,label,batch,epochs):
    in_func = tf.estimator.inputs.pandas_input_fn(
    x=features,
    y=label,
    num_epochs=epochs,
    batch_size = batch,
    shuffle = True,
    )
    return in_func()

def my_eval_input_fn(features,label): 
    in_func = tf.estimator.inputs.pandas_input_fn(
    x=features,
    y=label,
    shuffle = False
    )
    return in_func()

def my_pred_input_fn(features): 
    in_func = tf.estimator.inputs.pandas_input_fn(
    x=features,
    shuffle = False
    )
    return in_func()


# train the model
classifier.train(
     input_fn=lambda:my_train_input_fn(df_train[['Name','Sex','Pclass','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']], 
                                                 df_train['Survived'],128,15000),steps=15000)
# Check accuracy on train dataset
eval_result = classifier.evaluate(
    input_fn=lambda:my_eval_input_fn(df_train[['Name','Sex','Pclass','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']], 
                                                 df_train['Survived']))
print('\nTrain set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# generate predictions
predictions = classifier.predict(
     input_fn=lambda:my_pred_input_fn(df_test[['Name','Sex','Pclass','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']]))
print ('Predictions: {}'.format(str(predictions)))
#generate and format outputs and write to the file
res=[]
for pid, pred_dict in zip(df_test['PassengerId'],predictions):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    res.append ([pid,class_id])
res_df = pd.DataFrame.from_records(res,columns = ['PassengerId','Survived'])
res_df.head()
res_df.to_csv("submission_draft.csv",index=False)
