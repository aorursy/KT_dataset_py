import pandas as pd

import tensorflow as tf

import numpy as np


CONTINOUS = ["Pclass","Age","Fare","Parch","SibSp"]

LABEL = "Survived"

NEW_COLUMN = "Family_members"
def input_fn(df,frame_type):

    if frame_type == "train":

        continuous_columns = {k: tf.constant(df[k].values) for k in CONTINOUS}

        categorical_columns = {"Sex":tf.SparseTensor(indices=[[i, 0] for i in range(df["Sex"].size)],values = df["Sex"].values,dense_shape = [df["Sex"].size,1])}

    

        feature_cols = continuous_columns    

        feature_cols.update(categorical_columns)

        labels = tf.constant(df["Survived"].values)

        return feature_cols, labels

    else:

        continuous_columns = {k: tf.constant(df[k].values) for k in CONTINOUS}

        categorical_columns = {"Sex":tf.SparseTensor(indices=[[i, 0] for i in range(df["Sex"].size)],

                                            values = df["Sex"].values,

                                            dense_shape = [df["Sex"].size,1])}

    

        feature_cols = continuous_columns

        feature_cols.update(categorical_columns)        



        return feature_cols
def build_fn(model_dir, model_type):

    

    #Processing categorical data

    Sex = tf.contrib.layers.sparse_column_with_keys("Sex",keys = ["male","female"])



    # Processing of the continous columns

    Age = tf.contrib.layers.real_valued_column("Age")

    Pclass = tf.contrib.layers.real_valued_column("Pclass")

    Fare = tf.contrib.layers.real_valued_column("Fare")

    Family_members = tf.contrib.layers.real_valued_column("Family_members")

    Embarked_modify = tf.contrib.layers.real_valued_column("Embarked_modify")

    Parch =  tf.contrib.layers.real_valued_column("Parch")

    SibSp =  tf.contrib.layers.real_valued_column("SibSp")

    

    #Transform Age

    age_buckets = tf.contrib.layers.bucketized_column(Age,boundaries = [18,25,30,35,40,45,50,55,60,65])



    

    #Cross Columns

    combined_1 = tf.contrib.layers.crossed_column([age_buckets,Sex],hash_bucket_size = 1e4)

    

    

    deep_columns = [Fare,age_buckets,tf.contrib.layers.embedding_column(Sex, dimension =8),Parch,SibSp]

    wide_columns = [Pclass,Family_members,Sex,combined_1]

    

    # Wide model

    if model_type == "wide":

        m = tf.contrib.learn.LinearClassifier(model_dir = model_dir,feature_columns = wide_columns)

    # Deep model

    elif model_type == "deep":

        m = tf.contrib.learn.DNNClassifier(model_dir = model_dir, feature_columns = deep_columns, hidden_units = [50,50])

    

    return m

def train_and_eval_fn(model_dir, model_type,train_dir,test_dir,train_steps):

    df = pd.read_csv(train_dir,skipinitialspace = True)

    df_train = df._slice(slice(0,700))

    df_test = df._slice(slice(700,890))

    

    # Processing df_train

    df_train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

    median_age = df_train['Age'].dropna().median()

    if len(df_train.Age[ df_train.Age.isnull() ]) > 0:

        df_train.loc[ (df_train.Age.isnull()), 'Age'] = median_age

    df_train.dropna(how="any",axis=0)

    df_train['Family_members'] = df_train['Parch'] + df_train['SibSp']

    

    

    # Processing df_test

    df_test.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

    median_age2 = df_train['Age'].dropna().median()

    if len(df_test.Age[ df_test.Age.isnull() ]) > 0:

        df_test.loc[ (df_test.Age.isnull()), 'Age'] = median_age2

    df_test.dropna(how="any",axis=0)

    df_test['Family_members'] = df_test['Parch'] + df_test['SibSp']    

    

    test_file = pd.read_csv(test_dir,skipinitialspace=True)

    del test_file['PassengerId']

    del test_file['Name']

    del test_file['Ticket']

    del test_file['Cabin']

    

    median_age3 = test_file['Age'].dropna().median()

    if len(test_file.Age[ test_file.Age.isnull() ]) > 0:

        test_file.loc[ (test_file.Age.isnull()), 'Age'] = median_age3

        

    test_file.dropna(how="any",axis=0)

    features = input_fn(test_file,"test")



    m = build_fn(model_dir,model_type)

    y,_ = input_fn(df_train,"train")



    m.fit(input_fn = lambda: input_fn(df_train,"train"),steps = train_steps)

    results = m.evaluate(input_fn = lambda: input_fn(df_test,"train"),steps = 1) 

    

    

    # I am having problem predicting it...Please help!

    New_results = m.predict(x=features)

    

    with open("new.txt","w") as f:

        f.write(New_results)

    

    for key in sorted(results):

        print("%s : %s" % (key,results[key]))

        

FLAGS = None      

    

    

    

    

import argparse

import sys

def main(_):

    train_and_eval_fn(FLAGS.model_dir, FLAGS.model_type,FLAGS.train_data,FLAGS.test_data,FLAGS.train_steps)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.register("type","bool",lambda v: v.lower()=="true")

    parser.add_argument("--model_dir",

                       type = str,

                       default ="/kaggle/",

                        help ="Base directory for output models"

                       )

    parser.add_argument("--model_type",

                       type =str,

                       default = "deep",

                       help="Valid model tyoes: {'wide','deep','wide-deep'}")

    parser.add_argument("--train_steps",

                       type = int,

                       default = 200,

                       help="Number of training steps.")

    parser.add_argument(



      "--train_data",



      type=str,



      default="/kaggle/input/train.csv",



      help="Path to the training data."



      )



    parser.add_argument(



          "--test_data",



          type=str,



          default="/kaggle/input/test.csv",



          help="Path to the test data."



        )

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main =main,argv = [sys.argv[0]] + unparsed)



    # It will show 'Permission denied error here'.

    # Run it in a jupyter notebook and please help me predict. 

    

    