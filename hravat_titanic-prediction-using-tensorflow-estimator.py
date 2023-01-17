###Acccuracy improvemeent from 0.55 to 0.61244
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_main = pd.read_csv("../input/train.csv")
df_main['Embarked'].fillna('S', inplace=True)
df_main['Pclass'] = df_main['Pclass'].astype(str)
df_main = df_main.loc[:,['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
#df_main = df_main.dropna(how="any", axis=0)
df_main.head()
df_main_stats_summary = df_main.loc[:,['Age','Fare','SibSp','Parch']]
df_main_stats_summary.describe()
import tensorflow as tf
import shutil
print(tf.__version__)
def make_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df['Survived'],
    batch_size = 10,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 10,
    num_threads = 1
  )
def make_prediction_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = None,
    batch_size = 10,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 10,
    num_threads = 1  
  )
Pclass_list =  df_main.Pclass.unique()
Pclass_list
gender_list =  df_main.Sex.unique()
gender_list
embarked_list =  df_main.Embarked.unique()
embarked_list 
def make_feature_cols():
 Pclass_categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Pclass',
        vocabulary_list=Pclass_list)
    
 Gender_categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Sex',
        vocabulary_list=gender_list)
 
 Embarked_categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Embarked',
        #vocabulary_list=embarked_list,
        vocabulary_list=embarked_list,
        default_value=0
        )


    
    
 return [  
    #Pclass_feature_column =
    tf.feature_column.embedding_column(Pclass_categorical_column,dimension=3),
    
     
    #gender_feature_column =
    tf.feature_column.embedding_column(Gender_categorical_column,dimension=2), 
    
    #age_feature_column =  
    tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column('Age',dtype=tf.float32), boundaries = np.arange(0.0, 100 , 10).tolist()
        ),

    #SibSp_feature_column =  
    tf.feature_column.numeric_column('SibSp',dtype=tf.float32),
      
    #Parch_feature_column =  
    tf.feature_column.numeric_column('Parch',dtype=tf.float32), 
      
    #Fare_feature_column =  
    tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column('Fare',dtype=tf.float32), boundaries = np.arange(0.0, 700 , 100).tolist()
    ),
        
    #embarked_feature_column =
    tf.feature_column.embedding_column(Embarked_categorical_column,dimension=3),
         
  ]
np.random.seed(seed=1) #makes result reproducible
msk = np.random.rand(len(df_main)) < 0.8
traindf = df_main[msk]
evaldf = df_main[~msk]
# Create estimator train and evaluate function# Creat 
def train_and_evaluate(output_dir, num_train_steps):
    
    estimator = tf.estimator.DNNClassifier(
    #model_dir = output_dir,
    feature_columns=make_feature_cols(),
    hidden_units=[32,16,4],
    n_classes=2,    
    optimizer=tf.train.AdamOptimizer(
      learning_rate=0.01
    ))
    
    train_spec = tf.estimator.TrainSpec(input_fn = make_input_fn(traindf, 1000), 
                                      max_steps = num_train_steps )
    
    eval_spec = tf.estimator.EvalSpec(input_fn = make_input_fn(evaldf, 1), 
                                    steps = 1, 
                                    start_delay_secs = 60, # start evaluating after N seconds, 
                                    throttle_secs = 60 )  # evaluate every N seconds
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
   
    return estimator
#OUTDIR = '../Titanic Prediction/Model_Details'
tf.logging.set_verbosity(tf.logging.INFO)
shutil.rmtree(OUTDIR, ignore_errors = True)
estimator = train_and_evaluate(OUTDIR, 100)
a = estimator.predict(make_prediction_input_fn(evaldf,1))
df_test = pd.read_csv("../input/test.csv")
df_test['Embarked'].fillna('S', inplace=True)
df_test['Pclass'] = df_test['Pclass'].astype(str)
df_test = df_test.loc[:,['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
#df_main = df_main.dropna(how="any", axis=0)
df_test.head()
test = estimator.predict(make_prediction_input_fn(df_test,1))
pred_list = []

for i in test:
    for a in i['classes']:
        pred_list.append(str(a)[1:][1:2])   
df_submit = pd.read_csv("../input/test.csv")
df_submit = pd.DataFrame(df_submit['PassengerId'])
df_pred = pd.DataFrame({'Survived':pred_list})
df_submit['Survived'] = df_pred
df_submit.to_csv('hravat_titanic_pred_titl.csv',sep=',',index = False)
