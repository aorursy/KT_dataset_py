import pandas as pd 
import numpy as np

census = pd.read_csv( "../input/adult.csv" )

census.head()
census[ 'income' ].unique()
def label_fix( label ):
    if label == '<=50K':
        return 0
    else:
        return 1
    
census[ 'income' ] = census[ 'income' ].apply( label_fix )

census.head()
from sklearn.model_selection import train_test_split
x_data = census.drop( 'income', axis = 1 )
y_labels = census[ 'income' ]
X_train, X_test, y_train, y_test = train_test_split( x_data, y_labels, test_size = 0.3, random_state = 101 )

import tensorflow as tf

# for these categorical columns we know there are a few known values
sex = tf.feature_column.categorical_column_with_vocabulary_list( "sex", ["Female", "Male"] )
race = tf.feature_column.categorical_column_with_vocabulary_list( "race", ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'] )
relationship = tf.feature_column.categorical_column_with_vocabulary_list( "relationship", ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'] )

# hash is good for converting columns with large number of values or unknown values
occupation = tf.feature_column.categorical_column_with_hash_bucket( 'occupation', hash_bucket_size= 1000 )
marital_status = tf.feature_column.categorical_column_with_hash_bucket( 'marital.status', hash_bucket_size= 1000 )
education = tf.feature_column.categorical_column_with_hash_bucket( 'education', hash_bucket_size= 1000 )
workclass = tf.feature_column.categorical_column_with_hash_bucket( 'workclass', hash_bucket_size= 1000 )
native_country = tf.feature_column.categorical_column_with_hash_bucket( 'native.country', hash_bucket_size= 1000 )

# columns with continuous values
age = tf.feature_column.numeric_column( 'age' )
education_num = tf.feature_column.numeric_column( 'education.num' )
capital_gain = tf.feature_column.numeric_column( 'capital.gain' )
capital_loss = tf.feature_column.numeric_column( 'capital.loss' )
hours_per_week = tf.feature_column.numeric_column( 'hours.per.week' )

feat_cols = [ sex, race, relationship, occupation, marital_status, education, workclass, native_country, age, education_num, capital_gain, capital_loss, hours_per_week ]

input_func = tf.estimator.inputs.pandas_input_fn( x= X_train, y= y_train, batch_size= 100, num_epochs= None, shuffle= True )
model = tf.estimator.LinearClassifier( feature_columns= feat_cols )
model.train( input_fn= input_func, steps= 5000 )
pred_fn = tf.estimator.inputs.pandas_input_fn( x= X_test, batch_size= len( X_test ), shuffle= False )
predictions = list( model.predict( input_fn= pred_fn ) )
predictions[ 0 ]
final_preds = []
for pred in predictions: 
    final_preds.append( pred[ 'class_ids' ][ 0 ] )
    
final_preds[ : 10 ]
from sklearn.metrics import classification_report
print( classification_report( y_test, final_preds ) )
from sklearn import metrics

# roc_curve not working:  UndefinedMetricWarning: No positive samples in y_true, true positive value should be meaningless
#fpr, tpr, thresholds = metrics.roc_curve( y_test.values, final_preds, pos_label= 2 )
#metrics.auc(fpr, tpr)

metrics.roc_auc_score( y_test.values, final_preds )
metrics.average_precision_score( y_test.values, final_preds )  
import math as m
#y_test.to_frame().query( 'isna( income )' )
y_test[ y_test.isna() ]
