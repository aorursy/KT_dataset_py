import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf



%matplotlib inline
dftrain = pd.read_csv('../input/titanic_train (1).csv')
dftrain.head()
print(dftrain.info())
dftrain.isnull().values.any()
dftrain.columns
dftrain['embark_town'].unique()
dftrain['deck'].unique()
def finding_unknown(df):

    for col in df.columns:

        number = 0

        for i in range(0,len(df[col])):

            if i != None:

                if df[col][i] == 'unknown':

                    number+=1

        print(col,'=',number,'\n')
finding_unknown(dftrain)
#as the number of unknown values in deck is huge, it is very unlikely that it is going to help in predicting the output

dftrain = dftrain.drop('deck',axis=1)
dftrain.head()
dftrain.loc[dftrain['embark_town'] == 'unknown']
#as we have just one row, its better to remove the unknown data

dftrain = dftrain.drop(dftrain.index[48])
dftrain.info()
dftrain.head()
X = dftrain.drop(['survived','n_siblings_spouses','parch'],axis=1)
y = dftrain['survived']
#creating numeric feature columns

feature_columns = []

feature_columns.append(tf.feature_column.numeric_column('age',dtype=tf.float32))

feature_columns.append(tf.feature_column.numeric_column('fare',dtype=tf.float32))
#creating categorical feature columns

categorical_columns = ['sex','class','embark_town','alone']

tc = tf.feature_column

def create_cat_featcol(cat_cols):

    for feature_name in cat_cols:

        vocab = dftrain[feature_name].unique()

        feature_columns.append(tc.indicator_column(tc.categorical_column_with_vocabulary_list(feature_name,vocab)))
create_cat_featcol(categorical_columns)
feature_columns
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y)
#creating an input function

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=5,shuffle=True)
#defining a classifier

classifier = tf.estimator.DNNClassifier(hidden_units=[20,20,20,20], n_classes=2,feature_columns=feature_columns)
#training the classifier on the input function

classifier.train(input_fn=input_func,steps=50)
#creating a prediction function

pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
#creating a list of the predictions

note_predictions = list(classifier.predict(input_fn=pred_fn))
#a sample prediction in the list

note_predictions[0]
final_preds  = []

for pred in note_predictions:

    final_preds.append(pred['class_ids'][0])
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,final_preds))
#using Boosted Tress as a classifier

#adding L2 regularisation and decreasing the num of tress immensely improved the results 

#because the data is very less and it is very probable of the model to overfit

classifier_2 = tf.estimator.BoostedTreesClassifier(feature_columns = feature_columns,n_batches_per_layer=1,l2_regularization=0.1,n_trees=50)
classifier_2.train(input_fn=input_func,steps=50)
note_predictions_2 = list(classifier_2.predict(input_fn=pred_fn))

    
final_preds_2  = []

for pred in note_predictions_2:

    final_preds_2.append(pred['class_ids'][0])
print(classification_report(y_test,final_preds_2))