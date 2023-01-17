import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
%matplotlib inline
data = pd.read_csv("../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv")
data.head()
# drop the object id columns, they are of no use in the analysis
data.drop(['objid','specobjid'], axis=1, inplace=True)
data.head()
sns.countplot(x=data['class'])
def change_category_to_number(classCat):
    if classCat=='STAR':
        return 0
    elif classCat=='GALAXY':
        return 1
    else:
        return 2
# assign a numerical value to the categorical field of class, by using the above function
data['classCat'] = data['class'].apply(change_category_to_number)
data.head()
sns.pairplot(data[['u','g','r','i']])
data.drop(['run','rerun','camcol','field','class'],axis=1,inplace=True)
data.head()
data.dtypes
X = data.drop('classCat', axis=1)
y = data['classCat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=128)
dtClassifer = DecisionTreeClassifier(max_leaf_nodes=15,random_state=0)
dtClassifer.fit(X_train, y_train)
prediction = dtClassifer.predict(X_test)
prediction[:10]
y_test[:10]
accuracy_score(y_true=y_test, y_pred=prediction)
lrClassifier = LogisticRegression()
lrClassifier.fit(X_train,y_train)
prediction = lrClassifier.predict(X_test)
prediction[:10]
y_test[:10]
accuracy_score(y_true=y_test, y_pred=prediction)
featcols = [tf.feature_column.numeric_column('ra'),
            tf.feature_column.numeric_column('dec'),
            tf.feature_column.numeric_column('u'),
            tf.feature_column.numeric_column('g'),
            tf.feature_column.numeric_column('r'),
            tf.feature_column.numeric_column('i'),
            tf.feature_column.numeric_column('z'),
            tf.feature_column.numeric_column('redshift'),
            tf.feature_column.numeric_column('plate'),
            tf.feature_column.numeric_column('mjd'),
            tf.feature_column.numeric_column('fiberid')
           ]
model = tf.estimator.LinearClassifier(n_classes=3,
                                      optimizer=tf.train.FtrlOptimizer(l2_regularization_strength=0.1,learning_rate=0.01),
                                     feature_columns=featcols)
data.head()
def get_input_fn(num_epochs,n_batch,shuffle):
    return tf.estimator.inputs.pandas_input_fn(
        x=X_train,
        y=y_train,
        batch_size=n_batch,
        num_epochs=num_epochs,
        shuffle=shuffle
    )
model.train(input_fn=get_input_fn(100,128,True),steps=1000)
def evaluate_fn(num_epochs,n_batch,shuffle):
    return tf.estimator.inputs.pandas_input_fn(
        x=X_test,
        y=y_test,
        batch_size=n_batch,
        num_epochs=num_epochs,
        shuffle=shuffle
    )
model.evaluate(input_fn=evaluate_fn(100,128,True),steps=1000)
dnn_model = tf.estimator.DNNClassifier(n_classes=3,
                                       feature_columns=featcols,
                                       hidden_units=[1024,512,256,32,3],
                                       activation_fn=tf.nn.relu,
                                       optimizer='Adam',
                                       dropout=0.2,
                                      )
dnn_model.train(input_fn=get_input_fn(100,128,True),steps=1000)
dnn_model.evaluate(input_fn=evaluate_fn(100,128,True),steps=1000)
dnnlcc_model = tf.estimator.DNNLinearCombinedClassifier(n_classes=3,dnn_activation_fn='relu',dnn_dropout=0.2,dnn_hidden_units=[1024,512,256,32,3],dnn_optimizer='Adam',dnn_feature_columns=featcols,linear_feature_columns=featcols)
dnnlcc_model.train(input_fn=get_input_fn(100,128,True),steps=1000)
dnnlcc_model.evaluate(input_fn=evaluate_fn(100,128,True),steps=1000)
