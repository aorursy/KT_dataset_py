import pandas as pd

import random

from datetime import datetime

import re

import tensorflow as tf

import numpy as np

#from sklearn import metrics, cross_validation

from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from tensorflow.contrib import layers

from tensorflow.contrib import learn

random.seed(42)



test_orig = pd.read_csv("../input/test.csv")

train_orig = pd.read_csv("../input/train.csv")

train = train_orig.copy()

test = test_orig.copy()
def isNaN(x):

    return x != x

#fileds present in the data 

train.columns


for key in train.columns:

    print("Number of Classes in ", key, " = ", len(train[key].unique()), 

          " Number of NAN = " ,sum(isNaN(train[key])))

# Age has  177 missing

# Cabin has 687 missing

# Embarked 2 missing

#Replace Missing Embarked value, there are only 2

for k in ['Embarked', 'Sex']:

    for tr2 in [train, test]:

        tr2[k] = tr2[k].fillna('N')
# lets check name next 

def getPName(st):

    if isNaN(st):

        return st

    dt = st.split(',')

    lastName = dt[0]

    fullName = dt[1].split(' ')

    fullName = list(filter(lambda f:f!='', fullName))

    pn = fullName[0]

    if pn == 'Mlle.':

        pn = 'Miss.'

    elif pn == 'Mme.':

        pn = 'Mrs.'

    elif pn == 'Ms.':

        pn = 'Miss.'

    return pn

def getLName(st):

    if isNaN(st):

        return st

    dt = st.split(',')

    lastName = dt[0]

    return lastName

lnameMap = {}

cnt = 0

for tr2 in [train, test]:

    lname = tr2.apply(lambda r:getLName(r['Name']), axis=1).unique()

    for idx in range(0, len(lname)):

        if not lname[idx] in lnameMap:

            lnameMap[lname[idx]] = cnt

            cnt +=1

for tr2 in [train, test]:

    tr2['LName'] = tr2.apply(lambda r:lnameMap[getLName(r['Name'])], axis=1)



train['LName_t'] = learn.ops.categorical_variable(train['LName'], 

                                                  len(train['LName'].unique()),

                                                 embedding_size=6, name='LName')
titleMap = {}

cnt =0

for tr2 in [train, test]:

    titles = tr2.apply(lambda r:getPName(r['Name']), axis=1).unique()

    for idx in range(0, len(titles)):

        if not titles[idx] in titleMap:

            titleMap[titles[idx]] = cnt

            cnt+=1

for tr2 in [train, test]:

    tr2['Titles'] = tr2.apply(lambda r:titleMap[getPName(r['Name'])], axis=1)

print(train['Titles'].unique())

print(titleMap)
def single(dt):

    if (dt['SibSp'] == 0) and (dt['Parch'] == 0):

        return 0

    return 1

for tr2 in [train, test]:

    tr2['Pclass_t'] = tf.one_hot(tr2['Pclass'], 3, 1.0, 0.0)

    tr2['Single'] = tr2.apply(lambda r:single(r), axis=1)



# Create a single column indicating people without kids. (Single) Maybe it will help

# Seperated PClass into one hot encoding
# Age has lot of missing data  (177 entries)

# Let's try to fill them based on average data based on different titles 

titleMapAvgAge = {}

for k in titleMap:

    num_valid = sum((train['Titles'] == titleMap[k]) & ~isNaN(train['Age'] ))

    num_nan = sum((train['Titles'] == titleMap[k]) & isNaN(train['Age'] ))

    sum_age   = sum(train.loc[(train['Titles'] == titleMap[k]) & ~isNaN(train['Age']),'Age'])

    if (num_valid != 0):

        avg = sum_age/num_valid

    else:

        avg = 0

    titleMapAvgAge[titleMap[k]] = avg

    print('title=',k, ' NumberOfItems:',num_valid,'/',num_nan, ' Average Age:',avg)

# Plugin all the missing ages based on title 



def plugAge(r):

    if not isNaN(r['Age']):

        return r['Age']

    return titleMapAvgAge[r['Titles']]

train['Age'] = train.apply(lambda r:plugAge(r), axis=1)

test['Age'] = test.apply(lambda r:plugAge(r), axis=1)

#Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',

#       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Embarked_t', 'Sex_t',

#       'Titles', 'LName', 'Pclass_t', 'Single', 'Age_t', 'SibSp_t', 'Parch_t',

#       'Fare_t', 'LName_t'],



# Features to use 





featureListKey =[ 'Embarked', 'Sex', 'SibSp', 'Parch', 'Titles', 'LName', 'Age', 'Fare', 

                 'Pclass','Single']

                 

fullDt = pd.concat([train, test])

X = train[featureListKey].copy()

y = train['Survived'].copy()



#normalize Age

for key in ['Age', 'Fare']:

    X[key]=((X[key] - X[key].mean())/(X[key].max() - X[key].min()))

X['Pclass'] = X['Pclass'] - 1



XTrain, XTest_, yTrain, yTest_ = train_test_split(X, y, test_size=0.2, random_state=42)

XTest, XDev, yTest, yDev = train_test_split(X, y, test_size=0.5, random_state=42)



XEval=test[featureListKey].copy()

for key in ['Age', 'Fare']:

    XEval[key]=((XEval[key] - XEval[key].mean())/(XEval[key].max() - XEval[key].min()))

XEval['Pclass'] = XEval['Pclass'] - 1
f_embarked = tf.feature_column.indicator_column(

              tf.feature_column.categorical_column_with_vocabulary_list('Embarked',

                        train['Embarked'].unique()))

f_sex = tf.feature_column.indicator_column(

              tf.feature_column.categorical_column_with_vocabulary_list('Sex',

                        train['Sex'].unique()))

#f_sex = tf.feature_column.numeric_column(key='Sex')

f_sibsp = tf.feature_column.numeric_column(key='SibSp')

f_parch = tf.feature_column.numeric_column(key='Parch')

f_age = tf.feature_column.numeric_column(key='Age')

f_titles = tf.feature_column.numeric_column(key='Titles')

f_single = tf.feature_column.indicator_column(

            tf.feature_column.categorical_column_with_identity('Single', 2))

#f_fare = tf.feature_column.numeric_column(key='Fare')

f_pclass = tf.feature_column.embedding_column(

    tf.feature_column.categorical_column_with_identity(key='Pclass',num_buckets=3), 3)

#f_titles = tf.feature_column.embedding_column(

#    tf.feature_column.categorical_column_with_hash_bucket(key='Titles',hash_bucket_size=20), 10)

f_columns= [ f_embarked, f_sex, f_sibsp, f_parch, f_pclass, f_age, f_titles, f_single]

#f_columns= [ f_embarked, f_sex]

train_input_fn = tf.estimator.inputs.pandas_input_fn(x=XTrain, 

                                                y=yTrain,

                                                batch_size=100,

                                                num_epochs=None,

                                                shuffle=True)

train_accuracy_input_fn = tf.estimator.inputs.pandas_input_fn(x=XTrain, 

                                                y=yTrain,

                                                num_epochs=1,

                                                shuffle=False)

dev_input_fn = tf.estimator.inputs.pandas_input_fn(x=XDev, 

                                                y=yDev,

                                                num_epochs=1,

                                                shuffle=False)

test_input_fn = tf.estimator.inputs.pandas_input_fn(x=XTest, 

                                                y=yTest,

                                                num_epochs=1,

                                                shuffle=False)

def model_fn(features, labels, mode, params):

    is_training = True

    if mode == tf.estimator.ModeKeys.PREDICT:

        is_training = False

    input_layer = tf.feature_column.input_layer(features=features, feature_columns=params['fc'])

    first_hidden_layer_ = tf.layers.dense(input_layer,        40, activation=tf.nn.relu)       #40

    first_hidden_layer  = tf.layers.dropout(first_hidden_layer_, rate=params['dropout'], training=is_training)

    second_hidden_layer_= tf.layers.dense(first_hidden_layer, 20, activation=tf.nn.relu)#20

    second_hidden_layer  = tf.layers.dropout(second_hidden_layer_, rate=params['dropout'], training=is_training)

    # Connect the output layer to second hidden layer (no activation fn)

    output_layer = tf.layers.dense(second_hidden_layer, 1) # , activation=tf.nn.sigmoid)

    # Reshape output layer to 1-dim Tensor to return predictions

    predictions = tf.reshape(output_layer, [-1])

    predictions_threshold = tf.cast(tf.greater(predictions,0.5), tf.float32)

    # Calculate loss using mean squared error

    if not is_training:

        return tf.estimator.EstimatorSpec(

            mode = mode,

            predictions={'Survived':predictions_threshold}

        )

    loss = tf.losses.mean_squared_error(labels, predictions)

    #    loss = tf.losses.log_loss(labels, predictions)

    #    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(

    #                    labels=tf.cast(labels, tf.float32), 

    #                    logits=predictions))

    optimizer = tf.train.FtrlOptimizer(

                learning_rate=params['learning_rate'],

                l2_regularization_strength=params['l2_regularization'])



    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    eval_metric_ops = {

      "rmse": tf.metrics.root_mean_squared_error(

          tf.cast(labels, tf.float32), predictions),

      "accuracy":tf.metrics.accuracy(labels, predictions_threshold)



    }



    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.

    return tf.estimator.EstimatorSpec(

      mode=mode,

      loss=loss,

      train_op=train_op,

      eval_metric_ops=eval_metric_ops)
def do_training(lr, drp, l2r, fc, steps):

    random.seed(42)

    model_params={"learning_rate": lr, 'l2_regularization':l2r, 'dropout':drp, 'fc': f_columns}

    my_nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

    r = my_nn.train(input_fn=train_input_fn, steps=steps )

    evtrain=my_nn.evaluate(input_fn=train_accuracy_input_fn)

    evdev=my_nn.evaluate(input_fn=dev_input_fn)

    evtest=my_nn.evaluate(input_fn=test_input_fn)

    print(r)

    print("Loss:%s" %evtest["loss"], " Accuracy Test:", evtest['accuracy']," Accuracy Train:", evtrain['accuracy'])

    return {

        'learningRate':lr,

        'dropout': drp,

        'l2Reg': l2r,

        'AccuracyDev': evdev['accuracy'],

        'AccuracyTest': evtest['accuracy'],

        'AccuracyTrain':evtrain['accuracy'],

        'Loss': evtest['loss'],

        'Steps': steps

    }
acc={}

acnt = 0

samp = [ # lr, l2, drp, steps

    (0.01, 0.1, .1, 100)

]

l2r = 0.1 # 0.2

drp = 0.1 # 0.2

lr = 0.2



# Pick learning rate

acc={}

for lr in [0.1,0.15,0.2, 0.25]:

    acc[acnt] = do_training(lr, drp, l2r, f_columns, 500)

    acnt +=1

for idx in acc:

    print('Learning Rate:', acc[idx]['learningRate'], 'Train Dev Accuracy:',acc[idx]['AccuracyDev'], ' TrainAccuracy:', acc[idx]['AccuracyTrain'] )

#Learning Rate: 0.2 Train Dev Accuracy: 0.836323  TrainAccuracy: 0.825843

#Learning Rate: 0.15 Train Dev Accuracy: 0.831839  TrainAccuracy: 0.831461

l2r = 0.15

drp = 0.1 # 0.2

lr = 0.2



# Pick dropout

acc={}

for drp in [0.05,0.1,0.15, 0.2, 0.25]:

    acc[acnt] = do_training(lr, drp, l2r, f_columns, 500)

    acnt +=1

for idx in acc:

    print('Dropout:', acc[idx]['dropout'], 'Train Dev Accuracy:',acc[idx]['AccuracyDev'], ' TrainAccuracy:', acc[idx]['AccuracyTrain'] )

#Dropout: 0.1 Train Dev Accuracy: 0.829596  TrainAccuracy: 0.831461   

#Dropout: 0.05 Train Dev Accuracy: 0.831839  TrainAccuracy: 0.828652
l2r = 0.15

drp = 0.15 # 0.2

lr = 0.2



# Pick dropout

acc={}

for l2r in [0.05,0.1,0.15, 0.2, 0.25]:

    acc[acnt] = do_training(lr, drp, l2r, f_columns, 500)

    acnt +=1

for idx in acc:

    print('l2Regularization:', acc[idx]['l2Reg'], 'Train Dev Accuracy:',acc[idx]['AccuracyDev'], ' TrainAccuracy:', acc[idx]['AccuracyTrain'] )

#l2Regularization: 0.15 Train Dev Accuracy: 0.82287  TrainAccuracy: 0.825843  

#l2Regularization: 0.15 Train Dev Accuracy: 0.820628  TrainAccuracy: 0.817416
l2r = 0.15

drp = 0.15 # 0.2

lr = 0.15

# Pick steps

acc={}

for stp in [ 2500, 3000, 3500, 4000]:

    acc[acnt] = do_training(lr, drp, l2r, f_columns, stp)

    acnt +=1



for idx in acc:

    print('idx:', idx, 'steps:', acc[idx]['Steps'], 'Train Dev Accuracy:',

          acc[idx]['AccuracyDev'], ' TestAccuracy',acc[idx]['AccuracyTest'],' TrainAccuracy:', acc[idx]['AccuracyTrain'] )

#final parameter selection

l2r = 0.15

drp = 0.15 # 0.2

lr = 0.15

steps = 4000
t = datetime.now()

print("Running with random Seed:", t)

random.seed(t)

model_params={"learning_rate": lr, 'l2_regularization':l2r, 'dropout':drp, 'fc': f_columns}

my_nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

r = my_nn.train(input_fn=train_input_fn, steps=steps )

evtrain=my_nn.evaluate(input_fn=train_accuracy_input_fn)

evdev=my_nn.evaluate(input_fn=dev_input_fn)

evtest=my_nn.evaluate(input_fn=test_input_fn)

print('Train Dev Accuracy:',evdev['accuracy'],

      ' TestAccuracy',evtest['accuracy'],' TrainAccuracy:', evtrain['accuracy'] )

# Train Dev Accuracy: 0.831839  TestAccuracy 0.85618  TrainAccuracy: 0.860955
pred_input_fn = tf.estimator.inputs.pandas_input_fn(x=XEval, 

                                                num_epochs=1,

                                                shuffle=False)

pred = my_nn.predict(input_fn=pred_input_fn)

v=[]

for i, p in enumerate(pred):

    v.append(int(p['Survived']))

result = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':v})

result.to_csv('result-d7.csv', index=False)