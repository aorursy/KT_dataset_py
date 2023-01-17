import pandas as pd

import random



import re

import tensorflow as tf

import numpy as np

#from sklearn import metrics, cross_validation

from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score

from tensorflow.contrib import layers

from tensorflow.contrib import learn



test_orig = pd.read_csv("../input/test.csv")

train_orig = pd.read_csv("../input/train.csv")

train = train_orig.copy()

test = test_orig.copy()

def isNaN(x):

    return x != x

#fileds present in the data 

train.columns
#Lets focus on converting all the column data to be numaric, We can see if missing data is going to cause problems after the first iteration is done

#easy to convert text data 

for key in train.columns:

    print("Number of Classes in ", key, " = ", len(train[key].unique()), 

          " Number of NAN = " ,sum(isNaN(train[key])))

# Age has  177 missing

# Cabin has 687 missing

# Embarked 2 missing

from sklearn.preprocessing import LabelEncoder
#Replace Missing Embarked value 

for k in ['Embarked', 'Sex']:

    for tr2 in [train, test]:

        tr2[k] = tr2[k].fillna('N')

        le = LabelEncoder().fit(tr2[k])

        tr2[k] = le.transform(tr2[k])
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

train.columns
featureListKey =[ 'Embarked', 'Sex', 'SibSp', 'Parch', 'Titles', 'LName', 'Age', 'Fare', 

                 'Pclass'  

            ]

fullDt = pd.concat([train, test])

X = train[featureListKey].copy()

y = train['Survived'].copy()



#normalize Age

for key in ['Age', 'Fare']:

    X[key]=((X[key] - X[key].mean())/(X[key].max() - X[key].min()))

X['Pclass'] = X['Pclass'] - 1



XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)



# Tensor flow feature setup 

f_embarked = tf.feature_column.numeric_column(key='Embarked')

f_sex = tf.feature_column.numeric_column(key='Sex')

f_sibsp = tf.feature_column.numeric_column(key='SibSp')

f_parch = tf.feature_column.numeric_column(key='Parch')

f_age = tf.feature_column.numeric_column(key='Age')

f_titles = tf.feature_column.numeric_column(key='Titles')

#f_fare = tf.feature_column.numeric_column(key='Fare')

f_pclass = tf.feature_column.embedding_column(

    tf.feature_column.categorical_column_with_identity(key='Pclass',num_buckets=3), 3)

#f_titles = tf.feature_column.embedding_column(

#    tf.feature_column.categorical_column_with_hash_bucket(key='Titles',hash_bucket_size=20), 10)

f_columns= [ f_embarked, f_sex, f_sibsp, f_parch, f_pclass, f_age, f_titles]

train_input_fn = tf.estimator.inputs.pandas_input_fn(x=XTrain, 

                                                y=yTrain,

                                                batch_size=100,

                                                num_epochs=None,

                                                shuffle=True)

test_input_fn = tf.estimator.inputs.pandas_input_fn(x=XTest, 

                                                y=yTest,

                                                num_epochs=1,

                                                shuffle=False)

starter_learning_rate = 0.02

global_step = tf.Variable(0, trainable=False)

learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,

                                           1000, 0.96, staircase=True)

my_nn = tf.estimator.DNNClassifier(feature_columns=f_columns,

                                  hidden_units=[40,40],

                                  activation_fn = tf.nn.relu,

                                  dropout=0.1,

                                  n_classes=2,

                                  optimizer=tf.train.FtrlOptimizer(

                                          learning_rate=0.02, 

                                           l2_regularization_strength=0.2

                                ))
my_nn.train(input_fn=train_input_fn, steps=500 )

ev=my_nn.evaluate(input_fn=test_input_fn)

print("Loss:%s" %ev["loss"], " Accuracy:", ev['accuracy'])
XEval=test[featureListKey].copy()

for key in ['Age', 'Fare']:

    XEval[key]=((XEval[key] - XEval[key].mean())/(XEval[key].max() - XEval[key].min()))

XEval['Pclass'] = XEval['Pclass'] - 1

pred_input_fn = tf.estimator.inputs.pandas_input_fn(x=XEval, 

                                                num_epochs=1,

                                                shuffle=False)

pred = my_nn.predict(input_fn=pred_input_fn)

v=[]

for i, p in enumerate(pred):

    v.append(p['class_ids'][0])

result = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':v})

result.to_csv('result.csv', index=False)