import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.python.framework import ops

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "99"
from tensorflow.python.client import device_lib

device_lib.list_local_devices()
from __future__ import absolute_import #파이썬 2,3버전문제로 인해 생기는 것들을 방지하고자 

from __future__ import division

from __future__ import print_function



import itertools

import warnings

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from pylab import rcParams

import matplotlib



from sklearn.model_selection import train_test_split

# 최소 최대를 0~1범위로 표준화 시키는, 전처리 과정에서 필요한 함수

from sklearn.preprocessing import MinMaxScaler



tf.logging.set_verbosity(tf.logging.INFO)

sess = tf.InteractiveSession()



train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

print('Shape of the train data with all features:', train.shape)

train = train.select_dtypes(exclude=['object'])

print("")

print('Shape of the train data with numerical features:', train.shape)

train.drop('Id',axis = 1, inplace = True)

train.fillna(0,inplace=True)
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test = test.select_dtypes(exclude=['object'])



#숫자형인 것들만 남겨놓고 여기 안에서 해결하겠다는 의지



ID = test.Id

test.fillna(0,inplace=True) #결측치 임의로 0 넣기

test.drop('Id',axis = 1, inplace = True)



print("")

print("List of features contained our dataset:",list(train.columns))
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest



clf = IsolationForest(max_samples = 100, random_state = 42)

clf.fit(train)

y_noano = clf.predict(train)

y_noano = pd.DataFrame(y_noano, columns = ['Top'])

y_noano[y_noano['Top'] == 1].index.values



train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]

train.reset_index(drop = True, inplace = True)

print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])

print("Number of rows without outliers:", train.shape[0])
train.head(10)
col_train = list(train.columns)

col_train_bis = list(train.columns)



# 예측해야 할 값은 처리할 필요 없으므로 제거

col_train_bis.remove('SalePrice')



mat_train = np.matrix(train)

mat_test  = np.matrix(test)

mat_new = np.matrix(train.drop('SalePrice',axis = 1))

mat_y = np.array(train.SalePrice).reshape((1314,1))



prepro_y = MinMaxScaler()

prepro_y.fit(mat_y)



prepro = MinMaxScaler()

prepro.fit(mat_train)



prepro_test = MinMaxScaler()

prepro_test.fit(mat_new)



train = pd.DataFrame(prepro.transform(mat_train),columns = col_train)

test  = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_bis)



train.head()
# List of features

COLUMNS = col_train

FEATURES = col_train_bis

LABEL = "SalePrice"



# Columns for tensorflow : 텐서플로에게 column값들을 제시

feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]



# Training set and Prediction set with the features to predict : 무엇을 학습해야 하는지 input, output 값 주기

training_set = train[COLUMNS]

prediction_set = train.SalePrice



# Train and Test : train 안에서 분리하기 (x_train > training_set, x_test > testing_set)

x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES] , prediction_set, test_size=0.33, random_state=42)

y_train = pd.DataFrame(y_train, columns = [LABEL])

training_set = pd.DataFrame(x_train, columns = FEATURES).merge(y_train, left_index = True, right_index = True)

training_set.head()



# Training for submission

training_sub = training_set[col_train]
# Same thing but for the test set : 나중에 test에도 실행되기 위해서 따로 test 셋도 처리하기

y_test = pd.DataFrame(y_test, columns = [LABEL])

testing_set = pd.DataFrame(x_test, columns = FEATURES).merge(y_test, left_index = True, right_index = True)

testing_set.head()
# Model  설정하기 

tf.logging.set_verbosity(tf.logging.ERROR)

regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, 

                                          activation_fn = tf.nn.relu, hidden_units=[256, 128, 64, 32, 16])

#optimizer = tf.train.GradientDescentOptimizer( learning_rate= 0.1 ))

    

# Reset the index of training : 인덱스 정리하기

training_set.reset_index(drop = True, inplace =True)
#

def input_fn(data_set, pred = False):

    

    if pred == False:

        

        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}

        labels = tf.constant(data_set[LABEL].values)

        

        return feature_cols, labels



    if pred == True:

        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}

        

        return feature_cols
# Deep Neural Network Regressor with the training set which contain the data split by train test split

regressor.fit(input_fn=lambda: input_fn(training_set), steps=2000)
# Evaluation on the test set created by train_test_split

ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)
# Display the score on the testing set

# 0.002X in average

loss_score1 = ev["loss"]

print("Final Loss on the testing set: {0:f}".format(loss_score1))
# Predictions

y = regressor.predict(input_fn=lambda: input_fn(testing_set))

predictions = list(itertools.islice(y, testing_set.shape[0]))
def leaky_relu(x):

    return tf.nn.relu(x) - 0.01 * tf.nn.relu(-x)
# Model

regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, 

                                          activation_fn = leaky_relu, hidden_units=[256, 128, 64, 32, 16])

    

# Deep Neural Network Regressor with the training set which contain the data split by train test split

regressor.fit(input_fn=lambda: input_fn(training_set), steps=2000)



# Evaluation on the test set created by train_test_split

ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)

# Display the score on the testing set

# 0.002X in average

loss_score2 = ev["loss"]

print("Final Loss on the testing set with Leaky Relu: {0:f}".format(loss_score2))



# Predictions

y_predict = regressor.predict(input_fn=lambda: input_fn(test, pred = True))
# Model

regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, 

                                          activation_fn = tf.nn.elu, hidden_units=[200, 100, 50, 25, 12])

    

# Deep Neural Network Regressor with the training set which contain the data split by train test split

regressor.fit(input_fn=lambda: input_fn(training_set), steps=2000)



# Evaluation on the test set created by train_test_split

ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)



loss_score3 = ev["loss"]
print("Final Loss on the testing set with Elu: {0:f}".format(loss_score3))

# Predictions

y_predict = regressor.predict(input_fn=lambda: input_fn(test, pred = True))
# Import and split : 저번과 동일한 방식

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train.drop('Id',axis = 1, inplace = True)

train_numerical = train.select_dtypes(exclude=['object'])

train_numerical.fillna(0,inplace = True)



train_categoric = train.select_dtypes(include=['object'])

train_categoric.fillna('NONE',inplace = True) #numerical과 다르게 적용

train = train_numerical.merge(train_categoric, left_index = True, right_index = True) 



#test셋도 동일하게 적용

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

ID = test.Id

test.drop('Id',axis = 1, inplace = True)

test_numerical = test.select_dtypes(exclude=['object'])

test_numerical.fillna(0,inplace = True)



test_categoric = test.select_dtypes(include=['object'])

test_categoric.fillna('NONE',inplace = True) #numerical과 다르게 적용



test = test_numerical.merge(test_categoric, left_index = True, right_index = True) 
# Removie the outliers: 저번과 동일한 방식

from sklearn.ensemble import IsolationForest



clf = IsolationForest(max_samples = 100, random_state = 42)

clf.fit(train_numerical)

y_noano = clf.predict(train_numerical)

y_noano = pd.DataFrame(y_noano, columns = ['Top'])

y_noano[y_noano['Top'] == 1].index.values



train_numerical = train_numerical.iloc[y_noano[y_noano['Top'] == 1].index.values]

train_numerical.reset_index(drop = True, inplace = True)



train_categoric = train_categoric.iloc[y_noano[y_noano['Top'] == 1].index.values]

train_categoric.reset_index(drop = True, inplace = True)



train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]

train.reset_index(drop = True, inplace = True)
#Preprocessing

col_train_num = list(train_numerical.columns)

col_train_num_bis = list(train_numerical.columns)



col_train_cat = list(train_categoric.columns)



col_train_num_bis.remove('SalePrice')



mat_train = np.matrix(train_numerical)

mat_test  = np.matrix(test_numerical)

mat_new = np.matrix(train_numerical.drop('SalePrice',axis = 1))

mat_y = np.array(train.SalePrice)



prepro_y = MinMaxScaler()

prepro_y.fit(mat_y.reshape(1314,1))



prepro = MinMaxScaler()

prepro.fit(mat_train)



prepro_test = MinMaxScaler()

prepro_test.fit(mat_new)



train_num_scale = pd.DataFrame(prepro.transform(mat_train),columns = col_train)

test_num_scale  = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_bis)
train[col_train_num] = pd.DataFrame(prepro.transform(mat_train),columns = col_train_num)

test[col_train_num_bis]  = test_num_scale
# Model 적용

COLUMNS = col_train_num

FEATURES = col_train_num_bis

LABEL = "SalePrice"



FEATURES_CAT = col_train_cat



engineered_features = []



for continuous_feature in FEATURES:

    engineered_features.append(

        tf.contrib.layers.real_valued_column(continuous_feature))



for categorical_feature in FEATURES_CAT:

    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(

        categorical_feature, hash_bucket_size=1000)



    engineered_features.append(tf.contrib.layers.embedding_column(sparse_id_column=sparse_column, dimension=16,combiner="sum"))

                                 

# Training set and Prediction set with the features to predict

training_set = train[FEATURES + FEATURES_CAT]

prediction_set = train.SalePrice



# Train and Test 

x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES + FEATURES_CAT] ,

                                                    prediction_set, test_size=0.20, random_state=42)

y_train = pd.DataFrame(y_train, columns = [LABEL])

training_set = pd.DataFrame(x_train, columns = FEATURES + FEATURES_CAT).merge(y_train, left_index = True, right_index = True)



# 제출하기

training_sub = training_set[FEATURES + FEATURES_CAT]

testing_sub = test[FEATURES + FEATURES_CAT]
# Same thing but for the test set

y_test = pd.DataFrame(y_test, columns = [LABEL])

testing_set = pd.DataFrame(x_test, columns = FEATURES + FEATURES_CAT).merge(y_test, left_index = True, right_index = True)
training_set[FEATURES_CAT] = training_set[FEATURES_CAT].applymap(str)

testing_set[FEATURES_CAT] = testing_set[FEATURES_CAT].applymap(str)



def input_fn_new(data_set, training = True):

    continuous_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}

    

    categorical_cols = {k: tf.SparseTensor(

        indices=[[i, 0] for i in range(data_set[k].size)], values = data_set[k].values, dense_shape = [data_set[k].size, 1]) for k in FEATURES_CAT}



    # Merges the two dictionaries into one.

    feature_cols = dict(list(continuous_cols.items()) + list(categorical_cols.items()))

    

    if training == True:

        # Converts the label column into a constant Tensor.

        label = tf.constant(data_set[LABEL].values)



        # Returns the feature columns and the label.

        return feature_cols, label

    

    return feature_cols



# Model

regressor = tf.contrib.learn.DNNRegressor(feature_columns = engineered_features, 

                                          activation_fn =leaky_relu, hidden_units=[256, 128, 64, 32, 16])

categorical_cols = {k: tf.SparseTensor(indices=[[i, 0] for i in range(training_set[k].size)], values = training_set[k].values, dense_shape = [training_set[k].size, 1]) for k in FEATURES_CAT}
# Deep Neural Network Regressor with the training set which contain the data split by train test split

regressor.fit(input_fn = lambda: input_fn_new(training_set) , steps=2000)
ev = regressor.evaluate(input_fn=lambda: input_fn_new(testing_set, training = True), steps=1)

loss_score4 = ev["loss"]

print("Final Loss on the testing set: {0:f}".format(loss_score4))
# Model

regressor = tf.contrib.learn.DNNRegressor(feature_columns = engineered_features, 

                                          activation_fn = tf.nn.relu, hidden_units=[2500])
# Deep Neural Network Regressor with the training set which contain the data split by train test split

regressor.fit(input_fn = lambda: input_fn_new(training_set) , steps=2000)
ev = regressor.evaluate(input_fn=lambda: input_fn_new(testing_set, training = True), steps=1)

loss_score5 = ev["loss"]
print("Final Loss on the testing set: {0:f}".format(loss_score5))
list_score = [loss_score1, loss_score2, loss_score3, loss_score4,loss_score5]

list_model = ['Relu_cont', 'LRelu_cont', 'Elu_cont', 'Relu_cont_categ','Shallow_1ku']
import matplotlib.pyplot as plt; plt.rcdefaults()



plt.style.use('ggplot')

objects = list_model

y_pos = np.arange(len(objects))

performance = list_score

 

plt.barh(y_pos, performance, align='center', alpha=0.9)

plt.yticks(y_pos, objects)

plt.xlabel('Loss ')

plt.title('Model compared without hypertuning')

 

plt.show()
def to_submit(pred_y,name_out):

    y_predict = list(itertools.islice(pred_y, test.shape[0]))

    y_predict = pd.DataFrame(prepro_y.inverse_transform(np.array(y_predict).reshape(len(y_predict),1)), columns = ['SalePrice'])

    y_predict = y_predict.join(ID)

    y_predict.to_csv(name_out + '.csv',index=False)
y_predict = regressor.predict(input_fn=lambda: input_fn_new(testing_sub, training = False))    

to_submit(y_predict, "submission_shallow")