import sys

print(sys.executable)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import time

import os

from IPython.display import clear_output

%matplotlib inline



import tensorflow as tf

# import tensorflow_addons as tfa



print('tensorflow version: {}'.format(tf.__version__))

print('GPU 사용 가능 여부: {}'.format(tf.test.is_gpu_available()))

print(tf.config.list_physical_devices('GPU'))
train_pandas = pd.read_csv("../input/train.csv")

test_pandas = pd.read_csv("../input/test.csv")
train_pandas
train_data = train_pandas

test_data = test_pandas

train_data.info()
train_data_without_label = train_data.drop(["SalePrice"],axis=1)

train_test_append = train_data_without_label.append(test_data)



dictionary_list = []

for col in train_test_append.select_dtypes(include=['object']).columns:

    tmp_dic = {}

    for number, key in enumerate(list(set(train_data_without_label[col].values))):

        if isinstance(key, float): # if key is 'nan' then value is also 'nan'

            tmp_dic[key] = key

        else:

            tmp_dic[key] = number

    dictionary_list.append(tmp_dic)

    print(col, tmp_dic, len(tmp_dic))



print(dictionary_list)
for index, col in enumerate(train_data_without_label.select_dtypes(include=['object']).columns):

    train_data[col] = train_data[col].map(dictionary_list[index]).astype(float)

    test_data[col] = test_data[col].map(dictionary_list[index]).astype(float)
for col in train_data:

    if train_data[col].isnull().any(): # fill nan to mean in that column

        train_data[col] = train_data[col].fillna(train_data[col].mean())

for col in test_data:

    if test_data[col].isnull().any(): # fill nan to mean in that column

        test_data[col] = test_data[col].fillna(test_data[col].mean())
train_data.info()
test_data.info()
train_data.drop('Id',axis = 1, inplace = True)



ID = test_data.Id

test_data.drop('Id',axis = 1, inplace = True)



print("")

print("List of features contained our dataset:",list(train_data.columns))
train_data.head(10)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler



col_train = list(train_data.columns)

col_train_bis = list(train_data.columns)



col_train_bis.remove('SalePrice')



prepro_label = MinMaxScaler() # use later to reverse transform

prepro_label.fit(train_data["SalePrice"].values.reshape(-1,1))



prepro_train = MinMaxScaler()

prepro_train.fit(train_data)



prepro_test = MinMaxScaler()

# prepro_test.fit(test_data)

prepro_test.fit(train_data.drop(["SalePrice"],axis=1))



train = pd.DataFrame(prepro_train.transform(train_data),columns = col_train)

test  = pd.DataFrame(prepro_test.transform(test_data),columns = col_train_bis)



train.head()
# List of features

COLUMNS = col_train

FEATURES = col_train_bis

LABEL = "SalePrice"



# Columns

feature_cols = FEATURES



# Training set and Prediction set with the features to predict

training_set = train[COLUMNS]

prediction_set = train.SalePrice



# Train and Test 

x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES] , prediction_set, test_size=0.2, random_state=2)

y_train = pd.DataFrame(y_train, columns = [LABEL])

training_set = pd.DataFrame(x_train, columns = FEATURES).merge(y_train, left_index = True, right_index = True)

training_set.head()
# Training for submission

training_sub = training_set[col_train]
# Same thing but for the test set

y_test = pd.DataFrame(y_test, columns = [LABEL])

testing_set = pd.DataFrame(x_test, columns = FEATURES).merge(y_test, left_index = True, right_index = True)

testing_set.head()
import numpy as np

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor



seed = 7

np.random.seed(seed)



# Model

model = Sequential()

model.add(Dense(200, input_dim=79, kernel_initializer='normal', activation='relu'))

model.add(Dense(100, kernel_initializer='normal', activation='relu'))

model.add(Dense(50, kernel_initializer='normal', activation='relu'))

model.add(Dense(25, kernel_initializer='normal', activation='relu'))

model.add(Dense(1, kernel_initializer='normal'))

# Compile model

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam())



feature_cols = training_set[FEATURES]

labels = training_set[LABEL].values



model.fit(np.array(feature_cols), np.array(labels), epochs=100, batch_size=10)
# Evaluation on the test set created by train_test_split

model.evaluate(np.array(feature_cols), np.array(labels))
# Predictions

feature_cols_test = testing_set[FEATURES]

labels_test = testing_set[LABEL].values



y = model.predict(np.array(feature_cols_test))

# predictions = list(itertools.islice(y, testing_set.shape[0]))

predictions = prepro_label.inverse_transform(y).reshape(-1)
reality = pd.DataFrame(prepro_train.inverse_transform(testing_set), columns = [COLUMNS])
reality = reality.loc[:, ['SalePrice']].values.reshape(-1)
import matplotlib



matplotlib.rc('xtick', labelsize=30) 

matplotlib.rc('ytick', labelsize=30) 



fig, ax = plt.subplots(figsize=(50, 40))



plt.style.use('ggplot')

plt.plot(predictions, reality, 'ro')

plt.xlabel('Predictions', fontsize = 30)

plt.ylabel('Reality', fontsize = 30)

plt.title('Predictions x Reality on dataset Test', fontsize = 30)

ax.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)

plt.show()
y_predict = model(np.array(test))



def to_submit(pred_y,name_out):

    y_predict = prepro_label.inverse_transform(np.array(pred_y).reshape(-1,1))

    y_predict = pd.DataFrame(y_predict, columns = ['SalePrice'])

    testId = pd.DataFrame(ID.values, columns= ['Id'])

    y_predict = testId.join(y_predict)

    y_predict.to_csv(name_out + '.csv',index=False)

    

to_submit(y_predict, "submission_continuous")