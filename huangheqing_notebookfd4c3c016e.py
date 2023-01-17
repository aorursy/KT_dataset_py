# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split
plt.figure(figsize=(20,20))  # 创建一个20 * 20 英寸的图像

plt.subplots_adjust(bottom=.05,top=.9,left=.05,right=.95)

 

plt.subplot(421)

plt.title("One informative feature, one cluster per class",fontsize='small')

X1,Y1= make_classification(n_samples=1000,n_features=2,n_redundant=0,n_informative=1,n_clusters_per_class=1)

plt.scatter(X1[:,0],X1[:,1],marker='o',c=Y1)

 

plt.subplot(422)

plt.title("Two informative features, one cluster per class", fontsize='small')

X2,Y2 = make_classification(n_samples=1000,n_features=2,n_redundant=0,n_informative=2)

plt.scatter(X2[:,0],X2[:,1],marker='o',c=Y2)

tf.compat.v1.disable_eager_execution()
def init_weights(shape):

    return tf.Variable(tf.compat.v1.random_normal(shape, stddev=0.01))
def model(X, w):

    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy
X =  tf.compat.v1.placeholder(tf.float32, [None, 2]) # create symbolic variables

Y =  tf.compat.v1.placeholder(tf.float32, [None])
w = init_weights([2, 1]) # like in logits regression, we need a shared variable weight matrix for logistic regression
py_x = model(X, w)

py_x = tf.squeeze(py_x,axis=1)
py_x
cost = tf.reduce_mean(

    tf.nn.sigmoid_cross_entropy_with_logits(logits=py_x, labels=Y)

) # compute mean cross entropy (softmax is applied internally)
train_op = tf.compat.v1.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
one = tf.ones_like(py_x)
zero = tf.zeros_like(py_x)
predict_op =  tf.where(py_x < 0.5, x=zero, y=one) # at predict time, evaluate the argmax of the logistic regression
X1,Y1= make_classification(n_samples=1000,n_features=2,n_redundant=0,n_informative=1,n_clusters_per_class=1)

X1
Y1
plt.scatter(X1[:,0],X1[:,1],marker='o',c=Y1)

 
trX, teX, trY, teY = train_test_split(X1,Y1,test_size=0.2,random_state=1)
trX.shape, teX.shape, trY.shape, teY.shape
# Launch the graph in a session

with tf.compat.v1.Session() as sess:

    # you need to initialize all variables

    tf.compat.v1.global_variables_initializer().run()



    for i in range(100):

        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):

            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        print(i, np.mean(np.where(teY < 0.5, 0, 1)  ==

                         sess.run(predict_op, feed_dict={X: teX})))
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!head -n 5 /kaggle/input/riiid-test-answer-prediction/questions.csv
import pandas as pd

import numpy as np

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

import tensorflow as tf

from sklearn.model_selection import train_test_split
dir_path = '/kaggle/input/riiid-test-answer-prediction/'

file_train = 'train.csv'

file_questions = 'questions.csv'
nrows =  100 * 10000

# nrows = None

train = pd.read_csv(

                    dir_path + file_train, 

                    nrows=nrows, 

                    usecols=['row_id', 'timestamp', 'user_id', 'content_id', 

                             'content_type_id', 'task_container_id', 'answered_correctly',

                            'prior_question_elapsed_time','prior_question_had_explanation'],

                    dtype={

                            'row_id': 'int64',

                            'timestamp': 'int64',

                            'user_id': 'int32',

                            'content_id': 'int16',

                            'content_type_id': 'int8',

                            'task_container_id': 'int8',

                            'answered_correctly': 'int8',

                            'prior_question_elapsed_time': 'float32',

                            'prior_question_had_explanation': 'str'

                        }

                   )

questions = pd.read_csv(

                        dir_path + file_questions, 

                        nrows=nrows,

                        usecols=['question_id','bundle_id','part'], 

                        dtype={

                           'question_id': 'int16',

                           'bundle_id': 'int16',

                           'part': 'int8',

                       }

                    )

train['prior_question_had_explanation'] = train['prior_question_had_explanation'].map({'True':1,'False':0}).fillna(-1).astype(np.int8)

train = train[train['content_type_id']==0]
import gc
gc.collect()
# 压缩内存

max_num = 100

train = train.groupby(['user_id']).tail(max_num)
train = pd.merge(

        left=train,

        right=questions,

        how='left',

        left_on='content_id',

        right_on='question_id'

        )
train
train = train.fillna(0)
train
class cat_deal:

    def __init__(self):

        self.max_len = 0

        self.dict_map = {}

    

    def fit(self, cat_list):

        index = 1 

        for cat_i in cat_list:

            if cat_i not in self.dict_map:

                self.dict_map[cat_i] = index

                index += 1

        self.max_len = index + 1

        

    def transform(self, cat_list):

        cat_transform_list = []

        for cat_i in cat_list:

            if cat_i in self.dict_map:

                cat_transform_list.append(self.dict_map[cat_i])

            else:

                cat_transform_list.append(0)

        return cat_transform_list
class float_deal:

    def __init__(self):

        self.max = 0

        self.min = 0

        self.max_min = 0 

        

    def fit(self, float_list):

        for float_i in float_list:

            if float_i < self.min:

                self.min = float_i

            if float_i > self.max:

                self.max = float_i

        self.max_min = self.max - self.min

        

    def transform(self, float_list):

        float_transform_list = []

        for float_i in float_list:

            if float_i < self.min:

                float_transform_list.append(0)

            elif float_i > self.max:

                float_transform_list.append(1)

            else:

                float_transform_list.append(float_i/self.max_min)

        return float_transform_list
dict_cat_class = {}

for columns in ['user_id','content_id',\

                'task_container_id','prior_question_had_explanation',\

                'bundle_id','part']:

    dict_cat_class[columns] = cat_deal()

    dict_cat_class[columns].fit(train[columns])



    train[columns] = dict_cat_class[columns].transform(train[columns])

    print(columns)
dict_float_class = {}

for columns in ['timestamp','prior_question_elapsed_time']:

    dict_float_class[columns] = float_deal()

    dict_float_class[columns].fit(train[columns])

    

    train[columns] = dict_float_class[columns].transform(train[columns])

    print(columns)
def squeeze(embedding):

    embedding = tf.squeeze(embedding,axis=1)

    return embedding

def concat(embedding_list):

    embedding = tf.concat(embedding_list, axis=1)

    return embedding

def multiply(multi_x_y):

    multi_x = multi_x_y[0]

    multi_y = multi_x_y[1]

    multi_x_y = tf.multiply(multi_x, multi_y)

    return multi_x_y
input_timestamp = tf.keras.Input(shape=(1,))

input_prior_question_elapsed_time = tf.keras.Input(shape=(1,))



# input int

input_user = tf.keras.Input(shape=(1,))

input_content = tf.keras.Input(shape=(1,))

input_task_container = tf.keras.Input(shape=(1,))

input_prior_question_had_explanation = tf.keras.Input(shape=(1,))

input_bundle = tf.keras.Input(shape=(1,))

input_part = tf.keras.Input(shape=(1,))



inputs = [input_timestamp,input_prior_question_elapsed_time,\

         input_user,input_content,\

         input_task_container,input_prior_question_had_explanation,\

         input_bundle,input_part]

# inputs = tf.keras.layers.Lambda(concat)(inputs)



# input session

# input_tags = Input(shape=(1))



# embedding float

embedding_timestamp = tf.keras.layers.Dense(64, activation=tf.nn.sigmoid)(input_timestamp)

embedding_prior_question_elapsed_time = tf.keras.layers.Dense(64, activation=tf.nn.sigmoid)(input_prior_question_elapsed_time)



# embedding int 

embedding_user = tf.keras.layers.Embedding(dict_cat_class['user_id'].max_len,

                                           64, input_length=1)(input_user)

embedding_user = tf.keras.layers.Lambda(squeeze)(embedding_user)



embedding_content = tf.keras.layers.Embedding(dict_cat_class['content_id'].max_len,

                                              64, input_length=1)(input_content)

embedding_content = tf.keras.layers.Lambda(squeeze)(embedding_content)



embedding_task_container = tf.keras.layers.Embedding(dict_cat_class['task_container_id'].max_len,

                                                     64, input_length=1)(input_task_container)

embedding_task_container = tf.keras.layers.Lambda(squeeze)(embedding_task_container)



embedding_prior_question_had_explanation = tf.keras.layers.Embedding(dict_cat_class['prior_question_had_explanation'].max_len, 

                                                                     64, input_length=1)(input_prior_question_had_explanation)

embedding_prior_question_had_explanation = tf.keras.layers.Lambda(squeeze)(embedding_prior_question_had_explanation)



embedding_bundle = tf.keras.layers.Embedding(dict_cat_class['bundle_id'].max_len,

                                             64, input_length=1)(input_bundle)

embedding_bundle = tf.keras.layers.Lambda(squeeze)(embedding_bundle)



embedding_part = tf.keras.layers.Embedding(dict_cat_class['part'].max_len,

                                           64, input_length=1)(input_part)

embedding_part = tf.keras.layers.Lambda(squeeze)(embedding_part)



embedding_all = [embedding_timestamp,embedding_prior_question_elapsed_time,\

                embedding_user, embedding_content, embedding_task_container,\

                embedding_prior_question_had_explanation, embedding_bundle, embedding_part]





nffm1, nffm2 = [], []

for i, embedding_i in enumerate(embedding_all):

    for j, embedding_j in enumerate(embedding_all):

        if i > j:

            nffm1.append(embedding_i), nffm2.append(embedding_j)

nffm1_layer = tf.keras.layers.Lambda(concat)(nffm1)

nffm2_layer = tf.keras.layers.Lambda(concat)(nffm2)     



nffm_all = tf.keras.layers.Lambda(multiply)([nffm1_layer,nffm2_layer])

    

logit = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(nffm_all)





model = tf.keras.models.Model(inputs=inputs, outputs=logit)



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['binary_crossentropy'])
nffm1, nffm2 = [], []

for i, embedding_i in enumerate(embedding_all):

    for j, embedding_j in enumerate(embedding_all):

        if i > j:

            nffm1.append(embedding_i), nffm2.append(embedding_j)

nffm1_layer = tf.keras.layers.Lambda(concat)(nffm1)

nffm2_layer = tf.keras.layers.Lambda(concat)(nffm2) 



nffm_all = tf.keras.layers.Lambda(multiply)([nffm1_layer,nffm2_layer])
nffm1
nffm2
nffm1, nffm2
nffm1_layer
model.summary()
plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',

                            verbose=0,

                            mode='min',

                            factor=0.1,

                            patience=6)



early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',

                               verbose=0,

                               mode='min',

                               patience=10)



# 保存

checkpoint = tf.keras.callbacks.ModelCheckpoint(f'fold.h5',

                             monitor='val_loss',

                             verbose=0,

                             mode='min',

                             save_best_only=True)
valid = pd.DataFrame()

for i in range(6):

    

    # 获取训练标签数据

    last_records = train.drop_duplicates('user_id', keep='last')

    

    # 获取训练标签以前的数据

    map__last_records__user_row = dict(zip(last_records['user_id'],last_records['row_id']))

    train['filter_row'] = train['user_id'].map(map__last_records__user_row)

    train = train[train['row_id']<train['filter_row']]



    # 特征加入训练集

    valid = valid.append(last_records)

    print(len(valid))
features_columns = ['timestamp','prior_question_elapsed_time',\

                    'user_id','content_id',\

                    'task_container_id','prior_question_had_explanation',\

                    'bundle_id','part']



X_test, y_test = [valid[columns].values for columns in features_columns], valid['answered_correctly'].values

# del valid



X_train, y_train = [train[columns].values for columns in features_columns], train['answered_correctly'].values

# del train
print('ok')
model.fit(X_train, y_train,

          epochs=10,

          batch_size=512 * 500 * 2,

          verbose=1,

          shuffle=True,

          validation_data=(X_test, y_test),

          callbacks=[plateau, early_stopping, checkpoint])



y_test_proba = model.predict(X_test, verbose=0, batch_size=512)

auc = roc_auc_score(y_test, y_test_proba)

print(auc)

y_test_proba = model.predict(X_test, verbose=0, batch_size=512)

auc = roc_auc_score(y_test, y_test_proba)

print(auc)
model.fit(X_train, y_train,

          epochs=1,

          batch_size=512 * 500,

          verbose=1,

          shuffle=True,

          validation_data=(X_test, y_test),

          callbacks=[plateau, early_stopping, checkpoint])



y_test_proba = model.predict(X_test, verbose=0, batch_size=512)

auc = roc_auc_score(y_test, y_test_proba)

print(auc)
y_test_proba = model.predict(X_test, verbose=0, batch_size=512)

auc = roc_auc_score(y_test, y_test_proba)

print(auc)
model.fit(X_train, y_train,

          epochs=1,

          batch_size=512 * 100,

          verbose=1,

          shuffle=True,

          validation_data=(X_test, y_test),

          callbacks=[plateau, early_stopping, checkpoint])



y_test_proba = model.predict(X_test, verbose=0, batch_size=512)

auc = roc_auc_score(y_test, y_test_proba)

print(auc)
model.fit(X_train, y_train,

          epochs=1,

          batch_size=512 * 500,

          verbose=1,

          shuffle=True,

          validation_data=(X_test, y_test),

          callbacks=[plateau, early_stopping, checkpoint])



y_test_proba = model.predict(X_test, verbose=0, batch_size=512)

auc = roc_auc_score(y_test, y_test_proba)

print(auc)
model.fit(X_train, y_train,

          epochs=2,

          batch_size=512 * 500,

          verbose=1,

          shuffle=True,

          validation_data=(X_test, y_test),

          callbacks=[plateau, early_stopping, checkpoint])



y_test_proba = model.predict(X_test, verbose=0, batch_size=512)

auc = roc_auc_score(y_test, y_test_proba)

print(auc)
model.fit(X_train, y_train,

          epochs=2,

          batch_size=512 * 500,

          verbose=1,

          shuffle=True,

          validation_data=(X_test, y_test),

          callbacks=[plateau, early_stopping, checkpoint])



y_test_proba = model.predict(X_test, verbose=0, batch_size=512)

auc = roc_auc_score(y_test, y_test_proba)

print(auc)
model.fit(X_train, y_train,

          epochs=1,

          batch_size=512 ,

          verbose=1,

          shuffle=True,

          validation_data=(X_test, y_test),

          callbacks=[plateau, early_stopping, checkpoint])



y_test_proba = model.predict(X_test, verbose=0, batch_size=512)

auc = roc_auc_score(y_test, y_test_proba)

print(auc)
import riiideducation

env = riiideducation.make_env()


iter_test = env.iter_test()



for (test_df, sample_prediction_df) in iter_test:



    test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].map({'True':1,'False':0}).fillna(-1).astype(np.int8)



    test_df = pd.merge(

        left=test_df,

        right=questions,

        how='left',

        left_on='content_id',

        right_on='question_id'

        )



    test_df = test_df.fillna(0)





    for columns in ['user_id','content_id',\

                    'task_container_id','prior_question_had_explanation',\

                    'bundle_id','part']:



        test_df[columns] = dict_cat_class[columns].transform(test_df[columns])

        print(columns)





    for columns in ['timestamp','prior_question_elapsed_time']:



        test_df[columns] = dict_float_class[columns].transform(test_df[columns])

        print(columns)



    X_test = [test_df[columns].values for columns in features_columns]



    test_df['answered_correctly'] =  model.predict(X_test, verbose=0, batch_size=512)

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

train.head()