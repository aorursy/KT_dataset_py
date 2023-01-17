import pandas as pd

import numpy as np

import zipfile as zf

from matplotlib import pyplot as plt
train_archive = zf.ZipFile('/kaggle/input/whats-cooking/train.json.zip', 'r')

test_archive = zf.ZipFile('/kaggle/input/whats-cooking/test.json.zip', 'r')



train_json = train_archive.read('train.json')

test_json = test_archive.read('test.json')



train = pd.read_json(train_json)

test = pd.read_json(test_json)



train_data = pd.DataFrame(train)

test_data = pd.DataFrame(test)



train_data = train_data.explode('ingredients')

test_data = test_data.explode('ingredients')
train_data
train_data = train_data.loc[train_data['ingredients'].isin(test_data['ingredients'].values)]

test_data = test_data.loc[test_data['ingredients'].isin(train_data['ingredients'].values)]
#t = train_data.groupby(['ingredients','cuisine']).size().to_frame('count')

#t = t.sort_values(by=['id']).where(t['id']==1).head(100)

#t = t.reset_index()

#t.pivot(index='ingredients', columns='cuisine', values='count')
train_data.insert(0,'bit_sign',1)
train_data = pd.pivot_table(train_data, values='bit_sign', index=['id','cuisine'],

                            columns=['ingredients']) #, aggfunc=np.sum)

train_data = train_data.reset_index()

train_data = train_data.fillna(0)
train_data
train_X = train_data.drop(['cuisine','id'], axis=1).values

train_X
train_X.shape
train_data['cuisine'] = train_data['cuisine'].map(

    { 

        'greek':0, 

        'southern_us':1, 

        'filipino':2, 

        'indian':3, 

        'jamaican':4,

        'spanish':5, 

        'italian':6, 

        'mexican':7, 

        'chinese':8, 

        'british':9, 

        'thai':10,

        'vietnamese':11, 

        'cajun_creole':12, 

        'brazilian':13, 

        'french':14, 

        'japanese':15,

        'irish':16, 

        'korean':17, 

        'moroccan':18, 

        'russian':19

    }

)



train_Y = train_data['cuisine'].values

train_Y
import tensorflow.keras as tf



model = tf.models.Sequential()





model.add(tf.layers.Dense(64, activation='relu'))

#model.add(tf.layers.Dropout(0.2))

#model.add(tf.layers.Dense(1024, activation='relu'))

#model.add(tf.layers.Dropout(0.5))

#model.add(tf.layers.Dense(256, activation='relu'))

#model.add(tf.layers.Dense(512, activation='relu'))

#model.add(tf.layers.Dropout(0.5))

#model.add(tf.layers.Dense(1024, activation='relu'))

#model.add(tf.layers.Dense(512, activation='relu'))

#model.add(tf.layers.BatchNormalization())

#model.add(tf.layers.Dropout(0.5))





model.add(tf.layers.Dense(20, activation='softmax'))

model.compile(optimizer='adam'

              ,loss='sparse_categorical_crossentropy'

              ,metrics=['accuracy']

             )



#model.summary()



res = model.fit(train_X, train_Y, epochs=5, validation_split=0.2)
test_data
test_data.insert(0,'bit_sign',1)
test_data = pd.pivot_table(test_data, values='bit_sign', index=['id'],

                            columns=['ingredients'])

test_data = test_data.reset_index()

test_data = test_data.fillna(0)

test_data
test_X = test_data.drop(['id'], axis=1).values

test_X.shape
prediction = np.argmax(model.predict(test_X), axis=1)
prediction = prediction.reshape(9944,1)
res_set = pd.DataFrame.from_records(prediction)

res_set.insert(0,'id',test_data['id'])

res_set.columns = ['id','cuisine']





res_set['cuisine'] = res_set['cuisine'].map(

    { 

        0:'greek', 

        1:'southern_us', 

        2:'filipino', 

        3:'indian', 

        4:'jamaican',

        5:'spanish', 

        6:'italian', 

        7:'mexican', 

        8:'chinese', 

        9:'british', 

        10:'thai',

        11:'vietnamese', 

        12:'cajun_creole', 

        13:'brazilian', 

        14:'french', 

        15:'japanese',

        16:'irish', 

        17:'korean', 

        18:'moroccan', 

        19:'russian'

    }

)







#res_set

res_set.to_csv('submission.csv', index=False)