import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from keras import backend as K

import tensorflow as tf

from category_encoders.binary import BinaryEncoder

import category_encoders as ce

import os

# adapted from https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
K.tensorflow_backend._get_available_gpus()
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 

sess = tf.Session(config=config) 

K.set_session(sess)
# fix random seed for reproducibility

seed = 42

np.random.seed(seed)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
print(os.listdir("../input"))
url = "../input/train_features.csv"

df = pd.read_csv(url, parse_dates=['date_recorded'],index_col='id' )
df['timestamp'] = df.date_recorded.apply(lambda d: d.timestamp()/ 10 ** 9)

df_ = df.drop(['date_recorded', 'region_code', 'district_code', 'region'], inplace = False, axis=1)

# df_ = df.drop(['date_recorded', 'district_code', 'region'], inplace = False, axis=1)

df_['region_district'] = df.apply(lambda row: f'{row.region}_{row.district_code}' , axis=1)

# df_ = df_.apply(lambda x: x.astype(str).str.lower())

train_input_columns = list(df_.columns)

train_numeric_columns = df_.select_dtypes(exclude=['object']).columns

df_.head()
dc = 'status_group'
yurl = '../input/train_labels.csv'

dfy = pd.read_csv(yurl, index_col='id' )

dfy.shape

y = dfy[dc]
encoder = LabelEncoder()

encoder.fit(y)

encoded_Y = encoder.transform(y)

# convert integers to dummy variables (i.e. one hot encoded)

dummy_y = np_utils.to_categorical(encoded_Y)
oc = df_.select_dtypes(include=['object']).columns

oc
hot = []

binary = []

for o in oc:

    if df_[o].unique().shape[0] > 127:

        print(df_[o].unique().shape[0], o)

        binary.append(o)

    else:

        hot.append(o)
test_url = "../input/test_features.csv"

df = pd.read_csv(test_url, parse_dates=['date_recorded'],index_col='id' )

df['timestamp'] = df.date_recorded.apply(lambda d: d.timestamp()/ 10 ** 9)

dft = df.drop(['date_recorded', 'region_code', 'district_code', 'region'], inplace = False, axis=1)

#dft = df.drop(['date_recorded', 'district_code', 'region'], inplace = False, axis=1)

dft['region_district'] = df.apply(lambda row: f'{row.region}_{row.district_code}' , axis=1)

# dft = dft.apply(lambda x: x.astype(str).str.lower())

test_input_columns = list(dft.columns)

# dft[dc] = ['fuctional'] * dft.shape[0]

dft.head()
test__ = list(dft.columns)

for c in train_input_columns:

    if c not in test_input_columns:

        print(f'{c} not in test')

    else:

        test__.remove(c)

print(test__)

        
encoders = Pipeline([

#                 ('vect', tfidf),

                ('binary', BinaryEncoder(cols=binary)),

                ('onehot', ce.OneHotEncoder(use_cat_names=True,cols=hot))

            ])

df_l = df_.shape[0]

both = pd.concat([df_,dft])

print(df_l)

both.head()
both_  = encoders.fit_transform(both)
df__  = both_.iloc[0:df_l]

dft_ = both_.iloc[df_l:]

df_l, both_.shape, df__.shape, dft_.shape
def pump_baseline_model():

	# create model

	model = Sequential()

	model.add(Dense(8, input_dim=404, activation='relu'))

	model.add(Dense(3, activation='softmax'))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model
estimator = KerasClassifier(build_fn=pump_baseline_model, epochs=6000, batch_size=5, verbose=0)
# results = cross_val_score(estimator, df__, dummy_y, cv=kfold)

# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
model = pump_baseline_model()

model.fit(x=df__, y=dummy_y,epochs=6) # tried 6000, result were disappionting 77% train 75% test 
p = model.predict(dft_, batch_size=None, verbose=0, steps=None)

p
p.shape
with open('testxgb.keras.csv', 'w') as f:

    f.write('id,status_group\n')

    for f_,i in zip(p, dft_.index):

        index = np.argmax(f_)

        d = 'non functional' if index == 2 else ('functional' if index == 0 else 'functional needs repair')

        f.write(f"{i},{d}")

        f.write('\n')