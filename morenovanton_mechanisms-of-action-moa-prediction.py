# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import skew

import itertools



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import FeatureUnion



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#sns.set(rc={'figure.figsize': (20, 5)})
test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

sample_submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')
test_features.head()
train_features.head()
print(train_features.shape[0])

print(test_features.shape[0])



print(train_features.shape[1])

print(test_features.shape[1])



train_features.shape[0] / test_features.shape[0]
missing_vals_train  = train_features.isnull().sum() / train_features.shape[0]

missing_vals_train[missing_vals_train > 0].sort_values(ascending=False) 
missing_vals_test  = test_features.isnull().sum() / test_features.shape[0]

missing_vals_test[missing_vals_test > 0].sort_values(ascending=False) 
train_features.info()
test_features.info()
train_features[['cp_time']] = train_features[['cp_time']].astype('object')

test_features[['cp_time']] = test_features[['cp_time']].astype('object')
train_features_object = train_features.select_dtypes(include = ['object'])

#train_features_object.loc[:, ('cp_time')] = train_features.cp_time



test_features_object = test_features.select_dtypes(include = ['object'])

#test_features_object.loc[:, ('cp_time')] = test_features.cp_time
print(train_features_object.shape)

train_features_object.head()
train_features_object_group_cp_type = train_features_object.groupby('cp_type').aggregate({'sig_id': 'count'}).reset_index()

train_features_object_group_cp_type['train/test'] = ['train', 'train']



test_features_object_group_cp_type = test_features_object.groupby('cp_type').aggregate({'sig_id': 'count'}).reset_index()

test_features_object_group_cp_type['train/test'] = ['test', 'test']



group_cp_type = pd.concat([train_features_object_group_cp_type, test_features_object_group_cp_type])

group_cp_type.head()
fig = px.bar(group_cp_type, x="cp_type", y="sig_id", color="train/test", title="cp_type")

fig.show()
train_features_object_group_cp_dose = train_features_object.groupby('cp_dose').aggregate({'sig_id': 'count'}).reset_index()

train_features_object_group_cp_dose['train/test'] = ['test', 'test']



test_features_object_group_cp_dose = test_features_object.groupby('cp_dose').aggregate({'sig_id': 'count'}).reset_index()

test_features_object_group_cp_dose['train/test'] = ['train', 'train']



group_cp_dose = pd.concat([train_features_object_group_cp_dose, test_features_object_group_cp_dose])

group_cp_dose.head()
fig = px.bar(group_cp_dose, x="cp_dose", y="sig_id", color="train/test", title="cp_dose")

fig.show()
train_features_object_group_cp_time = train_features_object.groupby('cp_time').aggregate({'sig_id': 'count'}).reset_index()

train_features_object_group_cp_time['train/test'] = ['train', 'train', 'train']



test_features_object_group_cp_time = test_features_object.groupby('cp_time').aggregate({'sig_id': 'count'}).reset_index()

test_features_object_group_cp_time['train/test'] = ['test', 'test', 'test']



group_cp_time = pd.concat([train_features_object_group_cp_time, test_features_object_group_cp_time])

group_cp_time.head()
fig = px.bar(group_cp_time, x="cp_time", y="sig_id", color="train/test", title="cp_time")

fig.show()
train_features_number = train_features.select_dtypes(include = ['float64', 'int64'])

#del train_features_number['cp_time']

test_features_number = test_features.select_dtypes(include = ['float64', 'int64'])

#del test_features_number['cp_time']
train_features_random = train_features_number[train_features_number.columns[

    np.random.randint(0, train_features_number.shape[1], 10)]]



train_features_random.hist(bins=40, figsize=(20,15))

plt.show()
skewed_feats = train_features_number.apply(lambda x: skew(x)) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats_index = skewed_feats.index



n_skewed = np.random.randint(0, skewed_feats_index.shape, 10)

random_g_skewed_feats = train_features_number[skewed_feats_index[n_skewed]]
random_g_skewed_feats.head()
colu = [[columns] * len(random_g_skewed_feats.iloc[:,ind]) for ind, columns in enumerate(random_g_skewed_feats.columns)]



values_random_g_skewed_feats_sp_g = []



for columns_split_g in range(random_g_skewed_feats.shape[1]):

    values_random_g_skewed_feats_sp_g.append(random_g_skewed_feats.iloc[:,columns_split_g])

    

d = {'g-': list(itertools.chain(*values_random_g_skewed_feats_sp_g)), 'indax_g': list(itertools.chain(*colu))}

df = pd.DataFrame(data=d)
df.head()
del train_features_object['sig_id']

cat_attr = train_features_object.columns

num_attr = np.array(train_features_number.columns)   

num_attr = np.delete(num_attr, np.argmax(num_attr == np.array(skewed_feats_index)[:, np.newaxis], axis=1))

len(num_attr)
len(num_attr) + len(cat_attr) + len(skewed_feats_index)
train_features.shape
# Create a class to select numerical or categorical columns 

class OldDataFrameSelector(BaseEstimator, TransformerMixin):

    

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        return X[self.attribute_names].values
# Create a class to skewness numerical of a data set

class Skewness_numericalSelector(BaseEstimator, TransformerMixin):

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        #min_nonzero = np.min(X[np.nonzero(X)])

        #X[X == 0] = min_nonzero

        #X_nonzero = np.where(X > 0.0000000001)

        #return np.log10p(X)

        return np.exp(X)
old_num_pipeline = Pipeline([

        ('selector', OldDataFrameSelector(num_attr)),

        ("scaler", StandardScaler())

    ])



old_cat_pipeline = Pipeline([

        ('selector', OldDataFrameSelector(cat_attr)),

        ('cat_encoder', OneHotEncoder(sparse=False)),

    ])



skew_num_pipeline = Pipeline([

        ('selector', OldDataFrameSelector(skewed_feats_index)),

        ('skew_scaler', Skewness_numericalSelector())

    ])



old_full_pipeline = FeatureUnion(transformer_list=[

        ("cat_pipeline", old_cat_pipeline),

        ("num_pipeline", old_num_pipeline),

        ("skew_pipeline", skew_num_pipeline)

    ])

skewed_feats_index = ['skewed_'+feats_index for feats_index in skewed_feats_index]

cat_dummies_attr = ['cp_type_ctl_vehicle', 'cp_type_trt_cp', 'cp_dose_D1', 'cp_dose_D2', 'cp_time_24', 'cp_time_48', 'cp_time_72']

columns_train_features = cat_dummies_attr + list(num_attr) + skewed_feats_index
len(columns_train_features)
del train_features['sig_id']

del test_features['sig_id']
train_features = pd.DataFrame(old_full_pipeline.fit_transform(train_features), columns=columns_train_features)

test_features = pd.DataFrame(old_full_pipeline.fit_transform(test_features), columns=columns_train_features)
print(train_features.shape)

print(test_features.shape)
import tensorflow as tf

from keras.models import Sequential

from keras.layers import BatchNormalization

from keras.layers.core import Dense, Flatten, Dropout, Lambda
del train_targets_scored['sig_id']
from sklearn.model_selection import train_test_split

#train_features = np.array(train_features.values)[:,1:]

#train_targets_scored = np.array(train_targets_scored.values)[:,1:]



X_train, y_train, X_test, y_test = train_test_split(train_features, train_targets_scored, test_size=0.3, random_state=42)
def ret(a):

    return  a 
879 *2
model= Sequential()



model.add(Lambda(ret, input_shape = [879]))



model.add(Dense(2637, activation = 'relu'))

model.add(BatchNormalization())



model.add(Dense(1758, activation = 'relu'))

model.add(BatchNormalization())



model.add(Dense(1200, activation = 'relu'))

model.add(BatchNormalization())



model.add(Dense(1000, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))



model.add(Dense(800, activation = 'relu'))

model.add(BatchNormalization())



model.add(Dense(400, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))



model.add(Dense(206, activation = 'sigmoid'))



model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = tf.keras.optimizers.Adam())
model_fit = model.fit(X_train, X_test, validation_data=(y_train, y_test), epochs=14) 
predictions = model.predict(test_features)
predictions
sample_submission.iloc[:,1:] = predictions
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False, header=True) 