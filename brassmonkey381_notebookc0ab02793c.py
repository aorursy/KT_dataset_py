#Neural Net using binary data only



print("Initialize libraries")

import pandas as pd

import sys

import numpy as np

import scipy as sp

import matplotlib.pyplot as plt

from sklearn.cross_validation import StratifiedKFold, KFold

from sklearn.metrics import log_loss

from sklearn.cluster import DBSCAN

from sklearn import metrics as skmetrics

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from collections import Counter

from keras.layers.advanced_activations import PReLU

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn import ensemble

from sklearn.decomposition import PCA

from sklearn.preprocessing import OneHotEncoder

import os

import gc

from scipy import sparse

from sklearn.cross_validation import train_test_split, cross_val_score

from sklearn.feature_selection import SelectPercentile, f_classif, chi2, SelectKBest

from sklearn import ensemble

from sklearn.neighbors import KNeighborsClassifier

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from keras.optimizers import SGD



from sklearn.cross_validation import cross_val_score

from sklearn.cross_validation import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn.cross_validation import train_test_split

from sklearn.metrics import log_loss
#------------------------------------------------ Read data from source files ------------------------------------

seed = 700

np.random.seed(seed)

datadir = '../input'



print("### ----- PART 1 ----- ###")



print("# Read people")

people = pd.read_csv(os.path.join(datadir,'people.csv'), dtype={'char_10' : np.int,

                                                               'char_11' : np.int,

                                                               'char_12' : np.int,

                                                               'char_13' : np.int,

                                                               'char_14' : np.int,

                                                               'char_15' : np.int,

                                                               'char_16' : np.int,

                                                               'char_17' : np.int,

                                                               'char_18' : np.int,

                                                               'char_19' : np.int,

                                                               'char_20' : np.int,

                                                               'char_21' : np.int,

                                                               'char_22' : np.int,

                                                               'char_23' : np.int,

                                                               'char_24' : np.int,

                                                               'char_25' : np.int,

                                                               'char_26' : np.int,

                                                               'char_27' : np.int,

                                                               'char_28' : np.int,

                                                               'char_29' : np.int,

                                                               'char_30' : np.int,

                                                               'char_31' : np.int,

                                                               'char_32' : np.int,

                                                               'char_33' : np.int,

                                                               'char_33' : np.int,

                                                               'char_34' : np.int,

                                                               'char_35' : np.int,

                                                               'char_36' : np.int,

                                                               'char_37' : np.int})

people['date'] = pd.to_datetime(people['date'])

people['date_increment'] = people['date'] - people['date'].min()

people = people.sort_values('date_increment')

print("reduce dimensions")

people.drop('char_2', axis=1, inplace=True) #duplicate of char_1

# rename people columns

people.rename(columns = {'char_1':'ppl_char_1',

                        'char_3':'ppl_char_3',

                        'char_4':'ppl_char_4',

                        'char_5':'ppl_char_5',

                        'char_6':'ppl_char_6',

                        'char_7':'ppl_char_7',

                        'char_8':'ppl_char_8',

                        'char_9':'ppl_char_9',

                        'char_10':'ppl_char_10',

                        'date':'ppl_date',

                        'date_increment': 'ppl_date_increment'}, inplace = True)

ppl_table_logi = people['group_1'].value_counts() == 1

people.group_1[people.group_1.isin(ppl_table_logi[ppl_table_logi == 1].index)] = 'group unique'

#people.head(5)
print("Read Train")

train = pd.read_csv(os.path.join(datadir,'act_train.csv'), dtype={'char_1' : np.str,

                                                               'char_2' : np.str,

                                                               'char_3' : np.str,

                                                               'char_4' : np.str,

                                                               'char_5' : np.str,

                                                               'char_6' : np.str,

                                                               'char_7' : np.str,

                                                               'char_8' : np.str,

                                                               'char_9' : np.str,

                                                               'char_10' : np.str

})

train['date'] = pd.to_datetime(train['date'])

train['date_increment'] = train['date'] - train['date'].min()

#train.head(5)
print("Read Test")

test = pd.read_csv(os.path.join(datadir,'act_test.csv'), dtype={'char_1' : np.str,

                                                               'char_2' : np.str,

                                                               'char_3' : np.str,

                                                               'char_4' : np.str,

                                                               'char_5' : np.str,

                                                               'char_6' : np.str,

                                                               'char_7' : np.str,

                                                               'char_8' : np.str,

                                                               'char_9' : np.str,

                                                               'char_10' : np.str

})

test['date'] = pd.to_datetime(test['date'])

test['date_increment'] = test['date'] - test['date'].min()

#test.head(5)
print("reduce dimensions of train and test char_10")

uni_char_10 = train[['people_id', 'char_10']]

uni_char_10 = uni_char_10.append(test[["people_id", "char_10"]])

x = uni_char_10.groupby('char_10').people_id.nunique()

uni_char_10 = x.index[x == 1]

train.char_10[train.char_10.isin(uni_char_10)] = 'type unique'

test.char_10[test.char_10.isin(uni_char_10)] = 'type unique'
split_len = len(train)



# Group Labels

Y = train["outcome"]

label_outcome = LabelEncoder()

Y = label_outcome.fit_transform(Y)

activity_id = test["activity_id"]



# Merge train and test
#train = pd.merge(train, people, on = 'people_id',how='left')

#train.drop('outcome', axis=1, inplace=True)

Df = pd.concat([train, test])
Df = pd.merge(Df, people, on = 'people_id', how = 'left')
Df.fillna("type 0", inplace=True)

#Df.head()
#change date_increment to numeric

Df['ppl_date_increment'] = Df.ppl_date_increment.astype(int) / 86400000000000

Df['date_increment'] = Df.date_increment.astype(int) / 86400000000000
not_categorical=[]

categorical=['activity_category',

             #'group_1',

             'char_1',

             'char_2',

             'char_3',

             'char_4',

             'char_5',

             'char_6',

             'char_7',

             'char_8',

             'char_9',

             'char_10',

             'ppl_char_1',

             'ppl_char_3',

             'ppl_char_4',

             'ppl_char_5',

             'ppl_char_6',

             'ppl_char_7',

             'ppl_char_8',

             'ppl_char_9']



for category in Df.columns:

    if category not in categorical:

        not_categorical.append(category)

        

#not_categorical

#CHAR_10 STILL HAD !4600 different types!
Df_cats = Df[categorical]

Df_cats.ix[:,0] = LabelEncoder().fit_transform(Df_cats.ix[:,0])

Df_cats.ix[:,1] = LabelEncoder().fit_transform(Df_cats.ix[:,1])

Df_cats.ix[:,2] = LabelEncoder().fit_transform(Df_cats.ix[:,2])

Df_cats.ix[:,3] = LabelEncoder().fit_transform(Df_cats.ix[:,3])

Df_cats.ix[:,4] = LabelEncoder().fit_transform(Df_cats.ix[:,4])

Df_cats.ix[:,5] = LabelEncoder().fit_transform(Df_cats.ix[:,5])

Df_cats.ix[:,6] = LabelEncoder().fit_transform(Df_cats.ix[:,6])

Df_cats.ix[:,7] = LabelEncoder().fit_transform(Df_cats.ix[:,7])

Df_cats.ix[:,8] = LabelEncoder().fit_transform(Df_cats.ix[:,8])

Df_cats.ix[:,9] = LabelEncoder().fit_transform(Df_cats.ix[:,9])

Df_cats.ix[:,10] = LabelEncoder().fit_transform(Df_cats.ix[:,10])

Df_cats.ix[:,11] = LabelEncoder().fit_transform(Df_cats.ix[:,11])

Df_cats.ix[:,12] = LabelEncoder().fit_transform(Df_cats.ix[:,12])

Df_cats.ix[:,13] = LabelEncoder().fit_transform(Df_cats.ix[:,13])

Df_cats.ix[:,14] = LabelEncoder().fit_transform(Df_cats.ix[:,14])

Df_cats.ix[:,15] = LabelEncoder().fit_transform(Df_cats.ix[:,15])

Df_cats.ix[:,16] = LabelEncoder().fit_transform(Df_cats.ix[:,16])

Df_cats.ix[:,17] = LabelEncoder().fit_transform(Df_cats.ix[:,17])

Df_cats.ix[:,18] = LabelEncoder().fit_transform(Df_cats.ix[:,18])
Df_cats.info()
#Df_cats.ix[:,0] =  (Df_cats.columns[0] + Df_cats.ix[:,0]).astype('category')

dec = LabelEncoder().fit_transform(Df_cats["activity_category"])

dec
#Df_cats.drop('char_10', 1, inplace=True)

data = sp.sparse.csr_matrix(Df_cats["activity_category"])
device_ids = FLS["device_id"].unique()

feature_cs = FLS["feature"].unique()



data = np.ones(len(FLS))

len(data)



dec = LabelEncoder().fit(FLS["device_id"])

row = dec.transform(FLS["device_id"])

col = LabelEncoder().fit_transform(FLS["feature"])
Df_cats.shape[0]
data = np.ones(Df_cats.shape[0])

row = LabelEncoder().fit_transform(Df["activity_id"])

len(data)

row



Df["activity_category"].nunique()
sparse_matrix = sparse.csr_matrix(

    (data, (row, Df_cats["activity_category"])), shape=(len(row), Df["activity_category"].nunique()))

sparse_matrix.shape

sys.getsizeof(sparse_matrix)
sparse_matrix
###################

#  Concat Feature

###################



print("# Concat all features")



f1 = Df[["activity_id", "char_1"]]

f2 = Df[["activity_id", "char_2"]]

f3 = Df[["activity_id", "char_3"]]

f4 = Df[["activity_id", "char_4"]]

f5 = Df[["activity_id", "char_5"]]

f6 = Df[["activity_id", "char_6"]]

f7 = Df[["activity_id", "char_7"]]

f8 = Df[["activity_id", "char_8"]]

f9 = Df[["activity_id", "char_9"]]

f10 = Df[["activity_id", "char_10"]]

f11 = Df[["activity_id", "char_11"]]

f12 = Df[["activity_id", "char_12"]]

f13 = Df[["activity_id", "char_13"]]

f14 = Df[["activity_id", "char_14"]]

f15 = Df[["activity_id", "char_15"]]

f16 = Df[["activity_id", "char_16"]]

f17 = Df[["activity_id", "char_17"]]

f18 = Df[["activity_id", "char_18"]]

f19 = Df[["activity_id", "char_19"]]

f20 = Df[["activity_id", "char_20"]]

f21 = Df[["activity_id", "char_21"]]

f22 = Df[["activity_id", "char_22"]]

f23 = Df[["activity_id", "char_23"]]

f24 = Df[["activity_id", "char_24"]]

f25 = Df[["activity_id", "char_25"]]

f26 = Df[["activity_id", "char_26"]]

f27 = Df[["activity_id", "char_27"]]

f28 = Df[["activity_id", "char_28"]]

f29 = Df[["activity_id", "char_29"]]

f30 = Df[["activity_id", "char_30"]]

f31 = Df[["activity_id", "char_31"]]

f32 = Df[["activity_id", "char_32"]]

f33 = Df[["activity_id", "char_33"]]

f34 = Df[["activity_id", "char_34"]]

f35 = Df[["activity_id", "char_35"]]

f36 = Df[["activity_id", "char_36"]]

f37 = Df[["activity_id", "char_37"]]

f38 = Df[["activity_id", "char_38"]]

f39 = Df[["activity_id", "people_id"]]

f40 = Df[["activity_id", "activity_category"]]

f41 = Df[["activity_id", "ppl_char_1"]]

#f42 = Df[["activity_id", "ppl_char_2"]]

f42 = Df[["activity_id", "group_1"]]

f43 = Df[["activity_id", "ppl_char_3"]]

f44 = Df[["activity_id", "ppl_char_4"]]

f45 = Df[["activity_id", "ppl_char_5"]]

f46 = Df[["activity_id", "ppl_char_6"]]

f47 = Df[["activity_id", "ppl_char_7"]]

f48 = Df[["activity_id", "ppl_char_8"]]

f49 = Df[["activity_id", "ppl_char_9"]]

f50 = Df[["activity_id", "ppl_char_10"]]

f51 = Df[["activity_id", "date_increment"]]

f52 = Df[["activity_id", "ppl_date"]]

f53 = Df[["activity_id", "ppl_date_increment"]]



f1.columns.values[1] = "feature"

f2.columns.values[1] = "feature"

f3.columns.values[1] = "feature"

f4.columns.values[1] = "feature"

f5.columns.values[1] = "feature"

f6.columns.values[1] = "feature"

f7.columns.values[1] = "feature"

f8.columns.values[1] = "feature"

f9.columns.values[1] = "feature"

f10.columns.values[1] = "feature"

f11.columns.values[1] = "feature"

f12.columns.values[1] = "feature"

f13.columns.values[1] = "feature"

f14.columns.values[1] = "feature"

f15.columns.values[1] = "feature"

f16.columns.values[1] = "feature"

f17.columns.values[1] = "feature"

f18.columns.values[1] = "feature"

f19.columns.values[1] = "feature"

f20.columns.values[1] = "feature"

f21.columns.values[1] = "feature"

f22.columns.values[1] = "feature"

f23.columns.values[1] = "feature"

f24.columns.values[1] = "feature"

f25.columns.values[1] = "feature"

f26.columns.values[1] = "feature"

f27.columns.values[1] = "feature"

f28.columns.values[1] = "feature"

f29.columns.values[1] = "feature"

f30.columns.values[1] = "feature"

f31.columns.values[1] = "feature"

f32.columns.values[1] = "feature"

f33.columns.values[1] = "feature"

f34.columns.values[1] = "feature"

f35.columns.values[1] = "feature"

f36.columns.values[1] = "feature"

f37.columns.values[1] = "feature"

f38.columns.values[1] = "feature"

f39.columns.values[1] = "feature"

f40.columns.values[1] = "feature"

f41.columns.values[1] = "feature"

f42.columns.values[1] = "feature"

f43.columns.values[1] = "feature"

f44.columns.values[1] = "feature"

f45.columns.values[1] = "feature"

f46.columns.values[1] = "feature"

f47.columns.values[1] = "feature"

f48.columns.values[1] = "feature"

f49.columns.values[1] = "feature"

f50.columns.values[1] = "feature"

f51.columns.values[1] = "feature"

f52.columns.values[1] = "feature"

f53.columns.values[1] = "feature"
FLS = pd.concat((f1, f2, f3, f4), axis=0, ignore_index=True)

FLS.info()
#FLS = pd.concat((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,

#                f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,

#                f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,

#                f31, f32, f33, f34, f35, f36, f37, f38, f39, f40,

#                f41, f42, f43, f44, f45, f46, f47, f48, f49, f50,

#                f51, f52, f53), axis=0, ignore_index=True)

#FLS.info()
FLS.head(5)
feature_cs = FLS["feature"].dropna()

activity_ids = Df["activity_id"].unique()
sparse_matrix = sparse.csr_matrix(

    (data, (row, col)), shape=(len(activity_ids), len(feature_cs)))

sparse_matrix.shape

sys.getsizeof(sparse_matrix)