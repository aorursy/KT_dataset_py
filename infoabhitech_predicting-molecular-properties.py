# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import math 

import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()
df_test = pd.read_csv('/kaggle/input/test.csv')

df_struct = pd.read_csv('/kaggle/input/structures.csv',dtype={'atom_index': np.int8,'x': np.float16,'y': np.float16,'z': np.float16})

df_train = pd.read_csv('/kaggle/input/train.csv',dtype={'id': np.int8,'atom_index0': np.int8,'atom_index1': np.int8,'scalar_coupling_constant': np.float16})
print("train datatype")

df_train.dtypes
print("structure datatype")

df_struct.dtypes
print("train length -", len(df_train))

print("test length -", len(df_test))

print("structure length -", len(df_struct))
print("train unique -",len(df_train['molecule_name'].unique()))

print("test unique -",len(df_test['molecule_name'].unique()))

print("structure unique -",len(df_struct['molecule_name'].unique()))
print("Scalar constant less than 0 -",len(df_train[df_train['scalar_coupling_constant'] < 0]))
print("train")

df_train.head(10)
print("struct")

df_struct.head()
print("test")

df_test.head()
# Merge x,y,z co-ordinates

df_struct['xyz'] = df_struct[['x','y','z']].values.tolist()

df_struct = df_struct.drop(['x','y','z'], axis=1)
print("train unique coupling type -",df_train['type'].unique())
print("test unique coupling type -",df_test['type'].unique())
print("struct unique atom type -",df_struct['atom'].unique())
print("test")

df_test.head()
# Molecule and atom index common in all dataset

df_struct.set_index(['molecule_name','atom_index'], inplace = True)
# Function to get atom

def funatom(x,y):

    return df_struct.loc[x,y]['atom']



# Function to get x,y,z co-ordinate of atom

def funxyz(x,y):

    return df_struct.loc[x,y]['xyz']
# Populate train data

df_train['atom_0'] = df_train.apply(lambda x: funatom(x['molecule_name'], x['atom_index_0']), axis=1)

df_train['atom_1'] = df_train.apply(lambda x: funatom(x['molecule_name'], x['atom_index_1']), axis=1)

df_train['xyz_0'] = df_train.apply(lambda x: funxyz(x['molecule_name'], x['atom_index_0']), axis=1)

df_train['xyz_1'] = df_train.apply(lambda x: funxyz(x['molecule_name'], x['atom_index_1']), axis=1)
# Populate test data

df_test['atom_0'] = df_test.apply(lambda x: funatom(x['molecule_name'], x['atom_index_0']), axis=1)

df_test['atom_1'] = df_test.apply(lambda x: funatom(x['molecule_name'], x['atom_index_1']), axis=1)

df_test['xyz_0'] = df_test.apply(lambda x: funxyz(x['molecule_name'], x['atom_index_0']), axis=1)

df_test['xyz_1'] = df_test.apply(lambda x: funxyz(x['molecule_name'], x['atom_index_1']), axis=1)
# Create new train columns

df_train['type_num'] = number.fit_transform(df_train['type'].astype('str'))

df_train['atom_1_num'] = number.fit_transform(df_train['atom_1'].astype('str'))

df_train['xyz'] = df_train['xyz_0'] + df_train['xyz_1']

df_train['atom_0_num'] = 1

df_train['atom_same_diff'] = (df_train['atom_0_num'] == df_train['atom_1_num']).astype(int)
# Create new test columns

df_test['type_num'] = number.fit_transform(df_test['type'].astype('str'))

df_test['atom_1_num'] = number.fit_transform(df_test['atom_1'].astype('str'))

df_test['xyz'] = df_test['xyz_0'] + df_test['xyz_1']

df_test['atom_0_num'] = 1

df_test['atom_same_diff'] = (df_test['atom_0_num'] == df_test['atom_1_num']).astype(int)
# Function to create additional features



def fundis(x):

    x1 = x[0] ; x2 = x[3] ; y1 = x[1] ; y2 = x[4] ; z1 = x[2] ; z2 = x[5]

    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2) + math.pow(z2 - z1, 2) * 1)



def fundotprod(x):

    x1 = x[0] ; x2 = x[3] ; y1 = x[1] ; y2 = x[4] ; z1 = x[2] ; z2 = x[5]

    return (x1*x2 + y1*y2 + z1*z2)



def funmod(x): 

    x1 = x[0] ; y1 = x[1] ; z1 = x[2]

    return math.sqrt(math.pow(x1, 2) + math.pow(y1, 2) + math.pow(z1, 2))



def funangle(x):

    return math.degrees(math.acos(x))





def funsqdisx(x):

    x1 = x[0] ; x2 = x[3] 

    return math.sqrt(math.pow(x2 - x1, 2) * 1)



def funsqdisy(x):

    y1 = x[1] ; y2 = x[4] 

    return math.sqrt(math.pow(y2 - y1, 2) * 1)



def funsqdisz(x):

    z1 = x[2] ; z2 = x[5] 

    return math.sqrt(math.pow(z2 - z1, 2) * 1)





def fundisx(x):

    x1 = x[0] ; x2 = x[3] 

    return (x2 - x1)



def fundisy(x):

    y1 = x[1] ; y2 = x[4] 

    return (y2 - y1)



def fundisz(x):

    z1 = x[2] ; z2 = x[5] 

    return (z2 - z1)





def funsumxyz(x):

    x1 = x[0] ; y1 = x[1] ; z1 = x[2]

    return(x1+y1+z1)





def funcross(x,y):

    return np.cross(x, y)
df_train['dis'] = df_train['xyz'].apply(fundis)

df_test['dis'] = df_test['xyz'].apply(fundis)



df_train['dotprod'] = df_train['xyz'].apply(fundotprod)

df_test['dotprod'] = df_test['xyz'].apply(fundotprod)



df_train['modxyz0'] = df_train['xyz_0'].apply(funmod)

df_train['modxyz1'] = df_train['xyz_1'].apply(funmod)



df_test['modxyz0'] =df_test['xyz_0'].apply(funmod)

df_test['modxyz1'] = df_test['xyz_1'].apply(funmod)



df_train['anglecos'] = df_train['dotprod'] / (df_train['modxyz0'] * df_train['modxyz1'])

df_test['anglecos'] = df_test['dotprod'] / (df_test['modxyz0'] * df_test['modxyz1'])



df_train['angle'] = df_train['anglecos'].apply(funangle)

df_test['angle'] = df_test['anglecos'].apply(funangle)



df_train['sqdisx'] = df_train['xyz'].apply(funsqdisx)

df_train['sqdisy'] = df_train['xyz'].apply(funsqdisy)

df_train['sqdisz'] = df_train['xyz'].apply(funsqdisz)



df_train['disx'] = df_train['xyz'].apply(fundisx)

df_train['disy'] = df_train['xyz'].apply(fundisy)

df_train['disz'] = df_train['xyz'].apply(fundisz)



df_train['sumxyz0'] = df_train['xyz_0'].apply(funsumxyz)

df_train['sumxyz1'] = df_train['xyz_1'].apply(funsumxyz)



df_train['cross'] = df_train.apply(lambda x: funcross(x['xyz_0'], x['xyz_1']), axis=1)

df_train['modcross'] = df_train['cross'].apply(funmod)

df_train['sumcross'] = df_train['cross'].apply(funsumxyz)



df_test['sqdisx'] = df_test['xyz'].apply(funsqdisx)

df_test['sqdisy'] = df_test['xyz'].apply(funsqdisy)

df_test['sqdisz'] = df_test['xyz'].apply(funsqdisz)



df_test['disx'] = df_test['xyz'].apply(fundisx)

df_test['disy'] = df_test['xyz'].apply(fundisy)

df_test['disz'] = df_test['xyz'].apply(fundisz)



df_test['sumxyz0'] = df_test['xyz_0'].apply(funsumxyz)

df_test['sumxyz1'] = df_test['xyz_1'].apply(funsumxyz)



df_test['cross'] = df_test.apply(lambda x: funcross(x['xyz_0'], x['xyz_1']), axis=1)

df_test['modcross'] = df_test['cross'].apply(funmod)

df_test['sumcross'] = df_test['cross'].apply(funsumxyz)
df_train.head()
# Selecting train ,test and target to be predicted

train = df_train[['atom_1_num','dis','dotprod','modxyz0','modxyz1','angle','disx','disy','disz','sqdisx','sqdisy','sqdisz','sumxyz0','sumxyz1','modcross','sumcross']]

y = df_train[['scalar_coupling_constant']]

test = df_test[['atom_1_num','dis','dotprod','modxyz0','modxyz1','angle','disx','disy','disz','sqdisx','sqdisy','sqdisz','sumxyz0','sumxyz1','modcross','sumcross']]
# Converting train and test to numpy data type for performance

train['atom_1_num'] = train['atom_1_num'].astype(np.int8)

train['dis'] = train['dis'].astype(np.float16)

train['dotprod'] = train['dotprod'].astype(np.float16)

train['modxyz0'] = train['modxyz0'].astype(np.float16)

train['modxyz1'] = train['modxyz1'].astype(np.float16)

train['angle'] = train['angle'].astype(np.float16)

train['disx'] = train['disx'].astype(np.float16)

train['disy'] = train['disy'].astype(np.float16)

train['disz'] = train['disz'].astype(np.float16)

train['sqdisx'] = train['sqdisx'].astype(np.float16)

train['sqdisy'] = train['sqdisy'].astype(np.float16)

train['sqdisz'] = train['sqdisz'].astype(np.float16)

train['sumxyz0'] = train['sumxyz0'].astype(np.float16)

train['sumxyz1'] = train['sumxyz1'].astype(np.float16)

train['modcross'] = train['modcross'].astype(np.float16)

train['sumcross'] = train['sumcross'].astype(np.float16)



test['atom_1_num'] = test['atom_1_num'].astype(np.int8)

test['dis'] = test['dis'].astype(np.float16)

test['dotprod'] = test['dotprod'].astype(np.float16)

test['modxyz0'] = test['modxyz0'].astype(np.float16)

test['modxyz1'] = test['modxyz1'].astype(np.float16)

test['angle'] = test['angle'].astype(np.float16)

test['disx'] = test['disx'].astype(np.float16)

test['disy'] = test['disy'].astype(np.float16)

test['disz'] = test['disz'].astype(np.float16)

test['sqdisx'] = test['sqdisx'].astype(np.float16)

test['sqdisy'] = test['sqdisy'].astype(np.float16)

test['sqdisz'] = test['sqdisz'].astype(np.float16)

test['sumxyz0'] = test['sumxyz0'].astype(np.float16)

test['sumxyz1'] = test['sumxyz1'].astype(np.float16)

test['modcross'] = test['modcross'].astype(np.float16)

test['sumcross'] = test['sumcross'].astype(np.float16)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=42)
import lightgbm as lgb

model = lgb.sklearn.LGBMRegressor(num_leaves= 128,

          min_child_samples= 79,

          objective= 'regression',

          max_depth= 9,

          learning_rate= 0.2,

          boosting_type= 'gbdt',

          subsample_freq= 1,

          subsample= 0.9,

          bagging_seed= 11,

          metric= 'mae',

          verbosity= -1,

          reg_alpha= 0.1,

          reg_lambda= 0.3,

          colsample_bytree= 1.0)
model.fit(x_train,y_train)
lgb.plot_importance(model,max_num_features=20)
y_preds = model.predict(x_test)

y_preds=pd.Series(y_preds)
y_preds.head()
y_test = y_test['scalar_coupling_constant']
y_test.head()
'''

#Creating submission dataframe

y_preds=pd.Series(y_preds)

pred = pd.DataFrame(df_test['id'])

pred['scalar_coupling_constant'] = y_preds

pred.head()

'''
#pred.to_csv("submission.csv", columns = pred.columns, index=False)
#from IPython.display import FileLink, FileLinks

#FileLinks('.') #lists all downloadable files on server
from sklearn.metrics import r2_score

r2_score(y_test, y_preds)