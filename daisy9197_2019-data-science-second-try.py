# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette(n_colors=10)



import warnings

warnings.filterwarnings('ignore')



from matplotlib.colors import ListedColormap

from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LogisticRegression

from keras.models import Model

from keras.layers import Dense, Input, BatchNormalization, Activation

from keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold

import tensorflow as tf

import keras

from functools import reduce
# read data

train = pd.read_csv('../input/data-science-bowl-2019/train.csv')

train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

test = pd.read_csv('../input/data-science-bowl-2019/test.csv')

specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
# create a new column to capture the correct/incorrect info from specs data

specs['attempts'] = ''



for i in range(len(specs)):

    if ('(Correct)' in specs['info'][i]) or ('(Incorrect)' in specs['info'][i]):

        specs['attempts'][i] = 1

    else:        

        specs['attempts'][i] = 0

# It is clearly that some event_id are not in assessment maps, 

# so only some of the event_id have the value from attempts.

# Next, we drop the useless columns to make it clear bacause the next step will be merged with train data



specs_drop = specs.drop(['info','args'],axis=1)



# merge the specs_attempts data with train data

train_cor = pd.merge(train,specs_drop,on='event_id',how='left')

        

# merge the specs_attempts data with test data

test_cor = pd.merge(test,specs_drop,on='event_id',how='left')
def create_new_variables(data):

    

    # 1. create new variables towards 'timestamp'

    

    '''

    change the data format first, and then create new variables

    '''

    

    data['timestamp'] =  pd.to_datetime(data['timestamp'])

    data['hour'] = data['timestamp'].dt.hour

    data['weekday']  = data['timestamp'].dt.dayofweek

    

    data['weekend'] = ['yes' if index in([5,6])  else 'no' for  index in data['weekday']]

    data['evening'] = ['yes' if index in([17,23]) else 'no' for index in data['hour']]

    data['freetime'] = [1 if (index_1 =='yes' or index_2 == 'yes') else 0 \

                               for (index_1,index_2) in zip(data['weekend'],data['evening'])]

    

    # 2. # create a new variable named as assessment. 'yes' represents it is a assessment type.

    

    data['assessment'] = [1 if index =='Assessment'  else 0 for  index in data['type']]

    

    # 3. create five new variables towards titile

    titles = ['Bird Measurer', 'Cart Balancer', 'Cauldron Filler', 'Chest Sorter', 'Mushroom Sorter']

    for each_var in titles:

        data[each_var] = [1 if (each_var) in index else 0 for index in data['title']]

    

    # 4. create four new variables towards 

    world = ['NONE', 'TREETOPCITY','MAGMAPEAK','CRYSTALCAVES']

    for each_wor in world:

        data[each_wor] = [1 if (each_wor) in index else 0 for index in data['world']]
def merge_data(data):

    '''

    groupby different key to get different dataset,

    and then merge them together

    '''

    

    # 1. groupby attempts

    attempts = data.groupby(['installation_id','game_session'],as_index=False)['attempts'].sum()

    

    # 2. groupby freetime

    freetime = data.groupby(['installation_id','game_session'],as_index=False)['freetime'].last()

    

    # 3. groupby event_id

    eventid = data.groupby(['installation_id','game_session'])['event_id'].nunique().reset_index()

    

    # 4. groupby gametime

    gametime = data.groupby(['installation_id','game_session'],as_index=False)['game_time'].max()

    

    # 5. groupby event_count

    eventcount = data.groupby(['installation_id','game_session'],as_index=False)['event_count'].max()

    

    # 6. groupby type

    ass_type = data.groupby(['installation_id','game_session'],as_index=False)['assessment'].last()

    

    # 7. groupby event_code

    etcode = data.groupby(['installation_id','game_session'])['event_code'].nunique().reset_index()

    

    # 8. groupby all kinds of title

    title =  data.groupby(['installation_id','game_session'],as_index=False)\

    ['Bird Measurer', 'Cart Balancer', 'Cauldron Filler', 'Chest Sorter', 'Mushroom Sorter'].last()

    

    # 9. groupby all kinds of worlds

    world = data.groupby(['installation_id','game_session'],as_index=False)['NONE', 'TREETOPCITY','MAGMAPEAK','CRYSTALCAVES'].last()

    

    

    '''

    merge all data together

    '''

    

    datalist = [attempts,freetime,eventid,gametime,eventcount,ass_type,etcode,title,world]

    new_data = reduce(lambda x,y: pd.merge(x,y, on=['installation_id','game_session'], how='outer'), datalist)

    

    return new_data

    
create_new_variables(data=train_cor)

train_merge_data = merge_data(data=train_cor)
create_new_variables(data=test_cor)

test_merge_data = merge_data(data=test_cor)
trainset = pd.merge(train_merge_data,train_labels,on=['installation_id','game_session'],how='outer')

# fill nan values in dataset

trainset.fillna(0,inplace=True)
testset = test_merge_data.fillna(0)
train_X = trainset[['attempts','freetime','event_id','game_time','event_count','assessment',\

              'event_code','Bird Measurer','Cart Balancer','Cauldron Filler','Chest Sorter','Mushroom Sorter',

                    'NONE','TREETOPCITY','CRYSTALCAVES','MAGMAPEAK']].values.astype('int')  

train_ybin = trainset[['num_correct']].values.astype('int')

train_ynum = trainset[['num_incorrect']].values.astype('int')



test_X = testset[['attempts','freetime','event_id','game_time','event_count','assessment',\

              'event_code','Bird Measurer','Cart Balancer','Cauldron Filler','Chest Sorter','Mushroom Sorter',

                    'NONE','TREETOPCITY','CRYSTALCAVES','MAGMAPEAK']].values.astype('int')
one_input = Input(shape=(16,), name='one_input') # pass by one input



# show one output: y_bin

y_bin_output = Dense(1, activation='sigmoid', name='y_bin_output')(one_input)

# merge one output with all predictors from input

x = keras.layers.concatenate([one_input, y_bin_output]) 

# stack all other layers

x = Dense(64, activation='relu')(x)

x = BatchNormalization()(x)

#another output

y_num_output = Dense(1, activation='sigmoid', name='y_num_output')(x)





model = Model(inputs=one_input, outputs=[y_bin_output, y_num_output])

model.compile(optimizer='Adam', loss=['binary_crossentropy', 'mean_squared_error'])



model.fit(train_X, [train_ybin, train_ynum],epochs=30,verbose=0)
testset['num_correct'] = 0

testset['num_incorrect'] = 0



for i in range(len(testset)):

    value = testset.iloc[i:i+1,2:18].values

    pred_y = model.predict(value)

    testset['num_correct'][i] = pred_y[0].astype('int')

    testset['num_incorrect'][i] = np.around(pred_y[1])
testset['accuracy'] = testset['num_correct']/(testset['num_correct'] + testset['num_incorrect'])



# fill nan

testset.fillna(0,inplace=True)



# calculate accuracy_group

testset['accuracy_group'] = 0

for m in range(len(testset)):

    if testset['accuracy'][m] == 1:

        testset['accuracy_group'][m] =3

    elif 0.5 <= testset['accuracy'][m] < 1:

        testset['accuracy_group'][m] =2

    elif 0 < testset['accuracy'][m] < 0.5:

        testset['accuracy_group'][m] =1

    elif testset['accuracy'][m] == 0:

        testset['accuracy_group'][m] =0

        

final_pred_1 = testset[(testset['Bird Measurer'] !=0) | (testset['Cart Balancer'] !=0)| (testset['Cauldron Filler'] !=0)\

                    | (testset['Chest Sorter'] !=0)| (testset['Mushroom Sorter'] !=0)]



final_pred = final_pred_1.groupby('installation_id',as_index=False)['accuracy_group'].mean()

final_pred['accuracy_group'] = (np.around(final_pred['accuracy_group'])).astype('int')

final_pred
# save as csv

final_pred.to_csv('submission.csv',index=False)