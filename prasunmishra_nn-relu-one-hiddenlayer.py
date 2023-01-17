"""
1/16/2019
What has changed?
1) Being NN, converted  categorical features through one-hot-encoder (was Lable Encoded earlier). This should make job easier for NN.

##Attribution##
Thanks to Elo World, SRK and multiple other kernels. 
I have mixed data pre processing and feature engineering and tried ReLU with one hidden layer. 
I think we need to experiemnet with architecture and other hyperparameters.
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.linear_model import Ridge
import time
from sklearn import preprocessing
import warnings
import datetime
warnings.filterwarnings("ignore")
import gc
from tqdm import tqdm
from scipy.stats import describe
%matplotlib inline

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
# Any results you write to the current directory are saved as output.
PATH = "../input/"
print("Path is:",PATH)

#############################################################################################
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df
################################################################################################
train = pd.read_csv(PATH+'train.csv')
test = pd.read_csv(PATH+'test.csv')
hist_trans = pd.read_csv(PATH+'historical_transactions.csv')
new_merchant_trans = pd.read_csv(PATH+'new_merchant_transactions.csv')

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
hist_trans = reduce_mem_usage(hist_trans)
new_merchant_trans = reduce_mem_usage(new_merchant_trans)

train = pd.get_dummies(train, columns= ['feature_1','feature_2','feature_3'], dummy_na= True)
test = pd.get_dummies(test, columns= ['feature_1','feature_2','feature_3'], dummy_na= True)
train.head(5)
len(train.columns)
train.columns
hist_trans = pd.get_dummies(hist_trans, columns= ['category_1','category_2','category_3'], dummy_na= True)
new_merchant_trans = pd.get_dummies(new_merchant_trans, columns= ['category_1','category_2','category_3'], dummy_na= True)
new_merchant_trans.head(5)
len(new_merchant_trans.columns)
new_merchant_trans.columns


############################################################################################    

for df in [hist_trans,new_merchant_trans]:
    #df['category_2'].fillna(1.0,inplace=True)
    #df['category_3'].fillna('A',inplace=True)
    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
############################################################################################    
    
def get_new_columns(name,aggs):
    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
############################################################################################    


for df in [hist_trans,new_merchant_trans]:
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['year'] = df['purchase_date'].dt.year
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['month'] = df['purchase_date'].dt.month
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    ###
    #df['daysinmonth'] = df['purchase_date'].dt.daysinmonth
    df['day'] = df['purchase_date'].dt.day
    df['dayposinmon']=df['day']/(df['purchase_date'].dt.daysinmonth)
    
    
    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
    df['hour'] = df['purchase_date'].dt.hour
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
    #Handled by one_hot_encoder
    #df['category_1'] = df['category_1'].map({'Y':1, 'N':0}) 
    #https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244
    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']
    


hist_trans.head(5)
############################################################################################    
aggs = {}
for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:
    aggs[col] = ['nunique']

aggs['purchase_amount'] = ['sum','max','min','mean','var']
aggs['installments'] = ['sum','max','min','mean','var']
aggs['purchase_date'] = ['max','min']
aggs['month_lag'] = ['max','min','mean','var']
aggs['month_diff'] = ['mean']
aggs['authorized_flag'] = ['sum', 'mean']
aggs['weekend'] = ['sum', 'mean']
#aggs['category_1'] = ['sum', 'mean']
#aggs['category_1_N','category_1_Y','category_1_nan'] = ['sum', 'mean']
aggs['card_id'] = ['size']
"""
for col in ['category_2_1.0', 'category_2_2.0','category_2_3.0', 'category_2_4.0', 'category_2_5.0', 'category_2_nan','category_3_A', 'category_3_B', 'category_3_C', 'category_3_nan']:
    hist_trans[col+'_mean'] = hist_trans.groupby([col])['purchase_amount'].transform('mean')
    aggs[col+'_mean'] = ['mean']    
"""
new_columns = get_new_columns('hist',aggs)
hist_trans_group = hist_trans.groupby('card_id').agg(aggs)
hist_trans_group.columns = new_columns
hist_trans_group.reset_index(drop=False,inplace=True)
hist_trans_group['hist_purchase_date_diff'] = (hist_trans_group['hist_purchase_date_max'] - hist_trans_group['hist_purchase_date_min']).dt.days
hist_trans_group['hist_purchase_date_average'] = hist_trans_group['hist_purchase_date_diff']/hist_trans_group['hist_card_id_size']
hist_trans_group['hist_purchase_date_uptonow'] = (datetime.datetime.today() - hist_trans_group['hist_purchase_date_max']).dt.days
train = train.merge(hist_trans_group,on='card_id',how='left')
test = test.merge(hist_trans_group,on='card_id',how='left')
del hist_trans_group;gc.collect();gc.collect()


############################################################################################    
aggs = {}
for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:
    aggs[col] = ['nunique']
aggs['purchase_amount'] = ['sum','max','min','mean','var']
aggs['installments'] = ['sum','max','min','mean','var']
aggs['purchase_date'] = ['max','min']
aggs['month_lag'] = ['max','min','mean','var']
aggs['month_diff'] = ['mean']
aggs['weekend'] = ['sum', 'mean']
#Handled by one_hot_encoder
#aggs['category_1'] = ['sum', 'mean']
#aggs['category_1_N','category_1_Y','category_1_nan'] = ['sum', 'mean']
aggs['card_id'] = ['size']
"""
for col in ['category_2_1.0', 'category_2_2.0','category_2_3.0', 'category_2_4.0', 'category_2_5.0', 'category_2_nan','category_3_A', 'category_3_B', 'category_3_C', 'category_3_nan']:
    new_merchant_trans[col+'_mean'] = new_merchant_trans.groupby([col])['purchase_amount'].transform('mean')
    aggs[col+'_mean'] = ['mean']
"""  
new_columns = get_new_columns('new_hist',aggs)
hist_trans_group = new_merchant_trans.groupby('card_id').agg(aggs)
hist_trans_group.columns = new_columns
hist_trans_group.reset_index(drop=False,inplace=True)
hist_trans_group['new_hist_purchase_date_diff'] = (hist_trans_group['new_hist_purchase_date_max'] - hist_trans_group['new_hist_purchase_date_min']).dt.days
hist_trans_group['new_hist_purchase_date_average'] = hist_trans_group['new_hist_purchase_date_diff']/hist_trans_group['new_hist_card_id_size']
hist_trans_group['new_hist_purchase_date_uptonow'] = (datetime.datetime.today() - hist_trans_group['new_hist_purchase_date_max']).dt.days
train = train.merge(hist_trans_group,on='card_id',how='left')
test = test.merge(hist_trans_group,on='card_id',how='left')
del hist_trans_group;gc.collect();gc.collect()
############################################################################################    


del hist_trans;gc.collect()
del new_merchant_trans;gc.collect()
train.head(5)

train['outliers'] = 0
train.loc[train['target'] < -30, 'outliers'] = 1
train['outliers'].value_counts()
############################################################################################    


for df in [train,test]:
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['dayofweek'] = df['first_active_month'].dt.dayofweek
    df['weekofyear'] = df['first_active_month'].dt.weekofyear
    ###
    #df['daysinmonth'] = df['first_active_month'].dt.daysinmonth
    df['day'] = df['first_active_month'].dt.day
    df['dayposinmon']=df['day']/(df['first_active_month'].dt.daysinmonth)
    
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days
    for f in ['hist_purchase_date_max','hist_purchase_date_min','new_hist_purchase_date_max',\
                     'new_hist_purchase_date_min']:
        df[f] = df[f].astype(np.int64) * 1e-9
    df['card_id_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']  
        
    df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']

for f in ['feature_1_1.0','feature_1_2.0', 'feature_1_3.0', 'feature_1_4.0', 'feature_1_5.0','feature_1_nan', 'feature_2_1.0', 'feature_2_2.0', 'feature_2_3.0','feature_2_nan', 'feature_3_0.0', 'feature_3_1.0', 'feature_3_nan']:
        order_label = train.groupby([f])['outliers'].mean()
        train[f] = train[f].map(order_label)
        test[f] = test[f].map(order_label)
  
train_columns = [c for c in train.columns if c not in ['card_id', 'first_active_month','target','outliers']]
target = train['target']
del train['target']

###############################################################ANN World ########################################

from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2

#####Getting the same number of columns for Train, Test######

y = target

train_df = train[train_columns]
test_df = test[train_columns]

#####Handling Missing Values#####     

for i in range(len(train_df.columns)):
    train_df.iloc[:,i] = (train_df.iloc[:,i]).fillna(-1)

for i in range(len(test_df.columns)):
    test_df.iloc[:,i] = (test_df.iloc[:,i]).fillna(-1)    
    
#####Encoding the Categorical Variables#####
"""
lbl = LabelEncoder()

for c in train.columns:
    if train[c].dtype == 'object':
        lbl.fit(list(train[c].values))
        train[c] = lbl.transform(list(train[c].values))

for c in test.columns:
    if test[c].dtype == 'object':
        lbl.fit(list(test[c].values))
        test[c] = lbl.transform(list(test[c].values))     
        
print("Done with the Encoding")        
"""
####Normalizing the values####

mmScale = MinMaxScaler()

n = train_df.shape[1]
print('n is:',n)

x_train = mmScale.fit_transform(train_df)
x_test = mmScale.transform(test_df)


from keras import optimizers
from keras import layers
from keras.layers import LeakyReLU

#####################Early Stopping ########################
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
#####Artificial Neural Networks Implementation#####
print("Starting Neural Network")

model_n = Sequential()
#Want to use an expotential linear unit instead of the usual relu
#model_n.add( Dense( n, activation='relu', input_shape=(n,) ) )
model_n.add( Dense( n, activation='tanh', input_shape=(n,) ) )
model_n.add(BatchNormalization())
model_n.add( Dense( int(0.5*n), activation='relu' ) )
model_n.add(LeakyReLU(alpha=.0011))

model_n.add(Dropout(0.5))

"""
model_n.add( Dense( 512, activation='relu' ) )
model_n.add(BatchNormalization())
model_n.add(LeakyReLU(alpha=.001))
model_n.add(Dropout(0.5))

model_n.add( Dense( 256, activation='relu' ) )
model_n.add(BatchNormalization())
model_n.add(LeakyReLU(alpha=.001))
model_n.add(Dropout(0.5))

model_n.add( Dense( 128, activation='relu' ) )
model_n.add(BatchNormalization())
model_n.add(LeakyReLU(alpha=.001))
model_n.add(Dropout(0.5))

model_n.add( Dense( 1024, activation='relu' ) )
model_n.add(BatchNormalization())
model_n.add(LeakyReLU(alpha=.001))
model_n.add(Dropout(0.5))

model_n.add( Dense( 99, activation='relu' ) )
model_n.add(BatchNormalization())
model_n.add(LeakyReLU(alpha=.001))
model_n.add(Dropout(0.5))

model_n.add( Dense( 64, activation='relu' ) )
model_n.add(BatchNormalization())
model_n.add(LeakyReLU(alpha=.001))
model_n.add(Dropout(0.5))

model_n.add( Dense( 19, activation='relu' ) )
model_n.add(BatchNormalization())
model_n.add(LeakyReLU(alpha=.001))
model_n.add(Dropout(0.5))
"""
model_n.add(Dense(1, activation='linear'))
model_n.compile(loss='mse', optimizer='Adadelta',  metrics=['mse'])
        

model_n.fit(x_train, y, epochs=25,verbose=1, batch_size=32,validation_split=0.2, callbacks=[early_stopping])

predictions = model_n.predict(x_test)

sub_df = pd.DataFrame({"card_id":test["card_id"]})
sub_df["target"] = pd.DataFrame(predictions)
from datetime import datetime
sub_df.to_csv('Kaggle_ELO_CNNStarter_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False,float_format='%.4f')


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(x_train, y)
pred_test_y = clf.predict(x_test)
sub_df1 = pd.DataFrame({"card_id":test["card_id"]})
sub_df1["target"] = pd.DataFrame(pred_test_y)
from datetime import datetime
sub_df1.to_csv('Kaggle_ELO_LRStarter_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False,float_format='%.4f')
