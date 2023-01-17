import numpy as np

import pandas as pd



data=pd.read_json('../input/train.json')
#making a copy of the main train dataset

train=data.copy()



#creating a dataframe from band_1 and band_2 columns in train set.

band_1_df=pd.DataFrame(i for i in train['band_1'])

band_2_df=pd.DataFrame(i for i in train['band_2'])



#converting `inc_angle` to numeric values.

train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')



#concatenate last 3 dataframes

train_df=pd.concat([band_1_df,band_2_df,train['inc_angle']],axis=1)
null_num=train_df.isnull().sum()

train_df.columns[null_num>0]
from fancyimpute import KNN



#fancy impute removes column names.

train_df_cols=list(train_df)



train_df = pd.DataFrame(KNN(k=3).complete(train_df))



train_df.columns=train_df_cols
#checking if there is no null value anymore.

null_num=train_df.isnull().sum()

train_df.columns[null_num>0]