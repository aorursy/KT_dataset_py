import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor





from sklearn.metrics import mean_squared_error



from sklearn.model_selection import train_test_split


import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

    
training_set_path = '../input/bits-f464-l1/train.csv'

testing_set_path =  '../input/bits-f464-l1/test.csv'

df = pd.read_csv(training_set_path)

df_test = pd.read_csv(testing_set_path)
df.drop(['id'],axis=1,inplace=True)

#df.drop_duplicates(inplace=True)
df0=df[df['a0']==1]

df1=df[df['a1']==1]

df2=df[df['a2']==1]

df3=df[df['a3']==1]

df4=df[df['a4']==1]

df5=df[df['a5']==1]

df6=df[df['a6']==1]
def drop_agent_column(df):

    df = df.drop(labels=['a0','a1','a2','a3','a4','a5','a6','time'],axis=1)

    return df
def get_correlated_parameter (df):

    

    #### Compute the correlation matrix

    corr = df.corr()

    

    corr_target = abs(corr["label"])



    feature = corr_target[corr_target>=0]

#     feature =  corr_target[corr_target<n_te]

#     print(feature)

    return feature.index

    

def get_corr_para_df (df):

    feature = get_correlated_parameter (df)

    return df[feature]
def get_xy (df):

    

    x = df.drop('label',axis=1)

    y = df["label"].copy()

    return x,y
def get_train_text_sets(df,test_size = 0.3):

    x,y = get_xy(df)

#     return train_test_split(X0,Y0,test_size)

    X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=test_size,shuffle = "false")

    return X_train,X_test,Y_train,Y_test
def get_common_columns(df):

    vals = df.apply(set, axis = 0)

    res = vals[vals.map(len) == 1].index



    return res
corr_para_df0 = get_corr_para_df(df0)

corr_para_df1 = get_corr_para_df(df1)

corr_para_df2 = get_corr_para_df(df2)

corr_para_df3 = get_corr_para_df(df3)

corr_para_df4 = get_corr_para_df(df4)

corr_para_df5 = get_corr_para_df(df5)

corr_para_df6 = get_corr_para_df(df6)

# print (len(corr_para_df0.columns))
def train_model_rfr(df):

    rfr =  RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60, bootstrap=True)

    x,y = get_xy(df)

    rfr.fit(x,y)

    return rfr

#     np.sqrt(mean_squared_error(y_test,gdb.predict(x_test)))
mo0 = train_model_rfr(corr_para_df0)

mo1 = train_model_rfr(corr_para_df1)

mo2 = train_model_rfr(corr_para_df2)

mo3 = train_model_rfr(corr_para_df3)

mo4 = train_model_rfr(corr_para_df4)

mo5 = train_model_rfr(corr_para_df5)

mo6 = train_model_rfr(corr_para_df6)
df0_test = df_test[df_test['a0']==1]

df1_test = df_test[df_test['a1']==1]

df2_test = df_test[df_test['a2']==1]

df3_test = df_test[df_test['a3']==1]

df4_test = df_test[df_test['a4']==1]

df5_test = df_test[df_test['a5']==1]

df6_test = df_test[df_test['a6']==1]
df0_test = df0_test[corr_para_df0.drop('label',axis=1).columns ]

df1_test = df1_test[corr_para_df1.drop('label',axis=1).columns ]

df2_test = df2_test[corr_para_df2.drop('label',axis=1).columns ]

df3_test = df3_test[corr_para_df3.drop('label',axis=1).columns ]

df4_test = df4_test[corr_para_df4.drop('label',axis=1).columns ]

df5_test = df5_test[corr_para_df5.drop('label',axis=1).columns ]

df6_test = df6_test[corr_para_df6.drop('label',axis=1).columns ]
lable0 = mo0.predict(df0_test)

lable1 = mo1.predict(df1_test)

lable2 = mo2.predict(df2_test)

lable3 = mo3.predict(df3_test)

lable4 = mo4.predict(df4_test)

lable5 = mo5.predict(df5_test)

lable6 = mo6.predict(df6_test)
result0 = pd.DataFrame({'id' : df0_test.index + 1  , 'label': lable0})

result1 = pd.DataFrame({'id' : df1_test.index + 1  , 'label': lable1})

result2 = pd.DataFrame({'id' : df2_test.index + 1  , 'label': lable2})

result3 = pd.DataFrame({'id' : df3_test.index + 1  , 'label': lable3})

result4 = pd.DataFrame({'id' : df4_test.index + 1  , 'label': lable4})

result5 = pd.DataFrame({'id' : df5_test.index + 1  , 'label': lable5})

result6 = pd.DataFrame({'id' : df6_test.index + 1  , 'label': lable6})
results_array = [result0, result1, result2, result3, result4, result5, result6]



result = pd.concat(results_array)

result.sort_values(by=['id'])

result.to_csv('submission_lab_1.csv', header=True, index=False)