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
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
def RMLSE(pre,actual):
    return np.sqrt(np.abs(np.mean(np.log(pre+1)-np.log(actual+1))))
df_train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
df_test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')
df_train['Province_State']=df_train['Province_State'].fillna(df_train['Country_Region'])
df_test['Province_State']=df_test['Province_State'].fillna(df_test['Country_Region'])
df_train['Date']=df_train['Date'].apply(lambda x:x.replace('-','')).apply(lambda x:x[5:])
df_test['Date']=df_test['Date'].apply(lambda x:x.replace('-','')).apply(lambda x:x[5:])
df_test['Date']=df_test['Date'].astype(int)
df_train['Date']=df_train['Date'].astype(int)
df_train[['c_cummax', 'f_cummax']] = df_train.groupby(['Country_Region', 'Province_State'])[['ConfirmedCases', 'Fatalities']].\
                                            transform('sum') 
df_train[['c_mean', 'f_mean']] = df_train.groupby(['Country_Region', 'Province_State'])[['ConfirmedCases', 'Fatalities']].\
                                            transform('mean') 
lc=preprocessing.LabelEncoder()
countries=df_train['Country_Region'].unique()
countries_label=lc.fit_transform(countries)
c_dic={countries[i]:countries_label[i] for i in range(len(countries_label))}
provinces=df_train['Province_State'].unique()
provinces_label=lc.fit_transform(provinces)
p_dic={provinces[i]:provinces_label[i] for i in range(len(provinces_label))}

ct=df_test['Country_Region'].unique()
ct_label=lc.fit_transform(ct)
ct_dic={ct[i]:ct_label[i] for i in range(len(ct_label))}
pt=df_test['Province_State'].unique()
pt_label=lc.fit_transform(pt)
pt_dic={pt[i]:pt_label[i] for i in range(len(pt_label))}
df_train['Province_State']=df_train['Province_State'].apply(lambda x: p_dic[x])
df_train['Country_Region']=df_train['Country_Region'].apply(lambda x: c_dic[x])
df_test['Country_Region']=df_test['Country_Region'].apply(lambda x: ct_dic[x])
df_test['Province_State']=df_test['Province_State'].apply(lambda x: pt_dic[x])
params={           \
    'task': 'train',\
    'boosting_type': 'gbdt',   #设置提升类型
    'objective': 'regression',  #目标函数
    'metric': {'12','auc'},     #评估函数
    'num_leaves': 31,          #叶子节点数
    'max_depth': -1,
    'learning_rate': 0.05,      #学习速率
    'feature_fraction': 0.9,    #建树的特征选择比例
    'bagging_fraction': 0.8,    #建树的样本采样比例
    'bagging_freq': 5,          # k 意味着每 k 次迭代执行bagging
    'verbose': 1                # <0 显示致命的，=0 显示错误（警告），>0 显示信息
    
}
ForecastId=df_test['ForecastId'].values
c_pre=[]
f_pre=[]
for k in p_dic:
    temp1=df_train[df_train['Province_State']==p_dic[k]]
    temp2=df_test[df_test['Province_State']==pt_dic[k]]
    feature=['Province_State','Country_Region','Date']
    lgb_train1=lgb.Dataset(temp1[feature],temp1['ConfirmedCases'])
    lgb_train2=lgb.Dataset(temp1[feature],temp1['Fatalities'])
    #lgb_eval1=lgb.Dataset(temp1[feature].iloc[l:],temp1['ConfirmedCases'].iloc[l:],reference=lgb_train1)
    #lgb_eval2=lgb.Dataset(temp1[feature].iloc[l:],temp1['Fatalities'].iloc[l:],reference=lgb_train2)
    model1=lgb.train(params,lgb_train1,num_boost_round=15)
    model2=lgb.train(params,lgb_train2,num_boost_round=15)
    c_pre.extend(model1.predict(temp2[feature],num_iteration=model1.best_iteration))
    f_pre.extend(model2.predict(temp2[feature],num_iteration=model2.best_iteration))
    print('*',end='')
c_pre,f_pre=np.array(c_pre),np.array(f_pre)
results=pd.DataFrame({'ForecastId':ForecastId,'ConfirmedCases':c_pre,'Fatalities':f_pre})
results.to_csv('results.csv',encoding='utf_8_sig')