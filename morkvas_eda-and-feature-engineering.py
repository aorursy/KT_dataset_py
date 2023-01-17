import pandas as pd

import numpy as np

import random

import seaborn as sns

from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
sales_data = pd.read_csv("/kaggle/input/sales-analysis/SalesKaggle3.csv", index_col = 'SKU_number')

sales_data.head()
historical_data = sales_data[sales_data['File_Type'] == 'Historical'].replace({'MarketingType':{'S':0,'D':1}})

target_train = historical_data['SoldFlag']

sales_train = historical_data.drop(['Order', 'File_Type','SoldFlag','SoldCount'], axis=1)
train_x, test_x, train_y, test_y = train_test_split(sales_train, target_train, test_size=0.33, random_state=42)

train_x.head()
logit_cv = LogisticRegressionCV(cv=10, random_state=17, solver='lbfgs')
scaler = StandardScaler()
def scaling_function(data, scaler, not_scale_feats=[]):

    features_to_scale = data.drop(not_scale_feats, axis=1)

    scaled_features = scaler.fit_transform(features_to_scale)

    return np.hstack([scaled_features]+

                              [data[feature].values.reshape(-1,1) for feature in not_scale_feats])   
def model_cycle(train_x, train_y, test_x, test_y, reg_model, scaler, not_scale_feats=[], exclude_feats=[]):

    final_train_x = train_x.drop(exclude_feats, axis=1)

    final_test_x = test_x.drop(exclude_feats, axis=1)

    scaled_train = scaling_function(final_train_x, scaler,not_scale_feats)

    reg_model.fit(scaled_train, train_y);

    train_score = reg_model.score(scaled_train, train_y)

    scaled_test= scaling_function(final_test_x, scaler, not_scale_feats)

    test_score = reg_model.score(scaled_test, test_y)

    #here we print feature names and coefs, to see, which has the biggest impact to the model

    for col, coef in zip(final_train_x.columns, reg_model.coef_[0]):

        print(col, coef)

    return (train_score,  test_score)
model_cycle(train_x, train_y, test_x, test_y, logit_cv, scaler, ['New_Release_Flag', 'MarketingType'], [])
eda_data = train_x.copy();

eda_data['SoldFlag'] = train_y;

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(x='MarketingType', hue='SoldFlag', data=eda_data);
chart = sns.countplot(x='ReleaseYear', hue='SoldFlag', data=eda_data);

chart.set_xticklabels(chart.get_xticklabels(), rotation=90);
time_frames = ([1935, 1987], # - period with some insufficient data

               [1988, 1999], # - period with small growth of items,

               [2000, 2009], # - period with main rising trend

               [2010, 2017]) # - downtrend

time_frame_sold = [2001, 2014]
sns.boxplot(x='SoldFlag', y='StrengthFactor', data=eda_data);
def percentile_print(data, feature, percentile_list = [25, 50, 75, 80, 90]):

    for percentile in percentile_list:

        print ("Percentile",percentile,"Unsold",np.percentile(data[data['SoldFlag']==0][feature], percentile), "Sold", np.percentile(data[data['SoldFlag']==1][feature], percentile))
percentile_print(eda_data, "StrengthFactor")
#For example we can set **Strenght_Factor_bias = 1805919.4000000004** beacuse 90% of sold data are below this value. 

StrenghtFactor_const = 1805919.4000000004
sns.boxplot(x='SoldFlag', y='ReleaseNumber', data=eda_data);
percentile_print(eda_data, "ReleaseNumber")
#Here we can set ReleaseNumber_const = 9.0 to separate unsold items

ReleaseNumber_const = 9.0
sns.boxplot(x='SoldFlag', y='ItemCount', data=eda_data);
percentile_print(eda_data, "ItemCount")
#Here we set coinstant ItemCount_const = 72.0, because 90% of unsold items are below this value.

ItemCount_const = 72.0
sns.countplot(x='New_Release_Flag', hue='SoldFlag', data=eda_data);
sns.lineplot(x='ReleaseYear', y='LowUserPrice', hue='SoldFlag', data=eda_data);
sns.lineplot(x='ReleaseYear', y='LowNetPrice',hue='SoldFlag',  data=eda_data);
sns.lineplot(x='ReleaseYear', y='PriceReg',hue='SoldFlag',  data=eda_data);
eda_data.drop("SoldFlag", axis=1).apply(lambda x: x.corr(eda_data['SoldFlag']))
eda_data['LowNetDiff'] = eda_data['LowNetPrice'] - eda_data['PriceReg'];

eda_data['LowUserDiff'] = eda_data['LowUserPrice'] - eda_data['PriceReg'];
#we will plot our new features together just for fun

sns.lineplot(x='ReleaseYear', y='LowNetDiff',  data=eda_data);

sns.lineplot(x='ReleaseYear', y='LowUserDiff',  data=eda_data);
train_x = train_x.replace({'MarketingType':{1:2}});

test_x = test_x.replace({'MarketingType':{1:2}});
model_cycle(train_x, train_y, test_x, test_y, logit_cv, scaler, ['New_Release_Flag'],[])
train_x = train_x.replace({'MarketingType':{2:1}});

test_x = test_x.replace({'MarketingType':{2:1}});
frames_names = ['frame%s' % i for i in range(len(time_frames))]
for name,t in zip(frames_names,time_frames):

    train_x[name] = (train_x['ReleaseYear']>=t[0])&(train_x['ReleaseYear']<=t[1])

    test_x[name] = (test_x['ReleaseYear']>=t[0])&(test_x['ReleaseYear']<=t[1])
model_cycle(train_x, train_y, test_x, test_y, logit_cv, scaler,['New_Release_Flag','MarketingType']+frames_names, [])
train_x['time_frame_sold'] = (train_x['ReleaseYear']>=time_frame_sold[0])&(train_x['ReleaseYear']<=time_frame_sold[1])

test_x['time_frame_sold'] = (test_x['ReleaseYear']>=time_frame_sold[0])&(test_x['ReleaseYear']<=time_frame_sold[1])
model_cycle(train_x, train_y, test_x, test_y, logit_cv, scaler,['New_Release_Flag', 'MarketingType','time_frame_sold'], frames_names)
train_x['LowNetDiff'] = train_x['LowNetPrice'] - train_x['PriceReg'];

test_x['LowNetDiff'] = test_x['LowNetPrice'] - test_x['PriceReg'];

train_x['LowUserDiff'] = train_x['LowUserPrice'] - train_x['PriceReg'];

test_x['LowUserDiff'] = test_x['LowUserPrice'] - test_x['PriceReg'];
model_cycle(train_x, train_y, test_x, test_y, logit_cv, scaler,['New_Release_Flag', 'MarketingType']+frames_names, ['time_frame_sold'])
train_x['StrenghtFactor_const'] = train_x['StrengthFactor']<=StrenghtFactor_const;

test_x['StrenghtFactor_const'] = test_x['StrengthFactor']<=StrenghtFactor_const;

train_x['ReleaseNumber_const'] = train_x['ReleaseNumber']<=ReleaseNumber_const;

test_x['ReleaseNumber_const'] = test_x['ReleaseNumber']<=ReleaseNumber_const;

train_x['ItemCount_const'] = train_x['ItemCount']<=ItemCount_const;

test_x['ItemCount_const'] = test_x['ItemCount']<=ItemCount_const;
model_cycle(train_x, train_y, test_x, test_y, logit_cv, scaler,['New_Release_Flag', 'MarketingType',

                                                             'StrenghtFactor_const','ReleaseNumber_const','ItemCount_const']+frames_names, ['time_frame_sold'])