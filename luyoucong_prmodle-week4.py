# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

df_test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')
## 将所有地区唯一识别成place_id

def genplace_id(x):

    try:

        place_id=x['Country_Region']+'/'+x['Province_State']

    except:

        place_id=x['Country_Region']

    return place_id



df_train['place_id']=df_train.apply(lambda x:genplace_id(x),axis=1)

df_test['place_id']=df_test.apply(lambda x:genplace_id(x),axis=1)

print("地区个数==>"+str(len(df_train['place_id'].unique())))
df_train
df_train['Date']=pd.to_datetime(df_train['Date'])

df_test['Date']=pd.to_datetime(df_test['Date'])
train=df_train.copy()

test=df_test.copy()
## 生成多个变量，但是后来发现效果反而更差

# def create_features(df):

#     df['day'] = df['Date'].dt.day

#     df['month'] = df['Date'].dt.month

#     df['dayofweek'] = df['Date'].dt.dayofweek

#     df['dayofyear'] = df['Date'].dt.dayofyear

#     df['quarter'] = df['Date'].dt.quarter

#     df['weekofyear'] = df['Date'].dt.weekofyear

#     return df
# train=create_features(train)

# test=create_features(test)
## 只使用Day作为变量

train['Day']=train['Date'].apply(lambda x:x.dayofyear).astype('int')

test['Day']=test['Date'].apply(lambda x:x.dayofyear).astype('int')
print(train['Day'].max()) #5/15

print(test['Day'].min()) #4/02
print(train['Date'].max())

print(test['Date'].min())
test['Day'].min()-train['Day'].min()
from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
X_train=train[train['Day']<93] #4/02之前作为训练集

y1_train=train[train['Day']<93]

X_test=test
# # 如果使用多个x作为变量：

# pd_train=X_train[col_var][X_train['place_id']=='Italy']

# x = np.array(pd_train)# x:[[0],[1],[2]```[62]]

# x
y1_train
## 模型训练

places=train['place_id'].unique()

for place in places:

    print("trainning modle for ==>"+place)

    pd_train=X_train[X_train['place_id']==place]

    x = np.array(range(len(pd_train))).reshape((-1,1)) # x:[[0],[1],[2]```[70]]

    y=y1_train['ConfirmedCases'][y1_train['place_id']==place]

    model = Pipeline([('poly', PolynomialFeatures(degree=3)),

                         ('linear', LinearRegression(fit_intercept=False))])

    model = model.fit(x, y)

    pd_test=X_test[X_test['place_id']==place]

    X_pred=(np.array(range(len(pd_test)))+71).reshape((-1,1)) #71是下一个时间点的开始

    predit_case=model.predict(X_pred)

    test.loc[test['place_id']==place,'ConfirmedCases'] = predit_case


## 四月前两周的病例图形

place='Spain'

plot_train=train[train['place_id']==place]

plot_test=test[test['place_id']==place]



sns.lineplot(x=plot_train['Day'][  (plot_train['Day']<106)],y=plot_train['ConfirmedCases'][ (plot_train['Day']<106)],label='true_case')

sns.lineplot(x=plot_test['Day'][(plot_test['Day']>92)&(plot_test['Day']<106)],y=plot_test['ConfirmedCases'][(plot_test['Day']>92)& (plot_test['Day']<106)],label='pred_case')

print("Spain")

plt.show()
## 只显示2周14天的数据

place='Spain'

plot_train=train[train['place_id']==place]

plot_test=test[test['place_id']==place]



sns.lineplot(x=plot_train['Day'][ (plot_train['Day']>92) & (plot_train['Day']<106)],y=plot_train['ConfirmedCases'][ (plot_train['Day']>92) & (plot_train['Day']<106)],label='true')

sns.lineplot(x=plot_test['Day'][(plot_test['Day']>92)&(plot_test['Day']<106)],y=plot_test['ConfirmedCases'][(plot_test['Day']>92)& (plot_test['Day']<106)],label='pred')

print("Spain")

plt.show()
## 所有地区画图

places=train['place_id'].unique()

for place in places:

   

    data=train[train['place_id']==place].copy()

    data_test=test[test['place_id']==place].copy()



    sns.lineplot(x=data['Day'],y=data['ConfirmedCases'],label='true_case')

    sns.lineplot(x=data_test['Day'

                       ][data_test['Day']>92],y=data_test['ConfirmedCases'][(data_test['Day']>92)],label='pred_case')



    # pred2=rf.predict(test[col_var][(test['place_id']==place)])

    # sns.lineplot(x=data_test['Day'],y=pred2,label='pred2')

    plt.title(place)



    plt.show()
## 死亡病例
train.columns
# X_train没变

y2_train=train[train['Day']<93]
places=train['place_id'].unique()

for place in places:

    print("trainning modle for ==>"+place)

    pd_train=X_train[X_train['place_id']==place]

    x = np.array(range(len(pd_train))).reshape((-1,1)) # x:[[0],[1],[2]```[70]]

    y=y2_train['Fatalities'][y2_train['place_id']==place]

    model = Pipeline([('poly', PolynomialFeatures(degree=3)),

                         ('linear', LinearRegression(fit_intercept=False))])

    model = model.fit(x, y)

    pd_test=X_test[X_test['place_id']==place]

    X_pred=(np.array(range(len(pd_test)))+71).reshape((-1,1))

    predit_death=model.predict(X_pred)

    test.loc[test['place_id']==place,'Fatalities'] = predit_death
## 四月前两周的死亡病例图形

place='Spain'

plot_train=train[train['place_id']==place]

plot_test=test[test['place_id']==place]



sns.lineplot(x=plot_train['Day'][ (plot_train['Day']<106)],y=plot_train['Fatalities'][(plot_train['Day']<106)],label='true_death')

sns.lineplot(x=plot_test['Day'][(plot_test['Day']>92)&(plot_test['Day']<106)],y=plot_test['Fatalities'][(plot_test['Day']>92)& (plot_test['Day']<106)],label='pred_death')

print('Spain')

plt.show()


## 生成数据接口

test.rename(columns={'ConfirmedCases':'Pred_case','Fatalities':'Pred_death'},inplace=True)
test
result=pd.merge(train,test,how='left')

result
## 使用的数据文件

result.drop(['Province_State'],axis=1).to_csv('result.csv')
submit = pd.read_csv('../input/covid19-global-forecasting-week-4/submission.csv')

submit['Fatalities'] = test['Pred_death'].astype('int')

submit['ConfirmedCases'] = test['Pred_case'].astype('int')

submit.to_csv('submission.csv',index=False)