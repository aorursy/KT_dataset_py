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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

import numpy as np

plt.style.use('fivethirtyeight') 

import warnings

warnings.filterwarnings('ignore') 
os.curdir
!pwd
regular=pd.read_csv("/kaggle/input/Regular_Season_Batter.csv")
regular.head()
#regular 데이터 셋에서 Null element가 없는 데이터 셋으로 만들기

regular=regular.loc[~regular['OPS'].isnull(),]
submission=pd.read_csv("/kaggle/input/Batter.csv")
agg={}

for i in regular.columns:

    agg[i]=[]
for i in submission['batter_name'].unique():

    for j in regular.columns:

        if j in ['batter_id','batter_name','height/weight','year_born','position','starting_salary']:

            agg[j].append(regular.loc[regular['batter_name']==i,j].iloc[0])

        elif j=='year':

            agg[j].append(2019)

        else:

            agg[j].append(0)

regular=pd.concat([regular,pd.DataFrame(agg)])
regular.info()
regular.columns
corr = regular.loc[:,regular.dtypes == 'float64'].corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))
corr = regular.loc[:,regular.dtypes == 'int64'].corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))
sns.pairplot(regular.loc[:,regular.dtypes == 'float64'])
sns.distplot(regular['year'])
regular['year'].describe()
sns.distplot(regular['AB'])
sns.distplot(regular['OPS'].dropna())
regular['OPS'].describe()
plt.scatter(regular['AB'],regular['OPS'])

plt.xlabel('AB')

plt.ylabel('OPS')
plt.scatter(regular['year'],regular['AB'])

plt.xlabel('year')

plt.ylabel('AB')
def get_self_corr(var,regular=regular):

    x=[]

    y=[]

    regular1=regular.loc[regular['AB']>=50,]

    for name in regular1['batter_name'].unique():

        a=regular1.loc[regular1['batter_name']==name,].sort_values('year')

        k=[]

        for i in a['year'].unique():

            if (a['year']==i+1).sum()==1:

                k.append(i)

        for i in k:

            x.append(a.loc[a['year']==i,var].iloc[0])

            y.append(a.loc[a['year']==i+1,var].iloc[0])

    plt.scatter(x,y)

    plt.title(var)

    plt.show()

    print(pd.Series(x).corr(pd.Series(y))**2)



regular['1B']=regular['H']-regular['2B']-regular['3B']-regular['HR']
for i in ['avg','1B','2B','3B']:

    get_self_corr(i)
for i in ['HR','BB']:

    get_self_corr(i)
regular['1b_luck']=regular['1B']/(regular['AB']-regular['HR']-regular['SO'])

regular['2b_luck']=regular['2B']/(regular['AB']-regular['HR']-regular['SO'])

regular['3b_luck']=regular['3B']/(regular['AB']-regular['HR']-regular['SO'])
for j in ['avg', 'G', 'AB', 'R', 'H','2B', '3B', 'HR', 'TB', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP','SLG', 'OBP', 'E','1b_luck','2b_luck','3b_luck']:

    lag_1_avg=[]

    for i in range(len(regular)): 

        if len(regular.loc[(regular['batter_name']==regular['batter_name'].iloc[i])&(regular['year']==regular['year'].iloc[i]-1)][j])==0:

            lag_1_avg.append(np.nan)

        else:

            lag_1_avg.append(regular.loc[(regular['batter_name']==regular['batter_name'].iloc[i])&(regular['year']==regular['year'].iloc[i]-1)][j].iloc[0])

    

    regular['lag_1_'+j]=lag_1_avg

    print(j)
def get_nujuk(name,year,var):

    if (len(regular.loc[(regular['batter_name']==name)&(regular['year']<year-1),'H'])!=0):

        return regular.loc[(regular['batter_name']==name)&(regular['year']<year-1),var].sum()

    else:

        return np.nan



for i in ['G', 'AB', 'R', 'H','2B', '3B', 'HR', 'TB', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO']:

    regular['total_'+i]=regular.apply(lambda x: get_nujuk(x['batter_name'],x['year'],i),axis=1)
from sklearn.ensemble import RandomForestRegressor
train=regular.loc[regular['year']<=2017,]

test=regular.loc[regular['year']==2018,]

y_train=train['OPS']

X_train=train[[x for x in regular.columns if ('lag' in x)|('total' in x)]]



y_test=test['OPS']

X_test=test[[x for x in regular.columns if ('lag' in x)|('total' in x)]]
rf=RandomForestRegressor(n_estimators=500)

rf.fit(X_train.fillna(-1),y_train,sample_weight=train['AB'])
pred=rf.predict(X_test.fillna(-1))
real=test['OPS']

ab=test['AB']



from sklearn.metrics import mean_squared_error

mean_squared_error(real,pred,sample_weight=ab)**0.5
train=regular.loc[regular['year']<=2018,]

test=regular.loc[regular['year']==2019,]

y_train=train['OPS']

X_train=train[[x for x in regular.columns if ('lag' in x)|('total' in x)]]





rf=RandomForestRegressor(n_estimators=500)

rf.fit(X_train.fillna(-1),y_train,sample_weight=train['AB'])
test=regular.loc[regular['year']==2019,]
pred=rf.predict(test[[x for x in regular.columns if ('lag' in x)|('total' in x)]].fillna(-1))
pd.DataFrame({'batter_id':test['batter_id'],'OPS':pred}).to_csv("baseline_submission.csv",index=False)
a = pd.read_csv("baseline_submission.csv")

a
x = pd.read_csv("/kaggle/input/Batter.csv")

x
x.loc[:,'OPS'] = a['OPS']

x
x.to_csv("2019_Predict_OPS.csv")