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



df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')

df.head()
df.isnull().sum()
df.drop(['County'], axis = 1, inplace = True)

df.drop(['Province_State'], axis = 1, inplace = True)

df.info()
df['Date'] = pd.to_datetime(df['Date']).dt.strftime("%Y%m%d")
test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')
test_df.head()
test_df.isnull().sum()
test_df.drop(['County'], axis = 1, inplace = True)

test_df.drop(['Province_State'], axis = 1, inplace = True)
test_df.info()
test_df['Date'] = pd.to_datetime(test_df['Date']).dt.strftime("%Y%m%d")
main_df = df.copy()
final_df = pd.concat([df, test_df], axis = 0)
final_df['TargetValue']
columns = ['Country_Region','Target']
def category_onehot_multcols(multcolumns):

    df_final=final_df

    i=0

    for fields in multcolumns:

        print(fields)

        df1=pd.get_dummies(final_df[fields],drop_first=True)

        final_df.drop([fields],axis=1,inplace=True)

        if i==0:

            df_final=df1.copy()

        else:

            df_final=pd.concat([df_final,df1],axis=1)

        i+=1

    df_final=pd.concat([final_df,df_final],axis=1)

    return df_final
final_df=category_onehot_multcols(columns)
final_df.shape
final_df.head()
final_df =final_df.loc[:,~final_df.columns.duplicated()]

final_df.drop(['Id','ForecastId'],axis = 1, inplace = True)

df_Train=final_df.iloc[:734156,:]

df_Test=final_df.iloc[734156:,:]
df_Train.tail()
df_Test.head()
df_Test.drop(['TargetValue'],axis=1,inplace=True)
X_train=df_Train.drop(['TargetValue'],axis=1)

y_train=df_Train['TargetValue']
X_train.shape
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse',

                                  max_depth=None, max_features='sqrt', max_leaf_nodes=None,

                                  max_samples=None, min_impurity_decrease=0.0,

                                  min_impurity_split=None, min_samples_leaf=1,

                                  min_samples_split=2, min_weight_fraction_leaf=0.0,

                                  n_estimators=400, n_jobs=None, oob_score=False,

                                  random_state=None, verbose=0, warm_start=False)
regressor.fit(X_train, y_train)
X_test = df_Test.values
X_test.shape
y_pred = regressor.predict(X_test)
len(y_pred)
predict = [int(x) for x in y_pred]



output = pd.DataFrame({'Id': test_df.index, 'TargetValue': predict})

output
a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()

b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()

c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()
a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']

a=pd.concat([a,b['q0.5'],c['q0.95']],1)

a['q0.05']=a['q0.05'].clip(0,10000)

a['q0.5']=a['q0.5'].clip(0,10000)

a['q0.95']=a['q0.95'].clip(0,10000)

a
a['Id'] =a['Id']+ 1

a
sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.head()