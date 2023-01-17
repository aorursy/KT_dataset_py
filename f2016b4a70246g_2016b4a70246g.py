import pandas as pd

import numpy as np

import seaborn as sns

df = pd.read_csv('train.csv')

test_df = pd.read_csv('test.csv')
df['type'].replace("new",0,inplace=True)

df['type'].replace("old",1,inplace=True)

df.fillna(value=df.mean(),inplace=True)
X=df.loc[:,'feature1':'feature11']

y=df.loc[:,['rating']]
from sklearn.ensemble import ExtraTreesRegressor
y_1=np.array(y['rating'])

reg=ExtraTreesRegressor(n_estimators=1947)

reg.fit(X,y_1)

test_df['type'].replace("new",0,inplace=True)

test_df['type'].replace("old",1,inplace=True)

test_df.fillna(value=test_df.mean(),inplace=True)

test_X = test_df.loc[:,'feature1':'feature11']
predict_out = reg.predict(test_X)

out = [[test_df['id'][i],int(round(predict_out[i]))] for i in range(len(predict_out))]
out_df = pd.DataFrame(data=out,columns=['id','rating'])

out_df.to_csv(r'out_6_2.csv',index=False)
import pandas as pd

import numpy as np

import seaborn as sns

df = pd.read_csv('train.csv')

test_df = pd.read_csv('test.csv')
df['type'].replace("new",0,inplace=True)

df['type'].replace("old",1,inplace=True)

df.fillna(value=0,inplace=True)
X=df.loc[:,'feature1':'feature11']

y=df.loc[:,['rating']]
from sklearn.ensemble import ExtraTreesRegressor
y_1=np.array(y['rating'])

reg=ExtraTreesRegressor(n_estimators=2000,random_state=0)

reg.fit(X,y_1)

test_df['type'].replace("new",0,inplace=True)

test_df['type'].replace("old",1,inplace=True)

test_df.fillna(value=0,inplace=True)

test_X = test_df.loc[:,'feature1':'feature11']
predict_out = reg.predict(test_X)

out = [[test_df['id'][i],int(round(predict_out[i]))] for i in range(len(predict_out))]
out_df = pd.DataFrame(data=out,columns=['id','rating'])

out_df.to_csv(r'out_6_1.csv',index=False)