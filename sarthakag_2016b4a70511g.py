import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.preprocessing import MinMaxScaler
dftest = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')

# dftest.head()
df = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

df.head(5)
def enc(x):

    if x == "new":

        return 0;

    elif x =="old":

        return 1;

    else:

        return x

# df.dtypes

df['type'] = df['type'].apply(enc);

# df.head(10)

dftest['type'] = dftest['type'].apply(enc);
{np.all(np.isfinite(df)),np.any(np.isnan(df))}
{np.all(np.isfinite(dftest)),np.any(np.isnan(dftest))}
df.replace([np.inf, -np.inf], np.nan);

df.replace(np.nan, 0, inplace=True);

dftest.replace([np.inf, -np.inf], np.nan);

dftest.replace(np.nan, 0, inplace=True);

# can do df.mean(
{np.all(np.isfinite(df)),np.any(np.isnan(df))}
{np.all(np.isfinite(dftest)),np.any(np.isnan(dftest))}
df.drop_duplicates(keep=False,inplace=True) 
df.head()

X = df.iloc[:,1:-1]

# X.head()

Xtest =dftest.iloc[:,1:];

Xtest.head()
Y  = df.iloc[:,-1:]

# Y.head()
# regressor = RandomForestRegressor(n_estimators=2000,random_state = 0)

regressor = ExtraTreesRegressor(n_estimators=2000,random_state = 0)

# regressor = RandomForestRegressor(n_estimators=10)

regressor.fit(X,Y.values.ravel())
ylol = regressor.predict(X)
ylol = ylol.round().astype(int)

s = Y.values;

count = 0
for i in range(len(ylol)):

    if(ylol[i] == s[i]):

        count = count+1;
count/len(ylol)
regressor.score(X,Y)
Ypred = regressor.predict(Xtest);
Ypred
Ypred = Ypred.round().astype(int)
Ypred
dftemp = dftest['id']
dftemp.columns ={"id"};

dftemp.head()
dftemp2= pd.DataFrame(Ypred)

dftemp2.columns = {"rating"};

dftemp2.head()
dftemp2.replace([np.inf, -np.inf], np.nan);

dftemp2.replace(np.nan, 1, inplace=True);

{np.all(np.isfinite(dftemp2)),np.any(np.isnan(dftemp2))}
df_final = pd.concat([dftemp,dftemp2.astype(int)],axis = 1)
df_final.head()
df_final.to_csv("ninth_submission.csv",index = False)