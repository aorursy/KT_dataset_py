# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%matplotlib inline

import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df = pd.concat((df_train,df_test),axis=0)
df.info()
df.head()
sns.scatterplot(x='Electrical',y='SalePrice',data=df)
total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])

missing_data.head(30)
drop_features = list(missing_data.index[missing_data.Percent >= .00136])
drop_features.remove('SalePrice')
# drop columns with too many NaNs

df_cleaned = df.drop(columns=drop_features)
df_cleaned.Utilities.value_counts()
missing_rows_idx = df_cleaned.drop('SalePrice',axis=1).isnull().any(axis=1)
# if there is no Bsmt values then most like there is no basement at all

df_cleaned.loc[np.logical_and(df_cleaned.Id>1460,missing_rows_idx)] = df_cleaned.loc[np.logical_and(df_cleaned.Id>1460,missing_rows_idx)].fillna({"BsmtFinSF1":0,"BsmtFinSF2":0,"BsmtFullBath":0,"BsmtHalfBath":0,"BsmtUnfSF":0,"TotalBsmtSF":0})

# Basically all the Utilities are AllPub

df_cleaned.loc[np.logical_and(df_cleaned.Id>1460,missing_rows_idx)] = df_cleaned.loc[np.logical_and(df_cleaned.Id>1460,missing_rows_idx)].fillna({"Utilities":"AllPub"})

# Garage most likely to be zero if missing in report...

df_cleaned.loc[np.logical_and(df_cleaned.Id>1460,missing_rows_idx)] = df_cleaned.loc[np.logical_and(df_cleaned.Id>1460,missing_rows_idx)].fillna({"GarageArea":0,"GarageCars":0})

# Houses in the neighborhood Sawyer saletype are probably WD

df_cleaned.loc[np.logical_and(df_cleaned.Id>1460,missing_rows_idx)] = df_cleaned.loc[np.logical_and(df_cleaned.Id>1460,missing_rows_idx)].fillna({"SaleType":"WD"})

# Just go with typical functional

df_cleaned.loc[np.logical_and(df_cleaned.Id>1460,missing_rows_idx)] = df_cleaned.loc[np.logical_and(df_cleaned.Id>1460,missing_rows_idx)].fillna({"Functional":"Typ"})

# When the ExterCond and ExterQual are both TA, these are most likely values for that neighboord Edwards

df_cleaned.loc[np.logical_and(df_cleaned.Id>1460,missing_rows_idx)] = df_cleaned.loc[np.logical_and(df_cleaned.Id>1460,missing_rows_idx)].fillna({"Exterior1st":"Wd Sdng","Exterior2nd":"Wd Sdng"})

# Just go with typical 

df_cleaned.loc[np.logical_and(df_cleaned.Id>1460,missing_rows_idx)] = df_cleaned.loc[np.logical_and(df_cleaned.Id>1460,missing_rows_idx)].fillna({"KitchenQual":"TA"})
# Houses in the neighborhood Sawyer are probably WD...

df_cleaned[df_cleaned.loc[:,"Neighborhood"] == "Sawyer"].SaleType.value_counts()
df_cleaned.KitchenQual.value_counts()
df_cleaned.loc[np.logical_and(df_cleaned.Id>1460,missing_rows_idx)]
df_cleaned.loc[np.logical_and(np.logical_and(df_cleaned.ExterCond=="TA",df_cleaned.ExterQual=="TA"),df_cleaned.Neighborhood=="Edwards")].Exterior1st.value_counts()

df_cleaned.loc[np.logical_and(df_cleaned.Exterior1st=="Wd Sdng",np.logical_and(np.logical_and(df_cleaned.ExterCond=="TA",df_cleaned.ExterQual=="TA"),df_cleaned.Neighborhood=="Edwards"))].Exterior2nd.value_counts()
df_cleaned.loc[np.logical_and(df_cleaned.Id>1460,missing_rows_idx)]
df_cleaned[np.logical_and(df_cleaned.Id>1460,missing_rows_idx)].info()
# drop the few rows with NaNs

df_cleaned = df_cleaned.dropna(subset=[col for col in df_cleaned.columns if col != 'SalePrice'],how='any')
df_cleaned = pd.get_dummies(df_cleaned)
# too many plots to do pairwise, so just do individual plts one at a time

for name in df.columns:

    plt.figure

    sns.scatterplot(x=name,y='SalePrice',data=df)

    plt.xticks(rotation=90)

    plt.show()
plt.figure(figsize=(15,10));

sns.heatmap(data=df_train.corr(),vmin=-1,vmax=1,linewidths=.3,cmap='jet',square=True);
# check if pearson correlation is above .8

plt.figure(figsize=(15,10));

sns.heatmap(data=abs(df_train.corr())>.50,vmin=0,vmax=1,linewidths=.3,cmap='YlGnBu',square=True);
df_cleaned['TotalFlrSF'] = df_cleaned['1stFlrSF'] + df_cleaned['2ndFlrSF'] + df_cleaned['TotalBsmtSF']

df_cleaned['Total_Bathrooms'] = (df_cleaned['FullBath'] + (0.5*df_cleaned['HalfBath']) + 

                               df_cleaned['BsmtFullBath'] + (0.5*df_cleaned['BsmtHalfBath']))



df_cleaned['Total_porch_sf'] = (df_cleaned['OpenPorchSF'] + df_cleaned['3SsnPorch'] +

                              df_cleaned['EnclosedPorch'] + df_cleaned['ScreenPorch'] +

                             df_cleaned['WoodDeckSF'])
X_train_cleaned = df_cleaned[df_cleaned.Id <= 1460]

X_test_cleaned = df_cleaned[df_cleaned.Id > 1460]
X_test_Ids = X_test_cleaned.pop('Id')

X_test_cleaned.pop('SalePrice');

X_train_Ids = X_train_cleaned.pop('Id')
y_train_cleaned = X_train_cleaned.pop('SalePrice')
from scipy import stats

X_trained_cleaned = X_train_cleaned[(np.abs(stats.zscore(X_train_cleaned)) < 10).all(axis=1)]
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import mutual_info_regression

kbest = SelectKBest(mutual_info_regression,k=int(np.floor(len(X_train_cleaned.columns)/4))).fit(X_train_cleaned,y_train_cleaned)

kbest_idx = kbest.get_support(indices=True)
X_train = X_train_cleaned.iloc[:,kbest_idx]

X_test = X_test_cleaned.iloc[:,kbest_idx]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor 



from sklearn.metrics import r2_score
rff = RandomForestRegressor()
params = {

    "n_estimators":[10,100,500,1000,3000],

    "max_depth":[10,50,100,None],

    "min_samples_split":[2,5,10],

    "max_features":["auto","sqrt","log2",None]

}

gs = RandomizedSearchCV(rff,param_distributions=params,n_iter=100,scoring='neg_mean_squared_error')
gs.fit(X_train_scaled,y_train_cleaned)
# Utility function to report best scores

def report(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")
report(gs.cv_results_)
y_pred_train = gs.best_estimator_.predict(X_train_scaled)

sns.scatterplot(x=y_pred_train,y=y_train_cleaned)

print("R2: ",r2_score(y_train_cleaned,y_pred_train))
from sklearn.ensemble import GradientBoostingRegressor
gbb = GradientBoostingRegressor()
params = {

    "loss":['ls','lad','huber','quantile'],

    "learning_rate":[.1,.01,.005,.0005],

    "n_estimators":[100,500,3000],

    "min_samples_split":[2,5,10,20],

    "max_features":["auto","sqrt","log2",None]

}



gs2 = RandomizedSearchCV(gbb,param_distributions=params,n_iter=100,scoring='neg_mean_squared_error')

gs2.fit(X_train_scaled,y_train_cleaned)
report(gs2.cv_results_)
y_pred_train = gs2.best_estimator_.predict(X_train_scaled)

sns.scatterplot(x=y_pred_train,y=y_train_cleaned)

print("R2: ",r2_score(y_train_cleaned,y_pred_train))
from sklearn.linear_model import Lasso

params = {

    "alpha":[.1,.2,.3,.4,.5,.6,.7,.8,.9,1]

}

lasso = Lasso()

gs3 = RandomizedSearchCV(lasso,param_distributions=params,n_iter=100,scoring='neg_mean_squared_error',cv=5)

gs3.fit(X_train_scaled,y_train_cleaned)
y_pred_train = gs3.best_estimator_.predict(X_train_scaled)

sns.scatterplot(x=y_pred_train,y=y_train_cleaned)

print("R2: ",r2_score(y_train_cleaned,y_pred_train))
y_pred =.2*gs.best_estimator_.predict(X_test_scaled) + .6*gs2.best_estimator_.predict(X_test_scaled) + .2 * gs3.best_estimator_.predict(X_test_scaled)

submission = pd.DataFrame({"Id":X_test_Ids,"SalePrice":y_pred})

submission.to_csv("submission1",index=False)