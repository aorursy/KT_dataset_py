# Import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="darkgrid")

%matplotlib inline  



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_test = pd.read_csv('../input/test.csv')

df_train = pd.read_csv('../input/train.csv')

df_train.head(10)

df_train.describe()
df_train.info()
#ax = sns.countplot(hue=df_train.columns.values,y=df_train.isnull().sum())

ax,fig=plt.subplots(figsize = (6,10))

plt.rcdefaults()

mask=df_train.columns[df_train.isnull().any()]

hasnulls=df_train[mask].isnull().sum()

y_pos = np.arange(len(hasnulls))

#no_nans = df_train.isnull().sum()

ax=plt.barh(y_pos,hasnulls,tick_label=df_train[mask].isnull().columns)

plt.title('Number of NaNs per feature')

plt.xlabel('Count of NaNs')

plt.tight_layout()

#ax.yticks(y_pos)

#ax.set_yticklabels(df_train.columns.values)
corr=df_train.corr()["SalePrice"]

corr[np.argsort(corr, axis=0)[::-1]]
corr=df_train.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(10, 8))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
fig=sns.regplot(x='OverallQual',y='SalePrice', order=3,data=df_train,)
fig=sns.swarmplot(x='GarageCars',y='GarageArea',data=df_train)
sns.distplot(df_train.MoSold)
sns.boxplot(x='YearBuilt',y='HouseStyle',data=df_train)
sns.boxplot(x='GarageCars',y='YearBuilt',data=df_train)
# First join the dataframes of train and test. The goal variable is automatically dismissed

Y=df_train.SalePrice

x_all = pd.concat([df_train, df_test], join="inner")

#len(x_all) 2919

# lengths of train and test = 1460 + 1459



# Removed because of many NaN values

x_all_sel = x_all.drop(['MiscFeature', 'Fence', 'PoolQC', 'FireplaceQu', 'Alley', 'LotFrontage'], axis=1)



# Removed because of low correlation with target variable

mask = np.abs(df_train.corr()["SalePrice"])<0.1



x_all_sel.drop(mask[mask==True].index, axis=1, errors='ignore', inplace=True)

x_all_sel.info()
tmp=pd.DataFrame()

for cn in x_all_sel.select_dtypes(['object']).columns:

    #tmpdict.update({cn:x_all_sel[cn].unique().values})

    tmp[cn]=[x_all_sel[cn].unique(),x_all_sel[cn].nunique()]

tmp=tmp.transpose()

tmp.columns=['Unique values per feature','Number']

tmp.sort_values('Number', ascending=False)
tmp=x_all_sel.select_dtypes(['object'])

tmp=tmp.columns[tmp.isnull().any()]

x_all_sel[tmp].isnull().sum()
x_all_sel.MSZoning.fillna(x_all_sel.MSZoning.mode()[0], inplace=True)

x_all_sel.Utilities.fillna(x_all_sel.Utilities.mode()[0], inplace=True)

x_all_sel.Exterior1st.fillna(x_all_sel.Exterior1st.mode()[0], inplace=True)

x_all_sel.Exterior2nd.fillna(x_all_sel.Exterior2nd.mode()[0], inplace=True)

x_all_sel.MasVnrType.fillna(x_all_sel.MasVnrType.mode()[0], inplace=True)

x_all_sel.Electrical.fillna(x_all_sel.Electrical.mode()[0], inplace=True)

x_all_sel.KitchenQual.fillna(x_all_sel.KitchenQual.mode()[0], inplace=True)

x_all_sel.Functional.fillna(x_all_sel.Functional.mode()[0], inplace=True)

x_all_sel.SaleType.fillna(x_all_sel.SaleType.mode()[0], inplace=True)
x_all_sel[(x_all_sel.GarageType.isnull() == False) & (x_all_sel.GarageFinish.isnull() == True)].GarageType
x_all_sel.loc[(x_all_sel.GarageType.isnull() == False) & (x_all_sel.GarageFinish.isnull() == True), 'GarageType'] = 'None'

x_all_sel.GarageType.fillna('None', inplace=True)

x_all_sel.GarageCond.fillna('None', inplace=True)

x_all_sel.GarageQual.fillna('None', inplace=True)

x_all_sel.GarageFinish.fillna('None', inplace=True)
Bsmt_columns = ['BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

#x_all_sel.loc[(x_all_sel[Bsmt_columns].isnull()==True),Bsmt_columns]

x_all_sel[Bsmt_columns][x_all_sel[Bsmt_columns].isnull().any(axis=1)==True].shape

#x_all_sel[Bsmt_columns].isnull().any(axis=1)==True
x_all_sel.loc[(x_all_sel[Bsmt_columns].isnull().all(axis=1)==True),Bsmt_columns] = 'None'

x_all_sel[Bsmt_columns][x_all_sel[Bsmt_columns].isnull().any(axis=1)==True]

#x_all_sel[Bsmt_columns][x_all_sel[Bsmt_columns].isnull().any(axis=1)==True]
for col in Bsmt_columns:

    x_all_sel[col].fillna(x_all_sel[col].mode()[0],inplace=True)
tmp=x_all_sel.select_dtypes(['number'])

tmp=tmp.columns[tmp.isnull().any()]

x_all_sel[tmp].isnull().sum()
x_all_sel.fillna(0,inplace=True)

#for col in tmp:

    #x_all_sel[col].fillna(x_all_sel[col].mean(),inplace=True)
fig=sns.boxplot(x='Neighborhood',y='SalePrice',data=df_train)

for item in fig.get_xticklabels():

    item.set_rotation(60)
excl_score=(x_all_sel.Neighborhood.value_counts()/x_all_sel.Neighborhood.count()).to_dict()

x_all_sel.Neighborhood.replace(excl_score, inplace=True)
#import category_encoders as ce

#encoder = ce.OneHotEncoder(x_all_sel.select_dtypes(['object']).columns)

#df_onehot = encoder.fit_transform(x_all_sel)

df_onehot = pd.get_dummies(x_all_sel)
sns.distplot(Y,rug=True)
df_train[Y>500000]
mask = Y>500000

X=df_onehot[0:1460].copy()

test=df_onehot[1460:]

X.drop(X[mask==True].index,inplace=True)

Y.drop(Y[mask==True].index,inplace=True)
sns.distplot(np.log(Y))

Y=np.log(Y)
from sklearn import tree

from sklearn.model_selection import cross_val_score



regressor = tree.DecisionTreeRegressor(random_state=0,criterion='friedman_mse',max_depth=7,min_samples_split=2)

cross_val_score(regressor, X, Y, cv=10).mean()

#clf.fit(X,Y)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100,verbose=True,n_jobs=2)

cross_val_score(rf,X,Y,cv=10).mean()
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=500,max_depth=4,min_samples_split=2,learning_rate=0.01,loss='ls',verbose=True)

cross_val_score(gbr,X,Y,cv=8).mean()
gbr.fit(X,Y)

pred = gbr.predict(test)
predictions = np.exp(pred)

predictions

df_pred = pd.DataFrame(data={'Id':np.arange(1461,2920),'SalePrice':predictions})

#,index=range(1461,2920),columns=['SalePrice'])

df_pred.to_csv('submission.csv',index=False)
