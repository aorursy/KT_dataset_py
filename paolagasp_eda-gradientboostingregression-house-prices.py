# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#libraries for visualizations

import matplotlib.pyplot as plt

import seaborn as sns



color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999







sns.set_style("white")

import scipy.stats as stats



import numpy as np

from sklearn.impute import SimpleImputer





from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import OneHotEncoder



from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor





from sklearn.metrics import mean_squared_error



import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory











import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



trainset=pd.read_csv('../input/train.csv')

trainset.shape

trainset.info()
trainset.head()
testset=pd.read_csv('../input/test.csv')

testset.shape



testset.head()
trainset.describe()

testset.describe()
trainset_cat = trainset.select_dtypes(include=['object']).copy()



trainset_cat.head()



lista=list(trainset_cat)

print(lista)





testset_cat = testset.select_dtypes(include=['object']).copy()



testset_cat.head()



lista_test=list(testset_cat)

print(lista_test)



#trainset_onehot =trainset.copy()

#trainset_onehot_f = pd.get_dummies(trainset_onehot, columns=['Neighborhood'], prefix = ['Neighborhood'])



#print(trainset_onehot_f.head())
for name in lista:

 trainset[name]=trainset_cat[name].astype('category').cat.codes





for nametest in lista_test:

 testset[nametest]=testset_cat[nametest].astype('category').cat.codes
trainset.head()
print(trainset['SalePrice'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(trainset['SalePrice'], color='c', bins=100, hist_kws={'alpha': 0.4});
trainset_c=trainset[trainset.SalePrice<500000]

print(trainset_c['SalePrice'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(trainset_c['SalePrice'], color='c', bins=100, hist_kws={'alpha': 0.4});
labels = []

values = []

for col in trainset_c.columns:

    if col not in ["Id", "SalePrice"] and trainset_c[col].dtype!='object':

        labels.append(col)

        values.append(np.corrcoef(trainset_c[col].values, trainset_c["SalePrice"].values)[0,1])

corr_df = pd.DataFrame({'columns_labels':labels, 'corr_values':values})

corr_df = corr_df.sort_values(by='corr_values')

 

corr_df = corr_df[(corr_df['corr_values']>0.25) | (corr_df['corr_values']<-0.25)]

ind = np.arange(corr_df.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(10,6))

rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='gold')

ax.set_yticks(ind)

ax.set_yticklabels(corr_df.columns_labels.values, rotation='horizontal')

ax.set_xlabel("Correlation coefficient")

ax.set_title("Correlation coefficient of the variables")

plt.show()
missingdata = trainset_c.isnull().sum(axis=0).reset_index()

missingdata.columns = ['column_name', 'missing_count']

missingdata = missingdata.ix[missingdata['missing_count']>0]

missingdata = missingdata.sort_values(by='missing_count')



ind = np.arange(missingdata.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(12,18))

rects = ax.barh(ind, missingdata.missing_count.values, color='blue')

ax.set_yticks(ind)

ax.set_yticklabels(missingdata.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show()
temp_df = trainset_c[corr_df.columns_labels.tolist()]

corrmat = temp_df.corr(method='pearson')

f, ax = plt.subplots(figsize=(12, 12))



# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=1., square=True, cmap="YlOrRd")

plt.title("Correlation Matrix", fontsize=15)

plt.show()
trainset_c['OverallQual'].loc[trainset_c['OverallQual']>7] = 7

plt.figure(figsize=(12,8))

sns.violinplot(x='OverallQual', y='SalePrice', data=trainset_c)

plt.xlabel('Overall Quality', fontsize=12)

plt.ylabel('SalePrice', fontsize=12)

plt.show()
col = "GrLivArea"

ulimit = np.percentile(trainset_c[col].values, 99.5)

llimit = np.percentile(trainset_c[col].values, 0.5)

trainset_c[col].loc[trainset_c[col]>ulimit] = ulimit

trainset_c[col].loc[trainset_c[col]<llimit] = llimit



plt.figure(figsize=(12,12))

sns.jointplot(x=trainset_c[col].values, y=trainset_c.SalePrice.values, height=10, color=color[4])

plt.ylabel('SalePrice', fontsize=12)

plt.xlabel('GrLivArea', fontsize=12)

plt.title("GrLivArea Vs SalePrice", fontsize=15)

plt.show()
trainset_c.fillna( method ='ffill', inplace = True)



testset.fillna( method ='ffill', inplace = True)



trainset_y = trainset_c.SalePrice

x_col = ['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt','YearRemodAdd','TotRmsAbvGrd','Fireplaces','Foundation','BsmtFinSF1','OpenPorchSF','WoodDeckSF','GarageCond','2ndFlrSF','HalfBath','LotArea','LotShape','GarageFinish','HeatingQC','BsmtQual','KitchenQual','ExterQual']



trainset_x = trainset_c[x_col]







my_model = GradientBoostingRegressor()

my_model.fit(trainset_x, trainset_y)

testset_X = testset[x_col]

testset_X.head()

# Use the model to make predictions

predicted_prices = my_model.predict(testset_X)

# We will look at the predicted prices to ensure we have something sensible.

print(predicted_prices)
my_submission = pd.DataFrame({'Id': testset.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)