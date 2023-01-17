# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
train= pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test= pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.head()

test.head()
print('Dimensions of train data:', train.shape)

print('Dimensions of test data:', test.shape)
#labels = train.pop('SalePrice') 

#train.shape

null= np.array(train.isnull().sum())

print(null)
train.describe()
train.isnull().sum()

dataframe=[train,test]

data= pd.concat(dataframe)
data.isnull().sum()
null= np.array(data.isnull().sum())

plt.hist(null)
corr= train.corr()

f, ax = plt.subplots(figsize=(15, 10))

sns.heatmap(corr,linewidths=.5, vmin=0, vmax=1,)
data['WoodDeckSF'].value_counts()
data= data.drop([ 'MiscFeature', 'GarageArea'],axis=1)
for i in data.columns:

    print( data[i].isnull().sum())
data._get_numeric_data().columns
NaN_columns= data.columns[data.isnull().any()].values

print(NaN_columns)
listNumeric= ['BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','GarageCars','GarageYrBlt','LotFrontage'

             ,'TotalBsmtSF','MasVnrArea']



for m in listNumeric:

    data[m].fillna(data[m].mode()[0], inplace= True)
listNA= ['Alley', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageFinish'

         , 'GarageQual', 'GarageType', 'PoolQC']

for j in listNA:

    data[j].fillna('None', inplace= True)
listMODE= ['Electrical','Exterior1st','Exterior2nd','Functional','KitchenQual','MSZoning','SaleType','Utilities']



for k in listMODE:

    data[k].fillna(data[k].mode()[0], inplace= True)
m=0

for i in data.columns:

    print( data[i].isnull().sum())

    if (data[i].isnull().sum()>0):

        m=m+1

print(m)

plt.hist(data['LotFrontage'])
data.info()
data['MasVnrType'].fillna('None', inplace=True)
data.isnull().sum()
data.info()
s=np.array(data._get_numeric_data().columns)

print(s)

df_merged_cat = data.select_dtypes(include = ['object']).astype('category')





df_merged_cat.LotShape.replace(to_replace = ['IR3', 'IR2', 'IR1', 'Reg'], value = [0, 1, 2, 3], inplace = True)

df_merged_cat.LandContour.replace(to_replace = ['Low', 'Bnk', 'HLS', 'Lvl'], value = [0, 1, 2, 3], inplace = True)

df_merged_cat.Utilities.replace(to_replace = ['NoSeWa', 'AllPub'], value = [0, 1], inplace = True)

df_merged_cat.LandSlope.replace(to_replace = ['Sev', 'Mod', 'Gtl'], value = [0, 1, 2], inplace = True)

df_merged_cat.ExterQual.replace(to_replace = ['Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3], inplace = True)

df_merged_cat.ExterCond.replace(to_replace = ['Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)

df_merged_cat.BsmtQual.replace(to_replace = ['None', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)

df_merged_cat.BsmtCond.replace(to_replace = ['None', 'Po', 'Fa', 'TA', 'Gd'], value = [0, 1, 2, 3, 4], inplace = True)

df_merged_cat.BsmtExposure.replace(to_replace = ['None', 'No', 'Mn', 'Av', 'Gd'], value = [0, 1, 2, 3, 4], inplace = True)

df_merged_cat.BsmtFinType1.replace(to_replace = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)

df_merged_cat.BsmtFinType2.replace(to_replace = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)

df_merged_cat.HeatingQC.replace(to_replace = ['Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)

df_merged_cat.Electrical.replace(to_replace = ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'], value = [0, 1, 2, 3, 4], inplace = True)

df_merged_cat.KitchenQual.replace(to_replace = ['Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3], inplace = True)

df_merged_cat.Functional.replace(to_replace = ['Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)

df_merged_cat.FireplaceQu.replace(to_replace =  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)

df_merged_cat.GarageFinish.replace(to_replace =  ['None', 'Unf', 'RFn', 'Fin'], value = [0, 1, 2, 3], inplace = True)

df_merged_cat.GarageQual.replace(to_replace =  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)

df_merged_cat.GarageCond.replace(to_replace =  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)

df_merged_cat.PavedDrive.replace(to_replace =  ['N', 'P', 'Y'], value = [0, 1, 2], inplace = True)

df_merged_cat.PoolQC.replace(to_replace =  ['None', 'Fa', 'Gd', 'Ex'], value = [0, 1, 2, 3], inplace = True)

df_merged_cat.Fence.replace(to_replace =  ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'], value = [0, 1, 2, 3, 4], inplace = True)
data=pd.get_dummies(data)

data.shape
train= data[:1460]

test= data[1460:]
SPtrain= train.pop('SalePrice')

SPtest= test.pop('SalePrice')
print(train.shape)

print(train.shape)
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor as GBR
"""gsc= GridSearchCV(estimator= GBR(), 

                  param_grid= {'max_depth': range(8,11), 

                              'n_estimators': (150,100),

                               'subsample': (0.6,0.9)

                              }, cv=7, scoring= 'neg_mean_squared_error'

                 , verbose=0, n_jobs= -1)

grid_result= gsc.fit(train, SPtrain)

best_params= grid_result.best_params_"""
gbr= GBR(max_depth=10, n_estimators= 100, random_state=1, subsample= 0.6)
gbr.fit(train, SPtrain)
predictions= gbr.predict(test)

Submission= pd.DataFrame()

Submission['SalePrice']= predictions

Submission['Id']= test['Id']

Submission.head()

Output2 = Submission.to_csv('Output2.csv', index= False)

Submission.to_csv('Output2.csv', index=False)
from IPython.display import HTML

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

#df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))



# create a link to download the dataframe

create_download_link(Submission)



# ↓ ↓ ↓  Yay, download link!
sns.distplot(SPtrain, kde= True, bins= 35)



