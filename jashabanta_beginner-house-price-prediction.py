# importing necesssary packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from scipy.stats import norm



from sklearn.model_selection import cross_val_score ,train_test_split

from sklearn.metrics import mean_squared_error , accuracy_score,r2_score

from sklearn.linear_model import LinearRegression

from lightgbm import LGBMRegressor



import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.head()
data.shape
((data.isnull().sum()/data.shape[0])*100).sort_values(ascending = False)[:20]
list(((data.isnull().sum()/data.shape[0])*100).sort_values(ascending = False).index[:20])
data.drop(columns=['PoolQC',

'MiscFeature',

'Alley',

'Fence',

'FireplaceQu',

'LotFrontage'],inplace = True)
(((data.isnull().sum()/data.shape[0])*100).sort_values(ascending = False)[:20] !=0).value_counts()
list(((data.isnull().sum()/data.shape[0])*100).sort_values(ascending = False).index[:13])
data.loc[:,['GarageType',

 'GarageYrBlt',

 'GarageFinish',

 'GarageCond',

 'GarageQual',

 'BsmtExposure',

 'BsmtFinType2',

 'BsmtFinType1',

 'BsmtCond',

 'BsmtQual',

 'MasVnrType',

 'MasVnrArea',

 'Electrical']].info()
# counting the object datatype columns and numeric datatype columns and selecting that



data1=data.loc[:,['GarageType',

 'GarageYrBlt',

 'GarageFinish',

 'GarageCond',

 'GarageQual',

 'BsmtExposure',

 'BsmtFinType2',

 'BsmtFinType1',

 'BsmtCond',

 'BsmtQual',

 'MasVnrType',

 'MasVnrArea',

 'Electrical']]

data1.dtypes.value_counts()
data1.dtypes[:].sort_values(ascending = False)
list(data1.dtypes[:].sort_values(ascending = False).index)
# filling numeric columns according to the given data



data.MasVnrArea.fillna(value = np.mean(data.MasVnrArea),inplace = True)

data.GarageYrBlt.fillna(method = 'ffill',inplace = True)
# Selected the column names for objective data types



obj_type_list=['Electrical',

 'MasVnrType',

 'BsmtQual',

 'BsmtCond',

 'BsmtFinType1',

 'BsmtFinType2',

 'BsmtExposure',

 'GarageQual',

 'GarageCond',

 'GarageFinish',

 'GarageType']
# counting and replceing the caterogical columns data according to max count



replace_list=[]

for i in obj_type_list:

    print(i)

    print(data[i].value_counts())

    replace_list.append(data[i].value_counts().index[0])

    print()

    

print(replace_list)
for i in range(len(obj_type_list)):

    data[obj_type_list[i]].replace(to_replace = np.nan, value = replace_list[i] ,inplace = True )
data.dtypes.sort_values(ascending=False).value_counts()
new_obj_data=data.select_dtypes(include='object')

new_obj_data.dtypes.value_counts()
# selecting the object column index and putting them into a list



enc_obj = list(data.dtypes.sort_values(ascending=False).index[:38])
# transform the data from string to numeric



enc = LabelEncoder()

for i in enc_obj:

    data[i]=enc.fit_transform(data[i])
data_corr = data.corr()

abs(data_corr['SalePrice'].sort_values(ascending = False))[:20]
model_data = data.loc[:,['OverallQual',

'GrLivArea',    

'GarageArea',   

'TotalBsmtSF',            

'FullBath',      

'YearBuilt',

'YearRemodAdd',

'SalePrice']]
plt.figure(figsize=(14,7))

sns.heatmap(model_data.corr(),annot=True)
# usinng scatterplot to remove outliers



norm_plot=list(model_data.columns[:-1])

for i in norm_plot:

    plt.figure(figsize=(10,5))

    print(sns.scatterplot(x=model_data[i],y=model_data.SalePrice))
# Selectong the row numbers having outliers and putting them into a list



drop_index_row=(list(model_data.loc[model_data.TotalBsmtSF > 3000,'TotalBsmtSF'].index)

                +list(model_data.loc[model_data.GrLivArea > 4000,'GrLivArea'].index)

                +list(model_data.loc[model_data.GarageArea > 1200,'GarageArea'].index))

drop_index_row=pd.Series(data=drop_index_row).unique()
# Droping the rows having outliers 



for i in drop_index_row:

    model_data.drop(index=i,inplace = True,axis = 0)
# ploting the datd to see the normal distribution



norm_plot=list(model_data.columns)

for i in norm_plot:

    plt.figure(figsize=(10,5))

    print(sns.distplot(a=model_data[i],kde=True,fit=norm))
# Normalze the data to treat skew-ness and kurtosis



model_data1 = model_data

model_data1 = np.log1p(model_data1)
for i in norm_plot:

    model_data1.loc[np.isinf(model_data1[i]) != False,i] = np.mean(model_data1[i])


for i in norm_plot:    

    plt.figure(figsize=(15,5))

    plt.subplot(121)

    sns.distplot(a=model_data[i],fit=norm)

    plt.subplot(122)

    sns.distplot(a=model_data1[i],fit=norm)
model_data1.head()
# Spliting into 70% train and 30% test



X=model_data1.iloc[:,:-1]

y=model_data1.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
# Define model into object



model = LGBMRegressor()
model.fit(X_train,y_train)

ypred=model.predict(X_test)
r2_score(y_test,ypred)
model_score = cross_val_score(LGBMRegressor(),X,y=y,cv=5,verbose=1)
model_score.mean()
model_reg = cross_val_score(LGBMRegressor(boosting_type='gbdt',

                                           gamma=0.081334,

                                           learning_rate=0.120667,

                                           num_estimators=290,

                                           max_depth=2),X=X,y=y,cv=5)



print(model_reg.mean())