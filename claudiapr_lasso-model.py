import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
sample=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train_copy=train.copy()

test_copy=test.copy()
numeric=train.select_dtypes(include=[np.number]) #selecting the data asociate with the numerical variables
categorical=train.select_dtypes(exclude=[np.number]) #selecting the data asociate with the categorical variables
train['SalePrice'].hist()
y=train['SalePrice']
train.drop(['SalePrice','Id'], axis=1, inplace=True)
numeric.drop(['SalePrice','Id'], axis=1, inplace=True)

num_columns=numeric.columns
def select_skew_index(df):

    numeric=df.select_dtypes(include=[np.number])

    num_columns=numeric.columns

    skew_features = df[num_columns].skew(axis = 0, skipna = True)

    high_skewness = skew_features[skew_features > 0.5]

    skew_index = high_skewness.index

    return skew_index
skew_index=select_skew_index(train)
train[skew_index].hist(figsize=(15,15))
kurt_features = train[num_columns].kurtosis(axis = 0, skipna = True)
high_kurt = kurt_features[kurt_features > 3]

kurt_index = high_kurt.index

high_kurt
fig=plt.figure(figsize=(15,20))

for i in range(1,18):

    ax=fig.add_subplot(6,3,i)

    ax.scatter(x=train[kurt_index[i-1]], y=y)

    ax.set_xlabel(kurt_index[i-1])
train[train['BsmtFinSF1']>5000].index
train[train['TotalBsmtSF']>6000].index
train[train['GrLivArea']>5000].index

train.shape
train=train.drop(train.index[1298])

y=y.drop(y.index[1298])
train.shape
y=np.log1p(y)
y.hist()
def correct_skew(df,skew_index):

    for i in skew_index:

        df[i] = np.log1p(df[i])
correct_skew(train,skew_index)
train[skew_index].hist(figsize=(15,15))
train.isnull().sum().sort_values(ascending=False)[0:25]
def fill_miss(df):

    Nvalues=['FireplaceQu','GarageFinish','BsmtCond','Alley','BsmtExposure','GarageCond','PoolQC','BsmtQual',

             'MiscFeature','MasVnrType','BsmtFinType1','GarageType','Fence','GarageQual','BsmtFinType2']

    GarBsmt=['GarageYrBlt','GarageCars','GarageArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',

'BsmtFullBath','BsmtHalfBath']

    df_cat=df.select_dtypes(exclude=[np.number])

    stats_df=df_cat.describe() 

    for i in df.columns:

        if(i in Nvalues):

            df[i].replace(np.nan,"None", inplace=True )

        elif(i in GarBsmt):

            df[i].replace(np.nan,0, inplace=True )

        

    for i in df_cat.columns:

        top = stats_df[i].iloc[2]

        if(df[i].isnull().sum()!=0):

            df[i].replace(np.nan,top, inplace=True )

            

    df.interpolate(inplace=True)
fill_miss(train)
train.isnull().sum()
def uniq_cat(df):

    categoric=df.select_dtypes(exclude=[np.number])

    cat_col=categoric.columns

    high_val=[]

    for i in cat_col:

        for j in range(df[i].unique().shape[0]):

            if ((df[i].value_counts()[j])/1459 > 0.99):

                 high_val.append(i) 

    return high_val
uniq_cat(train)
train = train.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
def filling_ordinal(df):

    feat=['ExterQual','ExterCond','BsmtQual','BsmtCond','KitchenQual','HeatingQC','KitchenQual'

          ,'HeatingQC','GarageQual','GarageCond']

    for x in feat:  

        df[x][df[x] == 'Ex'] = 5

        df[x][df[x] == 'Gd'] = 4

        df[x][df[x] == 'TA'] = 3

        df[x][df[x] == 'Fa'] = 2

        df[x][df[x] == 'Po'] = 1

        df[x][df[x] == 'None'] = 0

        

    df['LandSlope'][df['LandSlope'] == 'Sev'] = 3

    df['LandSlope'][df['LandSlope'] == 'Mod'] = 2

    df['LandSlope'][df['LandSlope'] == 'Gtl'] = 1

    

    df['BsmtExposure'][df['BsmtExposure'] == 'Gd'] = 4

    df['BsmtExposure'][df['BsmtExposure'] == 'Av'] = 3

    df['BsmtExposure'][df['BsmtExposure'] == 'Mn'] = 2

    df['BsmtExposure'][df['BsmtExposure'] == 'No'] = 1

    df['BsmtExposure'][df['BsmtExposure'] == 'None'] = 0

    

    feat1=['BsmtFinType1','BsmtFinType2']

    

    for x in feat1:

        df[x][df[x] == 'GLQ'] = 6

        df[x][df[x] == 'ALQ'] = 5

        df[x][df[x] == 'BLQ'] = 4

        df[x][df[x] == 'Rec'] = 3

        df[x][df[x] == 'LwQ'] = 2

        df[x][df[x] == 'Unf'] = 1

        df[x][df[x] == 'None'] = 0

        

    df['CentralAir'][df['CentralAir'] == 'Y'] = 1

    df['CentralAir'][df['CentralAir'] == 'N'] = 0

    
filling_ordinal(train)
def feat_ing(X):

    X['TotalBath']=X['BsmtFullBath']+ (1/2)*X['BsmtHalfBath']+X['FullBath']+ (1/2)*X['HalfBath']

    X['TotalSF']=X['TotalBsmtSF']+X['1stFlrSF']+X['2ndFlrSF']
feat_ing(train)
X=pd.get_dummies(train)
def clean(rtrain,rtest):

    y=rtrain['SalePrice']

    testId=rtest['Id']

    rtrain.drop(['Id','SalePrice'],axis=1,inplace=True)

    rtest.drop(['Id'],axis=1,inplace=True)

    

    # selecting the indexes of the skew features

    skew_index=select_skew_index(rtrain)

    

    # Eliminate the outier

    rtrain=rtrain.drop(rtrain.index[1298])

    y=y.drop(rtrain.index[1298])

    

    # Drop the columns in the test data with all values equal to na

    rtest=rtest.dropna(axis=1,how='all')

    

    # preparing features and target values

    y=np.log1p(y)



    #Correct the skewness

    #correct_skew(rtrain,skew_index)

    #correct_skew(rtest,skew_index)

    

    #Filling missing values

    fill_miss(rtrain)

    fill_miss(rtest)

    

    #Drop the features with low information

    #rtrain = rtrain.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating', 'PoolQC'], axis=1)

    #rtest = rtest.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating', 'PoolQC'], axis=1)

    

    # Correcting categorical values that are ordinal

    filling_ordinal(rtrain)

    filling_ordinal(rtest)

    

    #Feature ingeneering

    feat_ing(rtrain)

    feat_ing(rtest)

    

    #One hot encoding

    rtrain=pd.get_dummies(rtrain)

    rtest=pd.get_dummies(rtest)

    

    # Update the training set

    rtrain=rtrain[rtest.columns]

    

    

    return(rtrain,rtest,y,testId)
X,Xtest,y,TestId=clean(train_copy,test_copy)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.33, random_state=77)
from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

parameters= {'alpha':[0.0001,0.0002,0.0003,0.0004,0.0009,0.001,0.002,0.003,0.01,0.1,1,10,100]}



lasso=Lasso()

lasso_reg=GridSearchCV(lasso, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)

lasso_reg.fit(X,y)



print('The best value of Alpha is: ',lasso_reg.best_params_,'neg_mean_squared_error',lasso_reg.best_score_)
from sklearn import linear_model

from sklearn.linear_model import Lasso



best_alpha=0.0009

lasso = Lasso(alpha=best_alpha,max_iter=10000)

lasso.fit(X,y)
ytest=lasso.predict(Xtest)

#ytest_ridge=ridge.predict(Xtest)

#ytest=(ytest_ridge+ytest_lasso)/2


ytest=np.expm1(ytest)
#FI_lasso = pd.DataFrame({"Feature":X.columns, 'Importance':lasso.coef_})
#FI_lasso=FI_lasso.sort_values("Importance",ascending=False)
#import seaborn as sns

#sns.barplot(x='Importance', y='Feature', data=FI_lasso.head(10),color='b')
prediction=pd.DataFrame({'Id': TestId, 'SalePrice': ytest})
prediction
prediction.to_csv('submission.csv', index=False)