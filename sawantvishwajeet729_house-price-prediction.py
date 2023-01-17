import os

import pandas as pd

import numpy as np
df=pd.read_csv("../input/train.csv")
#df = pd.read_csv('train.csv')
df.head()
df.shape
import matplotlib.pyplot as plt
plt.hist(df['SalePrice'])
df['sale_price_log'] = np.log(df['SalePrice'])

plt.hist(df['sale_price_log'])
import statsmodels.formula.api as smf

import statsmodels.stats.multicomp as multi
def uni_analysis(df):

    x = (df.select_dtypes(include=['O']))

    p_value = []

    for i in x:

        para = 'SalePrice ~ '+str(i)

        model = smf.ols(formula=para, data=df)

        results = model.fit()

        p_value.append(results.f_pvalue)

    df1 = pd.DataFrame(list(zip(x,p_value)), columns =['Variable', 'p_value'])

    df1['Drop_column'] = df1['p_value'].apply(lambda x: 'True' if x > 0.05 else 'False')

    return df1
uni_analysis(df)
drop_col = ['Street','PoolQC','Utilities','LandSlope','MiscFeature','Condition2']
df.drop(drop_col, axis=1, inplace=True)
#3SsnPorch, there is some issue with the column name, the ols algorithm shows error with this name. hence we rename it.

df.rename(columns={'3SsnPorch': "threessnporch"}, inplace=True)

# YearRemodAdd teslls us if the house was remodelled. if the yearbuilt and yearremodadd are same that means there was no modification

df['YearRemodAdd'] = df['YearRemodAdd'].astype(str)

df.loc[(df.YearRemodAdd == df.YearBuilt), 'YearRemodAdd'] = 'No_remodel'
def univar_cont2(df, threshold):

    x = (df.select_dtypes(include=['int64','float64']))

    p_value = []

    col_name = []

    for i in x:

        if (df[i].nunique())<threshold:

            df[i] = df[i].astype(str)

            col_name.append(i)

            para = 'SalePrice ~ '+str(i)

            model = smf.ols(formula=para, data=df)

            results = model.fit()

            p_value.append(results.f_pvalue)

        else:

            print('columns are truely continous:',i,'and the unique entries are', df[i].nunique())

    df1 = pd.DataFrame(list(zip(col_name,p_value)), columns =['Variable', 'p_value'])

    df1['Drop_column'] = df1['p_value'].apply(lambda x: 'True' if x > 0.05 else 'False')

    return df1

univar_cont2(df, 50)  #threshold value was earlier included as inside the function as 20. the above function was updated later.
df.drop(['LowQualFinSF','BsmtHalfBath','threessnporch','MiscVal','MoSold','YrSold'], axis=1, inplace=True)
x = (df.select_dtypes(include=['int64','float64']))

for i in x:

    if (df[i].nunique())<50:

        df[i] = df[i].astype(str)
df.dtypes.value_counts()
def univar_cont(df, target):

    x = (df.select_dtypes(include=['int64','float64']))

    print('There are ',len(x.columns),' columns with continous variable')

    mean_val =[]

    median_val = []

    min_val = []

    max_val = []

    variance_val = []

    std_val = []

    q1_val = []

    q3_val= []

    corelation = []

    

    for i in x:

        mean_val.append(df[i].mean())

        median_val.append(df[i].median())

        min_val.append(df[i].min())

        max_val.append(df[i].max())

        variance_val.append(df[i].var())

        std_val.append(df[i].std())

        

        q1,q3 = df[i].quantile([0.25,0.75])

        q1_val.append(q1)

        q3_val.append(q3)

        corelation.append(df[i].corr(df[target]))

    df1 = pd.DataFrame(list(zip(x,mean_val,median_val,min_val,max_val,variance_val,std_val,q1_val,q3_val, corelation)), 

                      columns=['variable','mean','median','minimum','maximum','variance','std_deviation','quantile_1','quantile_3','corelation'])

    return df1

   
univar_cont(df, 'SalePrice')
df.drop(['BsmtFinSF2','EnclosedPorch','BsmtUnfSF'], axis=1, inplace=True)
df.head()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



newdf = df.select_dtypes(include=numerics)
newdf.drop(['Id','sale_price_log'], axis=1, inplace=True)
newdf.head()
cor_tab = newdf.corr()

cor_tab.style.background_gradient(cmap='coolwarm')

df.drop('1stFlrSF', axis=1, inplace=True)
df.drop('GarageYrBlt', axis=1, inplace=True)
df.head()
miss_val = df.isna().sum()

miss_val = pd.DataFrame(miss_val, columns=['miss'])



miss_val = miss_val.loc[(miss_val!=0).any(axis=1)]
miss_val['miss_percent'] = (miss_val['miss']/1460)*100
miss_val
# since alley and fence have more than 80% of missing data we will drop these columns. note that alley has good p-value. so later we will try to include it to see if it will improve the model

df.drop(['Alley','Fence'], axis=1, inplace=True)
# just noticed that fireplace is actually a numerical variable, it can be used as categorical. but to reduce the number of dummy

#columns, we will make it as numberic

df['Fireplaces'] = df['Fireplaces'].astype('int32')
df.shape
df['GarageType'].isna().sum()
xx= df[['GarageType','GarageFinish','GarageQual','GarageCond']]
xx.fillna(-99, inplace=True)
xx
xy = xx.index[xx['GarageType']==-99].tolist()
df['GarageCars'][xy]
df['GarageCond'][xy]
df['GarageArea'][xy]
df['GarageType'].fillna('No_garage', inplace=True)
df['GarageQual'].fillna('No_garage', inplace=True)
df['GarageCond'].fillna('No_garage', inplace=True)
df['GarageFinish'].fillna('No_garage', inplace=True)
miss_val = df.isna().sum()

miss_val = pd.DataFrame(miss_val, columns=['miss'])



miss_val = miss_val.loc[(miss_val!=0).any(axis=1)]

miss_val
bs = df.index[df['BsmtQual'].isna()].tolist()
df['BsmtCond'][bs]
df['BsmtFinType1'][bs]
df['BsmtExposure'][bs]
# similar to garage, impute these columns with no basement
df['BsmtFullBath'][bs]
df['BsmtQual'].fillna('No_basement', inplace=True)

df['BsmtCond'].fillna('No_basement', inplace=True)

df['BsmtExposure'].fillna('No_basement', inplace=True)

df['BsmtFinType1'].fillna('No_basement', inplace=True)

df['BsmtFinType2'].fillna('No_basement', inplace=True)
miss_val = df.isna().sum()

miss_val = pd.DataFrame(miss_val, columns=['miss'])



miss_val = miss_val.loc[(miss_val!=0).any(axis=1)]

miss_val
ms = df.index[df['MasVnrArea'].isna()].tolist()
df['MasVnrArea'][ms]
df['MasVnrArea'].fillna('No_Masonry', inplace=True)

df['MasVnrType'].fillna('No_Masonry', inplace=True)
fi = df.index[df['FireplaceQu'].isna()].tolist()
df['Fireplaces'][fi]
# the missing values in fireplacequ is where the fireplace is 0
df['FireplaceQu'].fillna('No_Fireplace', inplace=True)
miss_val = df.isna().sum()

miss_val = pd.DataFrame(miss_val, columns=['miss'])



miss_val = miss_val.loc[(miss_val!=0).any(axis=1)]

miss_val
df=df.dropna(subset=['Electrical'])
miss_val = df.isna().sum()

miss_val = pd.DataFrame(miss_val, columns=['miss'])



miss_val = miss_val.loc[(miss_val!=0).any(axis=1)]

miss_val
df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)
miss_val = df.isna().sum()

miss_val = pd.DataFrame(miss_val, columns=['miss'])



miss_val = miss_val.loc[(miss_val!=0).any(axis=1)]

miss_val
def outlier_detect (df, col, treatment=False):

    from scipy.stats import iqr

    iqr_c = iqr(df[col])

    q1,q3 = df[col].quantile([0.25,0.75])

    upper_wis = q3 + (1.5*iqr_c)

    lower_wis = q1 - (1.5*iqr_c)

    out_liers_upper = df.index[df[col] > upper_wis]

    out_liers_lower = df.index[df[col] < lower_wis]

    

    col_num = df.columns.get_loc(col)

    if treatment==True:

        for x in out_liers_upper:

            df.iloc[x,col_num]= upper_wis

        for x in out_liers_lower:

            df.iloc[x,col_num]= lower_wis

    return out_liers_upper

    return out_liers_lower

    return df
outlier_detect(df, 'LotFrontage')
plt.boxplot(df['LotFrontage'])
df.head()
df.dtypes.value_counts()
for i in df.columns:

    if (df[i].nunique())<20:

        df[i] = df[i].astype(str)
df.dtypes.value_counts()
df = pd.get_dummies(df)
df.shape
df.dtypes.value_counts()
del df['SalePrice']
target = np.array(df['sale_price_log'])
features = df.drop('sale_price_log', axis=1)
feature_list = list(features.columns)
features = np.array(features)
from sklearn.model_selection import train_test_split

train_features, test_features, train_target, test_target = train_test_split(features, target, test_size = 0.20, random_state = 77)
print('Training Features Shape:', train_features.shape)

print('Training Labels Shape:', train_target.shape)

print('Testing Features Shape:', test_features.shape)

print('Testing Labels Shape:', test_target.shape)
# Import the modelg

from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 500 decision trees

rf = RandomForestRegressor(n_estimators = 500, random_state = 77)

# Train the model on training data

rf.fit(train_features, train_target);
# Use the forest's predict method on the test data

predictions = rf.predict(test_features)
# Calculate the absolute errors

errors = abs(predictions - test_target)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 5))
predictions[10]
np.exp(predictions[10])
np.exp(test_target[10])
# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / test_target)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')

test=pd.read_csv("../input/test.csv")
test.shape
test.drop(drop_col, axis=1, inplace=True)

test.columns
test.rename(columns={'3SsnPorch': "threessnporch"}, inplace=True)

test['YearRemodAdd'] = test['YearRemodAdd'].astype(str)

test.loc[(test.YearRemodAdd == test.YearBuilt), 'YearRemodAdd'] = 'No_remodel'

test.drop(['LowQualFinSF','BsmtHalfBath','threessnporch','MiscVal','MoSold','YrSold'], axis=1, inplace=True)

test.drop(['BsmtFinSF2','EnclosedPorch','BsmtUnfSF'], axis=1, inplace=True)

test.drop('GarageYrBlt', axis=1, inplace=True)

test.drop('1stFlrSF', axis=1, inplace=True)

test.drop(['Alley','Fence'], axis=1, inplace=True)

test['Fireplaces'] = test['Fireplaces'].astype('int32')



test['GarageType'].fillna('No_garage', inplace=True)

test['GarageQual'].fillna('No_garage', inplace=True)

test['GarageCond'].fillna('No_garage', inplace=True)

test['GarageFinish'].fillna('No_garage', inplace=True)

test['BsmtQual'].fillna('No_basement', inplace=True)

test['BsmtCond'].fillna('No_basement', inplace=True)

test['BsmtExposure'].fillna('No_basement', inplace=True)

test['BsmtFinType1'].fillna('No_basement', inplace=True)

test['BsmtFinType2'].fillna('No_basement', inplace=True)

test['MasVnrArea'].fillna('No_Masonry', inplace=True)

test['MasVnrType'].fillna('No_Masonry', inplace=True)

test['FireplaceQu'].fillna('No_Fireplace', inplace=True)

test=test.dropna(subset=['Electrical'])

test['LotFrontage'].fillna(test['LotFrontage'].median(), inplace=True)
test.shape
test.dtypes.value_counts()