import os

import pandas as pd

import numpy as np
df=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
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

univar_cont2(df, 50)  
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
df['MasVnrArea'].fillna(0, inplace=True)

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
df_linear_reg = df

df_linear_reg.head()
col_lis=[]

for i in df_linear_reg.columns:

    if (df_linear_reg[i].dtype in numerics):

        if (df_linear_reg[i].nunique())>20:

            col_lis.append(i)

col_lis
#outlier_detect(df_linear_reg, 'LotFrontage',treatment=True)

#outlier_detect(df_linear_reg, 'LotArea',treatment=True)

#outlier_detect(df_linear_reg, 'BsmtFinSF1',treatment=True)

#outlier_detect(df_linear_reg, 'TotalBsmtSF',treatment=True)

#outlier_detect(df_linear_reg, '2ndFlrSF',treatment=True)

#outlier_detect(df_linear_reg, 'GrLivArea',treatment=True)

#outlier_detect(df_linear_reg, 'GarageArea',treatment=True)

#outlier_detect(df_linear_reg, 'OpenPorchSF',treatment=True)

#outlier_detect(df_linear_reg, 'ScreenPorch',treatment=True)

del df['Id']
df.columns
for i in df.columns:

    if (df[i].nunique())<20:

        df[i] = df[i].astype(str)
df['MSSubClass']= df['MSSubClass'].astype(int)

df['PoolArea']= df['PoolArea'].astype(int)
df.dtypes.value_counts()
df.shape
features = pd.get_dummies(df)

features.shape
features.head()
features.dtypes.value_counts()
target = features.sale_price_log
target_actuals = features['SalePrice']

del features['sale_price_log']

del features['SalePrice']
features_list = list(features.columns)
features.shape
#features = np.array(features)
from sklearn.model_selection import train_test_split

train_features, test_features, train_target, test_target = train_test_split(features, target, test_size = 0.20, random_state = 77)
print('Training Features Shape:', train_features.shape)

print('Training Labels Shape:', train_target.shape)

print('Testing Features Shape:', test_features.shape)

print('Testing Labels Shape:', test_target.shape)
# Import the modelg

from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 500 decision trees

rf = RandomForestRegressor(n_estimators = 820, random_state = 77, oob_score=True, max_features=0.80, n_jobs=-1)

# Train the model on training data

rf.fit(train_features, train_target)
# predict on the test data splitted from the train

predictions = rf.predict(test_features)
pred_actuals = np.exp(predictions)
# Calculate the absolute errors

errors = abs(predictions - test_target)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 5))
def linear_report (y_actual, y_pred):

    from sklearn.metrics import mean_squared_error

    from sklearn.metrics import mean_absolute_error

    from sklearn.metrics import r2_score

    

    mae = mean_absolute_error(y_actual, y_pred)

    print('MAE value is: ',mae)

    mse = mean_squared_error(y_actual, y_pred)

    print('MSE value is: ', mse)

    import math

    rmse = math.sqrt(mse)

    print('RMSE value is: ', rmse)

    r2 = r2_score(y_actual, y_pred)

    print('R2 score is: ',r2)

    mape = 100*(mae/y_actual)

    accuracy = 100-np.mean(mape)

    print('Accuracy score is: ',accuracy)
#Regression report for log values

linear_report(test_target,predictions)
#Regression report for actuals values

target_actuals = np.exp(test_target)

linear_report(target_actuals,pred_actuals)
from sklearn.metrics import mean_absolute_error

n_estimators = [800,820,830,850,860,870,880,900]

train_results = []

test_results = []

for estimator in n_estimators:

    rf = RandomForestRegressor(n_estimators=estimator, n_jobs=-1)

    rf.fit(train_features, train_target)

    train_pred = rf.predict(train_features)

    mae = mean_absolute_error(train_target, train_pred)

    mape = 100*(mae/train_pred)

    accuracy = 100-np.mean(mape)

    train_results.append(accuracy)

    

    y_pred = rf.predict(test_features)

    mae_p = mean_absolute_error(test_target, y_pred)

    mape_p= 100*(mae/y_pred)

    accuracy_p = 100-np.mean(mape_p)

    test_results.append(accuracy_p)

    

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(n_estimators, train_results,'b', label='Train R2')

line2, = plt.plot(n_estimators, test_results,'r', label='Test R2')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('R2 score')

plt.xlabel('n_estimators')

plt.show()
## number of estimators will be set to 820
from sklearn.metrics import mean_absolute_error

max_dep = np.arange(300, 500, 20)

train_results = []

test_results = []

for max_d in max_dep:

    rf = RandomForestRegressor(n_estimators=820, max_depth=max_d, n_jobs=-1)

    rf.fit(train_features, train_target)

    train_pred = rf.predict(train_features)

    mae = mean_absolute_error(train_target, train_pred)

    mape = 100*(mae/train_pred)

    accuracy = 100-np.mean(mape)

    train_results.append(accuracy)

    

    y_pred = rf.predict(test_features)

    mae_p = mean_absolute_error(test_target, y_pred)

    mape_p= 100*(mae/y_pred)

    accuracy_p = 100-np.mean(mape_p)

    test_results.append(accuracy_p)

    

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_dep, train_results,'b', label='Train R2')

line2, = plt.plot(max_dep, test_results,'r', label='Test R2')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('R2 score')

plt.xlabel('max depth')

plt.show()
## the training and test does not show any overfitting hence we will keep it as none.
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)

train_results = []

test_results = []

for min_s in min_samples_splits:

    rf = RandomForestRegressor(n_estimators=820, max_depth=None, n_jobs=-1, min_samples_split=min_s)

    rf.fit(train_features, train_target)

    train_pred = rf.predict(train_features)

    mae = mean_absolute_error(train_target, train_pred)

    mape = 100*(mae/train_pred)

    accuracy = 100-np.mean(mape)

    train_results.append(accuracy)

    

    y_pred = rf.predict(test_features)

    mae_p = mean_absolute_error(test_target, y_pred)

    mape_p= 100*(mae/y_pred)

    accuracy_p = 100-np.mean(mape_p)

    test_results.append(accuracy_p)

    

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(min_samples_splits, train_results,'b', label='Train R2')

line2, = plt.plot(min_samples_splits, test_results,'r', label='Test R2')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('R2 score')

plt.xlabel('min sample splits')

plt.show()
## we will keep it as default, i.e int 2
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)

train_results = []

test_results = []

for min_sl in min_samples_leafs:

    rf = RandomForestRegressor(n_estimators=820, max_depth=None, n_jobs=-1, min_samples_split=2, min_samples_leaf=min_sl)

    rf.fit(train_features, train_target)

    train_pred = rf.predict(train_features)

    mae = mean_absolute_error(train_target, train_pred)

    mape = 100*(mae/train_pred)

    accuracy = 100-np.mean(mape)

    train_results.append(accuracy)

    

    y_pred = rf.predict(test_features)

    mae_p = mean_absolute_error(test_target, y_pred)

    mape_p= 100*(mae/y_pred)

    accuracy_p = 100-np.mean(mape_p)

    test_results.append(accuracy_p)

    

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(min_samples_leafs, train_results,'b', label='Train R2')

line2, = plt.plot(min_samples_leafs, test_results,'r', label='Test R2')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('R2 score')

plt.xlabel('min sample leafs')

plt.show()
##there is no overfitting, so we will keep this as default i.e 1
max_feat = np.linspace(0.1, 1, 10, endpoint=True)

train_results = []

test_results = []

for min_fe in max_feat:

    rf = RandomForestRegressor(n_estimators=820, max_depth=None, n_jobs=-1, min_samples_split=2, max_features=min_fe )

    rf.fit(train_features, train_target)

    train_pred = rf.predict(train_features)

    mae = mean_absolute_error(train_target, train_pred)

    mape = 100*(mae/train_pred)

    accuracy = 100-np.mean(mape)

    train_results.append(accuracy)

    y_pred = rf.predict(test_features)

    mae_p = mean_absolute_error(test_target, y_pred)

    mape_p= 100*(mae/y_pred)

    accuracy_p = 100-np.mean(mape_p)

    test_results.append(accuracy_p)

    

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_feat, train_results,'b', label='Train R2')

line2, = plt.plot(max_feat, test_results,'r', label='Test R2')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('R2 score')

plt.xlabel('max features')

plt.show()
## max feature at 0.5 performs better
rf = RandomForestRegressor(n_estimators=820, max_depth=None, n_jobs=-1, min_samples_split=2, min_samples_leaf=1, random_state = 77, oob_score=True, max_features=0.45, criterion="mse")

#rf = RandomForestRegressor(n_estimators = 820, random_state = 77, oob_score=True, max_features=0.80, n_jobs=-1)

rf.fit(train_features, train_target)

train_pred = rf.predict(train_features)

        

y_pred = rf.predict(test_features)
linear_report(test_target,y_pred)
# Get numerical feature importances

importances = list(rf.feature_importances_)

# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(features_list, importances)]

# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
# List of features sorted from most to least important

sorted_importances = [importance[1] for importance in feature_importances]

sorted_features = [importance[0] for importance in feature_importances]

# Cumulative importances

cumulative_importances = np.cumsum(sorted_importances)



# Make a line graph

x_values = list(range(len(importances)))

plt.plot(x_values, cumulative_importances, 'g-')



# Axis labels and title

plt.xlabel('Variable'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances')



#Draw line at 95% of importance retained

plt.hlines(y = 0.98, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')
# Find number of features for cumulative importance of 95%

# Add 1 because Python is zero-indexed

print('Number of features for 98% importance:', np.where(cumulative_importances > 0.98)[0][0] + 1)
important_feature_names = [feature[0] for feature in feature_importances[0:177]]

important_indices = [features_list.index(feature) for feature in important_feature_names]

important_indices
important_train_features=train_features.iloc[:, important_indices]

important_test_features=test_features.iloc[:, important_indices]
important_train_features.shape
important_test_features.shape
rf_imp = RandomForestRegressor(n_estimators=820, max_depth=None, n_jobs=-1, min_samples_split=2, min_samples_leaf=1, random_state = 77, oob_score=True, max_features=0.45, criterion="mse")

rf_imp.fit(important_train_features, train_target)

train_pred_imp = rf_imp.predict(important_train_features)

        

y_pred_imp = rf_imp.predict(important_test_features)
## Report

linear_report(test_target,y_pred_imp)
print('Number of features for 99% importance:', np.where(cumulative_importances > 0.99)[0][0] + 1)
important_feature_names = [feature[0] for feature in feature_importances[0:217]]

important_indices = [features_list.index(feature) for feature in important_feature_names]
important_train_features=train_features.iloc[:, important_indices]

important_test_features=test_features.iloc[:, important_indices]
rf_imp = RandomForestRegressor(n_estimators=820, max_depth=None, n_jobs=-1, min_samples_split=2, min_samples_leaf=1, random_state = 77, oob_score=True, max_features=0.45, criterion="mse")

rf_imp.fit(important_train_features, train_target)

train_pred_imp = rf_imp.predict(important_train_features)

        

y_pred_imp = rf_imp.predict(important_test_features)
## Report

linear_report(test_target,y_pred_imp)
test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
test.head()
test.drop(drop_col, axis=1, inplace=True)

test.rename(columns={'3SsnPorch': "threessnporch"}, inplace=True)

test['YearRemodAdd'] = test['YearRemodAdd'].astype(str)

test.loc[(test.YearRemodAdd == test.YearBuilt), 'YearRemodAdd'] = 'No_remodel'

test.drop(['LowQualFinSF','BsmtHalfBath','threessnporch','MiscVal','MoSold','YrSold'], axis=1, inplace=True)



test.drop(['BsmtFinSF2','EnclosedPorch','BsmtUnfSF'], axis=1, inplace=True)

test.drop('1stFlrSF', axis=1, inplace=True)

test.drop('GarageYrBlt', axis=1, inplace=True)





# missing values



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

test['MasVnrArea'].fillna(0, inplace=True)

test['MasVnrType'].fillna('No_Masonry', inplace=True)

test['FireplaceQu'].fillna('No_Fireplace', inplace=True)

test['LotFrontage'].fillna(test['LotFrontage'].median(), inplace=True)



del test['Id']

for i in df:

    if (df[i].nunique())<20:

        test[i] = test[i].astype(str)

test['MSSubClass']= test['MSSubClass'].astype(int)

test['PoolArea'] = test['PoolArea'].astype(int)

# one hot encoding

#final_test = pd.get_dummies(test)



test['GarageCars'].replace('nan',0, inplace=True)

test['GarageCars'] = test['GarageCars'].astype(float)

test['GarageCars'] = test['GarageCars'].astype(int)

test['GarageCars'] = test['GarageCars'].astype(str)
test['BsmtFullBath'].replace('nan',0, inplace=True)

test['BsmtFullBath'] = test['BsmtFullBath'].astype(float)

test['BsmtFullBath'] = test['BsmtFullBath'].astype(int)

test['BsmtFullBath'] = test['BsmtFullBath'].astype(str)
testmiss_val = test.isna().sum()

testmiss_val = pd.DataFrame(testmiss_val, columns=['miss'])



testmiss_val = testmiss_val.loc[(testmiss_val!=0).any(axis=1)]

testmiss_val
test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].median(), inplace=True)

test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].median(), inplace=True)

test['GarageArea'].fillna(test['GarageArea'].median(), inplace=True)

testmiss_val = test.isna().sum()

testmiss_val = pd.DataFrame(testmiss_val, columns=['miss'])



testmiss_val = testmiss_val.loc[(testmiss_val!=0).any(axis=1)]

testmiss_val
test.shape
df.shape

# the extra two columns are saleprice and log sale price
test.dtypes.value_counts()
# one hot encoding

final_test = pd.get_dummies(test)
col_1 = final_test.columns
col_2 = features.columns
len(col_2)
for dd in col_1:

    if dd in col_2:

        pass

    else:

        print(dd, 'missing')
for dd in col_2:

    if dd in col_1:

        pass

    else:

        print(dd, 'missing')
# removed the column MSZoning_nan
final_test.drop('MSZoning_nan', axis=1, inplace=True)
final_test.drop(['Exterior1st_nan','Exterior2nd_nan','FullBath_4','KitchenQual_nan','TotRmsAbvGrd_13','TotRmsAbvGrd_15','Functional_nan','Fireplaces_4','GarageCars_5','SaleType_nan'], axis=1, inplace=True)
df['HouseStyle'].nunique()
test['HouseStyle'].nunique()
col_to_add = set( col_2 )-set(col_1)
for c in col_to_add:

    final_test[c]=0
col_1 = final_test.columns

col_2 = features.columns

for dd in col_1:

    if dd in col_2:

        pass

    else:

        print(dd, 'missing')
for dd in col_2:

    if dd in col_1:

        pass

    else:

        print(dd, 'missing')
# now all the columns in train model and test dataset are matching.
set( train_features.columns ) - set( final_test.columns )
final_imp_test = final_test.loc[:, important_feature_names]
final_test_predictions = rf_imp.predict(final_imp_test)
final_test_predictions
final_test_price = np.exp(final_test_predictions)
final_test_price
ids=pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
ids['Price'] = final_test_price
ids.drop('SalePrice', axis=1, inplace=True)
ids