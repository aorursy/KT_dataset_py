#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
import string
import warnings
warnings.filterwarnings('ignore')
#current data directory and data file name
import os
for root, dir, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(root, filename))
#import library
import pandas as pd
# Read training data
train_data = pd.read_csv('../input/home-data-for-ml-course/train.csv')
# Read testing data
test_data = pd.read_csv('../input/home-data-for-ml-course/test.csv')
#look at first 5 rows of data
print(train_data.head())
#shape of data
print('train shape', train_data.shape)
print('test shape', test_data.shape)
#get variables
print(train_data.columns.tolist())
#missing values
print(train_data.isnull().sum().sort_values(ascending = False))
#percentage
print(train_data.isnull().mean().sort_values(ascending = False))                         
target = train_data['SalePrice']
full_data = pd.concat([train_data, test_data], sort = True).reset_index(drop = True)#together
combine_data = [train_data,test_data]
full_data.info()
column_name_noncategorical = full_data.select_dtypes(exclude=['object']).columns.tolist()
column_name_noncategorical.remove('SalePrice')
column_name_noncategorical
column_name_categorical = full_data.select_dtypes(include=['object']).columns.tolist()
column_name_categorical
print(len(column_name_noncategorical), len(column_name_categorical))
full_data_describe = full_data.select_dtypes(exclude=['object']).describe(percentiles = [.25, .5, .75,.90])
full_data_describe
full_data_describe[7:8]
full_data_describe[8:9]
fig = plt.figure(figsize=(18,24))
for index,col in enumerate(column_name_noncategorical):
    plt.subplot(10,4,index+1)
    sns.distplot(full_data[column_name_noncategorical].loc[:,col].dropna())
fig.tight_layout(pad=1.0)
column_name_distinct = []
column_name_continuous = []
for col_name in column_name_noncategorical:
    if len(full_data[column_name_noncategorical][col_name].unique().tolist())< 15:
        column_name_distinct.append(col_name)
    else:
        column_name_continuous.append(col_name)
print(column_name_distinct, column_name_continuous)
fig = plt.figure(figsize=(18,24))
for index,col in enumerate(column_name_noncategorical):
    plt.subplot(10,4,index+1)
    sns.boxplot(full_data[column_name_noncategorical].loc[:,col].dropna())
fig.tight_layout(pad=1.0)
fig = plt.figure(figsize=(18,24))
for index,col in enumerate(column_name_noncategorical):
    plt.subplot(10,4,index+1)
    sns.scatterplot(x = train_data[column_name_noncategorical].iloc[:,index].dropna(), y = target, data = train_data[column_name_noncategorical].dropna())
fig.tight_layout(pad=1.0)
#list for outlayers:
column_name_noncategorical[33]
#for categorical values
fig = plt.figure(figsize=(18,24))
for index,col in enumerate(column_name_categorical):
    plt.subplot(11,4,index+1)
    sns.countplot(x = full_data[column_name_categorical].iloc[:,index], data = full_data[column_name_categorical].dropna())
fig.tight_layout(pad=1.0)
column_name_cat_noncategorical = []
column_name_cat_categorical = []
for col_name in column_name_categorical:
    if len(full_data[column_name_categorical][col_name].unique().tolist())< 30:
        column_name_cat_categorical.append(col_name)
    else:
        column_name_continuous.append(col_name)
print(column_name_cat_noncategorical, column_name_continuous)
plt.figure(figsize=(20,20))
sns.heatmap(full_data[column_name_noncategorical].corr(), mask = full_data[column_name_noncategorical].corr() <0.8, linewidth=0.5, cmap='Reds')
full_data_corr = full_data[column_name_noncategorical].corr().abs().unstack().sort_values(kind = "quicksort", ascending = False).to_frame()
full_data_corr.iloc[38:46]# 37 columns with corr of 1
#important correlation with target
target_corr = full_data.corr()['SalePrice'].abs().sort_values(ascending = False)[1:]
target_corr.iloc[0:5]
np.unique(train_data['YrSold']).tolist()
train_data.shape
train_data = train_data.drop('SalePrice', axis = 1)
train_data.shape
drop_col = []
for col in column_name_noncategorical:
    if train_data[col].value_counts().iloc[0] / len(train_data[col].dropna()) > 0.9:
        drop_col.append(col)
for col in column_name_categorical:
    if train_data[col].value_counts().iloc[0] / len(train_data[col].dropna()) > 0.9:
        drop_col.append(col)
train_data = train_data.drop(drop_col, axis = 1) 
test_data = test_data.drop(drop_col, axis = 1)
train_data.shape
test_data.shape
full_data_null_precentage = full_data.isnull().sum().sort_values(ascending = False).divide(full_data.shape[0])
full_data_null_precentage
del_list = []# list for delete, when there is too many missing values
fill_Auto = []# list for fill directly, when missing value precentage between 0.2 and 0.75
fill_other = []#other columns
full_data = pd.concat([train_data, test_data], sort = True).reset_index(drop = True)#together
full_data_columns = full_data.columns.tolist()
for col in full_data_columns:
    if full_data_null_precentage[col] >0.75:
        del_list.append(col)
    if 0.2<full_data_null_precentage[col] <=0.75:
        fill_Auto.append(col)
    if 0<full_data_null_precentage[col]<=0.2:
        fill_other.append(col)
del_list,fill_Auto,fill_other
full_data.drop(del_list, axis = 1, inplace = True)
full_data.shape
missing_list = []
full_data_missing = full_data.isnull().sum().sort_values(ascending = False)
full_data_columns = full_data.columns.tolist()
for col in full_data_columns:
    if full_data_missing[col] !=0:
        missing_list.append(col)
missing_list
non_object_list = full_data[missing_list].select_dtypes(exclude=['object']).columns.tolist()#not object list
non_object_list
full_data_copy = full_data.copy()
full_data_copy['SalePrice'] = target
target_corr = full_data_copy.corr()['SalePrice'].abs().sort_values(ascending = False)[1:]
target_corr
full_data_copy.fillna('None')
full_data_copy = full_data_copy.apply(lambda series: pd.Series(
    LabelEncoder().fit_transform(series[series.notnull()]),
    index=series[series.notnull()].index))
target_corr = full_data_copy.corr()['SalePrice'].abs().sort_values(ascending = False)[1:]
sort_name = target_corr.index.tolist()
sort_name[0:6]
full_data_copy.GarageFinish
full_data.GarageFinish
distinct_name = []
continuous_name = []
for name in non_object_list:
    if name in column_name_distinct:
        distinct_name.append(name)
        if name in fill_Auto:
            full_data[name] = full_data[name].fillna(0)
        else:
            full_data[name] = full_data.groupby(sort_name[0])[name].apply(lambda x: x.fillna(x.mode()))
            full_data[name] = full_data.fillna(full_data[name].mode())
    if name in column_name_continuous:
        continuous_name.append(name)
        if name in fill_Auto:
            full_data[name] = full_data.groupby(sort_name[0])[name].apply(lambda x: x.fillna(x.mean()))
        else:
            full_data[name] = full_data.groupby(sort_name[0])[name].apply(lambda x: x.fillna(x.mean()))
distinct_name
full_data[non_object_list].isnull().sum().sort_values(ascending = False)
object_list = full_data[missing_list].select_dtypes(include=['object']).columns.tolist()# object list
object_list
# since no strange items(like names), directly fill the data.
sort_name = target_corr.index.tolist()
for name in object_list:
    if name in fill_Auto:
        full_data[name] = full_data[name].fillna('None')  
    else:
        full_data[name] = full_data.groupby(sort_name[0])[name].apply(lambda x: x.fillna(x.mode()[0]))
full_data.isnull().sum().sort_values(ascending = False)
#https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.html
#we can try methods from this website. Now just fill the data
full_data.drop('Id', axis = 1, inplace = True)
full_data.isnull().sum().sort_values(ascending = False)
non_object_list = full_data.select_dtypes(exclude=['object']).columns.tolist()
distinct_name = []
continuous_name = []
for name in non_object_list:
    if name in column_name_distinct:
        distinct_name.append(name)
    if name in column_name_continuous:
        continuous_name.append(name)
object_list = full_data.select_dtypes(include=['object']).columns.tolist()
full_data['SalePrice'] = target
full_data.shape
print(object_list)
print(distinct_name)
print(continuous_name)
print(len(object_list+distinct_name+continuous_name))
full_data.head()
#Drop outliers rows in train_data
train_data  = full_data.loc[:1459]
test_data = full_data.loc[1460:]
test_data.drop('SalePrice', axis = 1, inplace = True)
data_combine = [train_data, test_data]
train_data.shape
#ctrl C + ctrl V
train_data = train_data.drop(index = train_data['1stFlrSF'][train_data['1stFlrSF'] > 4000].index)
train_data = train_data.drop(index = train_data[train_data['BsmtFinSF1'] > 5000].index)
train_data = train_data.drop(index = train_data[train_data['BsmtFinSF2'] > 1400].index)
train_data = train_data.drop(index = train_data[train_data['EnclosedPorch'] > 500].index)
train_data = train_data.drop(index = train_data[train_data['GrLivArea'] > 4500].index)
train_data = train_data.drop(index = train_data[train_data['LotFrontage'] > 300].index)
train_data = train_data.drop(index = train_data[train_data['TotalBsmtSF'] > 5000].index)
train_data = train_data.drop(index = train_data[train_data['WoodDeckSF'] > 750].index)
train_data.reset_index(inplace = True)
train_data
#distinct numerical value does not need labelEncoder
for dataset in data_combine:
    dataset[distinct_name] = dataset[distinct_name].astype(int)
train_data.head(2)
train_data[object_list] = train_data[object_list].apply(lambda series: pd.Series(
    LabelEncoder().fit_transform(series[series.notnull()]),
    index=series[series.notnull()].index))
test_data[object_list] = test_data[object_list].apply(lambda series: pd.Series(
    LabelEncoder().fit_transform(series[series.notnull()]),
    index=series[series.notnull()].index))
train_data
test_data.head()
train_data.head(2)
OHEncoder_train = OneHotEncoder(handle_unknown='ignore', sparse=False)
OHEncoder_test = OneHotEncoder(handle_unknown='ignore', sparse=False)
oh_list = object_list + distinct_name
low_cardinality_cols = [col for col in oh_list if train_data[col].nunique() < 5]
high_cardinality_cols = [col for col in oh_list if train_data[col].nunique() >= 5]
print(low_cardinality_cols,high_cardinality_cols)
train_data.isnull().sum().sort_values(ascending = False)
Onehog_Dataframe = []
for col in low_cardinality_cols:
    Onehog_list = OneHotEncoder().fit_transform(train_data[col].values.reshape(-1, 1)).toarray()
    num_col = train_data[col].nunique()
    cols = ['{}_{}'.format(col, num_col) for num_col in range(1, num_col + 1)]
    Onehog_df = pd.DataFrame(Onehog_list, columns=cols)
    Onehog_df.index = train_data.index
    Onehog_Dataframe.append(Onehog_df)

train_data = pd.concat([*Onehog_Dataframe,train_data[continuous_name],train_data[['SalePrice']],train_data[high_cardinality_cols]], axis = 1)
train_data.tail()
train_data.shape
train_data.tail()
train_data.dropna(axis = 0, inplace = True)
train_data.tail()
train_data.isnull().sum().sort_values(ascending = False)
Onehog_Dataframe_test = []
for col in low_cardinality_cols:
    Onehog_list = OneHotEncoder().fit_transform(test_data[col].values.reshape(-1, 1)).toarray()
    num_col = test_data[col].nunique()
    cols = ['{}_{}'.format(col, num_col) for num_col in range(1, num_col + 1)]
    Onehog_df = pd.DataFrame(Onehog_list, columns=cols)
    Onehog_df.index = test_data.index
    Onehog_Dataframe_test.append(Onehog_df)
    
#test_data_onehog = pd.DataFrame(OHEncoder_test.fit_transform(test_data[low_cardinality_cols]), index = range(1460,2919))
test_data = pd.concat([*Onehog_Dataframe_test,test_data[continuous_name],test_data[high_cardinality_cols]], axis = 1)
test_data.tail()
test_data_columns = test_data.columns.tolist()
train_data_columns = train_data.columns.tolist()
train_data_columns.remove('SalePrice')
for name in test_data_columns:
    if name not in train_data_columns:
        test_data.drop(name, axis = 1, inplace = True)
for name in train_data_columns:
    if name not in test_data_columns:
        train_data.drop(name, axis = 1, inplace = True)
train_data.shape
test_data.shape
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.preprocessing import RobustScaler
columns = continuous_name
train_data[columns] = RobustScaler(quantile_range = (15,85)).fit_transform(train_data[columns])
test_data[columns] = RobustScaler(quantile_range = (15,85)).fit_transform(test_data[columns])
gbr = ensemble.GradientBoostingRegressor(learning_rate=0.02, n_estimators=2000,
                                           max_depth=5, min_samples_split=2,
                                           loss='ls', max_features=35)
#gbr.fit(train_data.drop('SalePrice',axis = 1),train_data['SalePrice'])
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,mean_absolute_error
#StratifiedKFold
train_X = train_data.drop('SalePrice',axis = 1)
train_y = train_data['SalePrice']
num = 0
N = 5
saleprice_lis = []
accuracy_score_lis = []
skf = StratifiedKFold(n_splits=N, random_state=5, shuffle=False)
saleprice = pd.DataFrame(np.zeros((len(test_data), N)), columns=['Fold_{}'.format(i) for i in range(1, N + 1)])
for train_index, test_index in skf.split(train_X, train_y):
    num +=1
    X_train1, X_test1 = train_X.iloc[train_index,:], train_X.iloc[test_index,:]
    y_train1, y_test1 = train_y[train_index], train_y[test_index]
    gbr.fit(X_train1, y_train1)
    
    saleprice.loc[:, 'Fold_{}'.format(num)] = gbr.predict(test_data)
    prediction = gbr.predict(X_test1)
    #auc score
    auc_score = mean_absolute_error(y_test1, prediction)
    accuracy_score_lis = accuracy_score_lis + [auc_score]
    print("MAE score: ", auc_score)
train_data.shape
test_data.shape
submit = saleprice.sum(axis=1) / N
submission = pd.DataFrame({'Id': np.array(list(range(1461,2920))),
                           'SalePrice': submit})
submission.to_csv("../../kaggle/working/submission.csv", index=False)
submission
#13878.07430
#imporvement methods:
#gridsearchCV
#test and add more prediction methods
#fillNA with best correlation for each columns