#Let us start by importing the libraries 



import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

import numpy as np 

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error 

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

%matplotlib inline
housing_data = pd.read_table('/kaggle/input/AmesHousing.tsv', delimiter= '\t')
housing_data.head()
#let us create the functions first that we will use in our pipeline

#for now let us create the functions simply and we can update them in the later cells



def tranform_features(df): 

    return df





def select_features(df): 

    return df[['Gr Liv Area', 'SalePrice']]





def train_and_test(df):

    train = df.iloc[0:1460,:]

    test = df.iloc[1460:,:]

    

    selected_columns = train.select_dtypes(include = ['integer','float'])

    #let'sonly select the columns that we get from the select_features function

    columns_to_be_selected = select_features(df).columns.values.tolist()

    selected_columns = selected_columns[columns_to_be_selected]

    #drop the SalePrice column since it is our target variable

    selected_columns = selected_columns.drop(columns = ['SalePrice'])

    final_features = selected_columns.columns.values.tolist()

    

    lr = LinearRegression()

    lr.fit(selected_columns[final_features], train['SalePrice'])

    predictions = lr.predict(test[final_features])

    mse = mean_squared_error(test['SalePrice'], predictions)

    rmse = mse ** (1/2)

    

    return rmse

    
#Let us quickly test our functions 

tranform_features(housing_data)

select_features(housing_data)

train_and_test(housing_data)
housing_data.info()
cutoff_point = housing_data.shape[0] * 0.05

cutoff_point
cleaning_data = housing_data.copy()
cleaning_data.isnull().sum()
missing_data = cleaning_data.isnull().sum()

columns_drop =[]



for x in missing_data.index.values.tolist(): 

    if missing_data[x] > cutoff_point:

        columns_drop.append(x)

    

print(columns_drop)



cleaning_data = cleaning_data.drop(columns = columns_drop)

cleaning_data.shape
categorical_data = cleaning_data.select_dtypes(include=['object'])
categorical_data.columns
categorical_data.isnull().sum()
categorical_missing_data_columns = categorical_data.isnull().sum()

categorical_missing_data_columns = categorical_missing_data_columns.sort_values(ascending=False)

drop_columns = categorical_missing_data_columns[categorical_missing_data_columns > 0]

cols = drop_columns.index.values.tolist()
cleaning_data = cleaning_data.drop(columns = cols)
cleaning_data.isnull().sum()
numeric_columns = cleaning_data.select_dtypes(include=['float', 'integer'])
columns_to_impute = numeric_columns.loc[:,(numeric_columns.isnull().sum() > 0)].columns
columns_to_impute
numerical_mode = housing_data[columns_to_impute].mode()

numerical_mode
numerical_mode = numerical_mode.to_dict(orient='records')[0]

numerical_mode
cleaning_data.fillna(numerical_mode, inplace=True)
cleaning_data.isnull().sum()
cleaning_data.info()
cleaning_data[['Overall Qual', 'Overall Cond']].head()
cleaning_data[['Year Built', 'Year Remod/Add']].head()
cleaning_data[['Year Built', 'Year Remod/Add']].dtypes
cleaning_data['years_until_remod'] = cleaning_data['Year Remod/Add'] - cleaning_data['Year Built']
cleaning_data['years_until_remod'].value_counts()
cleaning_data[cleaning_data['years_until_remod'] < 0]
cleaning_data.drop(index = 850, inplace=True)
cleaning_data.drop(columns=['Year Remod/Add','Year Built'], inplace=True)
cleaning_data[['Order', 'PID']].head()
cleaning_data.drop(columns = ['Order','PID','Mo Sold','Yr Sold','Sale Type','Sale Condition'], inplace=True)
cleaning_data.shape
numerical_cols = cleaning_data.select_dtypes(include = ['integer', 'float']) 
numerical_cols_corr = numerical_cols.corr()
plt.figure(figsize = (12,10))

sns.heatmap(numerical_cols_corr)
cleaning_data.drop(columns = ['Garage Cars', 'TotRms AbvGrd'], inplace=True)
cleaning_data.shape
new_numerical_cols = cleaning_data.select_dtypes(include = ['float', 'integer'])

new_numerical_cols = new_numerical_cols.corr()

new_numerical_cols['SalePrice'].abs().sort_values()
corr_below = new_numerical_cols['SalePrice'].abs().sort_values()

corr_below = corr_below[corr_below < 0.3].index.values.tolist()

corr_below
cleaning_data.drop(columns = corr_below, inplace = True)
cleaning_data.shape
categorical_cols = cleaning_data.select_dtypes(include=['object'])

categorical_cols.shape
for x in categorical_cols.columns.values: 

    freq = categorical_cols[x].value_counts()

    print('The unique values in ', x,' are :')

    print(freq)

    print('---------------------------------')
unique_heavy_cols = [x for x in categorical_cols.columns if len(categorical_cols[x].value_counts()) > 10]
unique_heavy_cols
cleaning_data.drop(columns = unique_heavy_cols, inplace=True)
cleaning_data.shape
categorical_cols.drop(columns = unique_heavy_cols, inplace=True)
categorical_cols.shape
categorical_cols['Street'].head()
for col in categorical_cols.columns: 

    categorical_cols[col] = categorical_cols[col].astype('category')

    cleaning_data[col] = cleaning_data[col].astype('category')
categorical_cols.info()
print('Unique values :')

print(categorical_cols['Street'].value_counts())

print('\n')

print('Category data type conversion :')

print(categorical_cols['Street'].cat.codes.value_counts())
dummy_values = pd.get_dummies(categorical_cols)
dummy_values.head()
for col in categorical_cols.columns: 

    dummies = pd.get_dummies(cleaning_data[col])

    cleaning_data = pd.concat([cleaning_data,dummies], axis = 1)

    del cleaning_data[col]

    
cleaning_data.head()
def tranform_features(df):

    

    df_copy = df.copy()

    cutoff = df_copy.shape[0] * 0.05

    

    #drop numerical columns with missing data

    numerical_cols = df_copy.select_dtypes(include = ['integer','float'])

    missing_numerical = numerical_cols.isnull().sum()

    drop_numerical_columns = []

    for x in missing_numerical.index.values.tolist():

        if missing_numerical[x] > cutoff:

            drop_numerical_columns.append(x)

    df_copy.drop(columns = drop_numerical_columns, inplace = True)

    

    #drop categorical data with missing values

    categorical_data = df_copy.select_dtypes(include = ['object'])

    categorical_missing_data = categorical_data.isnull().sum().sort_values(ascending = False)

    drop_columns = categorical_missing_data[categorical_missing_data > 0]

    cat_cols = drop_columns.index.values.tolist()

    df_copy.drop(columns = cat_cols , inplace=True)

    

    #impute the remaining missing values with summary statistic

    numerical = df_copy.select_dtypes(include = ['integer','float'])

    numerical_cols_impute = numerical.loc[:,(numerical.isnull().sum() > 0)].columns

    numerical_mode = df_copy[numerical_cols_impute].mode().to_dict(orient = 'records')[0]

    df_copy.fillna(numerical_mode, inplace = True)

    

    #feature engineering - adding new features

    df_copy['year_until_remod'] = df_copy['Year Remod/Add'] - df_copy['Year Built']

    df_copy.drop(index = 850, inplace = True)

    df_copy.drop(columns = ['Year Remod/Add','Year Built'], inplace = True)

    df_copy.drop(columns = ['Order','PID','Mo Sold','Yr Sold','Sale Type','Sale Condition'], inplace = True)

    

    

    return df_copy
def select_features(df_copy, correlation_cutoff, uniqueness_cutoff):

    #two factors to consider - Correlation with target column and correlation with other features

    df_copy2 = df_copy.copy()

    df_copy2 = df_copy2.drop(columns = ['Garage Cars','TotRms AbvGrd'])

    

    #select numerical cols and base on correlation values decide which columns to keep

    numerical_cols = df_copy2.select_dtypes(include = ['integer','float'])

    numerical_correlation = numerical_cols.corr()

    target_correl = numerical_correlation['SalePrice'].abs().sort_values()

    corr_below_cutoff = target_correl[target_correl < correlation_cutoff].index.values.tolist()

    df_copy2 = df_copy2.drop(columns = corr_below_cutoff)

    

    #select categorical columns and convert them to numerical variables

    #dropping columns with alot of unique values

    categorical_only = df_copy2.select_dtypes(include = ['object'])

    unique_heavy_columns = [col for col in categorical_only.columns if len(categorical_only[col].value_counts()) > uniqueness_cutoff]

    df_copy2 = df_copy2.drop(columns = unique_heavy_columns)

    

    #converting the remaining categorical columns to dummies

    categorical_columns_only = df_copy2.select_dtypes(include = ['object'])

    for columns in categorical_columns_only.columns: 

        df_copy2[columns] = df_copy2[columns].astype('category')

    

    #converting to dummies

    for columns in categorical_columns_only.columns:

        dummies = pd.get_dummies(df_copy2[columns])

        df_copy2 = pd.concat([df_copy2,dummies], axis = 1)

        del df_copy2[columns]

    

    return df_copy2

    

    

    
def train_and_test(df_copy2,k):

    df_copy3 = df_copy2.copy()

    numeric = df_copy3.select_dtypes(include = ['integer','float'])

    columns_numeric = numeric.drop(columns = ['SalePrice'])

    cols = columns_numeric.columns.values.tolist()

    lr = LinearRegression()

    

    

    if k == 0: 

        train = df_copy3[0:1460]

        test = df_copy3[1460:]

        lr.fit(train[cols], train['SalePrice'])

        predictions = lr.predict(test[cols])

        mse = mean_squared_error(test['SalePrice'],predictions)

        rmse = mse ** (1/2)

        

        return rmse

    

    if k == 1: 

        shuffle_df = df_copy3.sample(frac = 1)

        shuffle_df = shuffle_df.reset_index()

        shuffle_df = shuffle_df.drop(columns = ['index'])

        

        #When we call the reset_index function, a new index column is added. The rows are still sorted.

        #Hence we decided to drop the index column

        

        fold_one = shuffle_df[0:1460]

        fold_two = shuffle_df[1460:]

        

        lr.fit(fold_one[cols],fold_one['SalePrice'])

        predictions_one = lr.predict(fold_two[cols])

        mse_one = mean_squared_error(fold_two['SalePrice'], predictions_one)

        rmse_one = mse_one ** (1/2)

        

        lr.fit(fold_two[cols], fold_two['SalePrice'])

        predictions_two = lr.predict(fold_one[cols])

        mse_two = mean_squared_error(fold_one['SalePrice'], predictions_two)

        rmse_two = mse_two ** (1/2)

        

        avg_rmse = np.mean([rmse_one,rmse_two])

        

        return avg_rmse

    

    else:

        #if k is more than one, then we perform KFold cross validation

        kf = KFold(n_splits = k, shuffle = True)

        mses = cross_val_score(lr,df_copy3[cols], df_copy3['SalePrice'], scoring = 'neg_mean_squared_error', cv = kf)

        rmses = np.sqrt(np.absolute(mses))

        avg_rmse_k_fold = np.mean(rmses)

        

        return avg_rmse_k_fold

        
final_test_data = pd.read_table('/kaggle/input/AmesHousing.tsv', delimiter='\t')

transformed_data = tranform_features(final_test_data)
selected_features_data = select_features(transformed_data, 0.3, 10)
for x in range(0,10,1): 

    print(train_and_test(selected_features_data,k = x))

    