import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy import stats
import warnings
from math import sqrt
warnings.filterwarnings('ignore')
%matplotlib inline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV,Ridge
from sklearn.model_selection import train_test_split 

data = pd.read_csv("../input/train.csv", delimiter="\t")
data.head()
data_missing = data.isnull().sum()
data_missing.shape
data.columns
data.shape
#Dropping columns having more than 5% missing values
drop_missing_cols=data_missing[data_missing>len(data)/20].sort_values()
data=data.drop(drop_missing_cols.index,axis=1)
data.shape
data.columns
data.isnull().sum().sample(5)
## Column name -> number of missing values
text_mv_counts = data.select_dtypes(include=['object']).isnull().sum().sort_values(ascending=False)

## Dropping any columns with missing values
drop_missing_cols_2 = text_mv_counts[text_mv_counts > 0]

data = data.drop(drop_missing_cols_2.index, axis=1)
data.columns
## Identify columns which has missing values <5%
num_missing = data.select_dtypes(include=['int', 'float']).isnull().sum()
fixable_numeric_cols = num_missing[(num_missing < len(data)/20) & (num_missing > 0)].sort_values()
fixable_numeric_cols
data.isnull().any().sum()
## Compute the most common value for each column in `fixable_nmeric_missing_cols`.
replacement_values_dict = data[fixable_numeric_cols.index].mode().to_dict(orient='records')[0]
replacement_values_dict

## Use `pd.DataFrame.fillna()` to replace missing values.
data = data.fillna(replacement_values_dict)
replacement_values_dict
data.isnull().any().sum()
data.shape
data.columns
data.columns
years_sold = data['Yr Sold'] - data['Year Built']
years_sold[years_sold < 0] 

years_since_remod = data['Yr Sold'] - data['Year Remod/Add']
years_since_remod[years_since_remod < 0]
## Create new columns
data['Years Before Sale'] = years_sold
data['Years Since Remod'] = years_since_remod

## Drop rows with negative values for both of these new features
data = data.drop([1702, 2180, 2181], axis=0)

## Drop  year columns
data = data.drop(["Year Built", "Year Remod/Add"], axis = 1)
data.columns
features_to_drop=['Order','PID',"Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"]
data=data.drop(features_to_drop, axis=1)
data.shape
def transform_features(data):
    data_missing = data.isnull().sum()
    drop_missing_cols=data_missing[data_missing>len(data)/20].sort_values()
    data=data.drop(drop_missing_cols.index,axis=1)
    
    ## Column name -> number of missing values
    text_mv_counts = data.select_dtypes(include=['object']).isnull().sum().sort_values(ascending=False)

    ## Dropping any columns with missing values
    drop_missing_cols_2 = text_mv_counts[text_mv_counts > 0]
    data = data.drop(drop_missing_cols_2.index, axis=1)
    print(data.shape)
    ## Identify columns which has missing values <5%
    num_missing = data.select_dtypes(include=['int', 'float']).isnull().sum()
    print(num_missing.shape)
    fixable_numeric_cols = num_missing[(num_missing < len(data)/20) & (num_missing > 0)].sort_values()
    fixable_numeric_cols
    
    
    ## Compute the most common value for each column in `fixable_nmeric_missing_cols`.
    replacement_values_dict = data[fixable_numeric_cols.index].mode().to_dict(orient='records')[0]
    replacement_values_dict

    ## replace missing values with most common values
    data = data.fillna(replacement_values_dict)
    years_sold = data['Yr Sold'] - data['Year Built']
    years_sold[years_sold < 0] 
    years_since_remod = data['Yr Sold'] - data['Year Remod/Add']
    years_since_remod[years_since_remod < 0]
    ## Create new columns
    data['Years Before Sale'] = years_sold
    data['Years Since Remod'] = years_since_remod

    ## Drop rows with negative values for both of these new features
    data = data.drop([1702, 2180, 2181], axis=0)

    ## Drop  year columns
    data = data.drop(["Year Built", "Year Remod/Add"], axis = 1)
    features_to_drop=['Order','PID',"Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"]
    data=data.drop(features_to_drop, axis=1)
    
    return data

def select_features(data):
    return data[["Gr Liv Area", "SalePrice"]]

def train_and_test(data):
    train=data[:1460]
    test=data[1460:]
    num_train=train.select_dtypes(include=['integer','float'])
    print(num_train.head())
    num_test=test.select_dtypes(include=['integer','float'])
    features=num_train.columns.drop('SalePrice')
    lr=LinearRegression()
    lr.fit(train[features], train['SalePrice'])
    prediction=lr.predict(test[features])
    mse = mean_squared_error(test["SalePrice"], prediction)
    rmse = np.sqrt(mse)
    return rmse

data = pd.read_csv("../input/train.csv", delimiter="\t")
transform_df=transform_features(data)
features_df=select_features(transform_df)
rmse=train_and_test(features_df)
rmse

transform_df.columns
num_feature_df=transform_df.select_dtypes(include=['float','int64'])
num_feature_df.shape
num_feature_df.columns
cat_feature_df = transform_df.select_dtypes(include=['object'])
cat_feature_df.shape
transform_df['SalePrice'].dtype
cat_feature_df.columns
corr_score=num_feature_df.corr()['SalePrice'].abs().sort_values()
corr_score
num_feature_df=num_feature_df.drop(corr_score[corr_score<0.4].index, axis=1)
num_feature_df.shape
%matplotlib inline
import seaborn as sns
def plot_features(df):
    
    sns.set(color_codes=True)
    for i, col in enumerate(df.columns):
        plt.figure(i)
        sns.countplot(x=col, data=df)
plot_features(num_feature_df)
# Following variables are ordinal and not numeric. So removing those features from numeric features and moving them to categorical 
ordinal_cols=['Overall Qual',  'Full Bath', 'TotRms AbvGrd', 'Fireplaces','Garage Cars']
print(type(ordinal_cols))
print(type(cat_feature_df))
cat_feature_df=pd.concat([cat_feature_df, num_feature_df[ordinal_cols]],axis=1)
num_feature_df=num_feature_df.drop(ordinal_cols, axis=1)
print("numeric columns: ", num_feature_df.columns)
print("categorical columns: ", cat_feature_df.columns)
x = num_feature_df['SalePrice']
sns.distplot(x);
fig, ax = plt.subplots(figsize=(6,6))
sns.set_style("whitegrid")
sns.boxplot(x="SalePrice", data=num_feature_df,orient = 'v',ax = ax)
plt.show();
num_feature_df['SalePrice'][num_feature_df['SalePrice']>600000].count()
#box plot overallqual/saleprice
var = 'Overall Qual'
data1 = pd.concat([num_feature_df['SalePrice'], cat_feature_df[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data1)
fig.axis(ymin=0, ymax=800000);
def plot_target_n_features(df): 
    sns.set(color_codes=True)
    for i, col in enumerate(df.columns):
        plt.figure(i)
        sns.regplot(x=col, y=num_feature_df['SalePrice'], data=df)
plot_target_n_features(num_feature_df.iloc[:,num_feature_df.columns!='SalePrice'])
plot_features(num_feature_df)
pandas_profiling.ProfileReport(num_feature_df)
x = num_feature_df['SalePrice']
sns.distplot(x);
num_feature_df.hist()
plt.show()
num_feature_df.columns
#pandas_profiling.ProfileReport(df_log)
# Find out the number of levels each categorical column have
unique_count=transform_df[list(cat_feature_df)].apply(lambda x: len(x.value_counts())).sort_values()

#Drop columns having more than 10 levels
cols_to_drop=unique_count[unique_count>10].index
transform_df=transform_df.drop(cols_to_drop, axis=1)
transform_df.columns

# convert categorical features to dummy variables
text_cols=transform_df.select_dtypes(include=['object'])
for col in text_cols:
    transform_df[col]=transform_df[col].astype('category')
dummy_df=pd.get_dummies(transform_df.select_dtypes(include=['category']))

transform_df=pd.concat([transform_df, dummy_df],axis=1)
transform_df=transform_df.drop(text_cols, axis=1)
transform_df.shape

transform_df.columns
def transform_features(data):
    data_missing = data.isnull().sum()
    drop_missing_cols=data_missing[data_missing>len(data)/20].sort_values()
    data=data.drop(drop_missing_cols.index,axis=1)
    
    ## Column name -> number of missing values
    text_mv_counts = data.select_dtypes(include=['object']).isnull().sum().sort_values(ascending=False)

    ## Dropping any columns with missing values
    drop_missing_cols_2 = text_mv_counts[text_mv_counts > 0]
    data = data.drop(drop_missing_cols_2.index, axis=1)
    
    ## Identify columns which has missing values <5%
    num_missing = data.select_dtypes(include=['int', 'float']).isnull().sum()
    #print(num_missing.shape)
    #num_missing1 = data.select_dtypes(include=['int64', 'float']).isnull().sum()
    #print(num_missing1.shape)
    fixable_numeric_cols = num_missing[(num_missing < len(data)/20) & (num_missing > 0)].sort_values()
    fixable_numeric_cols
    
    
    ## Compute the most common value for each column in `fixable_nmeric_missing_cols`.
    replacement_values_dict = data[fixable_numeric_cols.index].mode().to_dict(orient='records')[0]
    replacement_values_dict

    ## Use `pd.DataFrame.fillna()` to replace missing values.
    data = data.fillna(replacement_values_dict)
    years_sold = data['Yr Sold'] - data['Year Built']
    years_sold[years_sold < 0] 
    years_since_remod = data['Yr Sold'] - data['Year Remod/Add']
    years_since_remod[years_since_remod < 0]
    ## Create new columns
    data['Years Before Sale'] = years_sold
    data['Years Since Remod'] = years_since_remod

    ## Drop rows with negative values for both of these new features
    data = data.drop([1702, 2180, 2181], axis=0)

    ## Drop  year columns
    data = data.drop(["Year Built", "Year Remod/Add"], axis = 1)
    features_to_drop=['Order','PID',"Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"]
    data=data.drop(features_to_drop, axis=1)
    
    return data

def select_features(df):
    #Extract numeric and categorical features out of the dataset
    num_feature_df=df.select_dtypes(include=['float','int64'])
    cat_feature_df = df.select_dtypes(include=['object'])
    
    #find correlation between target and the features
    corr_score=num_feature_df.corr()['SalePrice'].abs().sort_values()
    
    #remove targets with correlation less than 0.4
    num_feature_df=num_feature_df.drop(corr_score[corr_score<0.4].index, axis=1)
    
    #move ordinal variables to categorical features
    
    ordinal_cols=['Overall Qual',  'Full Bath', 'TotRms AbvGrd', 'Fireplaces','Garage Cars']
    cat_feature_df=pd.concat([cat_feature_df, num_feature_df[ordinal_cols]],axis=1)
    num_feature_df=num_feature_df.drop(ordinal_cols, axis=1)

    #Extract categorical features out of the dataset
    text_cols = df.select_dtypes(include=['object'])
    
    # Find out the number of levels each categorical column have
    unique_count=df[list(text_cols)].apply(lambda x: len(x.value_counts())).sort_values()
    
    #Drop columns having more than 10 levels
    cols_to_drop=unique_count[unique_count>10].index
    df=df.drop(cols_to_drop, axis=1)
    
    # convert categorical features to dummy variables
    text_cols=df.select_dtypes(include=['object'])
    for col in text_cols:
        df[col]=df[col].astype('category')
    dummy_df=pd.get_dummies(df.select_dtypes(include=['category']))

    df=pd.concat([df, dummy_df],axis=1)
    df=df.drop(text_cols, axis=1)
    return df

def train_and_test(df):
    features=df.columns.drop('SalePrice')
    model=LinearRegression()
    kf=KFold(10, shuffle=True, random_state=1)
    mses=cross_val_score(model, df[features], df["SalePrice"], cv=kf, scoring='neg_mean_squared_error')
    rmses=np.sqrt(np.absolute(mses))
    avg_rmse=np.mean(rmses)
    return avg_rmse
    

data = pd.read_csv("../input/train.csv", delimiter="\t")
transform_df=transform_features(data)
features_df=select_features(transform_df)
rmse=train_and_test(features_df)
rmse

num_feature_df.shape
# Remove outliers from the target
def remove_outliers(df,column_name, cutoff):
    #[num_feature_df['SalePrice']>600000].count()
    df=df[df[column_name]<cutoff]
    return df
rem_out_df=remove_outliers(num_feature_df, 'SalePrice',600000)
rem_out_df.shape
def transform_features(data):
    data_missing = data.isnull().sum()
    drop_missing_cols=data_missing[data_missing>len(data)/20].sort_values()
    data=data.drop(drop_missing_cols.index,axis=1)
    
    ## Column name -> number of missing values
    text_mv_counts = data.select_dtypes(include=['object']).isnull().sum().sort_values(ascending=False)

    ## Dropping any columns with missing values
    drop_missing_cols_2 = text_mv_counts[text_mv_counts > 0]
    data = data.drop(drop_missing_cols_2.index, axis=1)
    
    ## Identify columns which has missing values <5%
    num_missing = data.select_dtypes(include=['int', 'float']).isnull().sum()
    fixable_numeric_cols = num_missing[(num_missing < len(data)/20) & (num_missing > 0)].sort_values()
    fixable_numeric_cols
    
    
    ## Compute the most common value for each column in `fixable_nmeric_missing_cols`.
    replacement_values_dict = data[fixable_numeric_cols.index].mode().to_dict(orient='records')[0]
    replacement_values_dict

    ## Use `pd.DataFrame.fillna()` to replace missing values.
    data = data.fillna(replacement_values_dict)
    years_sold = data['Yr Sold'] - data['Year Built']
    years_sold[years_sold < 0] 
    years_since_remod = data['Yr Sold'] - data['Year Remod/Add']
    years_since_remod[years_since_remod < 0]
    ## Create new columns
    data['Years Before Sale'] = years_sold
    data['Years Since Remod'] = years_since_remod

    ## Drop rows with negative values for both of these new features
    data = data.drop([1702, 2180, 2181], axis=0)

    ## Drop  year columns
    data = data.drop(["Year Built", "Year Remod/Add"], axis = 1)
    features_to_drop=['Order','PID',"Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"]
    data=data.drop(features_to_drop, axis=1)
    
    return data

def select_features(df):
    #Extract numeric features out of the dataset
    num_feature_df=df.select_dtypes(include=['float','int64'])
    
    #find correlation between target and the features
    corr_score=num_feature_df.corr()['SalePrice'].abs().sort_values()
    
    #remove targets with correlation less than 0.4
    num_feature_df=num_feature_df.drop(corr_score[corr_score<0.4].index, axis=1)
    
    #Extract categorical features out of the dataset
    text_cols = df.select_dtypes(include=['object'])
    
    # Find out the number of levels each categorical column have
    unique_count=df[list(text_cols)].apply(lambda x: len(x.value_counts())).sort_values()
    
    #Drop columns having more than 10 levels
    cols_to_drop=unique_count[unique_count>10].index
    df=df.drop(cols_to_drop, axis=1)
    
    # convert categorical features to dummy variables
    text_cols=df.select_dtypes(include=['object'])
    for col in text_cols:
        df[col]=df[col].astype('category')
    dummy_df=pd.get_dummies(df.select_dtypes(include=['category']))

    df=pd.concat([df, dummy_df],axis=1)
    df=df.drop(text_cols, axis=1)
    return df

def RF_train_and_test(df):
    features=df.columns.drop('SalePrice')
    kf=KFold(10, shuffle=True, random_state=1)
    rmse_values = list()
    hyp_rmse = dict()
    hyper_params = [x for x in range(1,31)]
    for k in hyper_params:
        model=RandomForestRegressor(n_jobs=-1, random_state=0, n_estimators=20, max_depth=k, criterion='mse')
        mses=cross_val_score(model, df[features], df["SalePrice"], cv=kf, scoring='neg_mean_squared_error')
        rmses=np.sqrt(np.absolute(mses))
        avg_rmse=np.mean(rmses)
        rmse_values.append(avg_rmse)
    min_rmse=rmse_values[0]
    for index, rmse in enumerate(rmse_values): 
        if min_rmse>rmse:
            min_rmse=rmse
            key=index+1
    rmse=min_rmse
    hyp_rmse[key]=rmse
    print('Best max_depth and corresponding rmse for the Random Forest is: ',hyp_rmse)
    %matplotlib inline
    plt.plot(hyper_params,rmse_values)
    plt.show()
    return key,rmse

def train_and_test(df):
    features=df.columns.drop('SalePrice')
    model=LinearRegression()
    kf=KFold(10, shuffle=True, random_state=1)
    mses=cross_val_score(model, df[features], df["SalePrice"], cv=kf, scoring='neg_mean_squared_error')
    rmses=np.sqrt(np.absolute(mses))
    avg_rmse=np.mean(rmses)
    return avg_rmse

data = pd.read_csv("../input/train.csv", delimiter="\t")

transform_df=transform_features(data)
rem_out_df=remove_outliers(transform_df, 'SalePrice',600000)
features_df=select_features(rem_out_df)
rmse=train_and_test(features_df)
rmse
def transform_features(data):
    drop_missing_cols=data_missing[data_missing>len(data)/20].sort_values()
    data=data.drop(drop_missing_cols.index,axis=1)
    
    ## Column name -> number of missing values
    text_mv_counts = data.select_dtypes(include=['object']).isnull().sum().sort_values(ascending=False)

    ## Dropping any columns with missing values
    drop_missing_cols_2 = text_mv_counts[text_mv_counts > 0]
    data = data.drop(drop_missing_cols_2.index, axis=1)
    
    ## Identify columns which has missing values <5%
    num_missing = data.select_dtypes(include=['int', 'float']).isnull().sum()
    fixable_numeric_cols = num_missing[(num_missing < len(data)/20) & (num_missing > 0)].sort_values()
    fixable_numeric_cols
    
    
    ## Compute the most common value for each column in `fixable_nmeric_missing_cols`.
    replacement_values_dict = data[fixable_numeric_cols.index].mode().to_dict(orient='records')[0]
    replacement_values_dict

    ## Use `pd.DataFrame.fillna()` to replace missing values.
    data = data.fillna(replacement_values_dict)
    years_sold = data['Yr Sold'] - data['Year Built']
    years_sold[years_sold < 0] 
    years_since_remod = data['Yr Sold'] - data['Year Remod/Add']
    years_since_remod[years_since_remod < 0]
    ## Create new columns
    data['Years Before Sale'] = years_sold
    data['Years Since Remod'] = years_since_remod

    ## Drop rows with negative values for both of these new features
    data = data.drop([1702, 2180, 2181], axis=0)

    ## Drop  year columns
    data = data.drop(["Year Built", "Year Remod/Add"], axis = 1)
    features_to_drop=['Order','PID',"Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"]
    data=data.drop(features_to_drop, axis=1)
  
    
    return data

def select_features(df):
    #Extract numeric features out of the dataset
    num_feature_df=df.select_dtypes(include=['float','int64'])
    
    #find correlation between target and the features
    corr_score=num_feature_df.corr()['SalePrice'].abs().sort_values()
    
    #remove targets with correlation less than 0.4
    num_feature_df=num_feature_df.drop(corr_score[corr_score<0.4].index, axis=1)
    
    #log transform
    target=num_feature_df['SalePrice']
    num_feature_df = num_feature_df.applymap(lambda x: np.log(x+1))
    num_feature_df['SalePrice']=target
    
    #Extract categorical features out of the dataset
    text_cols = df.select_dtypes(include=['object'])
    
    # Find out the number of levels each categorical column have
    unique_count=df[list(text_cols)].apply(lambda x: len(x.value_counts())).sort_values()
    
    #Drop columns having more than 10 levels
    cols_to_drop=unique_count[unique_count>10].index
    df=df.drop(cols_to_drop, axis=1)
    
    # convert categorical features to dummy variables
    text_cols=df.select_dtypes(include=['object'])
    for col in text_cols:
        df[col]=df[col].astype('category')
    dummy_df=pd.get_dummies(df.select_dtypes(include=['category']))

    df=pd.concat([df, dummy_df],axis=1)
    df=df.drop(text_cols, axis=1)
    return df

def train_and_test(df):
    features=df.columns.drop('SalePrice')
    model=LinearRegression()
    kf=KFold(10, shuffle=True, random_state=1)
    mses=cross_val_score(model, df[features], df["SalePrice"], cv=kf, scoring='neg_mean_squared_error')
    rmses=np.sqrt(np.absolute(mses))
    avg_rmse=np.mean(rmses)

    return avg_rmse

data = pd.read_csv("../input/train.csv", delimiter="\t")

transform_df=transform_features(data)
rem_out_df=remove_outliers(transform_df, 'SalePrice',600000)
features_df=select_features(rem_out_df)
rmse=train_and_test(features_df)
rmse
def ridge_train_and_test(df): 
    features=df.columns.drop('SalePrice')
    X_train,X_test,y_train,y_test = train_test_split(df[features],df['SalePrice'],test_size = 0.1, random_state = 42)
    alpha = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60]
    for a in alpha:
        ridge = Ridge(fit_intercept=True, alpha=a, random_state=42)
        # computing the RMSE on test data
        ridge.fit(X_train,y_train)
        y_pred = ridge.predict(X_test)
        rmse = sqrt(mean_squared_error(y_test, y_pred))
    return(rmse)

def ridge10K_train_and_test(df): 
    features=df.columns.drop('SalePrice')
    alpha = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60]
    for alpha in alpha:
        model = Ridge(alpha = alpha)
        # computing RMSE using 10-fold cross validation
        kf = KFold(n_splits=10,shuffle=True, random_state=42)
        mses=cross_val_score(model, df[features], df["SalePrice"], cv=kf, scoring='neg_mean_squared_error')
        rmses=np.sqrt(np.absolute(mses))
        avg_rmse=np.mean(rmses)
    return avg_rmse
def ridgecv_train_and_test(df): 
    features=df.columns.drop('SalePrice')
    X_train,X_test,y_train,y_test = train_test_split(df[features],df['SalePrice'],test_size = 0.33, random_state = 42)

    ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60], store_cv_values=False, cv= 10)
    ridge.fit(X_train,y_train)
    alpha = ridge.alpha_
    print('best alpha for ridgeCV',alpha)
    print("Try again for more precision with alphas centered around " + str(alpha))
    ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],store_cv_values=False, cv=10)
    ridge.fit(X_train, y_train)
    alpha = ridge.alpha_
    print("Best alpha in focused search for ridgeCV :", alpha)
    y_test_rdg = ridge.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_test_rdg))
    return rmse


def transform_features(data):
    drop_missing_cols=data_missing[data_missing>len(data)/20].sort_values()
    data=data.drop(drop_missing_cols.index,axis=1)
    
    ## Column name -> number of missing values
    text_mv_counts = data.select_dtypes(include=['object']).isnull().sum().sort_values(ascending=False)

    ## Dropping any columns with missing values
    drop_missing_cols_2 = text_mv_counts[text_mv_counts > 0]
    data = data.drop(drop_missing_cols_2.index, axis=1)
    
    ## Identify columns which has missing values <5%
    num_missing = data.select_dtypes(include=['int', 'float']).isnull().sum()
    fixable_numeric_cols = num_missing[(num_missing < len(data)/20) & (num_missing > 0)].sort_values()
    fixable_numeric_cols
    
    
    ## Compute the most common value for each column in `fixable_nmeric_missing_cols`.
    replacement_values_dict = data[fixable_numeric_cols.index].mode().to_dict(orient='records')[0]
    replacement_values_dict

    ## Use `pd.DataFrame.fillna()` to replace missing values.
    data = data.fillna(replacement_values_dict)
    years_sold = data['Yr Sold'] - data['Year Built']
    years_sold[years_sold < 0] 
    years_since_remod = data['Yr Sold'] - data['Year Remod/Add']
    years_since_remod[years_since_remod < 0]
    ## Create new columns
    data['Years Before Sale'] = years_sold
    data['Years Since Remod'] = years_since_remod

    ## Drop rows with negative values for both of these new features
    data = data.drop([1702, 2180, 2181], axis=0)

    ## Drop  year columns
    data = data.drop(["Year Built", "Year Remod/Add"], axis = 1)
    features_to_drop=['Order','PID',"Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"]
    data=data.drop(features_to_drop, axis=1)
    
    return data

def select_features(df):
    #Extract numeric features out of the dataset
    num_feature_df=df.select_dtypes(include=['float','int64'])
    
    #find correlation between target and the features
    corr_score=num_feature_df.corr()['SalePrice'].abs().sort_values()
    
    #remove targets with correlation less than 0.4
    num_feature_df=num_feature_df.drop(corr_score[corr_score<0.4].index, axis=1)
    
    #Extract categorical features out of the dataset
    text_cols = df.select_dtypes(include=['object'])
    
    # Find out the number of levels each categorical column have
    unique_count=df[list(text_cols)].apply(lambda x: len(x.value_counts())).sort_values()
    
    #Drop columns having more than 10 levels
    cols_to_drop=unique_count[unique_count>10].index
    df=df.drop(cols_to_drop, axis=1)
    
    # convert categorical features to dummy variables
    text_cols=df.select_dtypes(include=['object'])
    for col in text_cols:
        df[col]=df[col].astype('category')
    dummy_df=pd.get_dummies(df.select_dtypes(include=['category']))

    df=pd.concat([df, dummy_df],axis=1)
    df=df.drop(text_cols, axis=1)
    return df

def train_and_test(df):
    features=df.columns.drop('SalePrice')
    model=LinearRegression()
    kf=KFold(10, shuffle=True, random_state=1)
    mses=cross_val_score(model, df[features], df["SalePrice"], cv=kf, scoring='neg_mean_squared_error')
    rmses=np.sqrt(np.absolute(mses))
    avg_rmse=np.mean(rmses)
    return avg_rmse

data = pd.read_csv("../input/train.csv", delimiter="\t")

transform_df=transform_features(data)
rem_out_df=remove_outliers(transform_df, 'SalePrice',600000)
features_df=select_features(rem_out_df)
LR_score = train_and_test(features_df)
print("RMSE for linear regression:",LR_score)
ridge_score=ridge_train_and_test(features_df)
print("RMSE for ridge regression:",ridge_score)
ridge10K_score=ridge10K_train_and_test(features_df)
print("RMSE for ridge regression with 10-K validation:",ridge10K_score)
ridgecv_score=ridgecv_train_and_test(features_df)
print("RMSE for RidgeCV regression:",ridgecv_score)
def transform_features(data):
    drop_missing_cols=data_missing[data_missing>len(data)/20].sort_values()
    data=data.drop(drop_missing_cols.index,axis=1)
    
    ## Column name -> number of missing values
    text_mv_counts = data.select_dtypes(include=['object']).isnull().sum().sort_values(ascending=False)

    ## Dropping any columns with missing values
    drop_missing_cols_2 = text_mv_counts[text_mv_counts > 0]
    data = data.drop(drop_missing_cols_2.index, axis=1)
    
    ## Identify columns which has missing values <5%
    num_missing = data.select_dtypes(include=['int', 'float']).isnull().sum()
    fixable_numeric_cols = num_missing[(num_missing < len(data)/20) & (num_missing > 0)].sort_values()
    fixable_numeric_cols
    
    
    ## Compute the most common value for each column in `fixable_nmeric_missing_cols`.
    replacement_values_dict = data[fixable_numeric_cols.index].mode().to_dict(orient='records')[0]
    replacement_values_dict

    ## Use `pd.DataFrame.fillna()` to replace missing values.
    data = data.fillna(replacement_values_dict)
    years_sold = data['Yr Sold'] - data['Year Built']
    years_sold[years_sold < 0] 
    years_since_remod = data['Yr Sold'] - data['Year Remod/Add']
    years_since_remod[years_since_remod < 0]
    ## Create new columns
    data['Years Before Sale'] = years_sold
    data['Years Since Remod'] = years_since_remod

    ## Drop rows with negative values for both of these new features
    data = data.drop([1702, 2180, 2181], axis=0)

    ## Drop  year columns
    data = data.drop(["Year Built", "Year Remod/Add"], axis = 1)
    features_to_drop=['Order','PID',"Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"]
    data=data.drop(features_to_drop, axis=1)
    
    return data

def select_features(df):
    #Extract numeric features out of the dataset
    num_feature_df=df.select_dtypes(include=['float','int64'])
    
    #find correlation between target and the features
    corr_score=num_feature_df.corr()['SalePrice'].abs().sort_values()
    
    #remove targets with correlation less than 0.4
    num_feature_df=num_feature_df.drop(corr_score[corr_score<0.4].index, axis=1)
    
    #Extract categorical features out of the dataset
    text_cols = df.select_dtypes(include=['object'])
    
    # Find out the number of levels each categorical column have
    unique_count=df[list(text_cols)].apply(lambda x: len(x.value_counts())).sort_values()
    
    #Drop columns having more than 10 levels
    cols_to_drop=unique_count[unique_count>10].index
    df=df.drop(cols_to_drop, axis=1)
    
    # convert categorical features to dummy variables
    text_cols=df.select_dtypes(include=['object'])
    for col in text_cols:
        df[col]=df[col].astype('category')
    dummy_df=pd.get_dummies(df.select_dtypes(include=['category']))

    df=pd.concat([df, dummy_df],axis=1)
    df=df.drop(text_cols, axis=1)
    return df

def train_and_test(df):
    features=df.columns.drop('SalePrice')
    kf=KFold(10, shuffle=True, random_state=1)
    rmse_values = list()
    hyp_rmse = dict()
    hyper_params = [x for x in range(1,31)]
    for k in hyper_params:
        model=RandomForestRegressor(n_jobs=-1, random_state=0, n_estimators=20, max_depth=k, criterion='mse')
        mses=cross_val_score(model, df[features], df["SalePrice"], cv=kf, scoring='neg_mean_squared_error')
        rmses=np.sqrt(np.absolute(mses))
        avg_rmse=np.mean(rmses)
        rmse_values.append(avg_rmse)
    min_rmse=rmse_values[0]
    for index, rmse in enumerate(rmse_values): 
        if min_rmse>rmse:
            min_rmse=rmse
            key=index+1
    rmse=min_rmse
    hyp_rmse[key]=rmse
    print('Best max_depth and corresponding rmse for the Random Forest is: ',hyp_rmse)
    %matplotlib inline
    plt.plot(hyper_params,rmse_values)
    plt.show()
    return key,rmse


data = pd.read_csv("../input/train.csv", delimiter="\t")
transform_df=transform_features(data)
features_df=select_features(transform_df)
k, rmse=train_and_test(features_df)
rmse

def lasso_train_and_test(df):
    features=df.columns.drop('SalePrice')
    X_train,X_test,y_train,y_test = train_test_split(df[features],df['SalePrice'],test_size = 0.1, random_state = 42)
    alpha = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60]
    lasso = Lasso(max_iter=10000, normalize=True)
    coefs = []
    for a in alpha:
        lasso.set_params(alpha=a)
        lasso.fit(scale(X_train), y_train)
        coefs.append(lasso.coef_)
        pd.Series(lasso.coef_, index=X_train.columns)
        # computing the RMSE on test data
        lasso.fit(X_train,y_train)
        y_pred = lasso.predict(X_test)
        rmse = sqrt(mean_squared_error(y_test, y_pred))
    return(rmse)

def LassoCV_train_and_test(df): 
    coefs = []
    features=df.columns.drop('SalePrice')
    X_train,X_test,y_train,y_test = train_test_split(df[features],df['SalePrice'],test_size = 0.33, random_state = 42)

    lassocv = LassoCV(alphas=[0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60]
                      , cv=10, max_iter=1000, normalize=True)
    lassocv.fit(scale(X_train), y_train)
    alpha = lassocv.alpha_
    print('best alpha for lassocv',alpha)
    print("Try again for more precision with alphas centered around " + str(alpha))
    lassocv = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4]
                      , cv=10, max_iter=1000, normalize=True)
    lassocv.fit(scale(X_train), y_train)
    alpha = lassocv.alpha_
    print("Best alpha in focused search for ridgeCV :", alpha)
    #lassocv.set_params(alpha=alpha)
    coefs.append(lassocv.coef_)
    L_Coef = pd.Series(lassocv.coef_, index=X_train.columns)
    lassocv.fit(X_train, y_train)
    mean_squared_error(y_test, lassocv.predict(X_test))
    return (rmse,L_Coef)
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import scale
data = pd.read_csv("C:\\Users\\sadhana reddy\\Desktop\\Projects\\DataScience-DataAnalytics-master\\DataScience-DataAnalytics-master\\HousingPricePrediction\\AmesHousing.csv", delimiter="\t")

transform_df=transform_features(data)
rem_out_df=remove_outliers(transform_df, 'SalePrice',600000)
features_df=select_features(rem_out_df)

lasso_score=lasso_train_and_test(features_df)
print("RMSE for lasso regression:",lasso_score)

LassoCV_score,Coeff_series=LassoCV_train_and_test(features_df)
print("RMSE for LassoCV regression:",LassoCV_score)
Coeff_series.shape
Coeff_df = pd.DataFrame({'Features':Coeff_series.index,'Coefficients':Coeff_series.values})
Coeff_df[Coeff_df['Coefficients'] == 0].shape
def lasso_train_and_test(df):
    rmses=[]
    features=df.columns.drop('SalePrice')
    X_train,X_test,y_train,y_test = train_test_split(df[features],df['SalePrice'],test_size = 0.2, random_state = 42)
    alpha = [10000]
    #alpha = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60,0.001, 0.0005, 0.00002, 5]
    for a in alpha:
        lasso = Lasso(fit_intercept=True, alpha=a, max_iter=1000, random_state=42)
        # computing the RMSE on test data
        lasso.fit(X_train,y_train)
        y_pred = lasso.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmses.append(rmse)
    #print(rmses)
    #return np.mean(rmses)
    return (rmses)

data = pd.read_csv("../input/train.csv", delimiter="\t")

transform_df=transform_features(data)
rem_out_df=remove_outliers(transform_df, 'SalePrice',600000)
features_df=select_features(rem_out_df)
lasso_score=lasso_train_and_test(features_df)
print(lasso_score)
