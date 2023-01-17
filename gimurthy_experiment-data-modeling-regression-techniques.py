import numpy as np    # Data manipulation

import pandas as pd   # Dataframe manipulation 



# Generic libraries

import os          # For os related operations

import warnings    # To suppress warnings



# Libraries for plotting

import matplotlib.pyplot as plt   # For graphics

import seaborn as sns             # For graphics



# Libraries for data processing and modeling

from sklearn.preprocessing import StandardScaler                # For scaling dataset

from sklearn.preprocessing import LabelEncoder, OneHotEncoder   # For Dummy variable conversion

from sklearn import decomposition                               # For dimensionality reduction thru PCA

from sklearn.model_selection import train_test_split            # For Splitting data

from sklearn import metrics                                     # For loss functions



from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from sklearn.linear_model import Ridge



# Ignore warnings

warnings.filterwarnings("ignore")



# To show the graphs inline

%matplotlib inline
# 1.1

def cleanseNAs(df):

    """

    Function to fill the NAs

    Set the missing categorical value to 'other'

    set the missing numerical value to the Mean of the column

    

    Input:

        df - dataset to fill

        

    Return:

        df - filled dataset

        cat_cols - list of the index value of the categorical columns

    """

    na_cols = (df.isnull().sum().sort_values(ascending=False)).index.values

    col_names = df.index.values

    cat_cols = []



    for var in na_cols : 

        if df[var].dtype == np.object:

            df[var].fillna('other', inplace=True)

            cat_cols.append(df.columns.get_loc(var))

        else:

            df[var].fillna(df[var].mean(), inplace=True)

    

    return df, cat_cols



# 1.2

def setDummyVars(df, cat_cols):

    """

    Function to covert categorical variables to dummy vars

    

    Input:

        df - dataset to covert

        cat_cols - list of the index value of the categorical columns

        

    Return:

        df - converted dataset

    """

    

    print(' Categorical columns to convert - ',cat_cols)

    

    le = LabelEncoder()

    y = df.apply(le.fit_transform)

    

    ohe = OneHotEncoder(categorical_features = cat_cols)

    ohe.fit(y)

    

    dv_df = ohe.transform(y)

    

    return pd.DataFrame(dv_df.toarray())



# 1.3

def scaleDataset(df):

    """

    Function to scale the dataset

    

    Input:

        df - dataset to scale

                

    Return:

        df - Scaled dataset

    """

    

    col_names = df.columns.values

    ss = StandardScaler()

    

    return pd.DataFrame(ss.fit_transform(df), columns=col_names)



# 1.4

def applyPCA(df):

    """

    Reduce the dimensions of the dataset, by identifing the principle components.

    The number of the PC's are identified by plotting the Elbow curve of the 

    variance ratio for all the variables

    

    Input:

        df - dataset to appky PCA

                

    Return:

        df - Dataset after applying PCA 

    """

    

    # Initially run the PCA for all dimensions

    pca = decomposition.PCA()

    pca.fit(df)

    

    # Cumulative Variance that each PC explains

    var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

    

    # Plot the Elbow curve to identify the right dimenionality cutoff

    plt.plot(var)

    

    print('There is an Elbow at 200. Considering 180 variables as it provides coverage of ~90%')

    

    #Looking at above plot, considering 180 variables with 90% dimensionality coverage

    pca = decomposition.PCA(n_components=180)

    pca.fit(df)

       

    return pd.DataFrame(pca.fit_transform(df))



# 1.5

def get_train_test_data(df, Y, split_point, ratio):

    """

    Split the dataset into test, valid and test datasets along with the target data.

    

    Input:

        df - dataset to split

        Y - target dataset

        split_point - the number of rows at which to split into train and test data

        ratio - ratio of the split for train and valid data from train data

                

        Return:

        X_train - train data

        X_valid - valid data

        y_train - target data for modeling

        y_valid - target data for validation

        df_test - test data

    """

    

    df_train = df.loc[df.Id < (split_point + 1)] 

    df_test = df.loc[df.Id > (split_point)]

    

    X_train, X_valid, y_train, y_valid = train_test_split(

                                     df_train, Y,

                                     test_size=0.30,

                                     random_state=101

                                     )



    return X_train, X_valid, y_train, y_valid, df_test   



# 1.6

def run_regression(regr, X_train, y_train, X_valid, y_valid, rtype):

    """

    Function to execute the regression model and print the loss function results

        

    Input:

        regr - regression model

        X_train - train data

        y_train - train target data

        X_valid - validation data

        y_valid - validation target data

        rtype - Type of Regression model

        

    Return:

        regr - fitted regression model

    """

    

    regr.fit(X_train,y_train)



    predictions=regr.predict(X_valid)



    print(rtype + ' Regression Output - ')

    print('- MAE:', metrics.mean_absolute_error(y_valid, predictions))

    print('- MSE:', metrics.mean_squared_error(y_valid, predictions))

    print('- RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, predictions)))



    return regr
#  Import the data files

hp_train_data = pd.read_csv("../input/train.csv", header=0)

hp_test_data = pd.read_csv("../input/test.csv", header=0)



#   2.1 Check the differences in the train & test datasets

# Check the dimensions of the datasets & differences in the attributes

print('Dimensions of the Train Dataset - ' , hp_train_data.shape)

print('Dimensions of the Test Dataset - ' , hp_test_data.shape)



print('Additional column(s) in Train Dataset is - ', hp_train_data.columns.difference(hp_test_data.columns).tolist())



#   2.2 Extract Target column - Sale Price from the training dataset

saleprice = hp_train_data['SalePrice']

# Drop the column from the dataset

hp_train_data_up = hp_train_data.drop('SalePrice', axis=1)



#   2.3 Merge the datasets further cleaning

hp_all_data = pd.concat([hp_train_data_up, hp_test_data], ignore_index=True)



#   2.4 Extract the Id column from the combined dataset to a different attribute and drop the Id column

id = hp_all_data['Id']

hp_all_data.drop('Id', inplace=True, axis=1)



print('Dimensions of the Combined Dataset - ', hp_all_data.shape)
#   3.1 Set the chart parameters

sns.set(style="white")

sns.set(font_scale=1.25)

f, ax = plt.subplots(figsize=(10, 10))



#   3.2 Plot the heatmap to check the corelation

k = 10 # Number of variables to plot for heatmap

corr_data = hp_train_data.corr()

cols = corr_data.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(hp_train_data[cols].values.T)



hm = sns.heatmap(cm, cbar=True, annot=True, cbar_kws={"shrink": .5}, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()



#   3.3 Draw Scatter plots for SalesPrice with attributes which are strongly co-related to indentify the outliers.

# Some of the variables strongly corelated to SalePrice

cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

print('The following variables are better corelated to the Sales Price than te rest - ' , cols)



for i in cols:

    data = pd.concat([hp_train_data['SalePrice'], hp_train_data[i]], axis=1)

    data.plot.scatter(x=i, y='SalePrice', ylim=(0.800000));

print('\n')

print('There are some outliers in the data as evident from the scatter plot, but for now not handling them')
#   4.1 Number of NA entries for each of the columns

total_nas = hp_all_data.isnull().sum().sort_values(ascending=False)

percent_nas = (hp_all_data.isnull().sum()/hp_all_data.shape[0]).sort_values(ascending=False)

# Number of NA entries for each of the columns

missing_data = pd.concat([total_nas, percent_nas], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)



na_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageCond', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageType']

print ('This following columns have more than 5% NAs - ', cols, '. Deleting them.')



#   4.2 Drop the columns which have missing values more that 5% of the total dataset

hp_all_data.drop(na_cols, inplace=True, axis=1)

print('Dimensions of the dataset post variable deletion -', hp_all_data.shape)



#   4.3 Fill the missing values

hp_all_data, cat_cols = cleanseNAs(hp_all_data)
hp_all_data = setDummyVars(hp_all_data, cat_cols)

print('Dimensions of the dataset post Dummy Variable conversion -', hp_all_data.shape)
#   6.1 Scale data

hp_all_data = scaleDataset(hp_all_data)



#   6.2 Reduce dimensions thru PCA

hp_all_data = applyPCA(hp_all_data)

print('Dimensions of the dataset post Dimensionality Reduction -', hp_all_data.shape)



#   6.3 Add Id column back to the dataset

hp_all_data.insert(0, 'Id', id)
X_train, X_valid, y_train, y_valid, df_test = get_train_test_data(hp_all_data, saleprice, hp_train_data.shape[0], 0.3)



print('Dimensions of the Training Data -', X_train.shape)

print('Dimensions of the Validation Data -', X_valid.shape)

print('Dimensions of the Test Data -', df_test.shape)

rfregr = RandomForestRegressor(n_estimators=2000,         # No of trees in forest

                             criterion = "mse",       

                             max_features = "sqrt",     # no of features to consider for the best split

                             max_depth= 20,             # maximum depth of the tree

                             min_samples_split= 2,      # minimum number of samples required to split an internal node

                             min_impurity_decrease=0,   # Split node if impurity decreases greater than this value.

                             oob_score = True,          # whether to use out-of-bag samples to estimate error on unseen data.

                             n_jobs = -1,               # No of jobs to run in parallel

                             random_state=101,

                             verbose = 0                # Controls verbosity of process

                             )



rfregr = run_regression(rfregr, X_train, y_train, X_valid, y_valid, "Random Forest")
rregr = Ridge(alpha = 40.0,           # Regularization strength. 

              solver = "lsqr",        # auto,svd,cholesky,lsqr,sparse_cg,sag,saga

              fit_intercept=False     # Data is already normalized and centered



              )

rregr = run_regression(rregr, X_train, y_train, X_valid, y_valid, "Ridge")
xgregr = xgb.XGBRegressor(max_depth=3,

                min_child_weight=15,

                colsample_bytree=0.6,

                objective='reg:linear',

                n_estimators=4000,

                learning_rate=0.01)



xgregr = run_regression(xgregr, X_train, y_train, X_valid, y_valid, "XGBoost")
target_sp = xgregr.predict(df_test)

print(target_sp)