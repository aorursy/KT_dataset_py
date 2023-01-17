import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math



# import plot libraries

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline



# import stats to analyze and fit histograms

from scipy import stats 



# import ML libraries

from sklearn.compose import ColumnTransformer





from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print('importing:\n')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# open raw data

train_data_full = pd.read_csv('../input/home-data-for-ml-course/train.csv')

test_data_full = pd.read_csv('../input/home-data-for-ml-course/test.csv')



train_data_full.tail()
# target to model is the 'SalePrice' column



# Remove rows with missing target, separate target from predictors

train_data_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

# show statistics

train_data_full['SalePrice'].describe()
Q01 = train_data_full['SalePrice'].quantile(0.01)

Q99 = train_data_full['SalePrice'].quantile(0.99)

print("We keep house prices between %d $ and %d $"%(Q01, Q99))



train_data = train_data_full[(train_data_full['SalePrice'] > Q01) & (train_data_full['SalePrice'] < Q99)].copy(deep=True)

print("Size of new data set: ",train_data.shape)
fig = sns.distplot(train_data['SalePrice'])
train_data['log SalePrice'] = np.log(train_data['SalePrice'])



fig = sns.distplot(train_data['log SalePrice'])



print("Skewness of target was %f, reduced to %f after log transform"%(train_data['SalePrice'].skew(), train_data['log SalePrice'].skew()));
columns_to_drop = set() #will save the set of features to drop all along our analysis



columns_to_drop = {'SalePrice'} # drop as we took the log for the target and it should not be considered as a feature



# drop columns without interest

columns_to_drop = columns_to_drop | {'GarageYrBlt','Condition2', 'BsmtFinSF1','BsmtFinSF2', 'Exterior2nd', 'LotFrontage', 'MasVnrArea', \

                   'BsmtFullBath','BsmtHalfBath','HalfBath', 'GarageArea', 'PoolQC', 'MiscFeature', 'MiscVal', 'OverallCond'}



# make a deep copy to avoid modifying train_data  

new_train_data = train_data.copy(deep=True) 



# create new features by combining existing ones

new_train_data['newHouseStyle'] = train_data['BldgType'] + "_" + train_data['HouseStyle']

new_train_data['newExterQual'] = train_data['ExterQual'] + "_" + train_data['ExterCond']

new_train_data['newGarageQual'] = train_data['GarageQual'] + "_" + train_data['GarageCond'] + "_" + train_data['GarageFinish']

new_train_data['newLand'] = train_data['LandContour'] + "_" + train_data['LandSlope']

new_train_data['has1stfloor'] = (train_data['1stFlrSF'] > 0).astype(object)

new_train_data['has2ndfloor'] = (train_data['2ndFlrSF'] > 0).astype(object)

new_train_data['newPorchSF'] = train_data['OpenPorchSF'] +  train_data['EnclosedPorch'] + train_data['3SsnPorch'] + train_data['ScreenPorch']



# drop features used in building the new ones

columns_to_drop = columns_to_drop | {'BldgType', 'HouseStyle', 'ExterQual', 'ExterCond', 'GarageQual', 'GarageCond', 'GarageFinish',\

                                     'LandContour', 'LandSlope', '1stFlrSF', '2ndFlrSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'}



new_train_data.drop(columns=columns_to_drop, inplace=True)

new_train_data.head()
date_df = train_data[['YrSold','MoSold','SalePrice']].copy(deep=True)

date_df['Day'] = 1 # add a column with a day value



date_df.rename(columns={'YrSold' : 'Year', 'MoSold' : 'Month'}, inplace=True)



date_df['Date sold'] = pd.to_datetime(date_df[['Year','Month','Day']])

new_train_data['Date sold'] = date_df['Date sold'] # add to data

new_train_data.sort_values(by='Date sold', inplace=True) # sort the data by date

new_train_data['Date_sold_int'] = new_train_data['Date sold'].astype(np.int) # could be useful if we need a numerical feature related to the date



plt.figure()

sns.scatterplot(x=date_df['Date sold'],y=date_df['SalePrice'])

plt.xlim('2005-01-01', '2011-01-01')

plt.ylim(0,400000);
date_df.groupby(['Year', 'Month']).mean()[['SalePrice']].plot();

plt.title('Monthly average of sale price');
# check columns with missing data

missing_data_cols = set(new_train_data.columns[new_train_data.isna().any()].tolist())



# display the fraction of missing data

npts = len(new_train_data)

df_pct_missing = pd.DataFrame((npts - new_train_data[missing_data_cols].count())/npts)*100

df_pct_missing.columns = ['Missing data [%]']

df_pct_missing.sort_values('Missing data [%]')
# drop columns that have more than 75% values that are NAN



columns_to_drop = columns_to_drop | set(new_train_data.count()[new_train_data.count() < 0.25*max(new_train_data.count())].index.tolist())

print("We drop the following columns because more than 75% of the entries are missing: \n",new_train_data.count()[new_train_data.count() < 0.25*max(new_train_data.count())].index.tolist())
# Visualize histograms of features with missing data



categorical_cols = {cname for cname in new_train_data.columns if new_train_data[cname].dtype == "object"}

numerical_cols = {cname for cname in new_train_data.columns if new_train_data[cname].dtype in ['int64', 'float64']}



n=len(new_train_data[missing_data_cols - columns_to_drop].columns) # number of plots

f, axes = plt.subplots(nrows=(n-1)//4 +1,ncols=4,squeeze=False,figsize=(18,4*((n-1)//4 +1))) # represent them on 4 columnms

f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) # increase space between plots



for col, ax in zip(missing_data_cols - columns_to_drop, axes.flatten()[:n]):

    if col in categorical_cols:

        sns.countplot(data=new_train_data, x=col, ax=ax)

    else:

        sns.distplot(data=new_train_data, x=col, ax=ax)



plt.show()
new_train_data.fillna("missing", inplace=True)
# Select numerical columns

numerical_cols = {cname for cname in new_train_data.drop(columns=['log SalePrice']).columns if new_train_data[cname].dtype in ['int64', 'float64']}



# Visualize all numerical features

n=len(new_train_data[numerical_cols - columns_to_drop].columns) # number of plots

f, axes = plt.subplots(nrows=(n-1)//4 +1,ncols=4,squeeze=False,figsize=(18,4*((n-1)//4 +1))) # represent them on 4 columnms

f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) # increase space between plots



for col, ax in zip(numerical_cols - columns_to_drop, axes.flatten()[:n]):

    sns.regplot(data=new_train_data,x=col,y='log SalePrice', ax=ax)



plt.show()
new_train_data.drop(new_train_data['TotalBsmtSF'][new_train_data['TotalBsmtSF'] > 4000].index, inplace=True)

new_train_data.drop(new_train_data['LotArea'][new_train_data['LotArea'] > 100000].index, inplace=True)

new_train_data.drop(new_train_data['GrLivArea'][new_train_data['GrLivArea'] > 4000].index, inplace=True)



columns_to_drop = columns_to_drop | {"KitchenAbvGr", "Date_sold_int", "LowQualFinSF", "Id", "YrSold", "BsmtUnfSF", "MsSubClass", "MoSold",  'PoolArea', "newPorchSF"}
# Select numerical columns

numerical_cols = {cname for cname in new_train_data.drop(columns=['log SalePrice']).columns if new_train_data[cname].dtype in ['int64', 'float64']} 



# Visualize all numerical features

n=len(new_train_data[numerical_cols - columns_to_drop].columns) # number of plots

f, axes = plt.subplots(nrows=(n-1)//4 +1,ncols=4,squeeze=False,figsize=(18,4*((n-1)//4 +1))) # represent them on 4 columnms

f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) # increase space between plots



for col, ax in zip(numerical_cols - columns_to_drop, axes.flatten()[:n]):

    sns.distplot(new_train_data[col], ax=ax)



plt.show()
new_train_data['log GrLivArea'] = np.log(new_train_data['GrLivArea'])

new_train_data['log LotArea'] = np.log(new_train_data['LotArea'])



cols_to_log_tsf = ['GrLivArea', 'LotArea']

columns_to_drop = columns_to_drop | set(cols_to_log_tsf)

new_train_data.drop(cols_to_log_tsf, axis=1, inplace=True)





fig = plt.figure(figsize=(9,4))

ax0 = fig.add_subplot(121) # add subplot 1 (211 = 2 rows, 1 column, first plot)

ax1 = fig.add_subplot(122) # add subplot 2 

sns.distplot(new_train_data['log GrLivArea'], ax=ax0)

sns.distplot(new_train_data['log LotArea'], ax=ax1);
# sort the features as a function of their correlation to the target

numerical_cols = {cname for cname in new_train_data.drop(columns=['log SalePrice']).columns if new_train_data[cname].dtype in ['int64', 'float64']}



corr_df = new_train_data[list(numerical_cols - columns_to_drop | {'log SalePrice'})].corr()

corr_df['log SalePrice'] = np.abs(corr_df['log SalePrice']) # take absolute value

corr_df.sort_values('log SalePrice', ascending=False, inplace=True)

corr_df.drop('log SalePrice', axis=0, inplace=True)



# make a list that ranks the best features

ranked_num_cols = corr_df[['log SalePrice']].index.to_list()

print("Numerical features ranked by correlation to target: \n",ranked_num_cols)



corr_df[['log SalePrice']]
# plot a heatmap of the correlations between features

f, ax = plt.subplots(figsize=(14,12))

plt.title('Correlation between features', size=16)

sns.heatmap(new_train_data[list(numerical_cols - columns_to_drop)].corr()) 

plt.show()



new_train_data[list(numerical_cols - columns_to_drop)].corr()
columns_to_drop = columns_to_drop | set(corr_df[['log SalePrice']][corr_df['log SalePrice'].between(-0.4, 0.4)].index.to_list())
# Visualize all categorical features

categorical_cols = {cname for cname in new_train_data.columns if new_train_data[cname].dtype == "object" or new_train_data[cname].dtype == "bool"}



n=len(new_train_data[categorical_cols - columns_to_drop].columns) # number of plots

f, axes = plt.subplots(nrows=(n-1)//4 +1,ncols=4,squeeze=False,figsize=(18,4*((n-1)//4 +1))) # represent them on 4 columnms

f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) # increase space between plots



for col, ax in zip(categorical_cols - columns_to_drop, axes.flatten()[:n]):

    sns.countplot(x=col, data=new_train_data, ax=ax)



plt.show()
# remove features that have essentially one value

columns_to_drop = columns_to_drop | {"newLand", "Condition1", "PavedDrive", "Functional", "Electrical", "Utilities", "SaleCondition", "SaleType", "BsmtFinType2", "Street", "RoofMat1", "Heating", "has1stfloor"}



# remove features that do not have sufficient statistics (too many values)

columns_to_drop = columns_to_drop | {"newHouseStyle", "Exterior1st"}
# Visualize all categorical features

categorical_cols = {cname for cname in new_train_data.columns if new_train_data[cname].dtype == "object" or new_train_data[cname].dtype == "bool"}



n=len(new_train_data[categorical_cols - columns_to_drop].columns) # number of plots

f, axes = plt.subplots(nrows=(n-1)//4 +1,ncols=4,squeeze=False,figsize=(18,4*((n-1)//4 +1))) # represent them on 4 columnms

f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) # increase space between plots



for col, ax in zip(categorical_cols - columns_to_drop, axes.flatten()[:n]):

    sns.boxplot(data=new_train_data,x=col,y='log SalePrice', ax=ax)



plt.show()
# remove feature that do not seem very correlated with the price



columns_to_drop = columns_to_drop | {"LotConfig", "LotShape", "RoofStyle", "has2ndFloor", "NewGarageQual"}
nb_rows = new_train_data['log SalePrice'].count()



mean_target = new_train_data['log SalePrice'].mean()

std_target = new_train_data['log SalePrice'].std()



target_encoding_cols = set()

one_hot_cols = set()







for col in (categorical_cols - columns_to_drop) :

    

    value_list = []

    count_list = []

    mean_list = []

    

    # list of unique values

    value_list = new_train_data[col].unique()

    

    

    # loop through all possible values for the feature

    if len(value_list) == 1:

        # only 1 value --> no information

        columns_to_drop = columns_to_drop | {col}

    elif len(value_list) <=3:

        # low cardinality --> one-hot encoding is better

        

        for value in value_list:

            # count elements per value

            count_list = count_list + [new_train_data[col][new_train_data[col] == value].count()]    

           

        if max(count_list) < nb_rows * 0.95 : # check that all instances do not have the same value   

            one_hot_cols = one_hot_cols | {col}

        

    else:

        # target encoding

        

        for value in value_list:

            # count elements per value

            count_list = count_list + [new_train_data[col][new_train_data[col] == value].count()]

            mean_list = mean_list + [new_train_data['log SalePrice'][new_train_data[col] == value].mean()]

            

        delta_to_mean_list = [abs(mean - mean_target) for mean in mean_list]

        

        # select features that have a significantly different target values 

        if np.max(delta_to_mean_list) > std_target or np.mean(delta_to_mean_list) > std_target/3 : 

            target_encoding_cols = target_encoding_cols | {col}



                

print("Features that could be one-hot encoded: \n", one_hot_cols) 

print("\n")

print("Features that could be target encoded: \n", target_encoding_cols)
def data_engineering(raw_data, is_trainset):

    # preprocess DataFrame 'raw_data'    

    

    

    # replace nan 

    num_cols = {cname for cname in raw_data.columns if raw_data[cname].dtype in ['int64', 'float64']}

    cat_cols = {cname for cname in raw_data.columns if raw_data[cname].dtype == "object"}

    raw_data[num_cols].fillna(raw_data[num_cols].median(), inplace=True) # replace nan by median in numerical columns

    raw_data[cat_cols].fillna("missing", inplace=True) # replace nan by "missing" in categorical columns

    

    

    

    # leave input data unchanged

    data = pd.DataFrame(columns = raw_data.columns, index=raw_data.index)

    data = raw_data.copy(deep=True)

    

    if is_trainset:

        # remove outliers from target

        Q01 = data['SalePrice'].quantile(0.01)

        Q99 = data['SalePrice'].quantile(0.99)

        data = data[(data['SalePrice'] > Q01) & (data['SalePrice'] < Q99)]

        

        # remove outliers from training set

        data.drop(data['TotalBsmtSF'][data['TotalBsmtSF'] > 4000].index, inplace=True)

        data.drop(data['LotArea'][data['LotArea'] > 100000].index, inplace=True)

        data.drop(data['GrLivArea'][data['GrLivArea'] > 4000].index, inplace=True)

        

    

    # create new columns

    data['newHouseStyle'] = data['BldgType'] + "_" + data['HouseStyle']

    data['newExterQual'] = data['ExterQual'] + "_" + data['ExterCond']

    data['newGarageQual'] = data['GarageQual'] + "_" + data['GarageCond'] + "_" + data['GarageFinish']

    data['newLand'] = data['LandContour'] + "_" + data['LandSlope']

    data['has1stfloor'] = (data['1stFlrSF'] > 0).astype(bool)

    data['has2ndfloor'] = (data['2ndFlrSF'] > 0).astype(bool)

    data['newPorchSF'] = data['OpenPorchSF'] +  data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch']

    data['hasPool'] = (data['PoolArea'] > 0).astype(int)

    

    date_df = data[['YrSold','MoSold']].copy(deep=True)

    date_df['Day'] = 1 # add a column with a day value

    date_df.rename(columns={'YrSold' : 'Year', 'MoSold' : 'Month'}, inplace=True)

    date_df['Date sold'] = pd.to_datetime(date_df[['Year','Month','Day']])

    data['Date sold'] = date_df['Date sold'] # add to data

    data.sort_values(by='Date sold', inplace=True) # sort the data by date

    data['Date_sold_int'] = data['Date sold'].astype(np.int) # could be useful if we need a numerical feature related to the date

    

    # log transform features

    data['log GrLivArea'] = np.log(data['GrLivArea'])

    data['log LotArea'] = np.log(data['LotArea'])

    data.drop(['GrLivArea', 'LotArea'], axis=1, inplace=True)

    

    if is_trainset:

        # take log of target and split it from the features

        data['log SalePrice'] = np.log(data['SalePrice'])

        

        

    

    

    

    # drop columns based on analysis above

    data.drop(set(data) & columns_to_drop, axis=1, inplace=True)

    

    

    return data

train_data = data_engineering(train_data_full, is_trainset=True)



y = train_data['log SalePrice'].copy()

X = train_data.drop(['log SalePrice'], axis=1).copy();
import category_encoders as ce

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



def preprocess_data(my_cols):

    # Drop irrelevant columns from DataFrame before passing it to this function

    # and pass columns of dataframe as input "my_cols"

    

    

    # Preprocessing for numerical data

    numerical_transformer = SimpleImputer(strategy='median')



    # Preprocessing for categorical data

    one_hot_transformer = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

        ('onehot', ce.one_hot.OneHotEncoder(handle_unknown='ignore'))])

    

    target_transformer = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

        ('target', ce.TargetEncoder(handle_unknown='ignore')),

        ('imputer2', SimpleImputer(strategy='median'))]) # put second imputer as sometimes TargetEncoder seems to generate nan values. Maybe not optimal

    

    # Bundle preprocessing for numerical and categorical data

    preprocessor = ColumnTransformer(

    transformers = [('num', numerical_transformer, list(numerical_cols & set(my_cols) )), \

                 ('OOE', one_hot_transformer, list(one_hot_cols & set(my_cols) )), \

          ('target', target_transformer, list(target_encoding_cols & set(my_cols) ))], ) 

    

    return preprocessor
def MAE_score_model(X,y,model):

    # Compute the MAE by train-test split on the features X and target y (80-20 split)

    # model can be chosen for comparison

    

    preprocessor = preprocess_data(X.columns)



    # Bundle preprocessing and modeling code in a pipeline

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ],verbose=False)

    

    # Break off validation set from training data

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)

    

    

    # Preprocessing of training data, fit model 

    my_pipeline.fit(X_train, y_train)

    

    

    # Preprocessing of validation data, get predictions

    preds = my_pipeline.predict(X_valid)

    

    return mean_absolute_error(y_valid, preds)
def MAE_CV_score_model(X,y,model):

    # Compute the MAE by cross-validation on the features X and target y

    # model can be chosen for comparison

    

    

    preprocessor = preprocess_data(X.columns)



    # Bundle preprocessing and modeling code in a pipeline

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ],verbose=False)

    

    # Multiply by -1 since sklearn calculates *negative* MAE

    scores = -1 * cross_val_score(my_pipeline, X, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')

    

    

    return scores.mean()
def plot_predict_error(X,logy, model):

    # Make a plot of the prediction error on the validation data from a train-test split

    # the target logy is assumed log-transformed, but we plot the graphs for the original target y

    

    preprocessor = preprocess_data(X.columns)



    # Bundle preprocessing and modeling code in a pipeline

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ],verbose=False)

    

    # Break off validation set from training data

    X_train, X_valid, y_train, y_valid = train_test_split(X, logy, train_size=0.8, test_size=0.2,random_state=0)

    

    

    # Preprocessing of training data, fit model 

    my_pipeline.fit(X_train, y_train)



    # Preprocessing of validation data, get predictions

    preds = my_pipeline.predict(X_valid)

    

    # plot error

    fig = plt.figure(figsize=(18,4))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3) # increase space between plots

    ax0 = fig.add_subplot(131) # add subplot 1 (121 = 1 row, 2 columns, first plot)

    ax1 = fig.add_subplot(132) # add subplot 2 

    ax2 = fig.add_subplot(133) # add subplot 3 

    

    ax0.scatter(np.exp(y_valid), np.exp(preds)-np.exp(y_valid))

    ax0.set_title("error plot (absolute)")

    ax0.set_xlabel("price [$]")

    ax0.set_ylabel("error on price [$]")

    

    MSE = mean_absolute_error(np.exp(y_valid), np.exp(preds))

    print("MSE = ", MSE)

    ax1.scatter(np.exp(y_valid),(np.exp(preds)-np.exp(y_valid))/np.exp(y_valid))

    ax1.set_title("error plot (relative)")

    ax1.set_xlabel("price [$]")

    ax1.set_ylabel("error on price [%]")

    # put axis in percent

    vals = ax1.get_yticks()

    ax1.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

    

    

    ax2.hist(np.exp(preds)-np.exp(y_valid), bins = 20, range=(-3*MSE,3*MSE))

    ax2.set_title("error histogram")

    ax2.set_xlabel("error on price [$]")

    ax2.set_ylabel("counts")
from sklearn.linear_model import LinearRegression



model = LinearRegression()

MAE = MAE_CV_score_model(X[['OverallQual']], y, model)

print('MAE = %.2e '%(MAE))
plot_predict_error(X[['OverallQual']], y, model)
ranked_num_cols = list(set(ranked_num_cols) & set(X.columns))



for it in range(1, len(ranked_num_cols)):

    cols = ranked_num_cols[0:it]

    MAE = MAE_CV_score_model(X[cols],y,model)

    print('%d features: MAE = %.2e '%(it, MAE))
model = LinearRegression()

MAE = MAE_CV_score_model(X,y,model)

print('MAE = %.2e '%(MAE))
plot_predict_error(X, y, model)
# test impact of parameter n_estimators



for n_estimators in [50,100,500,1000,5000]:

    model = RandomForestRegressor(n_estimators, random_state=0)

    MAE = MAE_score_model(X,y,model)

    print('MAE = %.2e for n_estimators = %d'%(MAE, n_estimators))

    
n_estimators=100

model = RandomForestRegressor(n_estimators, random_state=0)

MAE = MAE_score_model(X[list(set(X.columns) & numerical_cols)],y,model)

print('Numerical features only: MAE = %.2e for n_estimators = %d'%(MAE, n_estimators))
from sklearn.linear_model import Lasso



for alpha in [0.0001, 0.0005, 0.001,0.1,0.2,0.5]:

    model = Lasso(random_state=0, alpha=alpha) 

    MAE = MAE_score_model(X,y,model)

    print('MAE = %.2e for alpha = %f '%(MAE, alpha))
from sklearn.linear_model import Ridge



for alpha in [0.0001, 0.001, 0.1, 0.2, 0.3, 0.5, 0.7, 1]:

    model = Ridge(random_state=0, alpha=alpha) 

    MAE = MAE_score_model(X,y,model)

    print('MAE = %.2e for alpha = %f '%(MAE, alpha))
# Plot error



model =  Ridge(random_state=0, alpha=0.3) 



plot_predict_error(X,y,model)
from sklearn.ensemble import GradientBoostingRegressor



# let' try gradient boost



for n_estimators in [10,50,100,500,1000]:

    learning_rate = 0.1

    model = GradientBoostingRegressor(random_state=0, n_estimators=n_estimators, learning_rate=learning_rate) 

    MAE = MAE_score_model(X,y,model)

    print('MAE = %.2e for n_estimators = %d and learning_rate = %f'%(MAE, n_estimators, learning_rate))
# cover over both parameters

# there are built-in methods like grid search that make this certainly better, 

# but for now we will keep it simple



n_estimators_list = [50,75,100,125,150]

learning_rate_list = [0.1, 0.15, 0.2, 0.25, 0.3]



score_mat = []



for n_estimators in n_estimators_list:

    for learning_rate in learning_rate_list:

        model = GradientBoostingRegressor(random_state=0, n_estimators=n_estimators, learning_rate=learning_rate) 

        score = MAE_score_model(X,y,model)

        score_mat.append({'n_estimators' : n_estimators, 'learning_rate': learning_rate, 'score': score})

        

score_df = pd.DataFrame(score_mat)



# plot heatmap

score_df = score_df.pivot("n_estimators", "learning_rate", "score")

sns.heatmap(score_df, annot=True);
model = GradientBoostingRegressor(random_state=0, n_estimators=75, learning_rate=0.2) 



plot_predict_error(X,y,model)
from sklearn.neighbors import KNeighborsRegressor



for n_neighbors in [3,5,10,15,20]:

    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance') 

    MAE = MAE_score_model(X ,y,model)

    print('MAE = %.2e for n_neighbors = %d'%(MAE, n_neighbors))
# check how many features give the optimal result



for nb_features in range(1, len(ranked_num_cols)):

    model = KNeighborsRegressor(n_neighbors=10, weights='distance') 

    # take categorical columns only

    MAE = MAE_score_model(X[ranked_num_cols[0:nb_features]] ,y,model)

    print('MAE = %.2e for %d best features '%(MAE, nb_features))
for n_neighbors in [3,5,10,15,20,25]:

    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance') 

    # take categorical columns only

    MAE = MAE_score_model(X[ranked_num_cols[0:3]] ,y,model)

    print('MAE = %.2e for n_neighbors = %d'%(MAE, n_neighbors))
# check if there are outliers in the test set

        

test_outlier_df = test_data_full[(test_data_full['TotalBsmtSF'] > 4000) | (test_data_full['LotArea'] > 100000) | (test_data_full['GrLivArea'] > 4000)]

rows_outliers = test_outlier_df.index.values

test_outlier_df[['TotalBsmtSF', 'LotArea', 'GrLivArea']]
test_data_full.loc[1089, 'TotalBsmtSF'] = 4000

test_data_full.loc[1089, 'GrLivArea'] = 4000
test_data = data_engineering(test_data_full, is_trainset=False)



model = GradientBoostingRegressor(random_state=0, n_estimators=75, learning_rate=0.2) 



preprocessor = preprocess_data(X.columns)



# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                  ('model', model)

                 ],verbose=False)



# Preprocessing of training data, fit model 

my_pipeline.fit(X, y)



# Preprocessing of validation data, get predictions

preds = my_pipeline.predict(test_data)

# compare with train data



fig = plt.figure(figsize=(9,5))

sns.distplot(train_data_full['SalePrice'])

sns.distplot(np.exp(preds));



plt.title('comparison between house prices in train and test sets')

plt.legend(['train', 'test']);

plt.ylabel('relative count');
# Put "Id" back in place

output = pd.DataFrame(columns = ['SalePrice'], index=test_data.index)

output['SalePrice'] = np.exp(preds)

output = output.merge(test_data_full[['Id']], left_index=True, right_index=True)



output.sort_values(by=['Id'], inplace=True)



output[['Id', 'SalePrice']].to_csv('submission.csv', index=False)