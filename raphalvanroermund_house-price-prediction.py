import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# import plot libraries

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline



# import stats to analyze and fit histograms

from scipy import stats 



# import ML libraries

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

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

train_data_full = pd.read_csv('../input/house-prices-dataset/train.csv')

test_data_full = pd.read_csv('../input/house-prices-dataset/test.csv')



train_data_full.tail()
# target to model is the 'SalePrice' column



# Remove rows with missing target, separate target from predictors

train_data_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = train_data_full.SalePrice # new df with target



# remove target from features dataframe

train_data = train_data_full.drop(['SalePrice'], axis=1)
fig = sns.distplot(train_data_full['SalePrice'])
train_data_full['SalePrice'].describe()
train_data_full.corr()['SalePrice'].sort_values(ascending=False).head(15)
# Select categorical columns

categorical_cols = {cname for cname in train_data.columns if train_data[cname].dtype == "object"}



# Select numerical columns

numerical_cols = {cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']}
# check columns with missing data

missing_data_cols = set(train_data.columns[train_data.isna().any()].tolist())



# display the fraction of missing data

df_pct_missing = pd.DataFrame((len(y) - train_data[missing_data_cols].count())/len(y))*100

df_pct_missing.columns = ['Missing data [%]']

df_pct_missing.sort_values('Missing data [%]')
# drop columns that have less than 25% values that are NAN



columns_to_drop = train_data.count()[train_data.count()<0.75*max(train_data.count())].index.tolist()

columns_to_drop = set(columns_to_drop) # convert to set to avoid multiple instances

print("We drop the following columns because more than 25% of the entries are missing: \n",columns_to_drop)
# Visualize all numerical features

n=len(train_data[numerical_cols].columns) # number of plots

f, axes = plt.subplots(nrows=(n-1)//4 +1,ncols=4,squeeze=False,figsize=(18,4*((n-1)//4 +1))) # represent them on 4 columnms

f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) # increase space between plots



for col, ax in zip(numerical_cols, axes.flatten()[:n]):

    sns.regplot(data=train_data_full,x=col,y='SalePrice', ax=ax)



plt.show()
# show numerical columns that have missing data but are not in dropped columns list

cols = (missing_data_cols & numerical_cols) - columns_to_drop

train_data[cols].head()
# let's have a look at the distribution of the columns that have missing data in order to make the best inference



# Visualize all features

n=len(train_data[cols].columns) # number of plots

f, axes = plt.subplots((n-1)//3 +1,3, figsize=(18,6*((n-1)//3 +1))) # represent them on 3 columnms



for col, ax in zip(cols, axes.flatten()[:n]):

    #sns.countplot(x=col, data=train_data_filtered, ax=ax)

    sns.distplot(a=train_data[col][(train_data[col].notnull())], ax=ax,fit=stats.norm)



plt.show()
# Visualize all categorical features

n=len(train_data[categorical_cols].columns) # number of plots

f, axes = plt.subplots(nrows=(n-1)//4 +1,ncols=4,squeeze=False,figsize=(18,4*((n-1)//4 +1))) # represent them on 4 columnms

f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) # increase space between plots



for col, ax in zip(categorical_cols, axes.flatten()[:n]):

    sns.boxplot(data=train_data_full,x=col,y='SalePrice', ax=ax)



plt.show()
# Columns that will be one-hot encoded

low_cardinality_cols = set([col for col in categorical_cols if train_data[col].nunique() <= 5])



# Columns that will be dropped from the dataset

high_cardinality_cols = categorical_cols - low_cardinality_cols



print('Categorical columns that will be one-hot encoded: \n', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset: \n', high_cardinality_cols)
# show low cardinality categorical columns that have missing data

cols = (missing_data_cols & low_cardinality_cols) 



# Visualize all features

n=len(train_data[cols].columns) # number of plots



f, axes = plt.subplots(nrows=(n-1)//4 +1,ncols=4,squeeze=False,figsize=(18,4*((n-1)//4 +1))) # represent them on 4 columnms

f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) # increase space between plots



for col, ax in zip(cols, axes.flatten()[:n]):

    sns.countplot(x=col, data=train_data, ax=ax)



plt.show()
col_val = 'Electrical'

val_list = train_data_full[col_val].unique()



print("Price variations for the feature %s: \n"%col_val)

df = train_data_full.groupby(col_val).mean()['SalePrice']

print(df.sort_values())
sns.boxplot(data=train_data_full,x=col_val,y='SalePrice')
# Function to control the columns to drop 



def drop_column(dataset, min_instances_per_feature = 0.75, max_cardinality = 5):

    """ 

    Select which columns have to be dropped from dataset

    

    inputs:

    dataset = dataframe 

    min_instances_per_feature = minimum fraction of data per features to keep it in data set

    max_cardinality = highest cardinality accepted for categorical columns

    

    outputs:

    columns_to_drop = set of columns to drop

    fitered_dataset = dataset without colmuns

    """

    

    missing_data_cols = set(dataset.columns[dataset.isna().any()].tolist())

    

    # 1) drop features that have too many missing values

    columns_to_drop = set(dataset.count()[dataset.count() < min_instances_per_feature*max(dataset.count())].index.tolist())



    # 2) drop categorical features with high cardinality

    low_cardinality_cols = set([col for col in categorical_cols if train_data[col].nunique() <= max_cardinality])

    high_cardinality_cols = categorical_cols - low_cardinality_cols

    columns_to_drop = columns_to_drop | high_cardinality_cols 



    # 3) drop categorical features with missing data

    missing_data_cat_cols = (missing_data_cols & categorical_cols) 

    columns_to_drop = columns_to_drop | missing_data_cat_cols

    

    fitered_dataset = dataset.drop(columns_to_drop,axis=1)

    

    return [columns_to_drop, fitered_dataset]
[columns_to_drop,fitered_train_data] = drop_column(train_data, min_instances_per_feature = 0.75, max_cardinality = 5)

print(columns_to_drop)
# Keep selected columns only

my_cols = list((categorical_cols | numerical_cols) - columns_to_drop)
def preprocess_data(data):

    # preprocess DataFrame 'data'

    # Drop irrelevant columns from DataFrame before passing it to this function

    

    # Select categorical and numerical columns from the datafrome 'data'

    cat_cols = {cname for cname in data.columns if data[cname].dtype == "object"}

    num_cols = {cname for cname in data.columns if data[cname].dtype in ['int64', 'float64']}

    

    

    # Preprocessing for numerical data

    numerical_transformer = SimpleImputer(strategy='most_frequent')



    # Preprocessing for categorical data

    categorical_transformer = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='most_frequent')),

        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    



    # Bundle preprocessing for numerical and categorical data

    # Let's put remainder='passthrough' so that if more columns are present than those specified in the transformers, we keep them in data set.

    # This might lead to errors but such errors will force us to have a closer look at the data

    preprocessor = ColumnTransformer(

    transformers=[('num', numerical_transformer, list(num_cols)),

        ('cat', categorical_transformer, list(cat_cols))],

                                  remainder='passthrough') 

    

    return preprocessor
def MAE_score_model(X,y,model):

    # Compute the MAE by train-test split on the features X and target y (80-20 split)

    # model can be chosen for comparison

    

    preprocessor = preprocess_data(X)



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
def plot_predict_error(X,y,model):

    # Make a plot of the prediction error on the validation  data from a train-test split

    

    preprocessor = preprocess_data(X)



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

    

    # plot error

    fig = plt.figure(figsize=(18,4))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3) # increase space between plots

    ax0 = fig.add_subplot(131) # add subplot 1 (121 = 1 row, 2 columns, first plot)

    ax1 = fig.add_subplot(132) # add subplot 2 

    ax2 = fig.add_subplot(133) # add subplot 3 

    

    ax0.scatter(y_valid,preds-y_valid)

    ax0.set_title("error plot")

    ax0.set_xlabel("price")

    ax0.set_ylabel("error on price")

    

    MSE = mean_absolute_error(y_valid, preds)

    ax1.scatter(y_valid,preds-y_valid)

    ax1.set_title("zoom on error plot")

    ax1.set_xlabel("price")

    ax1.set_ylabel("error on price")

    ax1.set_ylim((-3*MSE,3*MSE))

    

    ax2.hist(preds-y_valid, bins = 20, range=(-3*MSE,3*MSE))

    ax2.set_title("error histogram")

    ax2.set_xlabel("error on price")

    ax2.set_ylabel("counts")
def MAE_CV_score_model(X,y,model):

    # Compute the MAE by cross-validation on the features X and target y

    # model can be chosen for comparison

    

    

    preprocessor = preprocess_data(X)



    # Bundle preprocessing and modeling code in a pipeline

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ],verbose=False)

    

    # Multiply by -1 since sklearn calculates *negative* MAE

    scores = -1 * cross_val_score(my_pipeline, X, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')

    

    return scores.mean()
from sklearn.linear_model import LinearRegression



model = LinearRegression()

MAE = MAE_CV_score_model(train_data[['OverallQual']],y,model)

print('MAE = %.0f '%(MAE))
model = LinearRegression()

MAE = MAE_CV_score_model(train_data[list(set(my_cols) & numerical_cols)],y,model)

print('MAE = %.0f '%(MAE))
model = LinearRegression()

MAE = MAE_CV_score_model(train_data[my_cols],y,model)

print('MAE = %.0f '%(MAE))
# test impact of parameter n_estimators



for n_estimators in [50,100,500,1000,5000]:

    model = RandomForestRegressor(n_estimators, random_state=0)

    MAE = MAE_score_model(train_data[my_cols],y,model)

    print('MAE = %.0f for n_estimators = %d'%(MAE, n_estimators))

    
n_estimators=500

model = RandomForestRegressor(n_estimators, random_state=0)

MAE = MAE_score_model(train_data[list(set(my_cols) & numerical_cols)],y,model)

print('Numerical features only: MAE = %.0f for n_estimators = %d'%(MAE, n_estimators))
from sklearn.linear_model import Lasso



for alpha in [0,0.001,0.1,0.2,0.5]:

    model = Lasso(random_state=0, alpha=alpha, max_iter=10^6) 

    MAE = MAE_score_model(train_data[my_cols],y,model)

    print('MAE = %.0f for alpha = %f '%(MAE, alpha))
from sklearn.neighbors import KNeighborsRegressor



for n_neighbors in [3,5,10,20]:

    model = KNeighborsRegressor(n_neighbors=n_neighbors) 

    MAE = MAE_score_model(train_data[my_cols],y,model)

    print('MAE = %.0f for n_neighbors = %d '%(MAE, n_neighbors))
from sklearn.ensemble import GradientBoostingRegressor



# let' try gradient boost



for n_estimators in [10,50,100,500,1000]:

    learning_rate = 0.1

    model = GradientBoostingRegressor(random_state=0, n_estimators=n_estimators, learning_rate=learning_rate) 

    MAE = MAE_score_model(train_data[my_cols],y,model)

    print('MAE = %.0f for n_estimators = %d and learning_rate = %f'%(MAE, n_estimators, learning_rate))
for n_estimators in [10,50,100,500,1000]:

    learning_rate = 0.2

    model = GradientBoostingRegressor(random_state=0, n_estimators=n_estimators, learning_rate=learning_rate) 

    MAE = MAE_score_model(train_data[my_cols],y,model)

    print('MAE = %.0f for n_estimators = %d and learning_rate = %f'%(MAE, n_estimators, learning_rate))
plot_predict_error(train_data[my_cols],y,model)