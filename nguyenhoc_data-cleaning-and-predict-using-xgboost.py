# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Data Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
pd.set_option("display.max_columns", 100)



# load train data

data = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv", index_col="Id")



data.shape
data.head()
categorical_cols = data.select_dtypes(include=['object']).columns

numberical_cols = data.select_dtypes(exclude=['object']).columns

print("Numberical columns: ", len(numberical_cols))

print("Categorical columns: ", len(categorical_cols))
data[numberical_cols].describe()
data[categorical_cols].describe()
fig = plt.figure(figsize=(30,1))

numberical_col_corr = data[numberical_cols].corr().loc[["SalePrice"],:].sort_values(by="SalePrice",axis=1)

sns.heatmap(data=numberical_col_corr, annot=True)
weak_corr_threadhold = 0.2

weak_numberical_cols = list(numberical_col_corr[abs(numberical_col_corr) < weak_corr_threadhold].dropna(axis=1))

print("There are %d weak correlated(< %.2f) variables with SalePrice:" % (len(weak_numberical_cols), weak_corr_threadhold))

print(weak_numberical_cols)
strong_numberical_cols = numberical_col_corr.columns[len(weak_numberical_cols):-1][::-1]

strong_cols_num = len(strong_numberical_cols)



print("There are %d strong correlated(> %.2f) variables with SalePrice:" % (strong_cols_num-1, weak_corr_threadhold))

print(list(strong_numberical_cols))
numberical_null_count = data[numberical_cols].isnull().sum()

print("Missing values: ")

print(numberical_null_count[numberical_null_count>0])
plt.rcParams.update({'font.size': 14})



def explore_variable(col_name):

    strong_crr_cols = find_high_corelated_variables(col_name)

    draw_variable([col_name] + strong_crr_cols)





def draw_variable(col_names):

    num_cols = len(col_names)

    fig = plt.figure(figsize=(8, num_cols*4))

    i = 0

    for col in col_names:

        fig.add_subplot(num_cols, 2, 2*i+1)

        sns.regplot(x=data[col], y=data["SalePrice"])

        plt.xlabel(col)

        plt.title('Corr to SalePrice = %.2f'% numberical_col_corr[col])

        fig.add_subplot(num_cols, 2, 2*i+2)

        sns.distplot(data[col].dropna())

        plt.xlabel(col)

        i += 1

        

    plt.tight_layout()



        





variable_corr = data[list(set(numberical_cols)-set(["SalePrice"]))].corr()

high_corr_threadhold = 0.7



def find_high_corelated_variables(col_name):

    corr = variable_corr.loc[[col_name],:]

    strong_corr = corr[(corr>=high_corr_threadhold) & (corr<1)].dropna(axis=1)

    print("Strong corelated variables:")

    print(strong_corr)

    return list(strong_corr.columns)
explore_variable("OverallQual")
explore_variable("GrLivArea")
data["GrLivArea"].describe()
data["GrLivArea"].sort_values().tail()
explore_variable("GarageCars")
explore_variable("1stFlrSF")
explore_variable("TotalBsmtSF")
explore_variable("TotRmsAbvGrd")
explore_variable("YearBuilt")
explore_variable("YearRemodAdd")



explore_variable("GarageYrBlt")
explore_variable("MasVnrArea")
explore_variable("Fireplaces")
explore_variable("BsmtFinSF1")
explore_variable("LotFrontage")
explore_variable("WoodDeckSF")
explore_variable("2ndFlrSF")
explore_variable("OpenPorchSF")
explore_variable("HalfBath")
explore_variable("LotArea")

explore_variable("BsmtFullBath")

explore_variable("BsmtUnfSF")
fig = plt.figure(figsize=(6,12))

sns.lmplot(x="MasVnrArea", y="SalePrice", hue="KitchenQual", data=data) 
fig = plt.figure(figsize=(6,12))

sns.lmplot(x="2ndFlrSF", y="SalePrice", hue="KitchenQual", data=data) 
unique_val_num_dict = {col:len(data[col].unique())  for col in categorical_cols}

sorted(unique_val_num_dict.items(), key=lambda x: x[1])
null_count = data[categorical_cols].isnull().sum()

null_count[null_count>0].dropna(axis=0)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler





median_transformer = Pipeline([

    ('imputer', SimpleImputer(strategy="median")),

    ('std_scaler', StandardScaler()),

])



zero_transformer = Pipeline([

    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),

    ('std_scaler', StandardScaler()),

])

#zero_transformer = SimpleImputer(strategy='constant', fill_value=0)

#median_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



drop_cols =  ['Alley', 'PoolQC', 'Fence', 'MiscFeature'] + ['TotalBsmtSF', 'GarageYrBlt'] + ['KitchenAbvGr', 'EnclosedPorch', 'MSSubClass', 'OverallCond', 'YrSold', 'LowQualFinSF', 'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2', '3SsnPorch', 'MoSold', 'PoolArea', 'ScreenPorch', 'BedroomAbvGr']

zero_fill_cols  = ['MasVnrArea', 'Fireplaces', 'WoodDeckSF', '2ndFlrSF', 'OpenPorchSF', 'HalfBath', 'BsmtFullBath']

median_fill_cols =  ['OverallQual', 'GrLivArea', 'GarageCars', '1stFlrSF', 'TotRmsAbvGrd', 'BsmtFinSF1', 'LotFrontage', 'LotArea', 'BsmtUnfSF']



preprocessor = ColumnTransformer(

                transformers=[

                    ('num_zero', zero_transformer, zero_fill_cols),

                    ('median_zero', median_transformer, median_fill_cols),

                    ('cat', categorical_transformer, list(set(categorical_cols) - set(['Alley', 'PoolQC', 'Fence', 'MiscFeature'])))

                ])



train_data = data.copy()

train_data.drop(drop_cols, axis=1, inplace=True)

train_data.drop(train_data[(train_data['GrLivArea'] > 4500) |

                (train_data['1stFlrSF'] > 5000) |

                (train_data['BsmtFinSF1'] > 4000) |

                (train_data['LotFrontage'] > 300)].index

                ,axis=0, inplace=True)



y = train_data.SalePrice

X = train_data.drop('SalePrice', axis=1)

feature_cols = list(X.columns)
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

#plt.figure(figsize=(6,6))

#sns.distplot(X_valid_full["OverallQual"])

#sns.distplot(test_data["OverallQual"])

#plt.legend(["OverallQual valid", "OverallQual test"])


model = XGBRegressor(n_estimators=1000,

                         learning_rate=0.05, 

                         early_stopping_rounds=20, 

                         eval_set=[(X_valid_full, y_valid)],

                         random_state=0)



#model = XGBRegressor(n_estimators=1000,learning_rate = 0.01,random_state=0)



my_pipeline = Pipeline([

    ('preprocess', preprocessor),

    ('model', model)])



my_pipeline.fit(X_train_full,y_train)



mae = mean_absolute_error(y_valid, my_pipeline.predict(X_valid_full))

print("MAE:" ,mae)







test_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv", index_col="Id")

predict_result = my_pipeline.predict(test_data[feature_cols])

output = pd.DataFrame({'Id':test_data.index, 

                        'SalePrice':predict_result})

output.to_csv('submission.csv', index=False)