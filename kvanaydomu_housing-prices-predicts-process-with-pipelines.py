# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder, OrdinalEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor

from collections import Counter

from xgboost import XGBRegressor



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
filePath1 = "/kaggle/input/house-prices-advanced-regression-techniques/train.csv"

filePath2 = "/kaggle/input/house-prices-advanced-regression-techniques/test.csv"



home_train = pd.read_csv(filePath1,index_col="Id")

home_test = pd.read_csv(filePath2,index_col="Id")
home_train.head()
home_train.shape
home_test.shape
home_train.columns
numCol=home_train.select_dtypes(exclude="object").columns

numCol
len(numCol)
home_train.select_dtypes(exclude="object").describe()
catCol = home_train.select_dtypes(include="object").columns

catCol
home_train.select_dtypes(include="object").describe()
len(catCol)
target = home_train.SalePrice

sns.distplot(target)

plt.title("Examine sales prices and it's skew")

plt.show()
num_attributes=home_train.select_dtypes(exclude="object").drop("SalePrice",axis=1).copy()



fig = plt.figure(figsize=(12,18))



for i in range(len(num_attributes.columns)):

    try:

        fig.add_subplot(9,4,i+1)

        sns.distplot(num_attributes.iloc[:,i].dropna(),kde=True,rug=True)

        plt.xlabel(num_attributes.columns[i])

    except ValueError and RuntimeError:

        pass



plt.tight_layout()

plt.show()
f = plt.figure(figsize=(12,18))





for i in range(len(num_attributes.columns)):

    try:

        f.add_subplot(9,4,i+1)

        sns.boxplot(num_attributes.iloc[:,i])

    except ValueError and RuntimeError:

        pass



plt.tight_layout()

plt.show()
f = plt.figure(figsize=(12,18))





for i in range(len(num_attributes.columns)):

    try:

        f.add_subplot(9,4,i+1)

        sns.scatterplot(num_attributes.iloc[:,i],target)

    except ValueError and RuntimeError:

        pass



plt.tight_layout()

plt.show()
f = plt.figure(figsize=(15,10))

sns.heatmap(home_train.corr(),annot=False,fmt=".2f")

plt.show()
correlation = home_train.corr()



correlation["SalePrice"].sort_values(ascending=False).head(16)
catCol = home_train.select_dtypes(include="object").columns

catCol
g=sns.factorplot(x="OverallQual",y="SalePrice",data=home_train,kind="bar",size=6)

g.set_ylabels("Sale Prices")

g.add_legend()

plt.xticks(rotation=45)

plt.show()
g=sns.factorplot(x="Neighborhood",y="SalePrice",data=home_train,kind="box",size=6)

g.set_ylabels("Sale Prices")

g.add_legend()

plt.xticks(rotation=45)

plt.show()
g=sns.factorplot(x="Neighborhood",y="SalePrice",data=home_train,kind="bar",size=6)

g.set_ylabels("Sale Prices")

g.add_legend()

plt.xticks(rotation=45)

plt.show()
g=sns.factorplot(x="HouseStyle",y="SalePrice",data=home_train,kind="bar",size=6)

g.set_ylabels("Sale Prices")

g.add_legend()

plt.xticks(rotation=45)

plt.show()
def detectOutliers(df,features):

    outlier_indices=[]

    for c in features:

        Q1=np.percentile(df[c],25)

        Q2=np.percentile(df[c],75)

        IQR = Q2-Q1

        outlierStep = IQR*1.5

        outlierListCol = df[(df[c] < Q1-outlierStep) | (df[c]>Q2+outlierStep)].index

        outlier_indices.extend(outlierListCol)

    outlier_indices=Counter(outlier_indices)

    multiple_outliers=list(i for i,v in outlier_indices.items() if v>2)

    return multiple_outliers
outliers=home_train.loc[detectOutliers(home_train,numCol)]

outliers
outliers.shape
home_train.dropna(axis=0,subset=["SalePrice"],inplace=True)

y = home_train.SalePrice

home_train.drop(["SalePrice"],axis=1,inplace=True)

X_train_full, X_valid_full, y_train, y_valid = train_test_split(home_train,y,train_size=0.8,test_size=0.2,random_state=0)
print(home_train.shape[0],home_test.shape[0])
[col for col in home_train.columns if col not in home_test.columns]
categorical_col = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype=="object"]

numerical_col = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ["int64","float64"]]



missing_columns_values = X_train_full.isnull().sum()

print("Columns list which has missing values : \n{}".format(missing_columns_values[missing_columns_values>0]))
missing_numeric_columns_values = X_train_full[numerical_col].isnull().sum()

print("Numerical Columns list which has missing values : \n{}".format(missing_numeric_columns_values[missing_numeric_columns_values>0]))
constant_num_cols = ['GarageYrBlt', 'MasVnrArea',"LotFrontage"]

# I imputate them how using neededNumCols.I calculate mean of neededNumCols and assing to constant_num_cols



meanNumCols = list(set(numerical_col).difference(constant_num_cols))



constant_categorical_cols = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',

                             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

# Samely , I imputate how most frequently which categotycal data



mostFrqCol = list(set(categorical_col).difference(constant_categorical_cols))  # I imputate as most frequently with pipeline



my_cols = constant_num_cols+meanNumCols+constant_categorical_cols+mostFrqCol
X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = home_test[my_cols].copy()
numerical_transformer_m = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='mean')),

    ('scaler', StandardScaler())])



numerical_transformer_c = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),

    ('scaler', StandardScaler())])







categorical_transformer_mf = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown = 'ignore', sparse = False))

])





categorical_transformer_c = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),

    ('onehot', OneHotEncoder(handle_unknown = 'ignore', sparse = False))

])





preprocessor = ColumnTransformer(

    transformers=[

        ('num_mean', numerical_transformer_m, meanNumCols),

        ('num_constant', numerical_transformer_c, constant_num_cols),

        ('cat_mf', categorical_transformer_mf, mostFrqCol),

        ('cat_c', categorical_transformer_c, constant_categorical_cols)

    ])
model = RandomForestRegressor(n_estimators=100,random_state=0)



my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])



my_pipeline.fit(X_train,y_train)



preds = my_pipeline.predict(X_valid)



print('MAE:', mean_absolute_error(y_valid, preds))
scores = -1 * cross_val_score(my_pipeline,X_train,y_train,cv=5,scoring='neg_mean_absolute_error')

scores
def get_score(n_estimators):

    my_pipeline = Pipeline(steps=[

        ('preprocessor', preprocessor),

        ("model",RandomForestRegressor(n_estimators=n_estimators,random_state=0))

    ])

    scores = -1 * cross_val_score(my_pipeline,X_train,y_train,cv=3,scoring='neg_mean_absolute_error')

    return scores.mean()
estimators = np.arange(50,450,50)



results = {}



for i in range(1,9):

    results[i*50] = get_score(i*50)
results
plt.plot(list(results.keys()), list(results.values()))

plt.show()
model = RandomForestRegressor(n_estimators=350,random_state=0)



my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])



my_pipeline.fit(X_train,y_train)



preds = my_pipeline.predict(X_valid)



print('MAE:', mean_absolute_error(y_valid, preds))
my_second_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

my_second_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', my_second_model)

                     ])

my_second_pipeline.fit(X_train,y_train)



preds2 = my_second_pipeline.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds2))
def get_score2(n_estimators):

    my_pipeline2 = Pipeline(steps=[

        ('preprocessor', preprocessor),

        ("model",XGBRegressor(n_estimators=n_estimators,learning_rate=0.05, n_jobs=4))

    ])

    scores2 = -1 * cross_val_score(my_pipeline2,X_train,y_train,cv=3,scoring='neg_mean_absolute_error')

    return scores2.mean()
estimators2 = np.arange(50,1600,50)



results2 = {}



for i in range(1,30):

    results2[i*50] = get_score2(i*50)
sorted(results2.items(),key=lambda x:x[1])
plt.plot(list(results2.keys()),list(results2.values()))

plt.show()
my_second_model = XGBRegressor(n_estimators=350, learning_rate=0.05, n_jobs=4)

my_second_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', my_second_model)

                     ])

my_second_pipeline.fit(X_train,y_train)



preds2 = my_second_pipeline.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds2))
X=home_train.copy()

X_test = home_test.copy()



X_tr = X[my_cols].copy()

X_te = X_test[my_cols].copy()
my_second_model = XGBRegressor(n_estimators=350, learning_rate=0.05, n_jobs=4)

my_second_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', my_second_model)

                     ])

my_second_pipeline.fit(X_tr,y)



preds2 = my_second_pipeline.predict(X_te)

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds2})
compression_opts = dict(method="zip",archive_name="submission.csv")

output.to_csv("submission.zip",index=False,compression=compression_opts)