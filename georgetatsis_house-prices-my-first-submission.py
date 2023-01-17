# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
X_full = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)

len(X_full)
X_full.head()
X_full.columns
from sklearn import metrics

from sklearn.preprocessing import StandardScaler, normalize, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, accuracy_score, mean_squared_error

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#checking for columns with missing data

X_full.loc[:, X_full.isnull().any()].columns
X_full.loc[:, X_full.isnull().any()].isna().mean()

# drop columns with more thn 30% of missing data



X_full.drop(X_full.columns[X_full.isnull().mean() > 0.3], axis=1, inplace=True)

X_full.columns
# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)



# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]





# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()



# Preprocessing for numerical data

numerical_transformer = Pipeline(steps=[

    ('simpleimputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())

])



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])

#checking for correlations in numerical data

x =  numerical_transformer['simpleimputer'].fit_transform(X_train[numerical_cols])

X_normalized = normalize(x)

df_normalized = pd.DataFrame(X_normalized, columns=[numerical_cols])

df_normalized

plt.figure(figsize=(30,30))

heatmap = sns.heatmap(df_normalized.corr(), vmin=-1, vmax=1, annot=True)

heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
# Removing high correlated features

cor_matrix = df_normalized.corr().abs()

upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))

to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]

to_drop = [to_drop[i][0] for i in range(0,len(to_drop))]

numerical_cols = list(set(numerical_cols) - set(to_drop))
# Checking for best number of trees to use

MAE = []

MSE = []

score = []

for k in range(100, 650, 50):

    # Define model

    model = RandomForestRegressor(n_estimators=k, random_state=0)



    # Bundle preprocessing and modeling code in a pipeline

    clf = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', model)

                         ])

    

    clf.fit(X_train, y_train)





    # Preprocessing of validation data, get predictions

    preds = clf.predict(X_valid)



    MAE.append(mean_absolute_error(y_valid, preds))

    MSE.append(mean_squared_error(y_valid, preds,squared=False))

    score.append(clf.score(X_valid, y_valid))
plt.style.use("fivethirtyeight")

plt.plot(range(100, 650, 50), MAE)

plt.xlabel("Number of trees")

plt.ylabel("MAE")

plt.show()
plt.style.use("fivethirtyeight")

plt.plot(range(100, 650, 50), MSE)

plt.xlabel("Number of trees")

plt.ylabel("MSE")

plt.show()
plt.style.use("fivethirtyeight")

plt.plot(range(100, 650, 50), score)

plt.xlabel("Number of trees")

plt.ylabel("score")

plt.show()
# Define best model

model = RandomForestRegressor(n_estimators=550, random_state=0)



# Bundle preprocessing and modeling code in a pipeline

clf_final = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', model)

                         ])

    

clf_final.fit(X_train, y_train)





# Preprocessing of validation data, get predictions

preds = clf_final.predict(X_valid)



print('MAE:', mean_absolute_error(y_valid, preds))

print('MSE:', mean_squared_error(y_valid, preds, squared=False))

print('Score:',clf_final.score(X_valid, y_valid))
# Calculating each feature's importance

encoded_cat_cols = clf['preprocessor'].transformers_[1][1]['onehot'].get_feature_names(categorical_cols)  

features_importances = []

for ind in categorical_cols:

    bool_array = (np.core.defchararray.find(encoded_cat_cols.astype(str), ind)+1).astype(np.bool)

    bool_array = np.append(bool_array, [False for i in range(0,36)], axis=0)

    features_importances.append(clf.steps[1][1].feature_importances_[bool_array].mean())

    



features_importances = np.append(features_importances, clf_final.steps[1][1].feature_importances_[174:], axis=0)



features_importances
# plot feature's importance

fig = plt.figure(figsize=(40,20))

ax = fig.add_subplot(111)

ax.barh(X_train.columns, features_importances)

ax.set_xlabel("importance", size=30) 

ax.set_xticks(np.arange(min(features_importances), max(features_importances), 0.002))

ax.grid()

ax.set_ylabel("features", size=30) 
from xgboost import XGBRegressor
# XGBRegressor 

MAE = []

MSE = []

score = []

for k in range(100, 650, 50):

    # Define model

    model=XGBRegressor(n_estimators=k,learning_rate=0.05)



    # Bundle preprocessing and modeling code in a pipeline

    clf = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', model)

                         ])

    

    clf.fit(X_train, y_train)







    # Preprocessing of validation data, get predictions

    preds = clf.predict(X_valid)



    MAE.append(mean_absolute_error(preds,y_valid))

    MSE.append(mean_squared_error(preds,y_valid,squared=False))

    score.append(clf.score(X_valid, y_valid))
plt.style.use("fivethirtyeight")

plt.plot(range(100, 650, 50), MAE)

plt.xlabel("Number of trees")

plt.ylabel("MAE")

plt.show()
plt.style.use("fivethirtyeight")

plt.plot(range(100, 650, 50), MSE)

plt.xlabel("Number of trees")

plt.ylabel("MSE")

plt.show()
plt.style.use("fivethirtyeight")

plt.plot(range(100, 650, 50), score)

plt.xlabel("Number of trees")

plt.ylabel("score")

plt.show()
# Define best model

model=XGBRegressor(n_estimators=600,learning_rate=0.05)



# Bundle preprocessing and modeling code in a pipeline

clf_final = Pipeline(steps=[('preprocessor', preprocessor),

                          ('model', model)

                         ])

    

clf_final.fit(X_train, y_train)





# Preprocessing of validation data, get predictions

preds = clf_final.predict(X_valid)



print('MAE:', mean_absolute_error(y_valid, preds))

print('MSE:', mean_squared_error(y_valid, preds, squared=False))

print('Score:',clf_final.score(X_valid, y_valid))
# Preprocessing of test data, fit model

preds_test = clf_final.predict(X_test)
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)