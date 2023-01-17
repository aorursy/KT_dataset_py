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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/kaggle/input/exl-data/exl_data.csv')
data_dict = pd.read_excel('/kaggle/input/exl-dict/exl_dict.xlsx')
data.head()
data_dict
# Basic details about data
print("Number of features: ", data.shape[1])
print("Number of customers: ", data.shape[0])
data.nunique()
data['var30'].value_counts()
# Percentage of missing values in each column
for col in data.columns:
    if sum(data[col].isnull()) != 0:
        print(col, ' :', sum(data[col].isnull())/data.shape[0]*100, '%  ', sum(data[col].isnull()))
# Deciding Categorical & Numerical features
categorical_features = ['var16', 'var25', 'var32', 'var33', 'var34', 'var35', 'var36', 'var37', 'var39', 'var40', 'self_service_platform']

numerical_features = []
for col in data.columns:
    if col not in categorical_features:
        numerical_features.append(col)
        
print("Numerical features: ", numerical_features)
print("\nCategorical Features: ", categorical_features)
data.info()
# Describing features
data[numerical_features].describe()
# Visualizing Numerical Features
data[numerical_features].hist(bins=30, figsize=(30,30));
# Trying to remove zeros
new_data = data.replace({0:np.nan})
new_data[numerical_features].hist(bins=30, figsize=(30,30))
# Correlations in numerical features
sns.heatmap(data[numerical_features[1:]].corr())
data[numerical_features[1:]].corr()
# Categorical features Visualization
sns.set(font_scale=2)
fig = plt.figure(figsize=(50,50))
for i in range(1, len(categorical_features)+1):
    ax = fig.add_subplot(5, 3, i)
    sns.countplot(y=categorical_features[i-1], data=data, color="c")
data['var36'].value_counts()
data.head()
target = data['self_service_platform'].replace({'Desktop':1, 'Mobile App':2, 'Mobile Web':3, 'STB':4})
data.drop('self_service_platform', axis=1, inplace=True)
categorical_features.remove('self_service_platform')
numerical_features.remove('cust_id')
# Categorical Tranformations
def custom_transform(X):
    
    categorical_features = ['var16','var25','var32', 'var33', 'var34', 'var35', 'var36', 'var38', 'var37', 'var39', 'var40']
    categorical_data = X[categorical_features]
    
    # Treating var36
    # Removing Only in string
    new_var36 = categorical_data['var36'].str.split(pat=' ', expand=True)[0]
    # Seperating these variables
    new_var36 = new_var36.str.get_dummies(sep='/').add_prefix('var36_')
    
    # Treat var37 and var 40
    categorical_data[['var37', 'var40']] = categorical_data[['var37', 'var40']].replace(["Y", "N"], [1, 0])
    
    # Fill null values of var38 by 0
    categorical_data['var38'].fillna('Empty', inplace=True)
    
    # One hot encoding on others
    categorical_data.drop('var36', axis=1, inplace=True)
    categorical_data = pd.concat([new_var36, pd.get_dummies(categorical_data)], axis=1).fillna(0)
    
    X =  pd.concat([X.drop(categorical_features, axis=1), categorical_data], axis = 1)
    
    # Filling missing values in numerical data
    X.fillna(0, inplace=True)
    
    # Dropping var30
    X.drop(['var30', 'cust_id'], axis=1, inplace=True)
    
    return X
custom_transform(data)
custom_transform(data).columns
custom_transform(data).isnull().any()
custom_transform(data).info()
from sklearn.preprocessing import FunctionTransformer
data_preprocess_transformer = FunctionTransformer(custom_transform)
def generate_features(data):
    data['ratio_tickets'] = data['var5']/data['var26']
    data['ratio_emails'] = data['var7']/data['var24']
    data['ratio_calls'] = data['var10']/data['var28']
    
    # Handling division by zeros
    data.loc[~np.isfinite(data['ratio_calls']), 'ratio_calls'] = -1
    data.loc[~np.isfinite(data['ratio_emails']), 'ratio_emails'] = -1
    
    return data.fillna(0)

featgen_transformer = FunctionTransformer(generate_features)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500)
rf_pipeline = Pipeline([('preprocess', data_preprocess_transformer), ('feat_gen', featgen_transformer), ('model', rf)])
rf_pipeline.fit(X_train, y_train)
rf_pipeline.score(X_test, y_test)
rf_pipeline.score(X_train, y_train)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, rf_pipeline.predict(X_test))
features_importance = pd.DataFrame(pd.concat([pd.Series(generate_features(custom_transform(X_train)).columns), pd.Series(rf_pipeline.steps[2][1].feature_importances_*100)], axis=1))
features_importance
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

model = XGBClassifier()
def fit_model(model, X_train, y_train, X_test, y_test):
    rf_pipeline = Pipeline([('preprocess', data_preprocess_transformer), ('feat_gen', featgen_transformer), ('model', model)])
    rf_pipeline.fit(X_train, y_train)
    
    print("Training score: ", rf_pipeline.score(X_train, y_train))
    print("Testing score: ", rf_pipeline.score(X_test, y_test))
    print(confusion_matrix(y_test, rf_pipeline.predict(X_test)))
    return

fit_model(model, X_train, y_train, X_test, y_test)
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.arange(10, 20)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 20, 25, 30, 40, 50]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 6, 8, 10]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'model__n_estimators': n_estimators,
               'model__max_features': max_features,
               'model__max_depth': max_depth,
               'model__min_samples_split': min_samples_split,
               'model__min_samples_leaf': min_samples_leaf,
               'model__bootstrap': bootstrap}

print(random_grid)
# Use the random grid to search for best hyperparameters

rf_pipeline = Pipeline([('preprocess', data_preprocess_transformer), ('feat_gen', featgen_transformer), ('model', RandomForestClassifier())])


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations

rf_random = RandomizedSearchCV(estimator = rf_pipeline, param_distributions = random_grid, n_iter = 100, cv = 3, random_state=2)
# Fit the random search model

rf_random.fit(X_train, y_train)

print("Training score: ", rf_random.best_estimator_.score(X_train, y_train))
print("Testing score: ", rf_random.best_estimator_.score(X_test, y_test))


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr_pipeline = Pipeline([('preprocess', data_preprocess_transformer), ('feat_gen', featgen_transformer), ('model', lr)])
lr_pipeline.fit(X_train, y_train)

print("Training score: ", lr_pipeline.score(X_train, y_train))
print("Testing score: ", lr_pipeline.score(X_test, y_test))
# Applying PCA
transformed_train = generate_features(custom_transform(X_train))
from sklearn.decomposition import PCA
pca = PCA()

pca.fit(transformed_train)
pca.explained_variance_ratio_.cumsum()
pd.DataFrame(pca.transform(transformed_train))
rf_Pca_pipe = Pipeline([('preprocess', data_preprocess_transformer), ('feat_gen', featgen_transformer), ('pca', PCA(n_components=6)), ('model', RandomForestClassifier(max_depth=15))])
rf_Pca_pipe.fit(X_train, y_train)

print("Training score: ", rf_Pca_pipe.score(X_train, y_train))
print("Testing score: ", rf_Pca_pipe.score(X_test, y_test))
lr_Pca_pipe = Pipeline([('preprocess', data_preprocess_transformer), ('feat_gen', featgen_transformer), ('pca', PCA(n_components=10)), ('model', lr)])

lr_Pca_pipe.fit(X_train, y_train)

print("Training score: ", lr_Pca_pipe.score(X_train, y_train))
print("Testing score: ", lr_Pca_pipe.score(X_test, y_test))
from sklearn.feature_selection import SelectKBest, chi2, f_classif

rf_k_pipe = Pipeline([('preprocess', data_preprocess_transformer), ('feat_gen', featgen_transformer), ('select_k', SelectKBest(k=30, score_func= f_classif)), ('model', RandomForestClassifier(min_samples_split=200, n_estimators=100))])
rf_k_pipe.fit(X_train, y_train)

print("Training score: ", rf_k_pipe.score(X_train, y_train))
print("Testing score: ", rf_k_pipe.score(X_test, y_test))
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
le = LabelEncoder()
imp = SimpleImputer(strategy='constant', fill_value=0)
def new_transform(X):
    
    # Treating var36
    # Removing Only in string
    new_var36 = X['var36'].str.split(pat=' ', expand=True)[0]
    # Seperating these variables
    new_var36 = new_var36.str.get_dummies(sep='/').add_prefix('var36_')
    
    # Fill null values of var38 by Empty
    X['var38'].fillna('Empty', inplace=True)
    
    # Target encoding on others
    X.drop('var36', axis=1, inplace=True)
    X = pd.concat([new_var36, X], axis=1)
    
    # Dropping var30
    X.drop(['var30', 'cust_id'], axis=1, inplace=True)
    
    return X

new_transformer = FunctionTransformer(new_transform)

new_transform_pipe = Pipeline(steps = [('preprocess', new_transformer), ('LabelEncoder', le), ('imputer', imp)])
rfnew_pipe = Pipeline([('preprocess', new_transform_pipe), ('model', RandomForestClassifier(max_depth=14, n_estimators=200))])


rfnew_pipe.fit(X_train, y_train)

print("Training score: ", rfnew_pipe.score(X_train, y_train))
print("Testing score: ", rfnew_pipe.score(X_test, y_test))
import pandas as pd
exl_data = pd.read_csv("../input/exl-data/exl_data.csv")