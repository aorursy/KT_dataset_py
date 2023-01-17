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

data[numerical_features].hist(bins=30, figsize=(30,30))
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
data.columns
# Treating var36

# Removing Only in string (i.e seperate the variables on space)

new_var36 = data['var36'].str.split(pat=' ', expand=True)[0]

# Seperating these variables as one hot encoding by seperating at '/'

new_var36 = new_var36.str.get_dummies(sep='/').add_prefix('var36_')



new_data = pd.concat([new_var36, data.drop('var36', axis=1)], axis=1).fillna(0)
new_data.head()
new_data.columns
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.15)
X_train.head()
X_test.head()
new_data.columns
# Categorical Tranformations

def new_transform(X):

    

    categorical_features = ['var16','var25','var32', 'var33', 'var34', 'var35', 'var38', 'var39']

    

    # Dropping var30 and cust_id

    X = X.drop(['var30', 'cust_id'], axis=1)

    

    X[['var37', 'var40']] = X[['var37', 'var40']].replace({"Y":1, "N":0})

    

    # Treating var36

    # Removing Only in string (i.e seperate the variables on space)

    new_var36 = X['var36'].str.split(pat=' ', expand=True)[0]

    # Seperating these variables as one hot encoding by seperating at '/'

    new_var36 = new_var36.str.get_dummies(sep='/').add_prefix('var36_')



    X = pd.concat([new_var36, X.drop('var36', axis=1)], axis=1)

    X.fillna(0)

    X = pd.concat([pd.get_dummies(X[categorical_features]), X.drop(categorical_features, axis=1)], axis=1)

    

    

    # Feature Generation

    X['ratio_tickets'] = X['var5']/X['var26']

    X['ratio_emails'] = X['var7']/X['var24']

    X['ratio_calls'] = X['var10']/X['var28']

    X['ratio_video'] = X['var15']/X['var26']

    X['recurring_ratio'] = X['var21']/X['var27']

    X['credit_amount'] = X['var23']*X['var27']

    X['credit_recurring'] = X['var23']*X['var21']

    

    # Handling division by zeros

    X.loc[~np.isfinite(X['ratio_calls']), 'ratio_calls'] = 0

    X.loc[~np.isfinite(X['ratio_emails']), 'ratio_emails'] = 0

    X.loc[~np.isfinite(X['ratio_tickets']), 'ratio_tickets'] = 0

    X.loc[~np.isfinite(X['ratio_video']), 'ratio_video'] = 0

    X.loc[~np.isfinite(X['recurring_ratio']), 'recurring_ratio'] = 0

    X.loc[~np.isfinite(X['credit_amount']), 'credit_amount'] = 0

    X.loc[~np.isfinite(X['credit_recurring']), 'credit_recurring'] = 0

    

    

    # Some features on emails

    # The cust prefers email or not and the number of mails recieved by them

    X['email_likes'] = X['var37'].where(X['var37']==1, -1) # If he likes, +1 else -1

    X['email_likes_opened'] = X['email_likes']*X['var7']

    X['email_likes_sent'] = X['email_likes']*X['var24']

    

    return X.fillna(0)
new_X = new_transform(data)
new_X.head()
print((new_X!='other').isnull().any().sum())
new_transform(data).columns
new_transform(data).info()
# Adding missing values and one hot encoder pipeline

from sklearn.preprocessing import FunctionTransformer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import PowerTransformer

from sklearn.feature_selection import RFE

from sklearn.preprocessing import PolynomialFeatures



categorical_features = ['var25','var32', 'var33', 'var34', 'var35', 'var38', 'var39']



my_feature_transformer = FunctionTransformer(new_transform)

imp = SimpleImputer(strategy="most_frequent", add_indicator=True)

enc = OneHotEncoder(handle_unknown='ignore')

from category_encoders.target_encoder import TargetEncoder

target_enc = TargetEncoder()





col_transformer = Pipeline([("my_trans", my_feature_transformer), ("imp", imp)])
# Random Under Sampling

sampling_dict = {}

for i, target_count in enumerate(y_train.value_counts().values):

    if i==0:

        sampling_dict[i+1] = int(target_count*0.45)

    else:

        sampling_dict[i+1] = int(target_count)



from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy=sampling_dict, random_state=42)

sampling_dict
y_train.shape
X_train.shape, X_test.shape
from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import confusion_matrix

import xgboost as xgb

from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from xgboost.sklearn import XGBClassifier

from sklearn.neural_network import MLPClassifier

nn = MLPClassifier((10,), max_iter=1000)



#selector = RFE(estimator=LogisticRegression(),20, step=1)





from sklearn import preprocessing

from sklearn.feature_extraction import FeatureHasher





rf = RandomForestClassifier(random_state=2, min_samples_split=17)

rf_pipeline = Pipeline([('preprocess', my_feature_transformer), ('model', rf)])
X_res, y_res = rus.fit_resample(X_train, y_train)

rf_pipeline.fit(X_res, y_res)



print("Training score: ", rf_pipeline.score(X_train, y_train))

print("Testing score: ", rf_pipeline.score(X_test, y_test))

print(confusion_matrix(y_test, rf_pipeline.predict(X_test)))
# Training the same with whole dataset



rf_pipeline.score(data, target)
from sklearn.model_selection import cross_val_score



scores = cross_val_score(X=data, y=target, estimator=rf_pipeline, cv=4)



print(scores.mean())
scores
test_data = pd.read_excel('/kaggle/input/exl-test/exl_test.xlsx')
test_data.head()
test_data.shape
new_transform(test_data).head()
new_transform(data).head()
test_pred = pd.Series(rf_pipeline.predict(test_data))

test_pred.to_csv('test_pred_final.csv')
from IPython.display import FileLink

FileLink(r'test_pred_final.csv')
test_pred.value_counts()
test_pred.sum()
target.value_counts()
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
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, rf_pipeline.predict(X_test))
features_importance = pd.DataFrame(pd.concat([pd.Series(generate_features(custom_transform(X_train)).columns), pd.Series(rf_pipeline.steps[2][1].feature_importances_*100)], axis=1))

features_importance
from sklearn.model_selection import RandomizedSearchCV



# Number of trees in random forest

n_estimators = [100]



# Number of features to consider at every split

max_features = ['auto', 'sqrt']



# Maximum number of levels in tree

max_depth = [int(x) for x in np.arange(12, 18)]

max_depth.append(None)



# Minimum number of samples required to split a node

min_samples_split = [5, 10, 15, 20, 25, 30]

# Minimum number of samples required at each leaf node

min_samples_leaf = [2, 4, 6, 8, 10]

# Method of selecting samples for training each tree

bootstrap = [True]



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



rf_random = RandomizedSearchCV(estimator = rf_pipeline, param_distributions = random_grid, n_iter = 10, cv = 4, random_state=2)

# Fit the random search model



rf_random.fit(X_train, y_train)



print("Training score: ", rf_random.best_estimator_.score(X_train, y_train))

print("Testing score: ", rf_random.best_estimator_.score(X_test, y_test))
rf_random.best_estimator_
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