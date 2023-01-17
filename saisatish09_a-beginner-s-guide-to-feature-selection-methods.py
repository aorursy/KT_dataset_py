

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split



from sklearn.feature_selection import VarianceThreshold

# Loading the data

df = pd.read_csv('/kaggle/input/santander-customer-satisfaction/train.csv')

df.shape
# Check the presence of null data

[col for col in df.columns if df[col].isnull().sum() > 0] 
# Split the data into train and test data

x_train, x_test, y_train, y_test = train_test_split(

    df.drop(labels = ['TARGET'], axis = 1),

    df['TARGET'],

    test_size = 0.3,

    random_state = 0

)
constant_features = [

    features for features in x_train.columns if x_train[features].std() == 0

]

len(constant_features)
x_train.drop(labels = constant_features, axis = 1, inplace=True)

x_test.drop(labels = constant_features, axis = 1, inplace = True)
x_train, x_test, y_train, y_test = train_test_split(

    df.drop(labels = ['TARGET'], axis = 1),

    df['TARGET'],

    test_size = 0.3,

    random_state = 0

)
# Create a empty list

quasi_constant_feat = []



# Loop for searching all the columns in the data

for feature in x_train.columns:

    

    # find the predominant value

    predominant = (x_train[feature].value_counts() / np.float(

        len(x_train))).sort_values(ascending=False).values[0]

    

    # evaluate predominant feature

    if predominant > 0.999:

        quasi_constant_feat.append(feature)



len(quasi_constant_feat)
x_train.drop(labels = quasi_constant_feat, axis = 1, inplace=True)

x_test.drop(labels = quasi_constant_feat, axis = 1, inplace = True)

print(x_train.shape, x_test.shape)
X_train, X_test, y_train, y_test = train_test_split(

    df.drop(labels=['TARGET'], axis=1),

    df['TARGET'],

    test_size=0.3,

    random_state=0)

print(X_train.shape, X_test.shape)
# Create a empty list for duplicated features

duplicated_feat = []

for i in range(0, len(X_train.columns)):



    col_1 = X_train.columns[i]



    for col_2 in X_train.columns[i + 1:]:



        # if the features are duplicated

        if X_train[col_1].equals(X_train[col_2]):



            # and then append the duplicated one to a list

            duplicated_feat.append(col_2)
duplicated_features = set(duplicated_feat)

print(len(duplicated_features))
X_train.drop(labels = duplicated_features, axis = 1, inplace=True)

X_test.drop(labels = duplicated_features, axis = 1, inplace = True)

print(X_train.shape, X_test.shape)
# load dataset

data = pd.read_csv('/kaggle/input/bnp-paribas-cardif-claims-management/train.csv.zip')

data.shape
data.head()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerical_vars = list(data.select_dtypes(include=numerics).columns)

data = data[numerical_vars]

data.shape
# separate train and test sets

X_train, X_test, y_train, y_test = train_test_split(

    data.drop(labels=['target', 'ID'], axis=1),

    data['target'],

    test_size=0.3,

    random_state=0)



X_train.shape, X_test.shape
corrmat = X_train.corr()

fig, ax = plt.subplots()

fig.set_size_inches(16,16)

sns.heatmap(corrmat)
grouped_feature_ls = []

correlated_groups = []



def correlation(dataset, threshold):

    col_corr = set()  # Set of all the names of correlated columns

    corr_matrix = dataset.corr()

    for i in range(len(corr_matrix.columns)):

        for j in range(i):

            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value

                colname = corr_matrix.columns[i]  # getting the name of column

                col_corr.add(colname)

    return col_corr
corr_features = correlation(X_train, 0.9) # filter for all the features with correlation more than 0.9

correlated_features = set(corr_features) # Set statement is used to identify the unique feature in the list

print(len(correlated_features)) # length of set of correlated features
# Dropping all the correlated features from the data

X_train.drop(labels=correlated_features, axis=1, inplace=True)

X_test.drop(labels=correlated_features, axis=1, inplace=True)



X_train.shape, X_test.shape
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from sklearn.feature_selection import SelectKBest, SelectPercentile
df = pd.read_csv('/kaggle/input/bnp-paribas-cardif-claims-management/train.csv.zip')

df.shape
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerical_vars = list(df.select_dtypes(include=numerics).columns)

df = df[numerical_vars]

df.shape
# separate train and test sets

X_train, X_test, y_train, y_test = train_test_split(

    df.drop(labels=['target', 'ID'], axis=1),

    df['target'],

    test_size=0.3,

    random_state=0)



X_train.shape, X_test.shape
mi = mutual_info_classif(X_train.fillna(0), y_train)

mi
# let's add the variable names and order the features

# according to the MI for clearer visualisation

mi = pd.Series(mi)

mi.index = X_train.columns

mi.sort_values(ascending=False)
# and now let's plot the ordered MI values per feature

mi.sort_values(ascending=False).plot.bar(figsize=(20, 8))
# here I will select the top 10 features

# which are shown below

sel_ = SelectKBest(mutual_info_classif, k=10).fit(X_train.fillna(0), y_train)

X_train.columns[sel_.get_support()]
# load dataset

data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

data.shape
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerical_vars = list(data.select_dtypes(include=numerics).columns)

data = data[numerical_vars]

data.shape
# separate train and test sets

X_train, X_test, y_train, y_test = train_test_split(

    data.drop(labels=['SalePrice'], axis=1),

    data['SalePrice'],

    test_size=0.3,

    random_state=0)



X_train.shape, X_test.shape
# determine the mutual information

mi = mutual_info_regression(X_train.fillna(0), y_train)

mi = pd.Series(mi)

mi.index = X_train.columns

mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(20,8))
# here I will select the top 10 percentile

sel_ = SelectPercentile(mutual_info_regression, percentile=10).fit(X_train.fillna(0), y_train)

X_train.columns[sel_.get_support()]
from sklearn.feature_selection import chi2
# load dataset

data = pd.read_csv('/kaggle/input/titanic/train.csv')

data.shape
# the categorical variables in the titanic are PClass, Sex and Embarked

# first I will encode the labels of the categories into numbers



# for Sex / Gender

data['Sex'] = np.where(data.Sex == 'male', 1, 0)



# for Embarked

ordinal_label = {k: i for i, k in enumerate(data['Embarked'].unique(), 0)}

data['Embarked'] = data['Embarked'].map(ordinal_label)



# PClass is already ordinal
# separate train and test sets

X_train, X_test, y_train, y_test = train_test_split(

    data[['Pclass', 'Sex', 'Embarked']],

    data['Survived'],

    test_size=0.3,

    random_state=0)



X_train.shape, X_test.shape
f_score = chi2(X_train.fillna(0), y_train)

f_score
pvalues = pd.Series(f_score[1])

pvalues.index = X_train.columns

pvalues.sort_values(ascending = True)
from sklearn.feature_selection import f_classif, f_regression

from sklearn.feature_selection import SelectKBest, SelectPercentile
# load dataset

data = pd.read_csv('/kaggle/input/bnp-paribas-cardif-claims-management/train.csv.zip')

data.shape
data.head()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerical_vars = list(data.select_dtypes(include=numerics).columns)

data = data[numerical_vars]

data.shape
# separate train and test sets

X_train, X_test, y_train, y_test = train_test_split(

    data.drop(labels=['target', 'ID'], axis=1),

    data['target'],

    test_size=0.3,

    random_state=0)



X_train.shape, X_test.shape
univariate = f_classif(X_train.fillna(0), y_train)

univariate
# let's add the variable names and order it for clearer visualisation

univariate = pd.Series(univariate[1])

univariate.index = X_train.columns

univariate.sort_values(ascending=False, inplace=True)
# and now let's plot the p values

univariate.sort_values(ascending=False).plot.bar(figsize=(20, 8))
# here I will select the top 10 features

sel_ = SelectKBest(f_classif, k=10).fit(X_train.fillna(0), y_train)

X_train.columns[sel_.get_support()]
# load dataset

data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

data.shape
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerical_vars = list(data.select_dtypes(include=numerics).columns)

data = data[numerical_vars]

data.shape
# separate train and test sets

X_train, X_test, y_train, y_test = train_test_split(

    data.drop(labels=['SalePrice'], axis=1),

    data['SalePrice'],

    test_size=0.3,

    random_state=0)



X_train.shape, X_test.shape
univariate = f_regression(X_train.fillna(0), y_train) # fill all null values with 0 for now

univariate = pd.Series(univariate[1])

univariate.index = X_train.columns

univariate.sort_values(ascending=False, inplace=True)
univariate.sort_values(ascending=False).plot.bar(figsize=(20,8))
# here I will select the top 10 percentile

sel_ = SelectPercentile(f_regression, percentile=10).fit(X_train.fillna(0), y_train)

X_train.columns[sel_.get_support()]
X_train = sel_.transform(X_train.fillna(0))

X_train.shape
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import roc_auc_score, mean_squared_error
# load dataset

data = pd.read_csv('/kaggle/input/bnp-paribas-cardif-claims-management/train.csv.zip')

data.shape
data.head()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerical_vars = list(data.select_dtypes(include=numerics).columns)

data = data[numerical_vars]

data.shape
# separate train and test sets

X_train, X_test, y_train, y_test = train_test_split(

    data.drop(labels=['target', 'ID'], axis=1),

    data['target'],

    test_size=0.3,

    random_state=0)



X_train.shape, X_test.shape
# build decision tree for each feature 

roc_values = []

for feature in X_train.columns:

    clf = DecisionTreeClassifier()

    clf.fit(X_train[feature].fillna(0).to_frame(), y_train)

    y_scored = clf.predict_proba(X_test[feature].fillna(0).to_frame())

    roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))
roc_values = pd.Series(roc_values)

roc_values.index = X_train.columns

roc_values.sort_values(ascending=False)
# and now let's plot

roc_values.sort_values(ascending=False).plot.bar(figsize=(20, 8))
# a roc auc value of 0.5 indicates random decision

# let's check how many features show a roc-auc value

# higher than random

len(roc_values[roc_values > 0.5])
# load dataset

data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

data.shape
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerical_vars = list(data.select_dtypes(include=numerics).columns)

data = data[numerical_vars]

data.shape
# separate train and test sets

X_train, X_test, y_train, y_test = train_test_split(

    data.drop(labels=['SalePrice'], axis=1),

    data['SalePrice'],

    test_size=0.3,

    random_state=0)



X_train.shape, X_test.shape
# loop to build a tree, make predictions and get the mse

# for each feature of the train set

mse_values = []

for feature in X_train.columns:

    clf = DecisionTreeRegressor()

    clf.fit(X_train[feature].fillna(0).to_frame(), y_train)

    y_scored = clf.predict(X_test[feature].fillna(0).to_frame())

    mse_values.append(mean_squared_error(y_test, y_scored))
# let's add the variable names and order it for clearer visualisation

mse_values = pd.Series(mse_values)

mse_values.index = X_train.columns

mse_values.sort_values(ascending=False)
mse_values.sort_values(ascending=False).plot.bar(figsize=(20,8))
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import roc_auc_score



from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# load dataset

data = pd.read_csv('/kaggle/input/bnp-paribas-cardif-claims-management/train.csv.zip', nrows = 30000)

data.shape
data.head()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerical_vars = list(data.select_dtypes(include=numerics).columns)

data = data[numerical_vars]

data.shape
# separate train and test sets

X_train, X_test, y_train, y_test = train_test_split(

    data.drop(labels=['target', 'ID'], axis=1),

    data['target'],

    test_size=0.3,

    random_state=0)



X_train.shape, X_test.shape
# find and remove correlated features

# in order to reduce the feature space a bit

# so that the algorithm takes shorter



def correlation(dataset, threshold):

    col_corr = set()  # Set of all the names of correlated columns

    corr_matrix = dataset.corr()

    for i in range(len(corr_matrix.columns)):

        for j in range(i):

            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value

                colname = corr_matrix.columns[i]  # getting the name of column

                col_corr.add(colname)

    return col_corr



corr_features = correlation(X_train, 0.8)

print('correlated features: ', len(set(corr_features)) )
# removed correlated  features

X_train.drop(labels=corr_features, axis=1, inplace=True)

X_test.drop(labels=corr_features, axis=1, inplace=True)



X_train.shape, X_test.shape
sfs1 = SFS(RandomForestClassifier(n_jobs=4), 

           k_features=5, 

           forward=True, 

           floating=False, 

           verbose=2,

           scoring='roc_auc',

           cv=3)



sfs1 = sfs1.fit(np.array(X_train.fillna(0)), y_train)
selected_feat= X_train.columns[list(sfs1.k_feature_idx_)]

selected_feat
# load dataset

data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

data.shape
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerical_vars = list(data.select_dtypes(include=numerics).columns)

data = data[numerical_vars]

data.shape
# separate train and test sets

X_train, X_test, y_train, y_test = train_test_split(

    data.drop(labels=['SalePrice'], axis=1),

    data['SalePrice'],

    test_size=0.3,

    random_state=0)



X_train.shape, X_test.shape
# find and remove correlated features

def correlation(dataset, threshold):

    col_corr = set()  # Set of all the names of correlated columns

    corr_matrix = dataset.corr()

    for i in range(len(corr_matrix.columns)):

        for j in range(i):

            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value

                colname = corr_matrix.columns[i]  # getting the name of column

                col_corr.add(colname)

    return col_corr



corr_features = correlation(X_train, 0.8)

print('correlated features: ', len(set(corr_features)) )
# removed correlated  features

X_train.drop(labels=corr_features, axis=1, inplace=True)

X_test.drop(labels=corr_features, axis=1, inplace=True)



X_train.shape, X_test.shape
X_train.fillna(0, inplace=True)
# step forward feature selection



from mlxtend.feature_selection import SequentialFeatureSelector as SFS



sfs1 = SFS(RandomForestRegressor(), 

           k_features=5, 

           forward=True, 

           floating=False, 

           verbose=2,

           scoring='r2',

           cv=3)



sfs1 = sfs1.fit(np.array(X_train), y_train)
X_train.columns[list(sfs1.k_feature_idx_)]
from sklearn.linear_model import Lasso, LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectFromModel
# load dataset

data = pd.read_csv('/kaggle/input/bnp-paribas-cardif-claims-management/train.csv.zip')

data.shape
data.head()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerical_vars = list(data.select_dtypes(include=numerics).columns)

data = data[numerical_vars]

data.shape
# separate train and test sets

X_train, X_test, y_train, y_test = train_test_split(

    data.drop(labels=['target', 'ID'], axis=1),

    data['target'],

    test_size=0.3,

    random_state=0)



X_train.shape, X_test.shape
# linear models benefit from feature scaling



scaler = StandardScaler()

scaler.fit(X_train.fillna(0))
# l1 penalty is used for LASSO fitting

sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))

sel_.fit(scaler.transform(X_train.fillna(0)), y_train)
# this command let's me visualise those features that were kept

sel_.get_support()
# Now I make a list with the selected features

selected_feat = X_train.columns[(sel_.get_support())]



print('total features: {}'.format((X_train.shape[1])))

print('selected features: {}'.format(len(selected_feat)))

print('features with coefficients shrank to zero: {}'.format(

    np.sum(sel_.estimator_.coef_ == 0)))
# the number of features which coefficient was shrank to zero:

np.sum(sel_.estimator_.coef_ == 0)
# we can identify the removed features like this:

removed_feats = X_train.columns[(sel_.estimator_.coef_ == 0).ravel().tolist()]

removed_feats
# we can then remove the features from the training and testing set

# like this

X_train_selected = sel_.transform(X_train.fillna(0))

X_test_selected = sel_.transform(X_test.fillna(0))

print("Before Lasso Regularization :", X_train.shape, X_test.shape)

print("After Lasso Regularization :", X_train_selected.shape, X_test_selected.shape)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import RFE

from sklearn.metrics import roc_auc_score
# load dataset

data = pd.read_csv('/kaggle/input/bnp-paribas-cardif-claims-management/train.csv.zip', nrows=50000)

data.shape
data.head()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerical_vars = list(data.select_dtypes(include=numerics).columns)

data = data[numerical_vars]

data.shape
# separate train and test sets

X_train, X_test, y_train, y_test = train_test_split(

    data.drop(labels=['target', 'ID'], axis=1),

    data['target'],

    test_size=0.3,

    random_state=0)



X_train.shape, X_test.shape
sel_ = SelectFromModel(RandomForestClassifier(n_estimators=100))

sel_.fit(X_train.fillna(0), y_train)
# this command let's me visualise those features that were selected.

sel_.get_support()
# let's add the variable names and order it for clearer visualisation

selected_feat = X_train.columns[(sel_.get_support())]

len(selected_feat)
# let's display the list of features

selected_feat
pd.Series(sel_.estimator_.feature_importances_.ravel()).hist()
print('total features: {}'.format((X_train.shape[1])))

print('selected features: {}'.format(len(selected_feat)))

print('features with coefficients greater than the mean coefficient: {}'.format(

    np.sum(sel_.estimator_.feature_importances_ > sel_.estimator_.feature_importances_.mean())))