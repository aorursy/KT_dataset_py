%matplotlib inline

from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (10, 8)

import seaborn as sns

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

import collections

from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from ipywidgets import Image

from io import StringIO

import pydotplus #pip install pydotplus
# Create dataframe with dummy variables

def create_df(dic, feature_list):

    out = pd.DataFrame(dic)

    out = pd.concat([out, pd.get_dummies(out[feature_list])], axis = 1)

    out.drop(feature_list, axis = 1, inplace = True)

    return out



# Some feature values are present in train and absent in test and vice-versa.

def intersect_features(train, test):

    common_feat = list( set(train.keys()) & set(test.keys()))

    return train[common_feat], test[common_feat]
features = ['Looks', 'Alcoholic_beverage','Eloquence','Money_spent']
df_train = {}

df_train['Looks'] = ['handsome', 'handsome', 'handsome', 'repulsive',

                         'repulsive', 'repulsive', 'handsome'] 

df_train['Alcoholic_beverage'] = ['yes', 'yes', 'no', 'no', 'yes', 'yes', 'yes']

df_train['Eloquence'] = ['high', 'low', 'average', 'average', 'low',

                                   'high', 'average']

df_train['Money_spent'] = ['lots', 'little', 'lots', 'little', 'lots',

                                  'lots', 'lots']

df_train['Will_go'] = LabelEncoder().fit_transform(['+', '-', '+', '-', '-', '+', '+'])



df_train = create_df(df_train, features)

df_train
df_test = {}

df_test['Looks'] = ['handsome', 'handsome', 'repulsive'] 

df_test['Alcoholic_beverage'] = ['no', 'yes', 'yes']

df_test['Eloquence'] = ['average', 'high', 'average']

df_test['Money_spent'] = ['lots', 'little', 'lots']

df_test = create_df(df_test, features)

df_test
# Some feature values are present in train and absent in test and vice-versa.

y = df_train['Will_go']

df_train, df_test = intersect_features(train=df_train, test=df_test)

df_train
df_test
s0=(-(y.value_counts()/y.size)*np.log2(y.value_counts()/y.size)).sum()
print(df_train.Looks_handsome)

s1=(-(df_train['Looks_handsome'][:4].value_counts()/df_train['Looks_handsome'][:4].size)*np.log2(df_train['Looks_handsome'][:4].value_counts()/df_train['Looks_handsome'][:4].size)).sum()

s2=(-(df_train['Looks_handsome'][4:].value_counts()/df_train['Looks_handsome'][4:].size)*np.log2(df_train['Looks_handsome'][4:].value_counts()/df_train['Looks_handsome'][4:].size)).sum()

print ((-(df_train['Looks_handsome'][:4].value_counts()/df_train['Looks_handsome'][:4].size)*np.log2(df_train['Looks_handsome'][:4].value_counts()/df_train['Looks_handsome'][:4].size)).sum())

print ((-(df_train['Looks_handsome'][4:].value_counts()/df_train['Looks_handsome'][4:].size)*np.log2(df_train['Looks_handsome'][4:].value_counts()/df_train['Looks_handsome'][4:].size)).sum())

s0-(4/7)*s1-(3/7)*s2
# you code here
# you code here
balls = [1 for i in range(9)] + [0 for i in range(11)]
# two groups

balls_left  = [1 for i in range(8)] + [0 for i in range(5)] # 8 blue and 5 yellow

balls_right = [1 for i in range(1)] + [0 for i in range(6)] # 1 blue and 6 yellow
def entropy(a_list):

    # you code here

    pass
print(entropy(balls)) # 9 blue и 11 yellow

print(entropy(balls_left)) # 8 blue и 5 yellow

print(entropy(balls_right)) # 1 blue и 6 yellow

print(entropy([1,2,3,4,5,6])) # entropy of a fair 6-sided die
# information gain calculation

def information_gain(root, left, right):

    ''' root - initial data, left and right - two partitions of initial data'''

    

    # you code here

    pass
def best_feature_to_split(X, y):

    '''Outputs information gain when splitting on best feature'''

    

    # you code here

    pass
data_train = pd.read_csv('../input/adult_train.csv')
data_train.tail()
data_test = pd.read_csv('../input/adult_test.csv')
data_test.tail()
# necessary to remove rows with incorrect labels in test dataset

data_test = data_test[(data_test['Target'] == ' >50K.') | (data_test['Target']==' <=50K.')]



# encode target variable as integer

data_train.loc[data_train['Target']==' <=50K', 'Target'] = 0

data_train.loc[data_train['Target']==' >50K', 'Target'] = 1



data_test.loc[data_test['Target']==' <=50K.', 'Target'] = 0

data_test.loc[data_test['Target']==' >50K.', 'Target'] = 1
data_test.describe(include='all').T
data_train['Target'].value_counts()
fig = plt.figure(figsize=(25, 15))

cols = 5

rows = np.ceil(float(data_train.shape[1]) / cols)

for i, column in enumerate(data_train.columns):

    ax = fig.add_subplot(rows, cols, i + 1)

    ax.set_title(column)

    if data_train.dtypes[column] == np.object:

        data_train[column].value_counts().plot(kind="bar", axes=ax)

    else:

        data_train[column].hist(axes=ax)

        plt.xticks(rotation="vertical")

plt.subplots_adjust(hspace=0.7, wspace=0.2)
data_train.dtypes
data_test.dtypes
data_test['Age'] = data_test['Age'].astype(int)
data_test['fnlwgt'] = data_test['fnlwgt'].astype(int)

data_test['Education_Num'] = data_test['Education_Num'].astype(int)

data_test['Capital_Gain'] = data_test['Capital_Gain'].astype(int)

data_test['Capital_Loss'] = data_test['Capital_Loss'].astype(int)

data_test['Hours_per_week'] = data_test['Hours_per_week'].astype(int)
# choose categorical and continuous features from data



categorical_columns = [c for c in data_train.columns 

                       if data_train[c].dtype.name == 'object']

numerical_columns = [c for c in data_train.columns 

                     if data_train[c].dtype.name != 'object']



print('categorical_columns:', categorical_columns)

print('numerical_columns:', numerical_columns)
# fill missing data



for c in categorical_columns:

    data_train[c].fillna(data_train[c].mode(), inplace=True)

    data_test[c].fillna(data_train[c].mode(), inplace=True)

    

for c in numerical_columns:

    data_train[c].fillna(data_train[c].median(), inplace=True)

    data_test[c].fillna(data_train[c].median(), inplace=True)
data_train = pd.concat([data_train[numerical_columns],

    pd.get_dummies(data_train[categorical_columns])], axis=1)



data_test = pd.concat([data_test[numerical_columns],

    pd.get_dummies(data_test[categorical_columns])], axis=1)
set(data_train.columns) - set(data_test.columns)
data_train.shape, data_test.shape
data_test['Country_ Holand-Netherlands'] = 0
set(data_train.columns) - set(data_test.columns)
data_train.head(2)
data_test.head(2)
X_train = data_train.drop(['Target'], axis=1)

y_train = data_train['Target']



X_test = data_test.drop(['Target'], axis=1)

y_test = data_test['Target']
# you code here

# tree = 

# tree.fit
# you code here

# tree_predictions = tree.predict 
# you code here

# accuracy_score 
tree_params = {'max_depth': range(2,11)}



locally_best_tree = GridSearchCV # you code here                     



locally_best_tree.fit; # you code here 
# you code here 

# tuned_tree = 

# tuned_tree.fit 

# tuned_tree_predictions = tuned_tree.predict

# accuracy_score
# you code here 

# rf = 

# rf.fit # you code here 
# you code here 
# forest_params = {'max_depth': range(10, 21),

#                 'max_features': range(5, 105, 20)}



# locally_best_forest = GridSearchCV # you code here 



# locally_best_forest.fit # you code here 
# you code here 