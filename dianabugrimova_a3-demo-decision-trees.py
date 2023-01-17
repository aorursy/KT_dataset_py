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
df_train.head()
# Some feature values are present in train and absent in test and vice-versa.

y = df_train['Will_go']

df_train, df_test = intersect_features(train=df_train, test=df_test)

df_train
df_test
tree = DecisionTreeClassifier(criterion = 'entropy')
tree.fit(df_train, y)
import sklearn

import os     

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from IPython.display import SVG

from graphviz import Source

from IPython.display import display

from IPython.display import HTML

style = "<style>svg{width:70% !important;height:70% !important;}</style>"

HTML(style)
graph = Source(sklearn.tree.export_graphviz(tree, out_file=None,

                                   feature_names=list(df_train),

                                   filled = True))



display(SVG(graph.pipe(format='svg')))
tree_2 = DecisionTreeClassifier(criterion = 'entropy')

tree_2.fit(df_train[['Looks_handsome']], y)
graph = Source(sklearn.tree.export_graphviz(tree_2, out_file=None,

                                   filled = True))



display(SVG(graph.pipe(format='svg')))
ig = 0.128
tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=2)

tree.fit(df_train, y)
graph = Source(sklearn.tree.export_graphviz(tree, out_file=None,

                                   feature_names=list(df_train),

                                   filled = True))



display(SVG(graph.pipe(format='svg')))
balls = [1 for i in range(9)] + [0 for i in range(11)]
# two groups

balls_left  = [1 for i in range(8)] + [0 for i in range(5)] # 8 blue and 5 yellow

balls_right = [1 for i in range(1)] + [0 for i in range(6)] # 1 blue and 6 yellow
n=[1,2,3,4,5,6]
print(len(set(n)))

print(len(n))

print(n[1])

n.count(n[0])
def entropy(a_list):

    import math

    if len(set(a_list))==len(a_list) :

        entr=-(len(a_list)*(a_list.count(a_list[0])/len(a_list))*math.log2(a_list.count(a_list[0])/len(a_list)))

    else:

        entr=-(a_list.count(0)/len(a_list))*math.log2(a_list.count(0)/len(a_list))-(a_list.count(1)/len(a_list))*math.log2(a_list.count(1)/len(a_list))

    return(entr)
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
# we see some missing values

data_train.info()
# fill missing data



for c in categorical_columns:

    data_train[c].fillna(data_train[c].mode()[0], inplace=True)

    data_test[c].fillna(data_train[c].mode()[0], inplace=True)

    

for c in numerical_columns:

    data_train[c].fillna(data_train[c].median(), inplace=True)

    data_test[c].fillna(data_train[c].median(), inplace=True)
# no more missing values

data_train.info()
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
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=17) 

tree.fit(X_train, y_train)
graph = Source(sklearn.tree.export_graphviz(tree, out_file=None,

                                   feature_names=list(X_train),

                                   filled = True))



display(SVG(graph.pipe(format='svg')))
tree_predictions = tree.predict(X_test)
accuracy_score(y_test, tree_predictions)
tree_params = {'max_depth': range(2,11)}



locally_best_tree = GridSearchCV(tree, tree_params,

                         cv=5, n_jobs=-1, verbose=True)                   



locally_best_tree.fit(X_train, y_train)
locally_best_tree.best_params_ 
# you code here 

tuned_tree = DecisionTreeClassifier(criterion='entropy', max_depth=9, random_state=17) 

tuned_tree.fit(X_train, y_train) 

tuned_tree_predictions = tuned_tree.predict(X_test)

accuracy_score(y_test, tuned_tree_predictions)
# you code here 

rf = RandomForestClassifier(n_estimators=100, 

                                random_state=17)

rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)

accuracy_score(y_test, rf_predictions)
forest_params = {'max_depth': range(10, 21),

                 'max_features': range(5, 105, 20)}



locally_best_forest = GridSearchCV(rf, tree_params,

                         cv=5, n_jobs=-1, verbose=True) 



locally_best_forest.fit(X_train, y_train)
tuned_rf_predictions = locally_best_forest.predict(X_test)

accuracy_score(y_test, tuned_rf_predictions )