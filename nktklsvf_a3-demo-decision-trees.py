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

from math import log
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
df_train
df_test
will_go_counts = y.value_counts()

will_go_len = len(y)

f_probability = will_go_counts.get(0) / will_go_len

s_probability = will_go_counts.get(1) / will_go_len

initial_entropy = -f_probability * log(f_probability, 2) - s_probability * log(s_probability, 2)

print(initial_entropy) 
groups = [y[df_train["Looks_handsome"] == 1], y[df_train["Looks_handsome"] == 0]]

lens = [len(groups[0]), len(groups[1])]

entropies = [0, 0]

for i in range(2):

    counts = groups[i].value_counts()

    group_len = len(groups[i])

    f_probability = counts.get(0) / lens[i]

    s_probability = counts.get(1) / lens[i]

    entropies[i] = -f_probability * log(f_probability, 2) - s_probability * log(s_probability, 2)

ig = initial_entropy - (lens[0] / will_go_len) * entropies[0] - (lens[1] / will_go_len) * entropies[1]

print("Entropy of the left group:", entropies[0])

print("Entropy of the right group:", entropies[1])

print("IG:", ig)
clf_tree = DecisionTreeClassifier(random_state=17)

clf_tree.fit(df_train, y)
export_graphviz(clf_tree, feature_names=df_train.columns, out_file='tree.dot', filled=True)

!dot -Tpng 'tree.dot' -o 'tree.png'
balls = [1 for i in range(9)] + [0 for i in range(11)]
# two groups

balls_left  = [1 for i in range(8)] + [0 for i in range(5)] # 8 blue and 5 yellow

balls_right = [1 for i in range(1)] + [0 for i in range(6)] # 1 blue and 6 yellow
def entropy(a_list):

    list_len = len(a_list)

    result = 0.0

    for unique_value in set(a_list):

        count = 0

        for element in a_list:

            if unique_value == element:

                count += 1

        probability = count / list_len

        result -= probability * log(probability, 2)

    return result
print(entropy(balls)) # 9 blue и 11 yellow

print(entropy(balls_left)) # 8 blue и 5 yellow

print(entropy(balls_right)) # 1 blue и 6 yellow

print(entropy([1,2,3,4,5,6])) # entropy of a fair 6-sided die
# information gain calculation

def information_gain(root, left, right):

    ''' root - initial data, left and right - two partitions of initial data'''

    

    root_len = len(root)

    return entropy(root) - (len(left) / root_len) * entropy(left) - (len(right) / root_len) * entropy(right)
information_gain(balls, balls_left, balls_right)
def best_feature_to_split(X, y):

    '''Outputs information gain when splitting on best feature'''



    return max([information_gain(y,y[X[i] == 0], y[X[i] == 1]) for i in X])
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
tree = DecisionTreeClassifier(max_depth=3, random_state=17)

tree.fit(X_train, y_train)
tree_predictions = tree.predict(X_test)
accuracy_score(y_test, tree_predictions)
tree_params = {'max_depth': range(2,11)}



locally_best_tree = GridSearchCV(DecisionTreeClassifier(random_state=17), tree_params, cv=5) # you code here                     



locally_best_tree.fit(X_train, y_train) 
locally_best_tree.best_params_
tuned_tree = DecisionTreeClassifier(max_depth=9, random_state=17) 

tuned_tree.fit(X_train, y_train)

tuned_tree_predictions = tuned_tree.predict(X_test)

accuracy_score(y_test, tuned_tree_predictions)
rf = RandomForestClassifier(n_estimators=100, random_state=17)

rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)

accuracy_score(y_test, rf_predictions)
forest_params = {'max_depth': range(10, 21),

                'max_features': range(5, 105, 20)}



locally_best_forest = GridSearchCV(RandomForestClassifier(random_state=17), forest_params, cv=5)



locally_best_forest.fit(X_train, y_train) 
locally_best_tree.best_params_
lbf_predictions = locally_best_forest.predict(X_test)

accuracy_score(y_test, lbf_predictions)