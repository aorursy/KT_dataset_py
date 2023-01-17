%matplotlib inline
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10, 8)
import seaborn as sns
import numpy as np
import pandas as pd
import math
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
y.value_counts()
def S_entropy(p):
    return (-1)*sum([pi*math.log2(pi) for pi in p])
s0 = S_entropy([4/7,3/7])
s0
df_train['Will_go'] = y
df_train
handsome =  df_train[df_train['Looks_handsome']==1]
nothandsome =  df_train[df_train['Looks_handsome']==0]
handsome
nothandsome
def info_gain(s0,s,n,N):
    return s0-sum([ni/N*si for si,ni in zip(s,n)])
s = []
s.append(S_entropy([1/4,3/4]))
s.append(S_entropy([1/3,2/3]))

s
info_gain(s0,s,[4,3],7)
df_train = df_train.drop(['Will_go'],axis=1)
tree = DecisionTreeClassifier()
tree.fit(df_train,y)
y_pred = tree.predict(df_test)
# you code here
balls = [1 for i in range(9)] + [0 for i in range(11)]
balls
# two groups
balls_left  = [1 for i in range(8)] + [0 for i in range(5)] # 8 blue and 5 yellow
balls_right = [1 for i in range(1)] + [0 for i in range(6)] # 1 blue and 6 yellow

from math import log
    
def entropy(a_list):
    lst = list(a_list)
    size = len(lst) 
    entropy = 0
    set_elements = len(set(lst))
    if set_elements in [0, 1]:
        return 0
    for i in set(lst):
        occ = lst.count(i)
        entropy -= occ/size * log (occ/size,2)
    return entropy
print(entropy(balls)) # 9 blue и 11 yellow
print(entropy(balls_left)) # 8 blue и 5 yellow
print(entropy(balls_right)) # 1 blue и 6 yellow
# information gain calculation
def information_gain(root, left, right):
    return entropy(root) - entropy(left)*len(left)/len(root) - entropy(right)*len(right)/len(root) 
information_gain(balls,balls_left,balls_right)
X[X.iloc[:, best_feat_id] == 0]

len(df_train[df_train.iloc[:,0]==0])
def information_gains(X, y):
    '''Outputs information gain when splitting with each feature'''
    out = []
    for i in X.columns:
        out.append(information_gain(y, y[X[i] == 0], y[X[i] == 1]))
    return out
def best_feature_to_split(X, y):
    '''Outputs information gain when splitting on best feature'''
    out = []
    for i in X.columns:
        out.append(information_gain(y,y[X[i]==0],y[X[i]==1]))
    return out
    
best_feature_to_split(df_train,y)

def build_tree(X,y,feature_names):
    features_rate = best_feature_to_split(X,y)
    best_split_id = features_rate.index(max(features_rate))
    best_feature = feature_names[best_split_id]
    print('Best feature to split: {}'.format(best_feature))
    
    x_left = X[X.iloc[:,best_split_id]==0]
    x_right = X[X.iloc[:,best_split_id]==1]
    
    y_left = y[X.iloc[:,best_split_id]==0]
    y_right = y[X.iloc[:,best_split_id]==1]
    
    print('Splits: left leaf = {0}, right leaf = {1}'.format(len(x_left),len(x_right)))
    
    entropy_left = entropy(y_left)
    entropy_right = entropy(y_right)
    print('Entropy: left leaf = {0}, right leaf = {1}'.format(entropy_left,entropy_right))
    
    if entropy_left != 0:
        print('Splittiong left:\n\n')
        build_tree(x_left,y_left,feature_names)
    if entropy_right != 0:
        print('Splittiong right:\n\n')
        build_tree(x_right,y_right,feature_names)

build_tree(df_train,y,df_train.columns)

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
tree = DecisionTreeClassifier(max_depth=3,random_state=17)
tree.fit(X_train,y_train)
tree_preds = tree.predict(X_test)
accuracy_score(y_test,tree_preds)
tree_params = {'max_depth': range(2,11),'max_features':[1,2,5,10,20,25,40,50,70]}

locally_best_tree = GridSearchCV(tree,tree_params,cv=5)             

locally_best_tree.fit(X_train,y_train) 
locally_best_tree.best_params_
tuned_tree = DecisionTreeClassifier(max_depth=8,max_features=70,random_state=17)
tuned_tree.fit(X_train,y_train)
tuned_tree_predictions = tuned_tree.predict(X_test)
accuracy_score(y_test,tuned_tree_predictions)
# you code here 
# rf = 
# rf.fit # you code here 
# you code here 
# forest_params = {'max_depth': range(10, 21),
#                 'max_features': range(5, 105, 20)}

# locally_best_forest = GridSearchCV # you code here 

# locally_best_forest.fit # you code here 
# you code here 