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
# you code here
def entropy(series):
    import scipy as sc
    p = series.value_counts() / len(series)
    return sc.stats.entropy(p, base=2)

def entropy_manual(series):
    p_list = series.value_counts() / len(series)
    return np.sum([- p * np.log2(p) for p in p_list])

s0 = entropy(y)
print('initial entropy:', s0)
print(entropy(y) == entropy_manual(y))

# you code here
df_handsome = y[df_train['Looks_handsome']== 1]
s1 = entropy(df_handsome)

df_non_handsome = y[df_train['Looks_handsome'] == 0]
s2 = entropy(df_non_handsome)

print('left group entropy:', s1)
print('right group entropy:', s2)

# weighted entropy on split groups
split_entropy = s1 * len(df_handsome) / len(y) + s2 * len(df_non_handsome) / len(y)
print('split entropy:', split_entropy)

print('information gain:', s0 - split_entropy)
# you code here
dtree = DecisionTreeClassifier(criterion='entropy')

dtree.fit(df_train, y)
dtree.score(df_train, y)
# you code here
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
balls = [1 for i in range(9)] + [0 for i in range(11)]
# two groups
balls_left  = [1 for i in range(8)] + [0 for i in range(5)] # 8 blue and 5 yellow
balls_right = [1 for i in range(1)] + [0 for i in range(6)] # 1 blue and 6 yellow
balls_left
balls_right
def entropy(a_list):
    # you code here
    p_list = pd.Series(a_list).value_counts() / len(a_list)
    return np.sum([- p * np.log2(p) for p in p_list])
print(entropy(balls)) # 9 blue и 11 yellow
print(entropy(balls_left)) # 8 blue и 5 yellow
print(entropy(balls_right)) # 1 blue и 6 yellow
print(entropy([1,2,3,4,5,6])) # entropy of a fair 6-sided die
# information gain calculation
def information_gain(root, left, right):
    ''' root - initial data, left and right - two partitions of initial data'''
    
    # you code here
    init_entropy = entropy(root)
    entropy_left = entropy(left)
    entropy_right = entropy(right)
    
    # weighted entropy sum
    split_entropy = entropy_left * len(left) / len(root) + \
                    entropy_right * len(right) / len(root)
    
    return init_entropy - split_entropy


information_gain(balls, balls_left, balls_right)
def best_feature_to_split(X, y):
    '''Outputs information gain when splitting on best feature'''
    
    # you code here
    pass
data_train = pd.read_csv('../input/adult_train.csv')
print(data_train.shape)
data_train.tail()
data_test = pd.read_csv('../input/adult_test.csv')
print(data_test.shape)
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
print(data_train['Workclass'].value_counts())
print('================')
print('mode of Workclass:', data_train['Workclass'].mode())
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
tree = DecisionTreeClassifier(max_depth=3, criterion = 'entropy', random_state=17)
tree.fit(X_train, y_train)
tree.score(X_train, y_train)
# you code here
# accuracy_score 
#help(tree.score)
tree.score(X_test, y_test)
tree_params = {'max_depth': range(2,11)}

# default 3 folds
locally_best_tree = GridSearchCV(
    tree, param_grid=tree_params, n_jobs=-1, verbose=1)              

locally_best_tree.fit(X_train, y_train); # you code here 
print('best param:', locally_best_tree.best_params_)
print('best score:', locally_best_tree.best_score_)
print('final train score:', locally_best_tree.score(X_train, y_train))
print('final test score:', locally_best_tree.score(X_test, y_test))

# you code here 
# tuned_tree = 
# tuned_tree.fit 
# tuned_tree_predictions = tuned_tree.predict
# accuracy_score
# you code here 
rf = RandomForestClassifier(n_estimators=100, random_state=17, verbose=0)
rf.fit(X_train, y_train)
# you code here 
#rf.predict(X_test)
rf.score(X_test, y_test)
import math
print(math.sqrt(105))
print(np.log(105))
forest_params
forest_params = {'max_depth': range(10, 21),
                 #'max_features': range(5, 105, 20)
                }

locally_best_forest = GridSearchCV(rf, param_grid=forest_params,
                                   n_jobs=-1, verbose=1)

locally_best_forest.fit(X_train, y_train)
# you code here 
print('best params:', locally_best_forest.best_params_)
print('best scores:', locally_best_forest.best_score_)
print('final test score:', locally_best_forest.score(X_test, y_test))