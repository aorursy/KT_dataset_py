import numpy as np

import pandas as pd



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize': (9, 6)})
# Create dataframe with dummy variables

def create_df(dict_, feature_list):

    out = pd.DataFrame(dict_)

    out = pd.concat([out, pd.get_dummies(out[feature_list])], axis = 1)

    out.drop(feature_list, axis = 1, inplace = True)

    return out



# Some feature values are present in train and absent in test and vice-versa.

def intersect_features(train, test):

    common_feat = list( set(train.keys()) & set(test.keys()))

    return train[common_feat], test[common_feat]
features = ['Looks', 'Alcoholic_beverage','Eloquence','Money_spent']
from sklearn.preprocessing import LabelEncoder



df_train = {

    'Looks': ['handsome', 'handsome', 'handsome', 'repulsive',

              'repulsive', 'repulsive', 'handsome'],

    'Alcoholic_beverage': ['yes', 'yes', 'no', 'no',

                           'yes', 'yes', 'yes'],

    'Eloquence': ['high', 'low', 'average', 'average',

                  'low', 'high', 'average'],

    'Money_spent': ['lots', 'little', 'lots', 'little',

                    'lots', 'lots', 'lots']

}

df_train['Will_go'] = LabelEncoder().fit_transform(['+', '-', '+', '-',

                                                    '-', '+', '+'])



df_train = create_df(df_train, features)

df_train
df_test = {

    'Looks': ['handsome', 'handsome', 'repulsive'],

    'Alcoholic_beverage': ['no', 'yes', 'yes'],

    'Eloquence': ['average', 'high', 'average'],

    'Money_spent': ['lots', 'little', 'lots']

}



df_test = create_df(df_test, features)

df_test
# Some feature values are present in train and absent in test and vice-versa.

y = df_train['Will_go']

df_train, df_test = intersect_features(train=df_train, test=df_test)

df_train
df_test
probs = y.value_counts(normalize=True)

initial_info = -sum(np.multiply(probs, np.log2(probs)))

initial_info
# Split by 'Looks_handsome' >= 0.5:

#      True /       \ False

#    0,0,0,1        0,1,1

left_probs = pd.Series([0,0,0,1]).value_counts(normalize=True)

entropy_1 = -sum(np.multiply(left_probs, np.log2(left_probs)))



right_probs = pd.Series([0,1,1]).value_counts(normalize=True)

entropy_2 = -sum(np.multiply(right_probs, np.log2(right_probs)))



information_gained = initial_info - sum([4*entropy_1, 3*entropy_2])/len(df_train)

entropy_1, entropy_2, information_gained
from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(criterion='entropy', max_depth=4)

tree.fit(df_train, y)
# from sklearn.tree import plot_tree

# plot_tree(tree, filled=True, class_names=['No', 'Will go'])    # version >= 0.21



from sklearn.tree import export_graphviz

import graphviz



dot_data = export_graphviz(tree, out_file=None,

                           feature_names=list(df_train.columns),

                           class_names=['No', 'Will go'],

                           filled=True, rounded=True,

                           special_characters=True)

graphviz.Source(dot_data)
balls = [1 for i in range(9)] + [0 for i in range(11)]
# two groups

balls_left  = [1 for i in range(8)] + [0 for i in range(5)] # 8 blue and 5 yellow

balls_right = [1 for i in range(1)] + [0 for i in range(6)] # 1 blue and 6 yellow
def entropy(labels):

    """ Return Shannon entropy of list of labels: [0, 3, 1,...]. """

    probs = pd.Series(labels).value_counts(normalize=True)

    return -sum(np.multiply(probs, np.log2(probs)))
# Tests

print(entropy(balls)) # 9 blue и 11 yellow

print(entropy(balls_left)) # 8 blue и 5 yellow

print(entropy(balls_right)) # 1 blue и 6 yellow

print(entropy([1,2,3,4,5,6])) # entropy of a fair 6-sided die
def information_gain(root, left, right):

    """ root - initial data, left and right - two splits of initial data. """

    splitted = sum([len(left)*entropy(left), len(right)*entropy(right)])/len(root)

    return entropy(root) - splitted
information_gain(balls, balls_left, balls_right)
data_train = pd.read_csv('../input/adult_train.csv').rename(columns={'Martial_Status': 'Marital_Status'})

print(data_train.shape)

data_train.tail()
data_test = pd.read_csv('../input/adult_test.csv').rename(columns={'Martial_Status': 'Marital_Status'})

print(data_test.shape)

data_test.tail()
# necessary to remove rows with incorrect labels in test dataset

data_test.dropna(subset=['Target'], inplace=True)



# encode target variable as integer

data_train.loc[data_train['Target']==' <=50K', 'Target'] = 0

data_train.loc[data_train['Target']==' >50K', 'Target'] = 1



data_test.loc[data_test['Target']==' <=50K.', 'Target'] = 0

data_test.loc[data_test['Target']==' >50K.', 'Target'] = 1

data_test.tail(3)
data_test.describe(include='all').transpose()
data_train['Target'].value_counts()
fig = plt.figure(figsize=(25, 15))

cols = 5

rows = np.ceil(data_train.shape[1] / cols)

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
categorical_columns = [c for c in data_train.columns 

                       if data_train[c].dtype.name == 'object']

numerical_columns = [c for c in data_train.columns 

                     if data_train[c].dtype.name != 'object']



print('categorical_columns:', categorical_columns)

print('numerical_columns:', numerical_columns)
# we see some missing values

data_train.isna().sum()
# fill missing data

for c in categorical_columns:

    data_train[c].fillna(data_train[c].mode()[0], inplace=True)

    data_test[c].fillna(data_train[c].mode()[0], inplace=True)

    

for c in numerical_columns:

    data_train[c].fillna(data_train[c].median(), inplace=True)

    data_test[c].fillna(data_train[c].median(), inplace=True)
# no more missing values

print(data_train.isna().sum().sum())

data_test.isna().sum()
data_train = pd.concat([data_train[numerical_columns],

    pd.get_dummies(data_train[categorical_columns])], axis='columns')



data_test = pd.concat([data_test[numerical_columns],

    pd.get_dummies(data_test[categorical_columns])], axis='columns')

data_test.tail(3)
set(data_train.columns) - set(data_test.columns)
data_train.shape, data_test.shape
data_test['Country_ Holand-Netherlands'] = 0
set(data_train.columns) - set(data_test.columns)
data_train.head(2)
data_test.head(2)
X_train = data_train.drop(['Target'], axis='columns')

y_train = data_train['Target']



X_test = data_test.drop(['Target'], axis='columns')

y_test = data_test['Target']
from sklearn.metrics import accuracy_score



tree = DecisionTreeClassifier(max_depth=3, random_state=17)

tree.fit(X_train, y_train)

accuracy_score(y_test, tree.predict(X_test))
from sklearn.model_selection import GridSearchCV



params = {'max_depth': range(2,11)}



search = GridSearchCV(DecisionTreeClassifier(random_state=17),

                      param_grid=params, cv=5, n_jobs=-1)

search.fit(X_train, y_train)

search.best_score_, search.best_params_
best_tree = search.best_estimator_

accuracy_score(y_test, best_tree.predict(X_test))
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=50, random_state=17)

rf.fit(X_train, y_train)

accuracy_score(y_test, rf.predict(X_test))
%%time

params = {'max_depth': range(9, 21),

          'max_features': range(5, 105, 30)}



search = GridSearchCV(RandomForestClassifier(n_estimators=50, random_state=17),

                      param_grid=params, cv=5, n_jobs=-1)

search.fit(X_train, y_train)

print(search.best_score_, search.best_params_)
best_rf = search.best_estimator_

accuracy_score(y_test, best_rf.predict(X_test))