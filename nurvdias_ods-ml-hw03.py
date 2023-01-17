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



import seaborn as sns

import matplotlib.pyplot as plt



import plotly.graph_objs as go

import plotly.express as px



sns.set()
df_train = pd.read_csv("/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv", index_col = "uid")

df_test = pd.read_csv("/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv", index_col = "uid")



print("train: ", df_train.shape)

print("test: ", df_test.shape)
df_train.drop_duplicates(keep="first", inplace = True)





print("Shape after removing duplicate values")

print("train: ", df_train.shape)

print("test: ", df_test.shape)
df_train.head()
df_train.info()
df_train.nunique()
sns.distplot(df_train['age'])
bins= [0,25,45,65,100]

labels = ["Young", "Middle", "Senior", "Old"]

df_train['age'] = pd.cut(df_train['age'], bins=bins, labels=labels, right=False)

df_test['age'] = pd.cut(df_test['age'], bins=bins, labels=labels, right=False)
sns.countplot(df_train['age'])
sns.countplot(y = df_train['workclass'], hue=df_train['target'])
df_train['workclass'].replace({" Without-pay":"Not-working", 

                               " Never-worked":"Not-working", 

                               " ?": "UNKNOWN", 

                               " State-gov": "Gov", 

                               " Local-gov": "Gov", 

                               " Federal-gov": "Gov", 

                               " Self-emp-not-inc": "Self-emp", 

                               " Self-emp-inc": "Self-emp", 

                               " Private": "Private"}, inplace=True)

df_test['workclass'].replace({" Without-pay":"Not-working", 

                              " Never-worked": "Not-working", 

                              " ?":"UNKNOWN", 

                               " State-gov": "Gov", 

                               " Local-gov": "Gov", 

                               " Federal-gov": "Gov", 

                               " Self-emp-not-inc": "Self-emp", 

                               " Self-emp-inc": "Self-emp", 

                               " Private": "Private"}, inplace=True)
sns.countplot(y = df_train['workclass'], hue=df_train['target'])
df_train[['education-num', 'education']].drop_duplicates()
df_train.drop('education', axis=1, inplace=True)

df_test.drop('education', axis=1, inplace=True)
df_train['education-num'].replace(to_replace = [1,2,3,4,5,6,7,8], 

                                  value = 0,

                                  inplace=True)

df_train['education-num'].replace(to_replace = [9,10], 

                                  value = 1,

                                  inplace=True)

df_train['education-num'].replace(to_replace = [11,12], 

                                  value = 2,

                                  inplace=True)

df_train['education-num'].replace({13: 3, 14: 4, 15: 5, 16:6},

                                  inplace=True)





df_test['education-num'].replace(to_replace = [1,2,3,4,5,6,7,8], 

                                  value = 0,

                                  inplace=True)

df_test['education-num'].replace(to_replace = [9,10], 

                                  value = 1,

                                  inplace=True)

df_test['education-num'].replace(to_replace = [11,12], 

                                  value = 2,

                                  inplace=True)

df_test['education-num'].replace({13: 3, 14: 4, 15: 5, 16:6},

                                  inplace=True)
sns.countplot(df_train['education-num'], hue=df_train['target'])
df_train['marital-status']=df_train['marital-status'].str.strip()

df_train['marital-status'].unique()
df_train['marital-status'].replace(to_replace = ["Never-married","Divorced","Separated", "Widowed"], 

                              value = "Not-married",

                              inplace=True)

df_train['marital-status'].replace(to_replace = ["Married-civ-spouse","Married-spouse-absent","Married-AF-spouse"], 

                              value = "Married",

                              inplace=True)

df_test['marital-status'].replace(to_replace = ["Never-married","Divorced","Separated", "Widowed"], 

                              value = "Not-married",

                              inplace=True)

df_test['marital-status'].replace(to_replace = ["Married-civ-spouse","Married-spouse-absent","Married-AF-spouse"], 

                              value = "Married",

                              inplace=True)
df_train['marital-status'] = (df_train['marital-status']=='Married')

df_train['marital-status']*=1

df_train.rename(columns={"marital-status": "is-married"}, inplace=True)



df_test['marital-status'] = (df_test['marital-status']=='Married')

df_test['marital-status']*=1

df_test.rename(columns={"marital-status": "is-married"}, inplace=True)
sns.countplot(df_train['is-married'], hue=df_train['target'])
df_train['sex'] = df_train['sex'].str.strip()

df_test['sex'] = df_test['sex'].str.strip()
df_train['sex'] = (df_train['sex']=='Male')

df_train['sex']*=1

df_train.rename(columns={"sex": "is-male"}, inplace=True)



df_test['sex'] = (df_test['sex']=='Male')

df_test['sex']*=1

df_test.rename(columns={"sex": "is-male"}, inplace=True)
df_train['occupation'] = df_train['occupation'].str.strip()

df_test['occupation'] = df_test['occupation'].str.strip()
sns.catplot('target', col = 'occupation', data = df_train, kind="count", col_wrap=5)
df_train['occupation'].replace({"Adm-clerical": "Admin", 

                                "Armed-Forces": "Military", 

                                "Prof-specialty": "Professional", 

                                "Exec-managerial": "White-collar", 

                                "?":"UNKNOWN"}, inplace = True)                        

df_train['occupation'].replace(to_replace = ["Craft-repair", "Farming-fishing", 

                                             "Handlers-cleaners", "Machine-op-inspct", "Transport-moving"], 

                               value = "Blue-collar", inplace = True) 

df_train['occupation'].replace(to_replace = ["Other-service", "Priv-house-serv"], 

                               value = "Service", inplace = True)

df_train['occupation'].replace(to_replace = ["Protective-serv", "Tech-support", "Other-Occupations"], 

                               value = "Other-Occupations", inplace = True)





df_test['occupation'].replace({"Adm-clerical": "Admin", 

                                "Armed-Forces": "Military", 

                                "Prof-specialty": "Professional", 

                                "Exec-managerial": "White-collar", 

                                "?":"UNKNOWN"}, inplace = True)                        

df_test['occupation'].replace(to_replace = ["Craft-repair", "Farming-fishing", 

                                             "Handlers-cleaners", "Machine-op-inspct", "Transport-moving"], 

                               value = "Blue-collar", inplace = True) 

df_test['occupation'].replace(to_replace = ["Other-service", "Priv-house-serv"], 

                               value = "Service", inplace = True)

df_test['occupation'].replace(to_replace = ["Protective-serv", "Tech-support", "Other-Occupations"], 

                               value = "Other-Occupations", inplace = True)
sns.catplot("target", col="occupation", data = df_train, kind = "count", col_wrap=4)
sns.distplot(df_train['hours-per-week'])
bins= [0,20,40,60,100]

labels = ["Part-time", "Full-time", "Overtime", "Too-much"]

df_train['hours-per-week'] = pd.cut(df_train['hours-per-week'], bins=bins, labels=labels, right=False)

df_test['hours-per-week'] = pd.cut(df_test['hours-per-week'], bins=bins, labels=labels, right=False)
sns.countplot(df_train['hours-per-week'], hue = df_train['target'])
df_train['native-country'] = df_train['native-country'].str.strip()

df_test['native-country'] = df_test['native-country'].str.strip()
df_train['native-country'].unique()
df_train['native-country'].value_counts()
df_train['native-country'].replace(to_replace = ["Cambodia","Laos","Philippines","Thailand","Vietnam"], 

                                   value = "SE-Asia", inplace = True)

df_train['native-country'].replace(to_replace = ["Canada","India","England","Ireland","Scotland"], 

                                   value = "British-Commonwealth", inplace = True)

df_train['native-country'].replace(to_replace = ["Hong","Taiwan"], 

                                   value = "China", inplace = True)

df_train['native-country'].replace(to_replace = ["Columbia","Ecuador","El-Salvador","Peru"], 

                                   value = "South-America", inplace = True)

df_train['native-country'].replace(to_replace = ["Dominican-Republic","Guatemala","Haiti","Honduras",

                                                 "Jamaica","Mexico","Trinadad&Tobago","Nicaragua",

                                                 "Outlying-US(Guam-USVI-etc)","Puerto-Rico"], 

                                   value = "Latin-America", inplace = True)

df_train['native-country'].replace(to_replace = ["France","Italy","Holand-Netherlands","Germany"], 

                                   value = "Euro1", inplace = True)

df_train['native-country'].replace(to_replace = ["Greece","Hungary","Poland","Portugal","Yugoslavia","South"], 

                                   value = "Euro2", inplace = True)

df_train['native-country'].replace(to_replace = ["Iran","Cuba","Japan"], 

                                   value = "Other", inplace = True)

df_train['native-country'].replace(to_replace = ["?"], 

                                   value = "UNKNOWN", inplace = True)





df_test['native-country'].replace(to_replace = ["Cambodia","Laos","Philippines","Thailand","Vietnam"], 

                                   value = "SE-Asia", inplace = True)

df_test['native-country'].replace(to_replace = ["Canada","India","England","Ireland","Scotland"], 

                                   value = "British-Commonwealth", inplace = True)

df_test['native-country'].replace(to_replace = ["Hong","Taiwan"], 

                                   value = "China", inplace = True)

df_test['native-country'].replace(to_replace = ["Columbia","Ecuador","El-Salvador","Peru"], 

                                   value = "South-America", inplace = True)

df_test['native-country'].replace(to_replace = ["Dominican-Republic","Guatemala","Haiti","Honduras",

                                                 "Jamaica","Mexico","Trinadad&Tobago","Nicaragua",

                                                 "Outlying-US(Guam-USVI-etc)","Puerto-Rico"], 

                                   value = "Latin-America", inplace = True)

df_test['native-country'].replace(to_replace = ["France","Italy","Holand-Netherlands","Germany"], 

                                   value = "Euro1", inplace = True)

df_test['native-country'].replace(to_replace = ["Greece","Hungary","Poland","Portugal","Yugoslavia","South"], 

                                   value = "Euro2", inplace = True)

df_test['native-country'].replace(to_replace = ["Iran","Cuba","Japan"], 

                                   value = "Other", inplace = True)

df_test['native-country'].replace(to_replace = ["?"], 

                                   value = "UNKNOWN", inplace = True)
df_test['native-country'].value_counts()
sns.distplot(df_train[df_train['capital-loss']!=0]['capital-loss'])
med = df_train[df_train['capital-loss']!=0]['capital-loss'].median()

maxx = df_train[df_train['capital-loss']!=0]['capital-loss'].max()

bins = [-1, 0,med,maxx]

labels = ['None', 'Low', 'High']

df_train['capital-loss'] = pd.cut(df_train['capital-loss'], bins=bins, labels=labels, right=True)

df_test['capital-loss'] = pd.cut(df_test['capital-loss'], bins=bins, labels=labels, right=True)
sns.countplot(df_train['capital-loss'])
sns.distplot(df_train[df_train['capital-gain']!=0]['capital-gain'])
med = df_train[df_train['capital-gain']!=0]['capital-gain'].median()

maxx = df_train[df_train['capital-gain']!=0]['capital-gain'].max()

bins = [-1, 0,med,maxx]

labels = ['None', 'Low', 'High']

df_train['capital-gain'] = pd.cut(df_train['capital-gain'], bins=bins, labels=labels, right=True)

df_test['capital-gain'] = pd.cut(df_test['capital-gain'], bins=bins, labels=labels, right=True)
sns.countplot(df_train['capital-loss'])
df_train.head()
sns.countplot(df_train['target'])

print(df_train['target'].value_counts(normalize=True))
sns.countplot(y="age", hue='target', data=df_train)
df_train.columns
def facetgrid_countplot(x, y, **kwargs):

    sns.countplot(y=x, hue=y)

    #x = plt.xticks(rotation=90)





f = pd.melt(df_train[['target', 'age', 'education-num', 'workclass', 'is-married', 'occupation', 'is-male', 

                      'relationship', 'native-country', 'race', 'hours-per-week', 'capital-gain', 'capital-loss']], 

            id_vars=['target'])



g = sns.FacetGrid(f, col='variable', col_wrap=2,

                  sharex=False, sharey=False, aspect=2)

g = g.map(facetgrid_countplot, 'value', 'target')
df_train.info()
df_train['age'].unique()

to_change = {"Young": 1, "Middle": 2, "Senior": 3, "Old": 4}

df_train['age'] = df_train['age'].map(to_change)

df_train['age'].unique()



df_test['age'] = df_test['age'].map(to_change) 
df_temp = pd.DataFrame()

df_temp['count'] = df_train.groupby(['workclass']).count()['target']

df_temp['sum'] = df_train.groupby(['workclass']).sum()['target']

df_temp['mean'] = df_train.groupby(['workclass']).mean()['target']



df_temp.drop(columns=['sum', 'count'], inplace=True)



df_train['workclass'] = df_train['workclass'].map(df_temp.T.to_dict('list'))

df_train['workclass'] = df_train['workclass'].str[0]



df_test['workclass'] = df_test['workclass'].map(df_temp.T.to_dict('list'))

df_test['workclass'] = df_test['workclass'].str[0]
df_temp = pd.DataFrame()

df_temp['count'] = df_train.groupby(['occupation']).count()['target']

df_temp['sum'] = df_train.groupby(['occupation']).sum()['target']

df_temp['mean'] = df_train.groupby(['occupation']).mean()['target']



df_temp.drop(columns=['sum', 'count'], inplace=True)
df_train['occupation'] = df_train['occupation'].map(df_temp.T.to_dict('list'))

df_train['occupation'] = df_train['occupation'].str[0]



df_test['occupation'] = df_test['occupation'].map(df_temp.T.to_dict('list'))

df_test['occupation'] = df_test['occupation'].str[0]
df_temp = pd.DataFrame()

df_temp['count'] = df_train.groupby(['relationship']).count()['target']

df_temp['sum'] = df_train.groupby(['relationship']).sum()['target']

df_temp['mean'] = df_train.groupby(['relationship']).mean()['target']



df_temp.drop(columns=['sum', 'count'], inplace=True)
df_train['relationship'] = df_train['relationship'].map(df_temp.T.to_dict('list'))

df_train['relationship'] = df_train['relationship'].str[0]



df_test['relationship'] = df_test['relationship'].map(df_temp.T.to_dict('list'))

df_test['relationship'] = df_test['relationship'].str[0]
df_temp = pd.DataFrame()

df_temp['count'] = df_train.groupby(['race']).count()['target']

df_temp['sum'] = df_train.groupby(['race']).sum()['target']

df_temp['mean'] = df_train.groupby(['race']).mean()['target']



df_temp.drop(columns=['sum', 'count'], inplace=True)
df_train['race'] = df_train['race'].map(df_temp.T.to_dict('list'))

df_train['race'] = df_train['race'].str[0]



df_test['race'] = df_test['race'].map(df_temp.T.to_dict('list'))

df_test['race'] = df_test['race'].str[0]
df_temp = pd.DataFrame()

df_temp['count'] = df_train.groupby(['native-country']).count()['target']

df_temp['sum'] = df_train.groupby(['native-country']).sum()['target']

df_temp['mean'] = df_train.groupby(['native-country']).mean()['target']



df_temp.drop(columns=['sum', 'count'], inplace=True)
df_train['native-country'] = df_train['native-country'].map(df_temp.T.to_dict('list'))

df_train['native-country'] = df_train['native-country'].str[0]



df_test['native-country'] = df_test['native-country'].map(df_temp.T.to_dict('list'))

df_test['native-country'] = df_test['native-country'].str[0]
df_train['capital-loss'].unique()

to_change = {"None": 0, "Low": 1, "High": 2}

df_train['capital-loss'] = df_train['capital-loss'].map(to_change)

df_train['capital-gain'] = df_train['capital-gain'].map(to_change)



df_test['capital-loss'] = df_test['capital-loss'].map(to_change)

df_test['capital-gain'] = df_test['capital-gain'].map(to_change)
df_train['hours-per-week'].unique()

to_change = {"Part-time": 1, "Full-time": 2, "Overtime": 3, "Too-much": 4}



df_train['hours-per-week'] = df_train['hours-per-week'].map(to_change)

df_test['hours-per-week'] = df_test['hours-per-week'].map(to_change)
plt.figure(figsize=(25, 10))



sns.set(font_scale=1.2)



sns.heatmap(df_train.corr(), 

            cmap='YlGnBu',

            cbar=True, annot=True,

            square=True, fmt='.2f',

            annot_kws={'size': 10})
df_train.drop(columns='relationship', inplace=True)

df_test.drop(columns='relationship', inplace=True)
features = list(df_train.columns[:-1])

features
X_train, y_train = df_train[features], df_train['target']

X_test = df_test[features]
from sklearn.tree import DecisionTreeClassifier

clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=9, 

                                  random_state=17)



# training the tree

clf_tree.fit(X_train, y_train)



# some code to depict separating surface

predicted = clf_tree.predict(X_test)
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score



# function for fitting trees of various depths on the training data using cross-validation

def run_cross_validation_on_trees(X, y, tree_depths, cv=5, scoring='f1'):

    cv_scores_list = []

    cv_scores_std = []

    cv_scores_mean = []

    accuracy_scores = []

    for depth in tree_depths:

        tree_model = DecisionTreeClassifier(max_depth=depth)

        cv_scores = cross_val_score(tree_model, X, y, cv=cv, scoring=scoring)

        cv_scores_list.append(cv_scores)

        cv_scores_mean.append(cv_scores.mean())

        cv_scores_std.append(cv_scores.std())

        accuracy_scores.append(tree_model.fit(X, y).score(X, y))

    cv_scores_mean = np.array(cv_scores_mean)

    cv_scores_std = np.array(cv_scores_std)

    accuracy_scores = np.array(accuracy_scores)

    return cv_scores_mean, cv_scores_std, accuracy_scores

  

# function for plotting cross-validation results

def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, title):

    fig, ax = plt.subplots(1,1, figsize=(15,5))

    ax.plot(depths, cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)

    ax.fill_between(depths, cv_scores_mean-2*cv_scores_std, cv_scores_mean+2*cv_scores_std, alpha=0.2)

    ylim = plt.ylim()

    ax.plot(depths, accuracy_scores, '-*', label='train accuracy', alpha=0.9)

    ax.set_title(title, fontsize=16)

    ax.set_xlabel('Tree depth', fontsize=14)

    ax.set_ylabel('F1 score', fontsize=14)

    ax.set_ylim(ylim)

    ax.set_xticks(depths)

    ax.legend()



# fitting trees of depth 1 to 24

sm_tree_depths = range(1,25)

sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(X_train, y_train, sm_tree_depths)



# plotting accuracy

plot_cross_validation_on_trees(sm_tree_depths, sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores, 

                               'F1 score per decision tree depth on training data')
idx_max = sm_cv_scores_mean.argmax()

sm_best_tree_depth = sm_tree_depths[idx_max]

sm_best_tree_cv_score = sm_cv_scores_mean[idx_max]

sm_best_tree_cv_score_std = sm_cv_scores_std[idx_max]

print('The depth-{} tree achieves the best mean cross-validation f1 {} +/- {}% on training dataset'.format(

      sm_best_tree_depth, round(sm_best_tree_cv_score*100,5), round(sm_best_tree_cv_score_std*100, 5)))
modified = df_test.reset_index()

modified['target'] = predicted

df_submit = modified[['uid','target']]

df_submit.to_csv('/kaggle/working/submit.csv', index=False)