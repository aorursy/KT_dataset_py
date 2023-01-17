# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from datetime import datetime

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.head()
df['Churn'].value_counts(normalize = True)
#df.describe

#df.sum

df.isna().sum()
df.describe()
df.shape
sns.countplot(df['gender'])
sns.catplot(y="Churn", kind="count", data=df, height=2.6, aspect=2.5, orient='h')
sns.countplot(df['Churn'])
#MonthlyCarges and Contract

avg=(

    df.

     groupby(['Contract', 'Churn'])['MonthlyCharges'].

     sum().

    reset_index().

     sort_values(by = 'MonthlyCharges',

                ascending = False))

table = avg.pivot(index='Contract',

                 columns='Churn',

                 values = 'MonthlyCharges')

table
x = table.index

width = 0.35

for col in table.columns:

    plt.bar(x,table[col], width, label=col)

    plt.title('Churn per contract type')

    #plt.xticks(rotation=90)

plt.legend()

plt.show()
#MonthlyCarges and Gender

#, 'Churn'

avgg=(

    df.

     groupby(['gender', 'Churn'])['MonthlyCharges'].

    sum().#.agg({'MonthlyCharges':np.mean}).

    reset_index().

     sort_values(by = 'MonthlyCharges',

                ascending = False))

avgg

tableg = avgg.pivot(index='gender',

                 columns='Churn',

                 values = 'MonthlyCharges')

tableg
xg = tableg.index

for col in tableg.columns:

    plt.bar(xg,tableg[col], label=col)

    plt.title('Churn per gender')

    #plt.xticks(rotation=90)

plt.legend()

plt.show()
sns.distplot(df['tenure'])

sns.distplot(df['MonthlyCharges'])
for col in df.columns:

    print('{} unique element: {}'.format(col,df[col].nunique()))
def barplot_percentages(feature, orient='v', axis_name="percentage of customers"):

    ratios = pd.DataFrame()

    g = df.groupby(feature)["Churn"].value_counts().to_frame()

    g = g.rename({"Churn": axis_name}, axis=1).reset_index()

    g[axis_name] = g[axis_name]/len(df)

    if orient == 'v':

        ax = sns.barplot(x=feature, y= axis_name, hue='Churn', data=g, orient=orient)

        ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])

    else:

        ax = sns.barplot(x= axis_name, y=feature, hue='Churn', data=g, orient=orient)

        ax.set_xticklabels(['{:,.0%}'.format(x) for x in ax.get_xticks()])

    ax.plot()

barplot_percentages("SeniorCitizen")
df['churn_rate'] = df['Churn'].replace("No", 0).replace("Yes", 1)

g = sns.FacetGrid(df, col="SeniorCitizen", height=4, aspect=.9)

ax = g.map(sns.barplot, "gender", "churn_rate", palette = "Blues_d", order= ['Female', 'Male'])
df2 = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df2.head()
df.isna().any()
df['churn_rate'].value_counts()
categorylist = ['gender',

'Partner',

'Dependents',

'PhoneService',

'MultipleLines',

'InternetService',

'OnlineSecurity',

'OnlineBackup',

'DeviceProtection',

'TechSupport',

'StreamingTV',

'StreamingMovies',

'Contract',

'PaperlessBilling',

'PaymentMethod']

data = pd.get_dummies(df,columns=categorylist)

data.shape
data.head()
data.sample(frac=.1).plot('tenure','MonthlyCharges', subplots=True, kind='scatter')
# Installing more packages

data = data._get_numeric_data()
data.head()
corr = data.corr()

corr.style.background_gradient()
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.1)

train.shape, test.shape
test.head()
train['churn_rate'].value_counts(normalize = True)
test['churn_rate'].value_counts(normalize = True)
x_test = test.drop(['churn_rate'], axis=1)

y_test = test['churn_rate']
x_test.head()
y_test.head()
x = train.drop(['churn_rate'], axis=1)

y = train['churn_rate']
forest = ExtraTreesClassifier(n_estimators=100,

                              random_state=0)



forest.fit(x, y)

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(x.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest



plt.figure(figsize=(8, 6))



plt.title("Feature importances")

plt.bar(range(x.shape[1]), importances[indices],

       color="b", yerr=std[indices], align="center")

plt.xticks(range(x.shape[1]), indices)

plt.xlim([-1, x.shape[1]])
imp_features = []

for i in indices:

    imp_features.append("var_"+str(i))
imp_features[:20]
# Plot the TOP 20 feature importances of the forest



plt.figure(figsize=(8, 6))



plt.title("Feature importances")

plt.bar(range(x.shape[1]), importances[indices],

       color="b", yerr=std[indices], align="center")

plt.xticks(range(x.shape[1]), indices)

plt.xlim([-1, 19])
std
indices
feature_rank = list()

for i in range(1,len(indices)):

    feature_rank.append(x.columns[i])



print(feature_rank)
x.columns
drop = ['gender_Male', 'Partner_No', 'Dependents_No', 'PhoneService_No', 'PaperlessBilling_No'] 

x = x.drop(drop,axis=1)
x.head()
forest = ExtraTreesClassifier(n_estimators=100,

                              random_state=0)



forest.fit(x, y)

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(x.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest



plt.figure(figsize=(8, 6))



plt.title("Feature importances")

plt.bar(range(x.shape[1]), importances[indices],

       color="b", yerr=std[indices], align="center")

plt.xticks(range(x.shape[1]), indices)

plt.xlim([-1, x.shape[1]])
feature_rank = list()

for i in range(1,len(indices)):

    feature_rank.append(x.columns[i])



print(feature_rank)
# Plot the TOP 20 feature importances of the forest



plt.figure(figsize=(8, 6))



plt.title("Feature importances")

plt.bar(range(x.shape[1]), importances[indices], 

       color="r", yerr=std[indices], align="center")

plt.xticks(range(x.shape[1]), feature_rank, rotation=90)

plt.xlim([-1, 19])
# Plot the TOP 20 feature importances of the forest



plt.figure(figsize=(8, 6))



plt.title("Feature importances")

plt.bar(range(x.shape[1]), importances[indices], 

       color="r", yerr=std[indices], align="center")

plt.xticks(range(x.shape[1]), feature_rank, rotation=90)

#plt.xlim([-1, 19])
train, test = train_test_split(data, test_size = 0.1)

train.shape, test.shape
x_train = train.drop(['churn_rate'], axis=1)

y_train = train['churn_rate']
x_train.shape, y_train.shape, x_test.shape, y_test.shape
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(x_train, y_train)
# Predicting the Test set results

y_pred = classifier.predict(x_test)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
#Confusion Matrix Graph

from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm)

plt.show()
#Making the Confusion Matrix

# confusion_matrix is a function

from sklearn.metrics import confusion_matrix

from sklearn.metrics.classification import cohen_kappa_score

from sklearn.metrics import precision_recall_fscore_support

cm = confusion_matrix(y_test,y_pred)

k_stat = cohen_kappa_score(y_test,y_pred)

score = precision_recall_fscore_support(y_test,y_pred)
k_stat
score
roc_auc_score(y_test, y_pred)
#Applying k-fold crossvalidation

from sklearn.model_selection import cross_val_score, cross_validate

accuracies = cross_val_score(estimator = classifier,X = x_train, y = y_train, cv = 10)

stats = cross_validate(estimator = classifier,X = x_train, y = y_train, cv = 10)

mean_ac= accuracies.mean()

std_ac= accuracies.std()
print(stats)

print(accuracies)

print(mean_ac)

print(std_ac)
sns.countplot(y_pred)

sns.countplot(y_train)
import collections

collections.Counter(y_pred),collections.Counter(y_train)
#Performance

#Applying Grid Search to find the best model and best parameters

from sklearn.model_selection import GridSearchCV

# Include in the dictionaries the parameters we want to optimize

parameters = [ {'criterion':['entropy'],'min_impurity_decrease':[0.01,0.001,0.005,0.05],'min_impurity_split':[0.05,0.01,0.001,0.0001,0.0000001]}]

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters,scoring='accuracy',cv=10,n_jobs=-1)

grid_search = grid_search.fit(x_train,y_train)

best_accuracy = grid_search.best_score_

best_parmeters = grid_search.best_params_
grid_search
best_accuracy
best_parmeters