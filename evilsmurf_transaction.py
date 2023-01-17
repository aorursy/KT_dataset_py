# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# sklearn models & tools

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import roc_auc_score

from sklearn.metrics import make_scorer

from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA

train = pd.read_csv("/kaggle/input/customer-transaction-prediction/train.csv")

test = pd.read_csv("/kaggle/input/customer-transaction-prediction/test.csv")
train.info()
test.info()
train.isna().sum().sum()
test.isna().sum().sum()
train_summary = train.describe()

train_summary
train_summary_1 = train[train['target'] == 1].describe()

train_summary_1
train_summary_0 = train[train['target'] == 0].describe()

train_summary_0
test_summary = test.describe()

test_summary
fig = plt.figure(figsize=(25,5))

fig.subplots_adjust(hspace=0.3, wspace=0.3, )

ax = fig.add_subplot(1, 2, 1)

sns.distplot(train_summary_1.iloc[1], label="train_1" )

sns.distplot(train_summary_0.iloc[1], label="train_0" )

plt.title("Distribution of mean across train")

plt.legend()

ax = fig.add_subplot(1, 2, 2)

sns.distplot(train_summary.iloc[1], label="train")

sns.distplot(test_summary.iloc[1], label="test" )

plt.title("Distribution of mean train vs test")

plt.legend();
fig = plt.figure(figsize=(25,5))

fig.subplots_adjust(hspace=0.3, wspace=0.3, )

ax = fig.add_subplot(1, 2, 1)

sns.distplot(train_summary_1.iloc[2], label="train_1" )

sns.distplot(train_summary_0.iloc[2], label="train_0" )

plt.title("Distribution of Standard deviation across train")

plt.legend()

ax = fig.add_subplot(1, 2, 2)

sns.distplot(train_summary.iloc[2], label="train")

sns.distplot(test_summary.iloc[2], label="test" )

plt.title("Distribution of Standard deviation train vs test")

plt.legend();
fig = plt.figure(figsize=(25,5))

fig.subplots_adjust(hspace=0.3, wspace=0.3, )

ax = fig.add_subplot(1, 2, 1)

sns.distplot(train_summary_1.iloc[3], label="train_1" )

sns.distplot(train_summary_0.iloc[3], label="train_0" )

plt.title("Distribution of Min values across train")

plt.legend()

ax = fig.add_subplot(1, 2, 2)

sns.distplot(train_summary.iloc[3], label="train")

sns.distplot(test_summary.iloc[3], label="test" )

plt.title("Distribution of Min values train vs test")

plt.legend();
fig = plt.figure(figsize=(25,5))

fig.subplots_adjust(hspace=0.3, wspace=0.3, )

ax = fig.add_subplot(1, 2, 1)

sns.distplot(train_summary_1.iloc[7], label="train_1" )

sns.distplot(train_summary_0.iloc[7], label="train_0" )

plt.title("Distribution of Max values across train")

plt.legend()

ax = fig.add_subplot(1, 2, 2)

sns.distplot(train_summary.iloc[7], label="train")

sns.distplot(test_summary.iloc[7], label="test" )

plt.title("Distribution of Max values in train vs test")

plt.legend();

features = train.columns.values[2:202]
train_correlations = train.drop(["target"], axis=1).corr()

train_correlations = train_correlations.values.flatten()

train_correlations = train_correlations[train_correlations != 1]



test_correlations = test.corr()

test_correlations = test_correlations.values.flatten()

test_correlations = test_correlations[test_correlations != 1]



plt.figure(figsize=(20,5))

sns.distplot(train_correlations,  label="train")

sns.distplot(test_correlations,  label="test")

plt.xlabel("Correlation values found in train (except 1)")

plt.ylabel("Density")

plt.title("Are there correlations between features?"); 

plt.legend();
sns.countplot(train.target)
train.dtypes.value_counts()
test.dtypes.value_counts()
train.select_dtypes(include=['object'])
train.nunique(axis=0).sort_values()
test.nunique(axis=0).sort_values()
def density_plot(df1,df2,feat):

    plt.figure()

    fig,ax=plt.subplots(10,10,figsize=(20,20))

    for i,f in enumerate(feat):

        plt.subplot(10,10,i+1)

        sns.set_style('whitegrid')

        sns.kdeplot(df1[f])

        sns.kdeplot(df2[f])

        plt.legend(["class_0","class_1"])

        plt.xlabel(f,fontsize=9)

    plt.show()
train.columns
density_plot(train.loc[train.target==0],train.loc[train.target==1],train.columns[2:102])
density_plot(train.loc[train.target==0],train.loc[train.target==1],train.columns[102:])
def density_plt(df,feat):

    plt.figure()

    fig,ax=plt.subplots(10,10,figsize=(25,25))

    for i,f in enumerate(feat):

        plt.subplot(10,10,i+1)

        sns.kdeplot(df[f])

        plt.xlabel(f,fontsize=9)

    plt.show()
density_plt(test,test.columns[1:101])
density_plt(test,test.columns[101:])
fig = plt.figure(figsize=(25,10))

fig.subplots_adjust(hspace=0.3, wspace=0.3, )

ax = fig.add_subplot(2,3,1)

sns.kdeplot(train.loc[train.target==0]['var_0'])

sns.kdeplot(train.loc[train.target==1]['var_0'])

plt.legend(["class_0","class_1"])



ax = fig.add_subplot(2,3,2)

sns.kdeplot(train.loc[train.target==0]['var_2'])

sns.kdeplot(train.loc[train.target==1]['var_2'])

plt.legend(["class_0","class_1"])



ax = fig.add_subplot(2,3,3)

sns.kdeplot(train.loc[train.target==0]['var_6'])

sns.kdeplot(train.loc[train.target==1]['var_6'])

plt.legend(["class_0","class_1"])



ax = fig.add_subplot(2,3,4)

sns.kdeplot(train.loc[train.target==0]['var_9'])

sns.kdeplot(train.loc[train.target==1]['var_9'])

plt.legend(["class_0","class_1"])



ax = fig.add_subplot(2,3,5)

sns.kdeplot(train.loc[train.target==0]['var_53'])

sns.kdeplot(train.loc[train.target==1]['var_53'])

plt.legend(["class_0","class_1"])



ax = fig.add_subplot(2,3,6)

sns.kdeplot(train.loc[train.target==0]['var_99'])

sns.kdeplot(train.loc[train.target==1]['var_99'])

plt.legend(["class_0","class_1"])
train["Id"] = train.index.values

original_trainid = train.ID_code.values



train.drop("ID_code", axis=1, inplace=True)
train.head()
parameters = {'min_samples_leaf': [20, 25]}

forest = RandomForestClassifier(max_depth=15, n_estimators=15)

grid = GridSearchCV(forest, parameters, cv=3, n_jobs=-1, verbose=2, scoring=make_scorer(roc_auc_score))
grid.fit(train.drop(["target",], axis=1).values, train.target.values)
grid.best_score_

grid.best_params_
n_top = 20

importances = grid.best_estimator_.feature_importances_

idx = np.argsort(importances)[::-1][0:n_top]

feature_names = train.drop("target", axis=1).columns.values



plt.figure(figsize=(20,5))

sns.barplot(x=feature_names[idx], y=importances[idx]);

plt.title("What are the top important features to start with?");
fig, ax = plt.subplots(n_top,2,figsize=(20,5*n_top))



for n in range(n_top):

    sns.distplot(train.loc[train.target==0, feature_names[idx][n]], ax=ax[n,0],  norm_hist=True)

    sns.distplot(train.loc[train.target==1, feature_names[idx][n]], ax=ax[n,0],  norm_hist=True)

    sns.distplot(test.loc[:, feature_names[idx][n]], ax=ax[n,1], norm_hist=True)

    ax[n,0].set_title("Train {}".format(feature_names[idx][n]))

    ax[n,1].set_title("Test {}".format(feature_names[idx][n]))

    ax[n,0].set_xlabel("")

    ax[n,1].set_xlabel("")
top = train.loc[:, feature_names[idx]]

top.describe()
top = top.join(train.target)

# sns.pairplot(top, hue="target")
def woe(X, y):

    tmp = pd.DataFrame()

    tmp["variable"] = X

    tmp["target"] = y

    var_counts = tmp.groupby("variable")["target"].count()

    var_events = tmp.groupby("variable")["target"].sum()

    var_nonevents = var_counts - var_events

    tmp["var_counts"] = tmp.variable.map(var_counts).astype("float64")

    tmp["var_events"] = tmp.variable.map(var_events).astype("float64")

    tmp["var_nonevents"] = tmp.variable.map(var_nonevents).astype("float64")

    events = sum(tmp["target"] == 1)

    nonevents = sum(tmp["target"] == 0)

    tmp["woe"] = np.log(((tmp["var_nonevents"])/nonevents)/((tmp["var_events"])/events))

    tmp["woe"] = tmp["woe"].replace(np.inf, 0).replace(-np.inf, 0)

    tmp["iv"] = (tmp["var_nonevents"]/nonevents - tmp["var_events"]/events) * tmp["woe"]

    iv = tmp.groupby("variable")["iv"].last().sum()

    return tmp["woe"], tmp["iv"], iv
iv_values = []

feats = ["var_{}".format(i) for i in range(200)]

y = train["target"]

for f in feats:

    X = pd.qcut(train[f], 10, duplicates='drop')

    _, _, iv = woe(X, y)

    iv_values.append(iv)

    

iv_inds = np.argsort(iv_values)[::-1][:50]

iv_values = np.array(iv_values)[iv_inds]

feats = np.array(feats)[iv_inds]
plt.figure(figsize=(10, 20))

sns.barplot(y=feats, x=iv_values, orient='h')

plt.show()
imp = {}

imp["feature_name"] = feats

imp["iv_value"] = iv_values

imp_features = pd.DataFrame(data=imp)
imp_features.sort_values(by = ["iv_value"],ascending=False)
top_features = imp_features["feature_name"].iloc[0:20].values
top_features = top_features.tolist()

top_features
# adding feature from top to top_features

for i in top.columns:

    top_features.append(i)
#removing duplicates

top_features = list(set(top_features))

top_features
new_top = train[top_features]

new_top.head()
new_top.shape
encoder = LabelEncoder()

for your_feature in top_features:

    if(your_feature == "target"):

        pass

    elif(your_feature == "ID"):

        pass

    else:

        train[your_feature + "_qbinned"] = pd.qcut(

            train.loc[:, your_feature].values,

            q=10,

            duplicates='drop',

            labels=False

        )

        train[your_feature + "_qbinned"] = encoder.fit_transform(

            train[your_feature + "_qbinned"].values.reshape(-1, 1)

        )





        train[your_feature + "_rounded"] = np.round(train.loc[:, your_feature].values)

        train[your_feature + "_rounded_10"] = np.round(10*train.loc[:, your_feature].values)

        train[your_feature + "_rounded_100"] = np.round(100*train.loc[:, your_feature].values)
encoder = LabelEncoder()

for your_feature in top_features:

    if(your_feature == "target"):

        pass

    else:

        test[your_feature + "_qbinned"] = pd.qcut(

            test.loc[:, your_feature].values,

            q=10,

            duplicates='drop',

            labels=False

        )

        test[your_feature + "_qbinned"] = encoder.fit_transform(

            test[your_feature + "_qbinned"].values.reshape(-1, 1)

        )





        test[your_feature + "_rounded"] = np.round(test.loc[:, your_feature].values)

        test[your_feature + "_rounded_10"] = np.round(10*test.loc[:, your_feature].values)

        test[your_feature + "_rounded_100"] = np.round(100*test.loc[:, your_feature].values)
train.head()
# Deriving the New Features for the training dataset

mean_r=train.iloc[:,2:].mean(axis=1).values

min_r=train.iloc[:,2:].min(axis=1).values

max_r=train.iloc[:,2:].max(axis=1).values

std_r=train.iloc[:,2:].std(axis=1).values

skew_r=train.iloc[:,2:].skew(axis=1).values

kurto_r=train.iloc[:,2:].kurtosis(axis=1).values

median_r=train.iloc[:,2:].median(axis=1).values
## Adding the new features to the training dataset

train["mean_r"]=mean_r

train["min_r"]=min_r

train["max_r"]=max_r

train["std_r"]=std_r

train["skew_r"]=skew_r

train["kurto_r"]=kurto_r

train["median_r"]=median_r
# Deriving the New Features for the testing dataset

mean_r=test.iloc[:,2:].mean(axis=1).values

min_r=test.iloc[:,2:].min(axis=1).values

max_r=test.iloc[:,2:].max(axis=1).values

std_r=test.iloc[:,2:].std(axis=1).values

skew_r=test.iloc[:,2:].skew(axis=1).values

kurto_r=test.iloc[:,2:].kurtosis(axis=1).values

median_r=test.iloc[:,2:].median(axis=1).values



## Adding the new features to the testing dataset

test["mean_r"]=mean_r

test["min_r"]=min_r

test["max_r"]=max_r

test["std_r"]=std_r

test["skew_r"]=skew_r

test["kurto_r"]=kurto_r

test["median_r"]=median_r
test.shape
train.shape
for i in train.columns.values:

    if i not in test.columns.values:

        print( i)
iv_values = []

feats = train.columns.values

y = train["target"]

for f in feats:

    X = pd.qcut(train[f], 10, duplicates='drop')

    _, _, iv = woe(X, y)

    iv_values.append(iv)

    

iv_inds = np.argsort(iv_values)[::-1][:50]

iv_values = np.array(iv_values)[iv_inds]

feats = np.array(feats)[iv_inds]
plt.figure(figsize=(10, 20))

sns.barplot(y=feats, x=iv_values, orient='h')

plt.show()
train.to_csv("new_train.csv")

test.to_csv("new_test.csv")