import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set()



import os

print(os.listdir("../input"))



from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go
train = pd.read_csv("../input/train.csv")

train.head()
test = pd.read_csv("../input/test.csv")
train.shape[0] / test.shape[0]
train.info()
test.info()
column_types = train.dtypes

column_types[column_types==np.int64]
train.isnull().sum().sum()
test.isnull().sum().sum()
sns.countplot(train.target, palette="Set2");
magic = "wheezy-copper-turtle-magic"

train_corr = train.drop(["target", magic], axis=1).corr()

test_corr = test.drop(magic, axis=1).corr()
train_corr_flat = train_corr.values.flatten()

train_corr_flat = train_corr_flat[train_corr_flat != 1]



test_corr_flat = test_corr.values.flatten()

test_corr_flat = test_corr_flat[test_corr_flat != 1]



fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.distplot(train_corr_flat, ax=ax[0], color="tomato")

sns.distplot(test_corr_flat, ax=ax[1], color="limegreen");

ax[0].set_title("Off-diagonal train corr \n distribution")

ax[1].set_title("Off-diagonal test corr \n distribution");

ax[0].set_xlabel("feature correlation value")

ax[1].set_xlabel("feature correlation value");
plt.figure(figsize=(25,25))

sns.heatmap(train_corr, vmin=-0.016, vmax=0.016, cmap="RdYlBu_r");
plt.figure(figsize=(25,25))

sns.heatmap(test_corr, vmin=-0.016, vmax=0.016, cmap="RdYlBu_r");
target_medians = train.groupby("target").median()

sorted_target_distance = np.abs(target_medians.iloc[0]-target_medians.iloc[1]).sort_values(ascending=False)
sorted_target_distance.head()
sorted_target_distance.tail()
fig, ax = plt.subplots(2,2,figsize=(20,10))

sns.distplot(train.loc[train.target==0, "wheezy-myrtle-mandrill-entropy"], color="Blue", ax=ax[0,0])

sns.distplot(train.loc[train.target==1, "wheezy-myrtle-mandrill-entropy"], color="Red", ax=ax[0,0])

sns.distplot(train.loc[train.target==0, "wheezy-copper-turtle-magic"], color="Blue", ax=ax[0,1])

sns.distplot(train.loc[train.target==1, "wheezy-copper-turtle-magic"], color="Red", ax=ax[0,1])

ax[1,0].scatter(train["wheezy-myrtle-mandrill-entropy"].values,

                train["skanky-carmine-rabbit-contributor"].values, c=train.target.values,

                cmap="coolwarm", s=1, alpha=0.5)

ax[1,0].set_xlabel("wheezy-myrtle-mandrill-entropy")

ax[1,0].set_ylabel("skanky-carmine-rabbit-contributor")

ax[1,1].scatter(train["wheezy-myrtle-mandrill-entropy"].values,

                train["wheezy-copper-turtle-magic"].values, c=train.target.values,

                cmap="coolwarm", s=1, alpha=0.5)

ax[1,1].set_xlabel("wheezy-myrtle-mandrill-entropy")

ax[1,1].set_ylabel("wheezy-copper-turtle-magic");
n_splits=3

n_repeats=3



X=train.drop(["target", "id"], axis=1).values

y=train.target.values

XTest = test.drop("id", axis=1).values



scaler = StandardScaler()

X = scaler.fit_transform(X)

XTest = scaler.transform(XTest)
X, x_val, y, y_val = train_test_split(X,y,test_size=0.2, stratify=y, random_state=2019)
skf = RepeatedStratifiedKFold(n_repeats=n_repeats,

                              n_splits=n_splits,

                              random_state=2019)



p_val = np.zeros(y_val.shape)

pTest = np.zeros(XTest.shape[0])

for train_idx, test_idx in skf.split(X,y):

    

    x_train, x_test = X[train_idx], X[test_idx]

    y_train, y_test = y[train_idx], y[test_idx]

    

    lr=LogisticRegression(penalty="l1", C=1, solver="saga")

    lr.fit(x_train, y_train)

    p_test = lr.predict_proba(x_test)[:,1]

    p_val += lr.predict_proba(x_val)[:,1]

    pTest += lr.predict_proba(XTest)[:,1]

    print(roc_auc_score(y_test, p_test))



p_val /= (n_splits*n_repeats)

pTest /= (n_splits*n_repeats)

print(roc_auc_score(y_val, p_val))
feat1 = "wheezy-myrtle-mandrill-entropy"

feat2 = "skanky-carmine-rabbit-contributor"

feat3 = "wheezy-copper-turtle-magic"
N = 10000



trace1 = go.Scatter3d(

    x=train[feat1].values[0:N], 

    y=train[feat2].values[0:N],

    z=train[feat3].values[0:N],

    mode='markers',

    marker=dict(

        color=train.target.values[0:N],

        colorscale = "Jet",

        opacity=0.3,

        size=2

    )

)



figure_data = [trace1]

layout = go.Layout(

    title = 'The turtle place',

    scene = dict(

        xaxis = dict(title=feat1),

        yaxis = dict(title=feat2),

        zaxis = dict(title=feat3),

    ),

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    ),

    showlegend=True

)



fig = go.Figure(data=figure_data, layout=layout)

py.iplot(fig, filename='simple-3d-scatter')
fig, ax = plt.subplots(5,5,figsize=(20,25))

for turtle1 in range(5):

    for turtle2 in range(5):

        my_turtle=turtle2+turtle1*5

        ax[turtle1, turtle2].scatter(train.loc[train["wheezy-copper-turtle-magic"]==my_turtle, feat1].values,

                                     train.loc[train["wheezy-copper-turtle-magic"]==my_turtle, feat2].values,

                                     c=train.loc[train["wheezy-copper-turtle-magic"]==my_turtle, "target"].values, cmap="coolwarm", s=5, alpha=0.5)

        ax[turtle1, turtle2].set_xlim([-15,15])

        ax[turtle1, turtle2].set_ylim([-15,15])
names = list(train.drop(["id", "target"], axis=1).columns.values)
first_names = []

second_names = []

third_names = []

fourth_names = []



for name in names:

    words = name.split("-")

    first_names.append(words[0])

    second_names.append(words[1])

    third_names.append(words[2])

    fourth_names.append(words[3])
print(len(first_names), len(np.unique(first_names)))

print(len(second_names), len(np.unique(second_names)))

print(len(third_names), len(np.unique(third_names)))

print(len(fourth_names), len(np.unique(fourth_names)))
feature_names = pd.DataFrame(index=train.drop(["target", "id"], axis=1).columns.values, data=first_names, columns=["kind"])

feature_names["color"] = second_names

feature_names["animal"] = third_names

feature_names["goal"] = fourth_names

feature_names.head()
plt.figure(figsize=(20,5))

sns.countplot(x="kind", data=feature_names, order=feature_names.kind.value_counts().index, palette="Greens_r")

plt.xticks(rotation=90);
plt.figure(figsize=(20,5))

sns.countplot(x="animal", data=feature_names, order=feature_names.animal.value_counts().index, palette="Oranges_r")

plt.xticks(rotation=90);
plt.figure(figsize=(20,5))

sns.countplot(x="color", data=feature_names, order=feature_names.color.value_counts().index, palette="Purples_r")

plt.xticks(rotation=90);
plt.figure(figsize=(20,5))

sns.countplot(x="goal", data=feature_names, order=feature_names.goal.value_counts().index, palette="Reds_r")

plt.xticks(rotation=90);
feature_names[feature_names.goal=="learn"]
combined = train.drop(["id", "target"], axis=1).append(test.drop("id", axis=1))

combined[combined.duplicated()]
n_subsamples_test = test.groupby(magic).size() 

n_subsamples_train = train.groupby(magic).size() 



plt.figure(figsize=(20,5))

plt.plot(n_subsamples_test.values, '.-', label="test")

plt.plot(n_subsamples_train.values, '.-', label="train")

plt.plot(n_subsamples_test.values + n_subsamples_train.values, '.-', label="total")

plt.legend();

plt.xlabel(magic)

plt.ylabel("sample count");
my_magic=0



train_subset = train.loc[train[magic]==my_magic].copy()

test_subset = test.loc[test[magic]==my_magic].copy()
n_splits=20

n_repeats=5



X=train_subset.drop(["target", "id"], axis=1).values

y=train_subset.target.values

XTest = test_subset.drop("id", axis=1).values



#scaler = StandardScaler()

#X = scaler.fit_transform(X)

#XTest = scaler.transform(XTest)



X, x_val, y, y_val = train_test_split(X,y,test_size=0.2, stratify=y, random_state=2019)
skf = RepeatedStratifiedKFold(n_repeats=n_repeats,

                              n_splits=n_splits,

                              random_state=2019)



importances = np.zeros(shape=(n_splits*n_repeats, XTest.shape[1]))

p_val = np.zeros(y_val.shape)

pTest = np.zeros(XTest.shape[0])



m=0

for train_idx, test_idx in skf.split(X,y):

    

    x_train, x_test = X[train_idx], X[test_idx]

    y_train, y_test = y[train_idx], y[test_idx]

    

    lr=LogisticRegression(penalty="l1", C=0.1, solver="liblinear", max_iter=300)

    lr.fit(x_train, y_train)

    p_test = lr.predict_proba(x_test)[:,1]

    p_val += lr.predict_proba(x_val)[:,1]

    pTest += lr.predict_proba(XTest)[:,1]

    importances[m,:] += lr.coef_[0]

    print(roc_auc_score(y_test, p_test))

    m+=1



p_val /= (n_splits*n_repeats)

pTest /= (n_splits*n_repeats)