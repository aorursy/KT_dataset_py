import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score



# read in and split data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
drop = ["id", "target", "wheezy-copper-turtle-magic"]

feature_cols = [ c for c in train.columns if c not in drop ]



skf = StratifiedKFold(n_splits=11, random_state=42)

clf = QuadraticDiscriminantAnalysis(1.0, store_covariances=False)



# prep result dataframe

sub = test[["id"]].copy()

sub["target"] = None

num_sets = train['wheezy-copper-turtle-magic'].max() + 1



train_preds = np.zeros(train.shape[0])

preds = np.zeros(test.shape[0])



for i in range(num_sets):

    train_data = train[train['wheezy-copper-turtle-magic'] == i]

    test_data = test[test['wheezy-copper-turtle-magic'] == i]

    

    data = pd.concat([train_data[feature_cols], test_data[feature_cols]])



    vt = VarianceThreshold(threshold=1.5).fit(data)

    

    slim_train_features = vt.transform(train_data[feature_cols])

    slim_test_features = vt.transform(test_data[feature_cols])



    for train_index, test_index in skf.split(slim_train_features, train_data['target']):

        clf.fit(slim_train_features[train_index, :], train_data.iloc[train_index]['target'])

        train_preds[train_data.index[test_index]] += clf.predict_proba(slim_train_features[test_index, :])[:, 1]

        

        preds[test_data.index] += clf.predict_proba(slim_test_features)[:, 1] / skf.n_splits

        

print("current AUC: ", roc_auc_score(train['target'], train_preds))

sub["target"] = preds
sub[["id", "target"]].to_csv("qda_submission_v17.csv", index=False)
scores = []

straglers = []

for i in range(num_sets):

    idx = train[train['wheezy-copper-turtle-magic'] == i].index

    s = roc_auc_score(train.loc[idx, 'target'], train_preds[idx])

    

    if s < 0.95:

        print(i, s)

        straglers.append(i)

        

    scores.append(s)

print("total straglers: ", len(straglers))
import matplotlib.gridspec as gridspec

import matplotlib



j = 197

colors = ["#fbd808", "#f9530b"]



train_features = train[train['wheezy-copper-turtle-magic'] == j].drop(['target', 'id', 'wheezy-copper-turtle-magic'], axis=1)

train_label = train[train['wheezy-copper-turtle-magic'] == j]['target']

vt = VarianceThreshold(threshold=1.5)

train_features_slim = vt.fit_transform(train_features)



rows = int(train_features_slim.shape[1] / 4)

cols = 4



fig, ax = plt.subplots(nrows=rows, ncols=cols, squeeze=False, figsize=(24, 36))

fig.subplots_adjust(hspace=0.5)



c = 0

for row in ax:

    for col in row:

        c += 1

        col.scatter(np.arange(train_features_slim.shape[0]), train_features_slim[:, c], c=train_label, cmap=matplotlib.colors.ListedColormap(colors));

        col.title.set_text("feature: {0}".format(c))  
# for c in np.arange(train_features_slim.shape[1]):

import itertools

c_features = train_features_slim.shape[1]

for n, m in list(itertools.combinations(np.arange(train_features_slim.shape[1]), 2))[c_features*3:c_features*4]:

    plt.figure()

    plt.scatter(train_features_slim[:, n], train_features_slim[:, m], c=train_label, cmap=matplotlib.colors.ListedColormap(colors));

    plt.title("feature: {0} v {1}".format(n, m))

    plt.plot()
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(20, 15))

ax = fig.add_subplot(111, projection='3d')



combos = itertools.combinations(np.arange(0, train_features_slim.shape[1]), 3)

for xi, yi, zi in list(combos)[0:10]:

    if xi != yi or zi != yi or xi != zi:

        ax.scatter(train_features_slim[:, xi], train_features_slim[:, yi], train_features_slim[:, zi], c=train_label, cmap=matplotlib.colors.ListedColormap(colors));

ax.set_xlabel('X axis')

ax.set_ylabel('Y axis')

ax.set_zlabel('Z axis');
x = train_features_slim[:, 1] # first feature

y = train_features_slim[:, 2] # second feature



d = 0.1

t_x1, t_x4 = np.quantile(x, [d, 1-d])

t_y1, t_y4 = np.quantile(y, [d, 1-d])

idx = np.where(

    ((x < t_x1) & (y < t_y1)) | ((x > t_x4) & (y > t_y4)) 

)



fig = plt.figure(figsize=(20, 15))

ax = fig.add_subplot(111, projection='3d')



for j in range(2, train_features_slim.shape[1]):

    z = train_features_slim[:, j]

    x_slim = x[idx]

    y_slim = y[idx]

    z_slim = z[idx]



    ax.scatter(x_slim, y_slim, z_slim, c=train_label.iloc[idx], cmap=matplotlib.colors.ListedColormap(colors));

ax.set_xlabel('X axis')

ax.set_ylabel('Y axis')

ax.set_zlabel('Z axis');
from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



# use LDA for outlier classification

clf = QuadraticDiscriminantAnalysis(0.5, store_covariances=False)



skf = StratifiedKFold(n_splits=3, random_state=42)



def cutoffs(x, y):

    d = 0.15

    t_x1, t_x4 = np.quantile(x, [d, 1-d])

    t_y1, t_y4 = np.quantile(y, [d, 1-d])

    idx = np.where(

        ((x < t_x1) & (y < t_y1)) | ((x > t_x4) & (y > t_y4)) 

    )

    idx_other = np.where(

        ((x >= t_x1) & (y >= t_y1)) | ((x <= t_x4) & (y <= t_y4))

    )

    return idx, idx_other



edge_train_preds = np.zeros(train_features_slim.shape[0])

edge_test_preds = np.zeros(train_features_slim.shape[0])



preds = np.zeros(test.shape[0])



for j in straglers:

# for j in range(num_sets):

    scores = []

    

    test_data = test[test['wheezy-copper-turtle-magic'] == j]



    train_features = train[train['wheezy-copper-turtle-magic'] == j].drop(['target', 'id', 'wheezy-copper-turtle-magic'], axis=1)

    train_label = train[train['wheezy-copper-turtle-magic'] == j]['target']

    train_preds_j = np.zeros(train_features.shape[0])

    train_times_j = np.zeros(train_features.shape[0]) # keeps track of the number of times an index shows up in sample



    vt = VarianceThreshold(threshold=1.5)

    train_features_slim = vt.fit_transform(train_features)

    test_features_slim = vt.transform(test_data[feature_cols])



    for f in range(train_features_slim.shape[1] - 1):

        idx_train, idx_train_other = cutoffs(train_features_slim[:, 0], train_features_slim[:, 1])

        idx_test_other, idx_test = cutoffs(test_features_slim[:, 0], test_features_slim[:, 1])



        for train_index, test_index in skf.split(train_features_slim[idx_train_other], train_label.iloc[idx_train_other]):

        

            x_train = train_features_slim[idx_train_other[0][train_index]]

            x_test = train_features_slim[idx_train_other[0][test_index]]

            y_train = train_label.iloc[idx_train_other].iloc[train_index]

            y_test = train_label.iloc[idx_train_other].iloc[test_index]

        

            if len(set(y_train)) > 1:

                clf.fit(x_train, y_train)

                probs = clf.predict_proba(x_test)[:, 1]

                scores.append(roc_auc_score(y_test, probs))

            

            train_times_j[idx_train_other[0][test_index]] += 1 

            train_preds_j[idx_train_other[0][test_index]] += probs 

            

            preds[test_data.index[idx_test[0]]] += clf.predict_proba(test_features_slim[idx_test[0]])[:, 1] / (skf.n_splits * (train_features_slim.shape[1] - 1))
assert len(preds) == len(sub)



sub2 = sub.copy()

idx = np.where(preds != 0)[0]

# for how many did we predict the same?

print(preds[idx].shape)

print(np.where(np.rint(sub["target"].loc[idx]) == np.rint(preds[idx]))[0].shape)



# what's the different in previous sums to current?

print(np.sum(preds[idx]))

print(np.sum(sub["target"].loc[idx]))



# what's the total summation impact on the predictions?

sub2.loc[idx, "target"] = (sub2.loc[idx, "target"] + preds[idx]) / 2

print(sub2["target"].sum())

print(sub["target"].sum())
sub2[["id", "target"]].to_csv("qda_submission_v16.csv", index=False)