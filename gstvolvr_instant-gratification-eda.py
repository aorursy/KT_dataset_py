import numpy as np

import pandas as pd

import warnings

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc



import matplotlib.pyplot as plt



warnings.filterwarnings(action='ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



features = train.drop(["id", "target"], axis=1)

labels = train["target"]



x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

print("train size: ", train.shape)

print("test size: ", test.shape)
columns = x_train.columns

means = []



for i in columns:

    means.append((i, x_train[i].mean()))



s_means = sorted(means, key=lambda x: x[1])



print("min: ", s_means[0])

print("max: ", s_means[-1])
plt.style.use('seaborn')

%matplotlib inline



def performance(pred, true, name="", printed=True):

    # ROC curve

    fpr, tpr, _ = roc_curve(pred, true)

    area = auc(fpr, tpr)

    

    if printed:

        print("AUC for {0} is {1:.4f}".format(name, area))

    return (name, area)
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from xgboost import XGBClassifier

from matplotlib.colors import ListedColormap

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



num_sets = x_train['wheezy-copper-turtle-magic'].max() + 1



names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",

         "Decision Tree", "Random Forest",

         "Naive Bayes", "QDA"]



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    GaussianNB(),

    QuadraticDiscriminantAnalysis()]



x_train_2 = x_train.drop('wheezy-copper-turtle-magic', axis=1)

x_test_2 = x_test.drop('wheezy-copper-turtle-magic', axis=1)



for name, clf in zip(names, classifiers):

    total_pred = []

    total_test = []

    total_acc = []

    

    for i in range(num_sets):

        train_indices = x_train[x_train['wheezy-copper-turtle-magic'] == i].index

        test_indices = x_test[x_test['wheezy-copper-turtle-magic'] == i].index



        vt = VarianceThreshold(threshold=2.0).fit(x_train_2.loc[train_indices])



        slim_x_train = vt.transform(x_train_2.loc[train_indices])

        slim_x_test = vt.transform(x_test_2.loc[test_indices])



        clf.fit(slim_x_train, y_train.loc[train_indices])

        pred = clf.predict(slim_x_test)

        total_pred += list(pred.ravel())

        total_test += list(y_test.loc[test_indices].ravel())

    performance(total_pred, total_test, name)
clf = QuadraticDiscriminantAnalysis()



test_features = test.drop(["id", "wheezy-copper-turtle-magic"], axis=1)

thresholds_auc = []

for threshold in np.arange(0.0, 3.1, 0.1):

    total_pred = []

    total_test = []

    total_auc = []

    total_same_as = []

    

    for i in range(num_sets):

        train_indices = x_train[x_train['wheezy-copper-turtle-magic'] == i].index

        test_indices = x_test[x_test['wheezy-copper-turtle-magic'] == i].index



        vt_train = VarianceThreshold(threshold=threshold).fit(x_train_2.loc[train_indices])

        

        slim_x_train = vt_train.transform(x_train_2.loc[train_indices])

        slim_x_test = vt_train.transform(x_test_2.loc[test_indices])

        

        clf.fit(slim_x_train, y_train.loc[train_indices])

        pred = clf.predict(slim_x_test)        

        

        total_pred += list(pred.ravel())

        total_test += list(y_test.loc[test_indices].ravel())

        total_auc += [performance(pred.ravel(), y_test.loc[test_indices], i, False)]

    

    t_auc = performance(total_pred, total_test, "QDA @ {0:.2f} threshold".format(threshold), False)

    thresholds_auc.append(t_auc)
x = np.arange(0.0, 3.1, 0.1)

y = list(map(lambda x: x[1], thresholds_auc))

plt.title("AUC / variance threshold")

plt.scatter(x, y)

plt.xlabel("threshold");

plt.ylabel("AUC");
clf = QuadraticDiscriminantAnalysis()



test_features = test.drop(["id", "wheezy-copper-turtle-magic"], axis=1)

train_features = features.drop(["wheezy-copper-turtle-magic"], axis=1)

feature_indices = []

sim_score = []



for i in range(num_sets):

    train_indices = features[features['wheezy-copper-turtle-magic'] == i].index

    test_indices = test[test['wheezy-copper-turtle-magic'] == i].index

    

    # keep track of indices for each set

    feature_indices.append(np.argwhere(vt_train.get_support() == True))



    # are the same features important in the test and train sets?

    vt_train = VarianceThreshold(threshold=1.5).fit(train_features.loc[train_indices])

    vt_test = VarianceThreshold(threshold=1.5).fit(test_features.loc[test_indices])

    

    sim = np.sum(vt_train.get_support() == vt_test.get_support()) / train_features.shape[1]

    sim_score.append(sim)
prev = feature_indices[0]

same = []

for idx in feature_indices[1:]:

    same.append(np.all(prev == idx))



print("% of sets with same features as previous: {0:.3f}".format((np.sum(same) * 100.0) / len(same)))
plt.scatter(range(num_sets), sim_score);

plt.xlabel("wheezy-copper-turtle-magic")

plt.ylabel("similarity between test and train features");
columns = train_features.columns



total_pred = []

total_test = []

total_delta = 0

for i in range(num_sets):

    train_indices = x_train[x_train['wheezy-copper-turtle-magic'] == i].index

    test_indices = x_test[x_test['wheezy-copper-turtle-magic'] == i].index



    vt_train = VarianceThreshold(threshold=1.5).fit(x_train_2.loc[train_indices])



    slim_x_train = vt_train.transform(x_train_2.loc[train_indices])

    slim_x_test = vt_train.transform(x_test_2.loc[test_indices])



    # plot correlations for set 0

    corr = pd.DataFrame(slim_x_train).corr()

    corr_sum_per_colum = corr[np.abs(corr) > 0.31].fillna(0).sum()

    high_corr_idx = np.argwhere((corr_sum_per_colum > 1.0) == True)

    

    clf.fit(slim_x_train, y_train.loc[train_indices])

    pred = clf.predict(slim_x_test)

    

    if len(high_corr_idx) > 0:

        print("high correlation columns for block {0}: ".format(i), list(columns[high_corr_idx].ravel()))

        # these usually occur in pairs, so simply drop the first redundant feature

        n_features = slim_x_train.shape[1]

        slim_x_train = np.delete(slim_x_train, high_corr_idx[0], axis=1)

        slim_x_test = np.delete(slim_x_test, high_corr_idx[0], axis=1)

        

        _, s1 = performance(pred, y_test.loc[test_indices].ravel(), name="", printed=False)

        clf.fit(slim_x_train, y_train.loc[train_indices])

        pred2 = clf.predict(slim_x_test)

        _, s2 = performance(pred2, y_test.loc[test_indices].ravel(), name="", printed=False)

        delta = s2 - s1

        

        if delta > 0:

            pred = pred2

            total_delta += delta

        

    total_pred += list(pred.ravel())

    total_test += list(y_test.loc[test_indices].ravel())

    

print("\ntotal delta after removing highly correlated features: {0: .2f}".format(total_delta))

# performance(total_pred, total_test, "QDA @ {0} threshold".format(names[j]));
from sklearn.preprocessing import normalize, minmax_scale, scale



clf = QuadraticDiscriminantAnalysis()



test_features = test.drop(["id", "wheezy-copper-turtle-magic"], axis=1)

thresholds_auc = []

names = [ "min max scale", "scale"]

processes = [ 

    minmax_scale,

    scale

]

for j, process in enumerate(processes):

    total_pred = []

    total_test = []

    total_auc = []

    total_same_as = []

    

    for i in range(num_sets):

        train_indices = x_train[x_train['wheezy-copper-turtle-magic'] == i].index

        test_indices = x_test[x_test['wheezy-copper-turtle-magic'] == i].index



        vt_train = VarianceThreshold(threshold=2.0).fit(x_train_2.loc[train_indices])

        

        slim_x_train = process(vt_train.transform(x_train_2.loc[train_indices]))

        slim_x_test = process(vt_train.transform(x_test_2.loc[test_indices]))

        

        clf.fit(slim_x_train, y_train.loc[train_indices])

        pred = clf.predict(slim_x_test)        

        

        total_pred += list(pred.ravel())

        total_test += list(y_test.loc[test_indices].ravel())

        total_auc += [performance(pred.ravel(), y_test.loc[test_indices], i, False)]

    

    t_auc = performance(total_pred, total_test, "QDA @ {0} threshold".format(names[j]))

    thresholds_auc.append(t_auc)
import numpy as np

import pandas as pd



from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.mixture import GaussianMixture

# from sklearn.mixture import GMM



# read in and split data

# train = pd.read_csv('../input/train.csv')

# test = pd.read_csv('../input/test.csv')



drop = ["id", "target", "wheezy-copper-turtle-magic"]

feature_cols = [ c for c in train.columns if c not in drop ]



skf = StratifiedKFold(n_splits=11, random_state=42)

clf = QuadraticDiscriminantAnalysis(0.111)

clf = GaussianMixture(n_components=2)



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



    # print(roc_auc_score(train_data['target'], train_preds[train_data.index]))

print(roc_auc_score(train['target'], train_preds))