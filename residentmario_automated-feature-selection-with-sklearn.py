import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
chars = pd.concat([pd.read_csv("../input/ahcd1/csvTestImages 3360x1024.csv", header=None), pd.read_csv("../input/ahcd1/csvTrainImages 13440x1024.csv", header=None)])
chars = chars.assign(label=pd.concat([pd.read_csv("../input/ahcd1/csvTestLabel 3360x1.csv", header=None), pd.read_csv("../input/ahcd1/csvTrainLabel 13440x1.csv", header=None)]).values)

chars_X = chars.iloc[:, :-1]
chars_y = chars.iloc[:, -1]
sns.heatmap(chars.iloc[100, :-1].values.reshape((32, 32)).T, cmap='Greys', cbar=False)
plt.axis('off')
pass
def sp(p, ax):
    char_pixel_means = chars_X.mean(axis=0).values
    idx_selected = np.where(char_pixel_means > np.percentile(char_pixel_means, p))[0]
    arr = np.isin(np.array(list(range(32*32))), idx_selected).reshape((32, 32)).astype(int) == False
    sns.heatmap(chars_X.mean(axis=0).values.reshape(32, 32), ax=ax, cbar=False, mask=arr)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("{0}% Selected".format(100-p))

fig, axarr = plt.subplots(1, 6, figsize=(12, 3))
sns.heatmap(chars_X.mean(axis=0).values.reshape(32, 32), ax=axarr[0], cbar=False)
axarr[0].axis('off')
axarr[0].set_aspect('equal')
axarr[0].set_title("100% Selected")
plt.suptitle("Feature Curation by Mean Pixel Value", fontsize=16)

sp(50, axarr[1])
sp(60, axarr[2])
sp(70, axarr[3])
sp(80, axarr[4])
sp(90, axarr[5])
from sklearn.feature_selection import GenericUnivariateSelect

trans = GenericUnivariateSelect(score_func=lambda X, y: X.mean(axis=0), mode='percentile', param=50)
chars_X_trans = trans.fit_transform(chars_X, chars_y)
print("We started with {0} pixels but retained only {1} of them!".format(chars_X.shape[1], chars_X_trans.shape[1]))
pd.set_option('max_columns', None)
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler = (kepler
     .drop(['rowid', 'kepid'], axis='columns')
     .rename(columns={'koi_disposition': 'disposition', 'koi_pdisposition': 'predisposition'})
     .pipe(lambda df: df.assign(disposition=(df.disposition == 'CONFIRMED').astype(int), predisposition=(df.predisposition == 'CANDIDATE').astype(int)))
     .pipe(lambda df: df.loc[:, df.dtypes.values != np.dtype('O')])  # drop str columns
     .pipe(lambda df: df.loc[:, (df.isnull().sum(axis='rows') < 500).where(lambda v: v).dropna().index.values])  # drop columns with greater than 500 null values
     .dropna()
)

kepler_X = kepler.iloc[:, 1:]
kepler_y = kepler.iloc[:, 0]

kepler.head()
from sklearn.feature_selection import mutual_info_classif
kepler_mutual_information = mutual_info_classif(kepler_X, kepler_y)

plt.subplots(1, figsize=(26, 1))
sns.heatmap(kepler_mutual_information[:, np.newaxis].T, cmap='Blues', cbar=False, linewidths=1, annot=True)
plt.yticks([], [])
plt.gca().set_xticklabels(kepler.columns[1:], rotation=45, ha='right', fontsize=12)
plt.suptitle("Kepler Variable Importance (mutual_info_classif)", fontsize=18, y=1.2)
plt.gcf().subplots_adjust(wspace=0.2)
pass
trans = GenericUnivariateSelect(score_func=mutual_info_classif, mode='percentile', param=50)
kepler_X_trans = trans.fit_transform(kepler_X, kepler_y)
print("We started with {0} features but retained only {1} of them!".format(kepler_X.shape[1] - 1, kepler_X_trans.shape[1]))
columns_retained_Select = kepler.iloc[:, 1:].columns[trans.get_support()].values
pd.DataFrame(kepler_X_trans, columns=columns_retained_Select).head()
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(kepler_X, kepler_y)

pd.Series(clf.feature_importances_, index=kepler.columns[1:]).plot.bar(color='steelblue', figsize=(12, 6))
from sklearn.feature_selection import SelectFromModel

clf = DecisionTreeClassifier()
trans = SelectFromModel(clf, threshold='median')
kepler_X_trans = trans.fit_transform(kepler_X, kepler_y)
print("We started with {0} features but retained only {1} of them!".format(kepler_X.shape[1] - 1, kepler_X_trans.shape[1]))
columns_retained_FromMode = kepler.iloc[:, 1:].columns[trans.get_support()].values
from sklearn.feature_selection import RFE

clf = DecisionTreeClassifier()
trans = RFE(clf, n_features_to_select=20)
kepler_X_trans = trans.fit_transform(kepler_X, kepler_y)
columns_retained_RFE = kepler.iloc[:, 1:].columns[trans.get_support()].values
from sklearn.feature_selection import RFECV

clf = DecisionTreeClassifier()
trans = RFECV(clf)
kepler_X_trans = trans.fit_transform(kepler_X, kepler_y)
columns_retained_RFECV = kepler.iloc[:, 1:].columns[trans.get_support()].values
len(columns_retained_RFECV)
import itertools
pairs = {}
for (i, (a, b)) in enumerate(itertools.combinations([set(columns_retained_Select), set(columns_retained_FromMode), set(columns_retained_RFE), set(columns_retained_RFECV)], 2)):
    pairs.update({str(i): len(a.difference(b))})
    
print("Enumerating differences between  3!")
list(pairs.values())