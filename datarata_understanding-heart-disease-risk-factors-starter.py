import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import scipy as sci

import seaborn as sns

import math

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve, auc

from graphviz import Source

from IPython.display import Image  

from sklearn.tree import export_graphviz

%matplotlib inline
heart = pd.read_csv('../input/heart-disease-uci/heart.csv')

heart.head()


print(heart.shape)
heart.describe()
heart.dtypes
dup_count = heart.duplicated()

heart[dup_count]
dup = heart.loc[heart['age'] == 38]

dup
heart = heart.drop(163)

heart.shape
null_count = heart.isnull().sum()

null_count
print('ca value counts')

print(heart['ca'].value_counts())

print('\nthal value counts')

print(heart['thal'].value_counts())
heart[heart['ca'] == 4]
heart[heart['thal']==0]
heart = heart.drop(list(heart[heart['ca'] == 4].index))

heart = heart.drop(list(heart[heart['thal'] == 0].index))

heart.shape
male_age_dist = heart.loc[heart['sex'] == 1]['age'] #Male

female_age_dist = heart.loc[heart['sex'] == 0]['age'] #Female

sns.distplot(male_age_dist, kde = False, label = 'Male', \

             hist_kws={"histtype": "step", "linewidth": 2, "alpha": 0.5, "color": "g"})

sns.distplot(female_age_dist, kde = False, label = 'Female', \

             hist_kws={"histtype": "step", "linewidth": 2, "alpha": 0.5, "color": "b"})

plt.legend()

plt.title('Male vs Female Age distribution')

plt.show()
heart_corr = heart.corr()

heart_corr.style.background_gradient(cmap = plt.get_cmap('bwr'))
g = sns.FacetGrid(heart, col = 'restecg', row = 'target', hue = 'sex', legend_out = True, height = 3)

g.map(sns.kdeplot, 'age', shade = True).add_legend()



plt.show()
g4 = sns.FacetGrid(heart, col = 'restecg', row = 'target', hue = 'sex', legend_out = True, height = 3)

g4.map(plt.scatter, 'age', 'oldpeak', facecolors = 'none').add_legend()

plt.show()
g = sns.FacetGrid(heart, col = 'cp', row = 'target', hue = 'sex', legend_out = True, height = 3)

g.map(sns.kdeplot, 'age', shade = True).add_legend()

plt.legend()

plt.show()
g2 = sns.FacetGrid(heart, col = 'cp', row = 'target', hue = 'sex', legend_out = True, height = 3)

g2.map(plt.scatter, 'age', 'chol', facecolors = 'none').add_legend()



plt.show()
heart_male = heart[heart['sex'] == 1].groupby('target')[['trestbps', 'chol', 'thalach', 'oldpeak']]

heart_female = heart[heart['sex'] == 0].groupby('target')[['trestbps', 'chol', 'thalach', 'oldpeak']]
heart_male.agg([np.mean, np.max, np.min])
heart_female.agg([np.mean, np.max, np.min])
fig = plt.figure(figsize = (15,5))

fig.suptitle('Male')

ax = fig.subplots(nrows = 1, ncols = 2, sharey = True)

heart_male.boxplot(column = ['trestbps', 'chol', 'thalach'], rot = 90, ax = ax)

ax[0].set_title('>50% Narrowing')

ax[1].set_title('<50% Narrowing')
fig = plt.figure(figsize = (15,5))

fig.suptitle('Female')

ax = fig.subplots(nrows = 1, ncols = 2, sharey = True)

heart_female.boxplot(column = ['trestbps', 'chol', 'thalach'], rot = 90, ax = ax)

ax[0].set_title('>50% Narrowing')

ax[1].set_title('<50% Narrowing')
fig = plt.figure(figsize = (15,5))

ax = fig.subplots(nrows = 1, ncols = 2, sharey = True)

heart_male.boxplot(column = ['oldpeak'], rot = 90, ax = ax)

ax[0].set_title('>50% Narrowing')

ax[1].set_title('<50% Narrowing')
fig = plt.figure(figsize = (15,5))

ax = fig.subplots(nrows = 1, ncols = 2, sharey = True)

heart_female.boxplot(column = ['oldpeak'], rot = 90, ax = ax)

ax[0].set_title('>50% Narrowing')

ax[1].set_title('<50% Narrowing')
g5 = sns.FacetGrid(heart, col = 'restecg', row = 'target', hue = 'cp', legend_out = True, height = 3)

g5.map(plt.scatter, 'thalach', 'oldpeak', facecolors = 'none').add_legend()



plt.show()
g6 = sns.FacetGrid(heart, col = 'thal', row = 'target', hue = 'cp', legend_out = True, height = 3)

g6.map(plt.scatter, 'thalach', 'oldpeak', facecolors = 'none').add_legend()



plt.show()
g6 = sns.FacetGrid(heart, col = 'ca', row = 'target', hue = 'cp', legend_out = True, height = 3)

g6.map(plt.scatter, 'thalach', 'oldpeak', facecolors = 'none').add_legend()



plt.show()
g7 = sns.FacetGrid(heart, col = 'exang', row = 'target', hue = 'slope', legend_out = True, height = 3)

g7.map(plt.scatter, 'age', 'thalach', facecolors = 'none').add_legend()

plt.show()
sns.lmplot(x = 'age', y = 'trestbps', hue = 'sex', data = heart, col = 'target', markers=["o", "x"], palette="Set1")
mean_male_bp = heart[heart['sex']==1]['trestbps'].mean()

mean_female_bp = heart[heart['sex']==0]['trestbps'].mean()

print(mean_male_bp)

print(mean_female_bp)
bp_m_f = sci.stats.ttest_ind(heart[heart['sex']==1]['trestbps'], heart[heart['sex']==0]['trestbps'])

bp_m_f
sns.lmplot(x = 'age', y = 'thalach', hue = 'sex', data = heart, col = 'target', markers=["o", "x"], palette="Set1")
heart.loc[(heart['sex'] == 0) & (heart['age'] < 45) & (heart['target'] == 0)]
heart.loc[heart['chol'] == max(heart['chol'])]
heart = heart.drop([85, 215])

heart.shape
shuffled_index = np.random.permutation(heart.index)

train_max_row = math.floor(heart.shape[0] * .7)



heart_dt = heart.copy()

heart_dt['sex'] = heart_dt['sex'].astype('object')

heart_dt['cp'] = heart_dt['cp'].astype('object')

heart_dt['fbs'] = heart_dt['fbs'].astype('object')

heart_dt['restecg'] = heart_dt['restecg'].astype('object')

heart_dt['exang'] = heart_dt['exang'].astype('object')

heart_dt['slope'] = heart_dt['slope'].astype('object')

heart_dt['thal'] = heart_dt['thal'].astype('object')
heart_dt = pd.get_dummies(heart_dt, drop_first=True)

heart_dt.head()
columns = list(heart_dt.columns)

columns.remove('target')

columns
heart_dt = heart_dt.reindex(shuffled_index)

heart_data = heart_dt[columns]

heart_label = heart_dt['target']

train_heart_data = heart_data.iloc[:train_max_row]

test_heart_data = heart_data.iloc[train_max_row:]

train_heart_label = heart_label.iloc[:train_max_row]

test_heart_label = heart_label.iloc[train_max_row:]
print(heart_data.shape)

print(heart_label.shape)

print(train_heart_data.shape)

print(test_heart_data.shape)

print(train_heart_label.shape)

print(test_heart_label.shape)
train_heart_label.value_counts()
clf = DecisionTreeClassifier()

clf.fit(train_heart_data, train_heart_label)

predictions_train = clf.predict(train_heart_data)

predictions = clf.predict(test_heart_data)

errors_train = roc_auc_score(train_heart_label, predictions_train)

errors = roc_auc_score(test_heart_label, predictions)

print(errors_train)

print(errors)
export_graphviz(clf, out_file='tree_limited.dot', feature_names = columns,

                rounded = True, proportion = False, precision = 2, filled = True)

!dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=600

Image(filename = 'tree_limited.png')
clf = DecisionTreeClassifier(max_depth = 7, min_samples_split = 5, min_samples_leaf = 4)

clf.fit(train_heart_data, train_heart_label)

predictions_train = clf.predict(train_heart_data)

predictions = clf.predict(test_heart_data)

predictions_proba = clf.predict_proba(test_heart_data)[:,1]

errors_train = roc_auc_score(train_heart_label, predictions_train)

errors = roc_auc_score(test_heart_label, predictions)

print(errors_train)

print(errors)
export_graphviz(clf, out_file='tree_limited.dot', feature_names = columns,

                rounded = True, proportion = False, precision = 2, filled = True)

!dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=600

Image(filename = 'tree_limited.png')
fpr, tpr, thresholds = roc_curve(test_heart_label, predictions_proba)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for diabetes classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)