import numpy as np

import pandas as pd
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

sns.set_context('notebook')
# Load the target features

targets = pd.read_csv('/kaggle/input/trends-assessment-prediction/train_scores.csv').set_index('Id')

targets.head()
# How many null values does each target have

n_nulls = targets.isnull().sum()

display(n_nulls)

n_nulls.plot.barh();
targets.dropna(inplace=True)
sns.heatmap(targets.corr()*100, square=True, annot=True, fmt='.0f');
targets.plot(lw=0, marker='.', markersize=1, subplots=True, figsize=(14, 8));
targets['age'].plot(lw=0, marker='.', markersize=1, figsize=(14, 4));
targets['age'].nunique()
plt.plot(targets['age'].sort_values().values);
sns.pairplot(targets, plot_kws=dict(s=5, alpha=0.5));
plt.figure(figsize=(6, 6))

d2 = targets.dropna().iloc[:, 3:].values

plt.scatter(d2[:, 0], d2[:, 1], s=3);
def rotate_origin(x, y, radians):

    """Rotates a point around the origin (0, 0)."""

    xx = x * np.cos(radians) + y * np.sin(radians)

    yy = -x * np.sin(radians) + y * np.cos(radians)

    return np.array([xx, yy]).T
# Let's rotate the domain 2 variables by 1 radian

d2_rot = rotate_origin(d2[:, 0], d2[:, 1], 1)

plt.figure(figsize=(6, 6))

plt.scatter(d2_rot[:, 0], d2_rot[:, 1], s=3);
# Let's explore between 0.85 and 0.95 radians

n_uniques = []

for r in np.linspace(0.85, 0.95, 5000):

    d22_rot = rotate_origin(d2[:, 0], d2[:, 1], r)[:, 1]

    n_uniques.append([r, len(np.unique(np.round(d22_rot, 6)))])

n_uniques = np.array(n_uniques)



plt.figure(figsize=(14, 2))

plt.scatter(n_uniques[:, 0], n_uniques[:, 1], s=3);
# Let's explore between 0.905 and 0.910 radians

n_uniques = []

for r in np.linspace(0.905, 0.910, 5000):

    d22_rot = rotate_origin(d2[:, 0], d2[:, 1], r)[:, 1]

    n_uniques.append([r, len(np.unique(np.round(d22_rot, 6)))])

n_uniques = np.array(n_uniques)



plt.figure(figsize=(14, 2))

plt.scatter(n_uniques[:, 0], n_uniques[:, 1], s=3);
d2_rot
# Let's explore between 0.90771 and 0.907715 radians

n_uniques = []

for r in np.linspace(0.90771, 0.907715, 5000):

    d22_rot = rotate_origin(d2[:, 0], d2[:, 1], r)[:, 1]

    n_uniques.append([r, len(np.unique(np.round(d22_rot, 6)))])

n_uniques = np.array(n_uniques)



plt.figure(figsize=(14, 2))

plt.scatter(n_uniques[:, 0], n_uniques[:, 1], s=3);
rot = 0.90771256655



d22_rot = rotate_origin(d2[:, 0], d2[:, 1], rot)[:, 1]

n_unique_entries = len(np.unique(np.round(d22_rot, 6)))



print('Optimal rotation leads to %d unique entries on domain2_var2' % n_unique_entries)
# Let's rotate the domain 2 variables by 1 radian

d2_rot = rotate_origin(d2[:, 0], d2[:, 1], rot)

plt.figure(figsize=(8, 8))

plt.scatter(d2_rot[:, 0], d2_rot[:, 1], s=2);
# Let's rotate the domain 2 variables by 1 radian

d2_rot = rotate_origin(d2[:, 0], d2[:, 1], rot)

plt.figure(figsize=(8, 8))

plt.scatter(d2_rot[:, 0], d2_rot[:, 1], s=4)

plt.xlim(40, 100)

plt.ylim(-30, 0);
targets.loc[:, 'd21_rot'] = d2_rot[:, 0]

targets.loc[:, 'd22_rot'] = d2_rot[:, 1]



sns.heatmap(targets.corr()*100, square=True, annot=True, fmt='.0f');
plt.plot(targets['d22_rot'].sort_values().values);
from scipy.stats import norm

for col in targets.columns:

    plt.figure(figsize=(8, 2))

    sns.distplot(targets[col], fit=norm, kde=True)

    plt.show()
pow_age = 0.93

pow_d1v1 = 1.25

pow_d1v2 = 1.72

pow_d2v1 = 1.37

pow_d2v2 = 1.39

pow_d21 = 1.85

pow_d22 = 1



powers = [pow_age, pow_d1v1, pow_d1v2, pow_d2v1, pow_d2v2, pow_d21, pow_d22 ]
from scipy.stats import norm

for i, col in enumerate(targets.columns):

    plt.figure(figsize=(8, 2))

    sns.distplot(np.power(targets[col], powers[i]), fit=norm, kde=True)

    plt.show()
pow_age = 1.0

pow_d1v1 = 1.5

pow_d1v2 = 1.5

pow_d2v1 = 1.5

pow_d2v2 = 1.5

pow_d21 = 1.5

pow_d22 = 1



powers = [pow_age, pow_d1v1, pow_d1v2, pow_d2v1, pow_d2v2, pow_d21, pow_d22 ]



from scipy.stats import norm

for i, col in enumerate(targets.columns):

    plt.figure(figsize=(8, 2))

    sns.distplot(np.power(targets[col], powers[i]), fit=norm, kde=True)

    plt.show()