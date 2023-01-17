
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#!pip install -U imbalanced-learn
traindf=pd.read_csv('../input/trainingSetToMatchCustomTestSet111518.csv')
print(traindf.shape)
print(traindf.head())

def dropUselessFeatures(df):
    print(df.shape)
    df=df.drop(['Unnamed: 0','hephs','hepos','hepts','lephs','lepos','lepts',
                  'hmphs','hmpos','hmpts','lmphs','lmpos','lmpts',
                  'hlphs','hlpos','hlpts','llphs','llpos','llpts'], axis=1)
    
    #these boolean columns got rolled into the outlierScore
    df=df.drop(['highEnergy_transitory_1.5_TF',
       'lowEnergy_transitory_1.0_TF', 'lowEnergy_transitory_1.5_TF'], axis=1)
    
    
    print(df.shape)
    return df

traindf=dropUselessFeatures(traindf)

traindf.loc[:,'target']=traindf.loc[:,'target'].astype(str)

#from stacy's code
# move target to end
traindf = traindf[[c for c in traindf if c not in ['target']] + ['target']]
traindf.head()

#df[1].fillna(0, inplace=True)
traindf['deltaDetect'].fillna(0,inplace=True)
tigdf=traindf[traindf['hostgal_photoz']==0]
tegdf=traindf[traindf['hostgal_photoz']!=0]

print(tigdf.shape)
print(tegdf.shape)

tigdf=tigdf.drop(['hostgal_specz', 'hostgal_photoz', 'distmod', 'hostgal_photoz_err'], axis=1)
print(tigdf.shape)
tigdf.describe()
print('inter-galactic')
for theClass in tigdf.loc[:,'target'].unique():
    print('class ' + str(theClass) + ':')
    trueFilter=tigdf['target']==theClass
    print(trueFilter.sum())
print('extra-galactic')
for theClass in tegdf.loc[:,'target'].unique():
    print('class ' + str(theClass) + ':')
    trueFilter=tegdf['target']==theClass
    print(trueFilter.sum())
#https://www.kaggle.com/qianchao/smote-with-imbalance-data
#from sklearn.preprocessing import StandardScaler
Xig = np.array(tigdf.iloc[:, tigdf.columns != 'target'])
yig = np.array(tigdf.iloc[:, tigdf.columns == 'target'])
print('Shape of X: {}'.format(Xig.shape))
print('Shape of y: {}'.format(yig.shape))

Xeg = np.array(tegdf.iloc[:, tegdf.columns != 'target'])
yeg = np.array(tegdf.iloc[:, tegdf.columns == 'target'])
print('Shape of X: {}'.format(Xeg.shape))
print('Shape of y: {}'.format(yeg.shape))
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

Xig_train, Xig_test, yig_train, yig_test = train_test_split(Xig, yig, test_size=0.3, random_state=0)

print("Number transactions X_train dataset: ", Xig_train.shape)
print("Number transactions y_train dataset: ", yig_train.shape)
print("Number transactions X_test dataset: ", Xig_test.shape)
print("Number transactions y_test dataset: ", yig_test.shape)
print("Before OverSampling, counts of label '92': {}".format(sum(yig_train=='92')))
print("Before OverSampling, counts of label '65': {} \n".format(sum(yig_train=='65')))
print("Before OverSampling, counts of label '16': {}".format(sum(yig_train=='16')))
print("Before OverSampling, counts of label '6': {} \n".format(sum(yig_train=='6')))
print("Before OverSampling, counts of label '53': {}".format(sum(yig_train=='53')))

sm = SMOTE(random_state=2)
Xig_train_res, yig_train_res = sm.fit_sample(Xig_train, yig_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(Xig_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(yig_train_res.shape))

print("After OverSampling, counts of label '92': {}".format(sum(yig_train_res=='92')))
print("After OverSampling, counts of label '65': {}".format(sum(yig_train_res=='65')))
print("After OverSampling, counts of label '16': {}".format(sum(yig_train_res=='16')))
print("After OverSampling, counts of label '6': {}".format(sum(yig_train_res=='6')))
print("After OverSampling, counts of label '53': {}".format(sum(yig_train_res=='53')))

def smoteAdataset(Xig, yig, test_size=0.2, random_state=0):
    
    Xig_train, Xig_test, yig_train, yig_test = train_test_split(Xig, yig, test_size=test_size, random_state=random_state)
    print("Number transactions X_train dataset: ", Xig_train.shape)
    print("Number transactions y_train dataset: ", yig_train.shape)
    print("Number transactions X_test dataset: ", Xig_test.shape)
    print("Number transactions y_test dataset: ", yig_test.shape)

    classes=[]
    for i in np.unique(yig):
        classes.append(i)
        print("Before OverSampling, counts of label " + str(i) + ": {}".format(sum(yig_train==i)))
        
    sm=SMOTE(random_state=2)
    Xig_train_res, yig_train_res = sm.fit_sample(Xig_train, yig_train.ravel())

    print('After OverSampling, the shape of train_X: {}'.format(Xig_train_res.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(yig_train_res.shape))
    
    for eachClass in classes:
        print("After OverSampling, counts of label " + str(eachClass) + ": {}".format(sum(yig_train_res==eachClass)))
        
    return Xig_train_res, yig_train_res

Xeg_train_res, yeg_train_res=smoteAdataset(Xeg, yeg)

%matplotlib inline
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
import numpy as np

print(__doc__)

rng = np.random.RandomState(18)

f, ax = plt.subplots(1, 1, figsize=(8, 8))

# generate some data points
y = np.array([3.65284, 3.52623, 3.51468, 3.22199, 3.21])
z = np.array([0.43, 0.45, 0.6, 0.4, 0.211])
y_2 = np.array([3.3, 3.6])
z_2 = np.array([0.58, 0.34])

# plot the majority and minority samples
ax.scatter(z, y, label='Minority class', s=100)
ax.scatter(z_2, y_2, label='Majority class', s=100)

idx = rng.randint(len(y), size=2)
annotation = [r'$x_i$', r'$x_{zi}$']

for a, i in zip(annotation, idx):
    ax.annotate(a, (z[i], y[i]),
                xytext=tuple([z[i] + 0.01, y[i] + 0.005]),
                fontsize=15)

# draw the circle in which the new sample will generated
radius = np.sqrt((z[idx[0]] - z[idx[1]]) ** 2 + (y[idx[0]] - y[idx[1]]) ** 2)
circle = plt.Circle((z[idx[0]], y[idx[0]]), radius=radius, alpha=0.2)
ax.add_artist(circle)

# plot the line on which the sample will be generated
ax.plot(z[idx], y[idx], '--', alpha=0.5)

# create and plot the new sample
step = rng.uniform()
y_gen = y[idx[0]] + step * (y[idx[1]] - y[idx[0]])
z_gen = z[idx[0]] + step * (z[idx[1]] - z[idx[0]])

ax.scatter(z_gen, y_gen, s=100)
ax.annotate(r'$x_{new}$', (z_gen, y_gen),
            xytext=tuple([z_gen + 0.01, y_gen + 0.005]),
            fontsize=15)

# make the plot nicer with legend and label
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 10))
ax.set_xlim([0.2, 0.7])
ax.set_ylim([3.2, 3.7])
plt.xlabel(r'$X_1$')
plt.ylabel(r'$X_2$')
plt.legend()
plt.tight_layout()
plt.show()
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC
from imblearn.over_sampling import RandomOverSampler
from imblearn.base import BaseSampler

print(__doc__)
def create_dataset(n_samples=1000, weights=(0.01, 0.01, 0.98), n_classes=3,
                   class_sep=0.8, n_clusters=1):
    return make_classification(n_samples=n_samples, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters,
                               weights=list(weights),
                               class_sep=class_sep, random_state=0)
def plot_resampling(X, y, sampling, ax):
    X_res, y_res = sampling.fit_resample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor='k')
    # make nice plotting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    return Counter(y_res)
def plot_decision_function(X, y, clf, ax):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
X, y = create_dataset(n_samples=10000, weights=(0.01, 0.05, 0.94))
clf = LinearSVC().fit(X, y)
plot_decision_function(X, y, clf, ax1)
ax1.set_title('Linear SVC with y={}'.format(Counter(y)))
pipe = make_pipeline(RandomOverSampler(random_state=0), LinearSVC())
pipe.fit(X, y)
plot_decision_function(X, y, pipe, ax2)
ax2.set_title('Decision function for RandomOverSampler')
fig.tight_layout()
# Make an identity sampler
class FakeSampler(BaseSampler):

    _sampling_type = 'bypass'

    def _fit_resample(self, X, y):
        return X, y


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
X, y = create_dataset(n_samples=10000, weights=(0.01, 0.05, 0.94))
sampler = FakeSampler()
clf = make_pipeline(sampler, LinearSVC())
plot_resampling(X, y, sampler, ax1)
ax1.set_title('Original data - y={}'.format(Counter(y)))

ax_arr = (ax2, ax3, ax4)
for ax, sampler in zip(ax_arr, (RandomOverSampler(random_state=0),
                                SMOTE(random_state=0),
                                ADASYN(random_state=0))):
    clf = make_pipeline(sampler, LinearSVC())
    clf.fit(X, y)
    plot_resampling(X, y, sampler, ax)
    ax.set_title('Resampling using {}'.format(sampler.__class__.__name__))
fig.tight_layout()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
X, y = create_dataset(n_samples=5000, weights=(0.01, 0.05, 0.94),
                      class_sep=0.8)

ax_arr = ((ax1, ax2), (ax3, ax4))
for ax, sampler in zip(ax_arr, (SMOTE(random_state=0),
                                ADASYN(random_state=0))):
    clf = make_pipeline(sampler, LinearSVC())
    clf.fit(X, y)
    plot_decision_function(X, y, clf, ax[0])
    ax[0].set_title('Decision function for {}'.format(
        sampler.__class__.__name__))
    plot_resampling(X, y, sampler, ax[1])
    ax[1].set_title('Resampling using {}'.format(
        sampler.__class__.__name__))
fig.tight_layout()
fig, ((ax1, ax2), (ax3, ax4),
      (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(15, 30))
X, y = create_dataset(n_samples=5000, weights=(0.01, 0.05, 0.94),
                      class_sep=0.8)

ax_arr = ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8))
for ax, sampler in zip(ax_arr,
                       (SMOTE(random_state=0),
                        BorderlineSMOTE(random_state=0, kind='borderline-1'),
                        BorderlineSMOTE(random_state=0, kind='borderline-2'),
                        SVMSMOTE(random_state=0))):
    clf = make_pipeline(sampler, LinearSVC())
    clf.fit(X, y)
    plot_decision_function(X, y, clf, ax[0])
    ax[0].set_title('Decision function for {}'.format(
        sampler.__class__.__name__))
    plot_resampling(X, y, sampler, ax[1])
    ax[1].set_title('Resampling using {}'.format(sampler.__class__.__name__))
fig.tight_layout()
# create a synthetic data set with continuous and categorical features
rng = np.random.RandomState(42)
n_samples = 50
X = np.empty((n_samples, 3), dtype=object)
X[:, 0] = rng.choice(['A', 'B', 'C'], size=n_samples).astype(object)
X[:, 1] = rng.randn(n_samples)
X[:, 2] = rng.randint(3, size=n_samples)
y = np.array([0] * 20 + [1] * 30)

print('The original imbalanced dataset')
print(sorted(Counter(y).items()))
print('The first and last columns are containing categorical features:')
print(X[:5])

smote_nc = SMOTENC(categorical_features=[0, 2], random_state=0)
X_resampled, y_resampled = smote_nc.fit_resample(X, y)
print('Dataset after resampling:')
print(sorted(Counter(y_resampled).items()))
print('SMOTE-NC will generate categories for the categorical features:')
print(X_resampled[-5:])

plt.show()
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler,
                                     NearMiss,
                                     InstanceHardnessThreshold,
                                     CondensedNearestNeighbour,
                                     EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours,
                                     AllKNN,
                                     NeighbourhoodCleaningRule,
                                     OneSidedSelection)
print(__doc__)
def plot_resampling(X, y, sampling, ax):
    X_res, y_res = sampling.fit_resample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor='k')
    # make nice plotting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    return Counter(y_res)
def plot_decision_function(X, y, clf, ax):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
X, y = create_dataset(n_samples=5000, weights=(0.01, 0.05, 0.94),
                      class_sep=0.8)

clf = LinearSVC().fit(X, y)
plot_decision_function(X, y, clf, ax1)
ax1.set_title('Linear SVC with y={}'.format(Counter(y)))
sampler = ClusterCentroids(random_state=0)
clf = make_pipeline(sampler, LinearSVC())
clf.fit(X, y)
plot_decision_function(X, y, clf, ax2)
ax2.set_title('Decision function for {}'.format(sampler.__class__.__name__))
plot_resampling(X, y, sampler, ax3)
ax3.set_title('Resampling using {}'.format(sampler.__class__.__name__))
fig.tight_layout()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
X, y = create_dataset(n_samples=5000, weights=(0.01, 0.05, 0.94),
                      class_sep=0.8)

clf = LinearSVC().fit(X, y)
plot_decision_function(X, y, clf, ax1)
ax1.set_title('Linear SVC with y={}'.format(Counter(y)))
sampler = RandomUnderSampler(random_state=0)
clf = make_pipeline(sampler, LinearSVC())
clf.fit(X, y)
plot_decision_function(X, y, clf, ax2)
ax2.set_title('Decision function for {}'.format(sampler.__class__.__name__))
plot_resampling(X, y, sampler, ax3)
ax3.set_title('Resampling using {}'.format(sampler.__class__.__name__))
fig.tight_layout()
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2,
                                                         figsize=(15, 25))
X, y = create_dataset(n_samples=5000, weights=(0.1, 0.2, 0.7), class_sep=0.8)

ax_arr = ((ax1, ax2), (ax3, ax4), (ax5, ax6))
for ax, sampler in zip(ax_arr, (NearMiss(version=1),
                                NearMiss(version=2),
                                NearMiss(version=3))):
    clf = make_pipeline(sampler, LinearSVC())
    clf.fit(X, y)
    plot_decision_function(X, y, clf, ax[0])
    ax[0].set_title('Decision function for {}-{}'.format(
        sampler.__class__.__name__, sampler.version))
    plot_resampling(X, y, sampler, ax[1])
    ax[1].set_title('Resampling using {}-{}'.format(
        sampler.__class__.__name__, sampler.version))
fig.tight_layout()
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2,
                                                         figsize=(15, 25))
X, y = create_dataset(n_samples=500, weights=(0.2, 0.3, 0.5), class_sep=0.8)

ax_arr = ((ax1, ax2), (ax3, ax4), (ax5, ax6))
for ax, sampler in zip(ax_arr, (
        EditedNearestNeighbours(),
        RepeatedEditedNearestNeighbours(),
        AllKNN(allow_minority=True))):
    clf = make_pipeline(sampler, LinearSVC())
    clf.fit(X, y)
    plot_decision_function(X, y, clf, ax[0])
    ax[0].set_title('Decision function for {}'.format(
        sampler.__class__.__name__))
    plot_resampling(X, y, sampler, ax[1])
    ax[1].set_title('Resampling using {}'.format(
        sampler.__class__.__name__))
fig.tight_layout()
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2,
                                                         figsize=(15, 25))
X, y = create_dataset(n_samples=500, weights=(0.2, 0.3, 0.5), class_sep=0.8)

ax_arr = ((ax1, ax2), (ax3, ax4), (ax5, ax6))
for ax, sampler in zip(ax_arr, (
        CondensedNearestNeighbour(random_state=0),
        OneSidedSelection(random_state=0),
        NeighbourhoodCleaningRule())):
    clf = make_pipeline(sampler, LinearSVC())
    clf.fit(X, y)
    plot_decision_function(X, y, clf, ax[0])
    ax[0].set_title('Decision function for {}'.format(
        sampler.__class__.__name__))
    plot_resampling(X, y, sampler, ax[1])
    ax[1].set_title('Resampling using {}'.format(
        sampler.__class__.__name__))
fig.tight_layout()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
X, y = create_dataset(n_samples=5000, weights=(0.01, 0.05, 0.94),
                      class_sep=0.8)

clf = LinearSVC().fit(X, y)
plot_decision_function(X, y, clf, ax1)
ax1.set_title('Linear SVC with y={}'.format(Counter(y)))
sampler = InstanceHardnessThreshold(
    random_state=0, estimator=LogisticRegression(solver='lbfgs',
                                                 multi_class='auto'))
clf = make_pipeline(sampler, LinearSVC())
clf.fit(X, y)
plot_decision_function(X, y, clf, ax2)
ax2.set_title('Decision function for {}'.format(sampler.__class__.__name__))
plot_resampling(X, y, sampler, ax3)
ax3.set_title('Resampling using {}'.format(sampler.__class__.__name__))
fig.tight_layout()

plt.show()