import numpy as np

import scipy as sp

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



def warn(*args, **kwargs): pass

import warnings

warnings.warn = warn



%matplotlib inline



# from sklearn.preprocessing import LabelEncoder

# from sklearn.cross_validation import StratifiedShuffleSplit



train = pd.read_csv('../input/train.csv').drop('id',axis=1)

test = pd.read_csv('../input/test.csv')

test_ids = test['id']

test.drop('id',axis=1,inplace=True)
train.columns
print(train.isnull().any().any())

print(test.isnull().any().any())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
species = train['species']

train.drop('species',axis=1,inplace=True)

y_train = le.fit_transform(species)
print('There are {} categories in the label, each category has 10 observations\n'.format(species.unique()))

fig = plt.figure(figsize=(9.5,4));

ax = fig.add_subplot(1,1,1)

species.value_counts().plot(kind='bar');

ax.tick_params(axis='x',labelsize=6);
mColumns = [col for col in train.columns if col.startswith('margin')]

sColumns = [col for col in train.columns if col.startswith('shape')]

tColumns = [col for col in train.columns if col.startswith('texture')]
def setBoxplot(ax,ylim):

    ax.set_ylim(ylim);

    ax.tick_params(axis='y',labelsize=6);

    ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontsize=7);
fig = plt.figure(figsize = (9,4));

ax = fig.add_subplot(1,1,1);

train[mColumns].boxplot(whis=1.5);

setBoxplot(ax,[0,0.15])
fig=plt.figure(figsize=(9,6))

ax = fig.add_subplot(2,1,1)

train['margin6'].hist(bins=80,normed=True);

ax.set_title('margin6 - wide range')

ax = fig.add_subplot(2,1,2)

train['margin52'].hist(bins=80,normed=True);

ax.set_title('margin52 - narrow range');
fig = plt.figure(figsize = (9,4));

ax = fig.add_subplot(1,1,1);

train[sColumns].boxplot(whis=1.5);

setBoxplot(ax,[0,0.0015])
fig = plt.figure(figsize = (9,4));

ax = fig.add_subplot(1,1,1);

train[tColumns].boxplot(whis=1.5);

setBoxplot(ax,[0,0.13])
fig=plt.figure(figsize=(9,6))

ax = fig.add_subplot(2,1,1)

train['texture15'].hist(bins=80,normed=True);

ax.set_title('texture15 - wide range')

ax = fig.add_subplot(2,1,2)

train['texture16'].hist(bins=80,normed=True);

ax.set_title('texture16 - narrow range');
def correlation_matrix(df,ax1):

    import numpy as np

    from matplotlib import pyplot as plt

    from matplotlib import cm as cm



    cmap = cm.get_cmap('RdYlBu', 30)

    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap, vmin=-1.0, vmax=1.0)

    ax1.grid(True)

    labels = df.columns

    ax1.set_xticks(range(0,len(labels)))

    ax1.set_xticklabels(labels,fontsize=7,rotation=90)

    ax1.set_yticks(range(0,len(labels)))

    ax1.set_yticklabels(labels,fontsize=7)

    cbar = fig.colorbar(cax)

    cbar.ax.tick_params(labelsize=7)

    ax1.grid(b=False)
# correlation matrix for margin features

fig = plt.figure(figsize=(9.5,9))

ax = fig.add_subplot(1,1,1)

correlation_matrix(train[mColumns],ax)
# correlation matrix for shape features

fig = plt.figure(figsize=(9.5,9))

ax = fig.add_subplot(1,1,1)

correlation_matrix(train[sColumns],ax)
# correlation matrix for texture features

fig = plt.figure(figsize=(9.5,9))

ax = fig.add_subplot(1,1,1)

correlation_matrix(train[tColumns],ax)
from sklearn.preprocessing import MaxAbsScaler
x_data = np.vstack([train,test])

mas = MaxAbsScaler()

n_x_data = mas.fit_transform(x_data)

print(n_x_data.shape)

n_x_data
n_x_test = n_x_data[len(species):,:]

n_x_train = n_x_data[0:len(species),:]
print(n_x_train.shape)

print(y_train.shape)

print(n_x_test.shape)
# the best model get logloss = 0.04 or so with 'C':2000 and 'tol':0.0001

from sklearn.linear_model import LogisticRegression

from sklearn.grid_search import GridSearchCV

params = {'C':[0.001, 0.01, 1, 10, 100, 500, 1000, 2000], 'tol': [0.0001, 0.001, 0.005]}

log_reg = LogisticRegression(solver='newton-cg', multi_class='multinomial')

clf = GridSearchCV(log_reg, params, scoring='log_loss', refit='True', n_jobs=1, cv=5)

clf.fit(n_x_train, y_train)



print("best params: " + str(clf.best_params_))

for params, mean_score, scores in clf.grid_scores_:

    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std(), params))

    print(scores)
y_test_raw = clf.predict_proba(n_x_test)
submission = pd.DataFrame(y_test_raw, index=test_ids, columns=le.classes_)

submission.to_csv('./submission_raw.csv')