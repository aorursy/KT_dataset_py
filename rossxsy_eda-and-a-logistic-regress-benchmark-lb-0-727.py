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
train.head()
len(train)
len(test)
print(train.isnull().any().any())

print(test.isnull().any().any())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
gtype = train['type']

train.drop('type',axis=1,inplace=True)

y_train = le.fit_transform(gtype)
le.classes_
print('There are {} categories in the label, each category has 10 observations\n'.format(len(gtype.unique())))

plt.figure(figsize=(9,6));

gtype.value_counts().plot(kind='bar');
contColumns = train.columns[0:4]

catColumns = ['color']
def setBoxplot(ax,ylim):

    ax.set_ylim(ylim);

    ax.tick_params(axis='y',labelsize=20,);

    ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontsize=16);
fig = plt.figure(figsize = (9,5));

ax = fig.add_subplot(1,1,1);

train[contColumns].boxplot(whis=1.5);

setBoxplot(ax,[0,1.2])
fig=plt.figure(figsize=(9,15))

for i,col in enumerate(train[contColumns]):

    ax = fig.add_subplot(len(train.columns),1,i+1)

    train[col].hist(bins=50,normed=True);

    ax.set_title(col)



fig=plt.figure(figsize=(9,5))

for i,col in enumerate(train[catColumns]):

    train[col].value_counts().plot(kind='bar')
def correlation_matrix(df,ax1):

    import numpy as np

    from matplotlib import pyplot as plt

    from matplotlib import cm as cm



    cmap = cm.get_cmap('RdYlBu', 30)

    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap, vmin=-1.0, vmax=1.0)

    ax1.grid(True)

    labels = df.columns

    ax1.set_xticks(range(0,len(labels)))

    ax1.set_xticklabels(labels,fontsize=12,rotation=90)

    ax1.set_yticks(range(0,len(labels)))

    ax1.set_yticklabels(labels,fontsize=12)

    cbar = fig.colorbar(cax)

    cbar.ax.tick_params(labelsize=16)

    ax1.grid(b=False)
# correlation matrix for margin features

fig = plt.figure(figsize=(7,7))

ax = fig.add_subplot(1,1,1)

correlation_matrix(train[contColumns],ax)
# label encode the categorical features



x_data_df = pd.DataFrame(np.vstack([train,test]),columns = train.columns)

colLes = []

for col in catColumns:

    colLe = LabelEncoder()

    x_data_df[col] = colLe.fit_transform(x_data_df[col])

    colLes.append(colLe)

x_data_df.head()
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

x_cat = ohe.fit_transform(x_data_df[catColumns])

n_x_cat = x_cat

print(n_x_cat.shape)

type(n_x_cat)
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

x_cont = x_data_df[contColumns].values

n_x_cont = ss.fit_transform(x_cont)

print(n_x_cont.shape)

type(n_x_cont)
n_x_data = np.hstack([n_x_cont,n_x_cat.toarray()])

print(n_x_data.shape)

n_x_data
n_x_test = n_x_data[len(gtype):,:]

n_x_train = n_x_data[0:len(gtype),:]
print(n_x_train.shape)

print(y_train.shape)

print(n_x_test.shape)
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

y_test_raw_df = pd.DataFrame(y_test_raw, index=test_ids, columns=le.classes_)

submission = pd.DataFrame({'id':y_test_raw_df.idxmax(axis=1).index,'type':y_test_raw_df.idxmax(axis=1).values})



submission
submission.to_csv('./submission_raw.csv',index=False)