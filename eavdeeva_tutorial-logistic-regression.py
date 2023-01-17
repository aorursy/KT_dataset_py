import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



def sigmoid(x, w, w0):

    den = 1 + np.exp(-(x*w + w0))

    return 1.0/den



def plot_sigmoid(ax, w, w0):

    x = [0.1*i for i in range(-75, 76)]

    y = [sigmoid(x_i, w = w, w0 = w0) for x_i in x]

    out = ax.scatter(x = x, y = y)

    out = ax.axhline(y = 0.5, color = 'black', linestyle = '--')

    out = ax.axhline(y = 0, color = 'black', linestyle = '--')

    out = ax.axhline(y = 1, color = 'black', linestyle = '--')

    out = ax.axvline(x = 0, color = 'black', linestyle = '--')

    out = ax.set_title(

        'One-dimensional sigmoid \n p = sigmoid(wx + w0), w = ' + str(w) + ', w0 = ' + str(w0),

        fontsize = 16)

    out = ax.set_xlabel('x', fontsize = 14)

    out = ax.set_ylabel('p(x)', fontsize = 14)

    out = ax.grid()

    

fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, nrows = 1, figsize = (18,6))

plot_sigmoid(ax1, w = 1, w0 = 0)

plot_sigmoid(ax2, w = 10, w0 = 40)

plot_sigmoid(ax3, w = 0.001, w0 = 0)
import pandas as pd

train_df = pd.read_csv('../input/titanic/train.csv')
train_df.head()
train_df.describe()
train_df['is_female'] = train_df['Sex'].apply(lambda x: 1 if x == 'female' else 0)
train_df.drop(columns = ['PassengerId','Survived', 'Age']).describe()
def plot_one_col(df0, df1, col, ax, bins):

    ax.hist(df0[col], label = "didn't survive", density = True, bins = bins)

    ax.hist(df1[col], label = "survived", density = True, bins = bins, alpha = 0.5)

    ax.set_title(col, fontsize = 16)

    ax.set_xlabel(col + ' value', fontsize = 14)

    ax.set_ylabel('N entries per bin', fontsize = 14)

    ax.legend(fontsize = 14)
df0 = train_df.query('Survived == 0')

df1 = train_df.query('Survived == 1')

fig, ax = plt.subplots(ncols = 3, nrows = 2, figsize = (18, 12))

plot_one_col(df0, df1, col = 'Pclass', ax = ax[0,0], bins = [0.5, 1.5, 2.5, 3.5])

plot_one_col(df0, df1, col = 'SibSp', ax = ax[0,1], bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])

plot_one_col(df0, df1, col = 'Parch', ax = ax[0,2], bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])

plot_one_col(df0, df1, col = 'Fare', ax = ax[1,0], bins = [0, 15, 50, 75, 100, 150, 200, 300, 500])

plot_one_col(df0, df1, col = 'is_female', ax = ax[1,1], bins = [-0.5, 0.5, 1.5])
import numpy as np

from sklearn.model_selection import KFold

N_folds = 5

kf = KFold(n_splits = N_folds, random_state = 13, shuffle = True)

indexes = []

for train_index, valid_index in kf.split(train_df):

    print("TRAIN:", train_index[0:5], "VALID:", valid_index[0:5])

    indexes.append({'train':train_index, 'valid':valid_index})
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
clf = LogisticRegression(random_state = 13,  solver='lbfgs')

threshold = 0.5
def fit_on_feature_set(features):

    valid_acc = [0] * N_folds

    train_acc = [0] * N_folds

    acc = [0] * N_folds

    for fold in range(N_folds):

        inds_t = indexes[fold]['train']

        fold_train_df = train_df.loc[inds_t]

        inds_v = indexes[fold]['valid']

        fold_valid_df = train_df.loc[inds_v]

       

        clf.fit(fold_train_df[features], fold_train_df['Survived'])    

        predictions_train = clf.predict_proba(fold_train_df[features])[:,1] > threshold

        fold_train_df['predictions'] = predictions_train

        train_acc[fold] = accuracy_score(fold_train_df['Survived'], fold_train_df['predictions'])

    

        clf.predict_proba(fold_valid_df[features])

        predictions = clf.predict_proba(fold_valid_df[features])[:,1] > threshold

        fold_valid_df['predictions'] = predictions

        valid_acc[fold] = accuracy_score(fold_valid_df['Survived'], fold_valid_df['predictions'])

    

        acc[fold] = min(valid_acc[fold], train_acc[fold])

    return acc
num_features = ['Pclass', 'SibSp', 'Parch', 'Fare', 'is_female']

d_acc = {}

for feat in num_features:

    d_acc[feat] = fit_on_feature_set([feat])

df_acc = pd.DataFrame(d_acc)

df_acc['fold'] = list(x for x in range(N_folds))

df_acc
colors = ['blue', 'orange', 'green', 'red', 'purple', 'black']

fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (12,6))

for i in range(len(num_features)):

    col = num_features[i]

    ax.scatter(

        y = df_acc['fold'], x = df_acc[col], 

        label = col, s = 180, color = colors[i]

    )

    m = df_acc[col].mean()

    s = df_acc[col].std()/(N_folds**0.5)

    ax.axvline(

        x = m, 

        color = colors[i], linestyle = '--', alpha = 0.5

    )

    ax.axvline(

        x = m + s, 

        color = colors[i], alpha = 0.5

    )

    ax.axvline(

        x = m - s, 

        color = colors[i], alpha = 0.5

    )

    ax.axvspan(m-s, m+s, facecolor=colors[i], alpha=0.1)

    ax.set_xlim(0.5, 1.0)

    ax.set_ylabel('fold', fontsize = 20)

    ax.set_xlabel('accuracy', fontsize = 20)

    t1 = 'Compare log-reg models on one feature'

    t2 = 'Accuracy score vs fold'

    ax.set_title(t1 + '\n' + t2, fontsize = 20)

    ax.grid()

    ax.legend(fontsize = 16)
print('Mean accuracy score for one-feature based models: \n')

for col in num_features:

    print(

        col, 

        round(df_acc[col].mean(),3),'+-',

        round(df_acc[col].std()/(N_folds**0.5),3)

    )
clf.fit(train_df[['is_female']], train_df['Survived'])

test_df = pd.read_csv('../input/titanic/test.csv')

test_df['is_female'] = test_df['Sex'].apply(lambda x: 1 if x == 'female' else 0)

predictions = clf.predict_proba(test_df[['is_female']])[:,1] > threshold

test_df['Survived'] = predictions

test_df['Survived'] = test_df['Survived'].astype(int)

test_df[['PassengerId','Survived']].to_csv('submission.csv', index = False)