import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
#Load Data and replace na values with -1

train=pd.read_csv('../input/train.csv', na_values=-1)

test=pd.read_csv('../input/test.csv', na_values=-1)
#Check data sample

print(train.shape)

train.head(5)
train.isnull().values.any()
#Plot missing values for each column

missing_df = train.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.loc[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')

print(missing_df)



ind = np.arange(missing_df.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(8,6))

rects = ax.barh(ind, missing_df.missing_count.values, color='blue')

ax.set_yticks(ind)

ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show()
median_values = train.median(axis=0)

train = train.fillna(median_values, inplace=True)
features = train.drop(['id','target'], axis=1).values

targets = train.target.values
ax = sns.countplot(x = targets ,palette="Set2")

sns.set(font_scale=1.5)

ax.set_xlabel(' ')

ax.set_ylabel(' ')

fig = plt.gcf()

fig.set_size_inches(10,5)

ax.set_ylim(top=700000)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(targets)), (p.get_x()+ 0.3, p.get_height()+10000))



plt.title('Distribution of 595212 Targets')

plt.xlabel('Initiation of Auto Insurance Claim Next Year')

plt.ylabel('Frequency [%]')

plt.show()
train.head()
sns.set(style="white")



# Compute the correlation matrix

corr = train.corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897

def gini(actual, pred, cmpcol = 0, sortcol = 1):

    assert( len(actual) == len(pred) )

    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

    totalLosses = all[:,0].sum()

    giniSum = all[:,0].cumsum().sum() / totalLosses

    

    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)

 

def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return 'gini', gini_score
unwanted = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(unwanted, axis=1)  

test = test.drop(unwanted, axis=1)  
kfold = 5

skf = StratifiedKFold(n_splits=kfold, random_state=42)
# More parameters has to be tuned. Good luck :)

params = {

    'min_child_weight': 10.0,

    'objective': 'binary:logistic',

    'max_depth': 7,

    'max_delta_step': 1.8,

    'colsample_bytree': 0.4,

    'subsample': 0.8,

    'eta': 0.025,

    'gamma': 0.65,

    'num_boost_round' : 700

    }
X = train.drop(['id', 'target'], axis=1).values

y = train.target.values

test_id = test.id.values

test = test.drop('id', axis=1)
sub = pd.DataFrame()

sub['id'] = test_id

sub['target'] = np.zeros_like(test_id)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    print('[Fold %d/%d]' % (i + 1, kfold))

    X_train, X_valid = X[train_index], X[test_index]

    y_train, y_valid = y[train_index], y[test_index]

    # Convert our data into XGBoost format

    d_train = xgb.DMatrix(X_train, y_train)

    d_valid = xgb.DMatrix(X_valid, y_valid)

    d_test = xgb.DMatrix(test.values)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]



    # Train the model! We pass in a max of 1,600 rounds (with early stopping after 70)

    # and the custom metric (maximize=True tells xgb that higher metric is better)

    mdl = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, feval=gini_xgb, maximize=True, verbose_eval=100)



    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))

    # Predict on our test data

    p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)

    sub['target'] += p_test/kfold
sub.to_csv('kfold_stratified_xgboost.csv', index=False)