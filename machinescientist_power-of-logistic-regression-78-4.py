'''Import basic python modules'''

import pandas as pd

import numpy as np

import string

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from sklearn.metrics import roc_auc_score as auc

from sklearn.linear_model import LogisticRegression



'''import visualization libraries'''

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("ticks")

%matplotlib inline



'''Plotly visualization library .'''

import plotly.offline as py

import plotly.graph_objs as go

import plotly.tools as tls

py.init_notebook_mode(connected=True)



# markdown display formatted output

from IPython.display import Markdown

def bold(string):

    display(Markdown(string))
# Load data

train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')



print(train.shape)

print(test.shape)
train.sample(10)
## Variable description

def description(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.iloc[0].values

    summary['Second Value'] = df.iloc[1].values

    summary['Third Value'] = df.iloc[2].values

    return summary

bold('**Variable Description of  train Data:**')

description(train)
def replace_nan(data):

    for column in data.columns:

        if data[column].isna().sum() > 0:

            data[column] = data[column].fillna(data[column].mode()[0])





replace_nan(train)

replace_nan(test)
total = len(train)

plt.figure(figsize=(10,6))



g = sns.countplot(x='target', data=train, palette='coolwarm')

g.set_title("TARGET DISTRIBUTION", fontsize = 20)

g.set_xlabel("Target Vaues", fontsize = 15)

g.set_ylabel("Count", fontsize = 15)

sizes=[] # Get highest values in y

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=14) 

g.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights



plt.show()
bin_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']



import matplotlib.gridspec as gridspec # to do the grid of plots

grid = gridspec.GridSpec(3, 2) # The grid of chart

plt.figure(figsize=(16,20)) # size of figure



# loop to get column and the count of plots



for n, col in enumerate(train[bin_cols]): 

    ax = plt.subplot(grid[n]) # feeding the figure of grid

    sns.countplot(x=col, data=train, hue='target', palette='Set1') 

    ax.set_ylabel('Count', fontsize=15) # y axis label

    ax.set_title(f'{col} Distribution by Target', fontsize=18) # title label

    ax.set_xlabel(f'{col} values', fontsize=15) # x axis label

    sizes=[] # Get highest values in y

    for p in ax.patches: # loop to all objects

        height = p.get_height()

        sizes.append(height)

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/total*100),

                ha="center", fontsize=14) 

    ax.set_ylim(0, max(sizes) * 1.15) #set y limit based on highest heights

    

plt.show()
nom_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']





def ploting_cat_fet(df, cols, vis_row=5, vis_col=2):

    

    grid = gridspec.GridSpec(vis_row,vis_col) # The grid of chart

    plt.figure(figsize=(17, 35)) # size of figure



    # loop to get column and the count of plots

    for n, col in enumerate(train[cols]): 

        tmp = pd.crosstab(train[col], train['target'], normalize='index') * 100

        tmp = tmp.reset_index()

        tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)



        ax = plt.subplot(grid[n]) # feeding the figure of grid

        sns.countplot(x=col, data=train, order=list(tmp[col].values) , palette='Set3') 

        ax.set_ylabel('Count', fontsize=15) # y axis label

        ax.set_title(f'{col} Distribution by Target', fontsize=18) # title label

        ax.set_xlabel(f'{col} values', fontsize=15) # x axis label



        # twinX - to build a second yaxis

        gt = ax.twinx()

        gt = sns.pointplot(x=col, y='Yes', data=tmp,

                           order=list(tmp[col].values),

                           color='black', legend=False)

        gt.set_ylim(tmp['Yes'].min()-5,tmp['Yes'].max()*1.1)

        gt.set_ylabel("Target %True(1)", fontsize=16)

        sizes=[] # Get highest values in y

        for p in ax.patches: # loop to all objects

            height = p.get_height()

            sizes.append(height)

            ax.text(p.get_x()+p.get_width()/2.,

                    height + 3,

                    '{:1.2f}%'.format(height/total*100),

                    ha="center", fontsize=14) 

        ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights





    plt.subplots_adjust(hspace = 0.5, wspace=.3)

    plt.show()
ploting_cat_fet(train, nom_cols, vis_row=5, vis_col=2)
ord_cols = ['ord_0', 'ord_1', 'ord_2', 'ord_3']



#Ploting

ploting_cat_fet(train, ord_cols, vis_row=5, vis_col=2)
date_cols = ['day', 'month']



# Calling the plot function with date columns

ploting_cat_fet(train, date_cols, vis_row=5, vis_col=2)
train['ord_5_ot'] = 'Others'

train.loc[train['ord_5'].isin(train['ord_5'].value_counts()[:25].sort_index().index), 'ord_5_ot'] = train['ord_5']
target = train['target']

train_id = train['id']

test_id = test['id']

train.drop(['target', 'id','ord_5_ot'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)



print(train.shape)

print(test.shape)

print(target.shape)
traintest = pd.concat([train, test])

dummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)

train_ohe = dummies.iloc[:train.shape[0], :]

test_ohe = dummies.iloc[train.shape[0]:, :]



print(train_ohe.shape)

print(test_ohe.shape)
%%time

'''Covert dataframe to spare matrix'''

train_ohe = train_ohe.sparse.to_coo().tocsr()

test_ohe = test_ohe.sparse.to_coo().tocsr()

type(train_ohe)
%%time



# Model

def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):

    kf = KFold(n_splits=5)

    fold_splits = kf.split(train, target)

    cv_scores = []

    pred_full_test = 0

    pred_train = np.zeros((train.shape[0]))

    i = 1

    for dev_index, val_index in fold_splits:

        print('Started ' + label + ' fold ' + str(i) + '/5')

        dev_X, val_X = train[dev_index], train[val_index]

        dev_y, val_y = target[dev_index], target[val_index]

        params2 = params.copy()

        pred_val_y, pred_test_y = model_fn(dev_X, dev_y, val_X, val_y, test, params2)

        pred_full_test = pred_full_test + pred_test_y

        pred_train[val_index] = pred_val_y

        if eval_fn is not None:

            cv_score = eval_fn(val_y, pred_val_y)

            cv_scores.append(cv_score)

            print(label + ' cv score {}: {}'.format(i, cv_score))

        i += 1

    print('{} cv scores : {}'.format(label, cv_scores))

    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))

    print('{} cv std score : {}'.format(label, np.std(cv_scores)))

    pred_full_test = pred_full_test / 5.0

    results = {'label': label,

              'train': pred_train, 'test': pred_full_test,

              'cv': cv_scores}

    return results
def runLR(train_X, train_y, test_X, test_y, test_X2, params):

    print('Train LR')

    model = LogisticRegression(**params)

    model.fit(train_X, train_y)

    print('Predict 1/2')

    pred_test_y = model.predict_proba(test_X)[:, 1]

    print('Predict 2/2')

    pred_test_y2 = model.predict_proba(test_X2)[:, 1]

    return pred_test_y, pred_test_y2





lr_params = {'solver': 'liblinear', 'C':  0.1, 'max_iter': 1000}

results = run_cv_model(train_ohe, test_ohe, target, runLR, lr_params, auc, 'lr')
%%time

submission = pd.DataFrame({'id': test_id, 'target': results['test']})

submission.to_csv('submission.csv', index=False)