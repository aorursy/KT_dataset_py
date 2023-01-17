!pip install deeptables

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as plt

from deeptables.models.deeptable import DeepTable, ModelConfig

from tensorflow.keras.utils import plot_model

import tensorflow as tf

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv', index_col='id')

test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv', index_col='id')

train_copy = train.copy()
train.head()
sns.countplot(y = 'target',data = train, palette = 'Set2')
train.isnull().sum()
test.isnull().sum()
train.isna().sum()*100/len(train)
def heatmap(x, y, size):

    fig, ax = plt.pyplot.subplots()

    

    # Mapping from column names to integer coordinates

    x_labels = [v for v in sorted(x.unique())]

    y_labels = [v for v in sorted(y.unique())]

    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 

    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 

    

    size_scale = 500

    ax.scatter(

        x=x.map(x_to_num), # Use mapping for x

        y=y.map(y_to_num), # Use mapping for y

        s=size * size_scale, # Vector of square sizes, proportional to size parameter

        marker='s' # Use square as scatterplot marker

    )

    

    # Show column labels on the axes

    ax.set_xticks([x_to_num[v] for v in x_labels])

    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')

    ax.set_yticks([y_to_num[v] for v in y_labels])

    ax.set_yticklabels(y_labels)

    

corr = train.corr()

corr = pd.melt(corr.reset_index(), id_vars='index')

corr.columns = ['x', 'y', 'value']

heatmap(

    x=corr['x'],

    y=corr['y'],

    size=corr['value'].abs()

)
fig, ax = plt.pyplot.subplots(1,5, figsize=(30, 8))

for i in range(5): 

    sns.countplot(f'bin_{i}', data= train, ax=ax[i],palette= 'Set2')

    ax[i].set_ylim([0, 600000])

    ax[i].set_title(f'bin_{i}', fontsize=15)

fig.suptitle("Binary Feature Distribution (Train Data)", fontsize=20)

plt.pyplot.show()
fig, ax = plt.pyplot.subplots(1,5, figsize=(30, 8))

for i in range(5): 

    sns.countplot(f'bin_{i}', data= test, ax=ax[i], alpha=0.7,

                 order=test[f'bin_{i}'].value_counts().index,palette= 'Set2')

    ax[i].set_ylim([0, 600000])

    ax[i].set_title(f'bin_{i}', fontsize=15)

fig.suptitle("Binary Feature Distribution (Test Data)", fontsize=20)

plt.pyplot.show()
fig, ax = plt.pyplot.subplots(1,5, figsize=(30, 8))

for i in range(5): 

    sns.countplot(f'bin_{i}', hue='target', data= train, ax=ax[i],palette= 'Set2')

    ax[i].set_ylim([0, 500000])

    ax[i].set_title(f'bin_{i}', fontsize=15)

fig.suptitle("Binary Feature Distribution (Train Data)", fontsize=20)

plt.pyplot.show()
num_cols = test.select_dtypes(exclude=['object']).columns

fig, ax = plt.pyplot.subplots(2,3,figsize=(22,7))

for i, col in enumerate(num_cols):

    plt.pyplot.subplot(2,3,i+1)

    plt.pyplot.xlabel(col, fontsize=9)

    sns.kdeplot(train[col].values, bw=0.5,label='Train')

    sns.kdeplot(test[col].values, bw=0.5,label='Test')

   

plt.pyplot.show() 
target0 = train.loc[train['target'] == 0]

target1 = train.loc[train['target'] == 1]



fig, ax = plt.pyplot.subplots(2,3,figsize=(22,7))

for i, col in enumerate(num_cols):

    plt.pyplot.subplot(2,3,i+1)

    plt.pyplot.xlabel(col, fontsize=9)

    sns.kdeplot(target0[col].values, bw=0.5,label='Target: 0')

    sns.kdeplot(target1[col].values, bw=0.5,label='Target: 1')

    sns.kdeplot(test[col].values, bw=0.5,label='Test')

    

plt.pyplot.show() 
plt.pyplot.figure(figsize=(17, 35)) 

nom_cols = [f'nom_{i}' for i in range(5)]

fig, ax = plt.pyplot.subplots(2,3,figsize=(22,10))



for i, col in enumerate(train[nom_cols]): 

    tmp = pd.crosstab(train[col],train['target'], normalize='index') * 100

    tmp = tmp.reset_index()

    tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)



    ax = plt.pyplot.subplot(2,3,i+1)

    sns.countplot(x=col, data=train, order=list(tmp[col].values) , palette='Set2') 

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

    total = sum([p.get_height() for p in ax.patches])

    for p in ax.patches: # loop to all objects

        height = p.get_height()

        sizes.append(height)

        ax.text(p.get_x()+p.get_width()/2.,

                    height + 2000,

                    '{:1.2f}%'.format(height/total*100),

                    ha="center") 

    ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights





plt.pyplot.subplots_adjust(hspace = 0.5, wspace=.3)

plt.pyplot.show()
plt.pyplot.figure(figsize=(17, 35)) 

nom_cols = [f'nom_{i}' for i in range(5)]

fig, ax = plt.pyplot.subplots(2,3,figsize=(22,10))



for i, col in enumerate(train[nom_cols]): 

    tmp = pd.crosstab(train[col],train['target'], normalize='index') * 100

    tmp = tmp.reset_index()

    tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)



    ax = plt.pyplot.subplot(2,3,i+1)

    sns.countplot(x=col, data=train, order=list(tmp[col].values) , palette='Set2') 

    ax.set_ylabel('Count', fontsize=15) # y axis label

    ax.set_title(f'{col} Distribution by Target', fontsize=18) # title label

    ax.set_xlabel(f'{col} values', fontsize=15) # x axis label



    # twinX - to build a second yaxis

    gt = ax.twinx()

    gt = sns.pointplot(x=col, y='No', data=tmp,

                           order=list(tmp[col].values),

                           color='black', legend=False)

    gt.set_ylim(tmp['Yes'].min()-5,tmp['No'].max()*1.1)

    gt.set_ylabel("Target %False(0)", fontsize=16)

    sizes=[] # Get highest values in y

    total = sum([p.get_height() for p in ax.patches])

    for p in ax.patches: # loop to all objects

        height = p.get_height()

        sizes.append(height)

        ax.text(p.get_x()+p.get_width()/2.,

                    height + 2000,

                    '{:1.2f}%'.format(height/total*100),

                    ha="center") 

    ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights





plt.pyplot.subplots_adjust(hspace = 0.5, wspace=.3)

plt.pyplot.show()
ord_cols = [f'ord_{i}' for i in range(3)]

plt.pyplot.figure(figsize=(17, 35)) 

fig, ax = plt.pyplot.subplots(1,3,figsize=(22,10))



for i, col in enumerate(train[ord_cols]): 

    tmp = pd.crosstab(train[col],train['target'], normalize='index') * 100

    tmp = tmp.reset_index()

    tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)



    ax = plt.pyplot.subplot(2,3,i+1)

    sns.countplot(x=col, data=train, order=list(tmp[col].values) , palette='Set2') 

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

    total = sum([p.get_height() for p in ax.patches])

    for p in ax.patches: # loop to all objects

        height = p.get_height()

        sizes.append(height)

        ax.text(p.get_x()+p.get_width()/2.,

                    height + 2000,

                    '{:1.2f}%'.format(height/total*100),

                    ha="center") 

    ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights





plt.pyplot.subplots_adjust(hspace = 0.5, wspace=.3)

plt.pyplot.show()
plt.pyplot.figure(figsize=(17, 35)) 

ord_cols = [f'ord_{i}' for i in range(3)]

fig, ax = plt.pyplot.subplots(1,3,figsize=(22,10))



for i, col in enumerate(train[ord_cols]): 

    tmp = pd.crosstab(train[col],train['target'], normalize='index') * 100

    tmp = tmp.reset_index()

    tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)



    ax = plt.pyplot.subplot(2,3,i+1)

    sns.countplot(x=col, data=train, order=list(tmp[col].values) , palette='Set2') 

    ax.set_ylabel('Count', fontsize=15) # y axis label

    ax.set_title(f'{col} Distribution by Target', fontsize=18) # title label

    ax.set_xlabel(f'{col} values', fontsize=15) # x axis label



    # twinX - to build a second yaxis

    gt = ax.twinx()

    gt = sns.pointplot(x=col, y='No', data=tmp,

                           order=list(tmp[col].values),

                           color='black', legend=False)

    gt.set_ylim(tmp['Yes'].min()-5,tmp['No'].max()*1.1)

    gt.set_ylabel("Target %False(0)", fontsize=16)

    sizes=[] # Get highest values in y

    total = sum([p.get_height() for p in ax.patches])

    for p in ax.patches: # loop to all objects

        height = p.get_height()

        sizes.append(height)

        ax.text(p.get_x()+p.get_width()/2.,

                    height + 2000,

                    '{:1.2f}%'.format(height/total*100),

                    ha="center") 

    ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights





plt.pyplot.subplots_adjust(hspace = 0.5, wspace=.3)

plt.pyplot.show()
train = train_copy.copy()



ord_order = [

    [1.0, 2.0, 3.0],

    ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'],

    ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']

]



for i in range(1, 3):

    ord_order_dict = {i : j for j, i in enumerate(ord_order[i])}

    train[f'ord_{i}_en'] = train[f'ord_{i}'].fillna('NULL').map(ord_order_dict)

    test[f'ord_{i}_en'] = test[f'ord_{i}'].fillna('NULL').map(ord_order_dict)

    

for i in range(3, 6):

    ord_order_dict = {i : j for j, i in enumerate(sorted(list(set(list(train[f'ord_{i}'].dropna().unique()) + list(test[f'ord_{i}'].dropna().unique())))))}

    train[f'ord_{i}_en'] = train[f'ord_{i}'].fillna('NULL').map(ord_order_dict)

    test[f'ord_{i}_en'] = test[f'ord_{i}'].fillna('NULL').map(ord_order_dict)


cat_cols = [c for c in train.columns if '_en' not in c and c != 'target']

train[cat_cols] = train[cat_cols].astype('category')

test[cat_cols] = test[cat_cols].astype('category')
y = train['target']

X = train

X.drop(['target'], axis=1, inplace=True)



X_test = test

#X_test.drop(['id'], axis=1, inplace=True)

print(f'X shape: {X.shape}, y shape: {y.shape}, X_test shape: {X_test.shape}')
n_folds=3 #for faster demo, in the competition is 50

epochs=1 #for faster demo, in the competition is 100

batch_size=128


conf = ModelConfig(

    dnn_params={

        'hidden_units':((300, 0.3, True),(300, 0.3, True),), #hidden_units

        'dnn_activation':'relu',

    },

    fixed_embedding_dim=True,

    embeddings_output_dim=20,

    nets =['linear','cin_nets','dnn_nets'],

    stacking_op = 'add',

    output_use_bias = False,

    cin_params={

       'cross_layer_size': (200, 200),

       'activation': 'relu',

       'use_residual': False,

       'use_bias': True,

       'direct': True, 

       'reduce_D': False,

    },

)



dt = DeepTable(config = conf)

oof_proba, eval_proba, test_prob = dt.fit_cross_validation(

    X, y, X_eval=None, X_test=X_test, 

    num_folds=n_folds, stratified=False, iterators=None, 

    batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[], n_jobs=1)

submission = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv')

submission['target'] = test_prob

submission.to_csv('submission_linear_dnn_cin_kfold50.csv',index=False)