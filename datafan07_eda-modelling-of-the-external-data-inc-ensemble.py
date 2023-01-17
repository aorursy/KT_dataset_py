import pandas as pd

import numpy as np



import matplotlib

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



import seaborn as sns

import plotly.express as px



import os

import random

import re

import math

import time



import warnings



warnings.filterwarnings('ignore') # Disabling warnings for clearer outputs





seed_val = 42

random.seed(seed_val)

np.random.seed(seed_val)
# Setting color palette.

black_red = [

    '#1A1A1D', '#4E4E50', '#C5C6C7', '#6F2232', '#950740', '#C3073F'

]



# Setting plot styling.

plt.style.use('fivethirtyeight')
# loading datasets



train = pd.read_csv('../input/melanomaextendedtabular/external_upsampled_tabular.csv')

test = pd.read_csv('../input/melanomaextendedtabular/test_tabular.csv')

sample = pd.read_csv('../input/melanomaextendedtabular/sample_submission.csv')
train.sample(5)
# checking column names



print(

    f'Train data has {train.shape[1]} features, {train.shape[0]} observations and Test data {test.shape[1]} features, {test.shape[0]} observations.\nTrain features are:\n{train.columns.tolist()}\nTest features are:\n{test.columns.tolist()}'

)
# renaming column names for easier use



train.columns = [

    'img_name',  'sex', 'age', 'location', 'target','width','height'

]



test.columns = ['img_name', 'sex', 'age', 'location','width','height']



# Checking missing values:



def missing_percentage(df):



    total = df.isnull().sum().sort_values(

        ascending=False)[df.isnull().sum().sort_values(ascending=False) != 0]

    percent = (df.isnull().sum().sort_values(ascending=False) / len(df) *

               100)[(df.isnull().sum().sort_values(ascending=False) / len(df) *

                     100) != 0]

    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])





missing_train = missing_percentage(train)

missing_test = missing_percentage(test)



fig, ax = plt.subplots(1, 2, figsize=(16, 6))



sns.barplot(x=missing_train.index,

            y='Percent',

            data=missing_train,

            palette=black_red,

            ax=ax[0])



sns.barplot(x=missing_test.index,

            y='Percent',

            data=missing_test,

            palette=black_red,

            ax=ax[1])



ax[0].set_title('Train Data Missing Values')

ax[1].set_title('Test Data Missing Values')



plt.show()
# Creating a customized chart and giving in figsize etc.



fig = plt.figure(constrained_layout=True, figsize=(20, 9))



# Creating a grid:



grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)



ax1 = fig.add_subplot(grid[0, :2])



# Set the title.



ax1.set_title('Gender Distribution')



sns.countplot(train.sex.sort_values(ignore_index=True),

              alpha=0.9,

              ax=ax1,

              color='#C3073F',

              label='Train')

sns.countplot(test.sex.sort_values(ignore_index=True),

              alpha=0.7,

              ax=ax1,

              color='#1A1A1D',

              label='Test')

ax1.legend()



# Customizing the second grid.



ax2 = fig.add_subplot(grid[0, 2:])



# Plot the countplot.



sns.countplot(train.location,

              alpha=0.9,

              ax=ax2,

              color='#C3073F',

              label='Train',

              order=train['location'].value_counts().index)

sns.countplot(test.location,

              alpha=0.7,

              ax=ax2,

              color='#1A1A1D',

              label='Test',

              order=test['location'].value_counts().index), ax2.set_title(

                  'Anatom Site Distribution')



ax2.legend()

plt.xticks(rotation=20)



# Customizing the third grid.



ax3 = fig.add_subplot(grid[1, :])



# Set the title.



ax3.set_title('Age Distribution')



# Plot the histogram.



sns.distplot(train.age, ax=ax3, label='Train', color='#C3073F')

sns.distplot(test.age, ax=ax3, label='Test', color='#1A1A1D')



ax3.legend()



plt.show()
# Filling missing  values with 'unknown' and '-1' tags:



for df in [train, test]:

    df['location'].fillna('unknown', inplace=True)

    

train['sex'].fillna('unknown', inplace=True)



train['age'].fillna(-1, inplace=True)
# Double checking:



ids_train = train.location.values

ids_test = test.location.values

ids_train_set = set(ids_train)

ids_test_set = set(ids_test)



location_not_overlap = list(ids_train_set.symmetric_difference(ids_test_set))

n_overlap = len(location_not_overlap)

if n_overlap == 0:

    print(

        f'There are no different body parts occuring between train and test set...'

    )

else:

    print('There are some non-overlapping values between train and test set!\n')

    print(f'Different ones are:\n{pd.Series(np.setdiff1d((train.location.value_counts().index), pd.Series(test.location.value_counts().index)))}')

# merging detailed torso approach to torso only

train.replace(['anterior torso','lateral torso','posterior torso'], 'torso', inplace=True)
# Creating a customized chart and giving in figsize etc.



fig = plt.figure(constrained_layout=True, figsize=(20, 9))

# Creating a grid

grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)



# Customizing the first grid.



ax1 = fig.add_subplot(grid[1, :2])

# Set the title.

ax1.set_title('Scanned Body Parts - Female')



# Plot:



sns.countplot(

    train[train['sex'] == 'female'].location.sort_values(ignore_index=True),

    alpha=0.9,

    ax=ax1,

    color='#C3073F',

    label='Female',

    order=train['location'].value_counts().index)

ax1.legend()

plt.xticks(rotation=20)



# Customizing the second grid.



ax2 = fig.add_subplot(grid[1, 2:])



# Set the title.



ax2.set_title('Scanned Body Parts - Male')



# Plot.



sns.countplot(

    train[train['sex'] == 'male'].location.sort_values(ignore_index=True),

    alpha=0.9,

    ax=ax2,

    color='#1A1A1D',

    label='Male',

    order=train['location'].value_counts().index)



ax2.legend()

plt.xticks(rotation=20)



# Customizing the third grid.



ax3 = fig.add_subplot(grid[0, :])



# Set the title.



ax3.set_title('Malignant Ratio Per Body Part')



# Plot.



loc_freq = train.groupby('location')['target'].mean().sort_values(

    ascending=False)

sns.barplot(x=loc_freq.index, y=loc_freq, palette=black_red, ax=ax3)



ax3.legend()





plt.show()
# Plotting interactive sunburst:



fig = px.sunburst(data_frame=train,

                  path=['target', 'sex', 'location'],

                  color='sex',

                  color_discrete_sequence=black_red,

                  maxdepth=-1,

                  title='Sunburst Chart Benign/Malignant > Sex > Location')



fig.update_traces(textinfo='label+percent parent')

fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

fig.show()
# Plotting age vs sex vs target:



fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.lineplot(x='age',

             y='target',

             data=train,

             ax=ax[0],

             hue='sex',

             palette=black_red[2:5],

             ci=None)

sns.boxplot(x='target',

            y='age',

            data=train,

            ax=ax[1],

            hue='sex',

            palette=black_red[2:5]

           )



plt.legend(loc='upper right')



ax[0].set_title('Malignant Scan Frequency by Age')

ax[1].set_title('Scan Results by Age and Sex')



plt.show()
# Creating a customized chart and giving in figsize etc.



# Plotting age dist vs target and age dist vs datasets



fig = plt.figure(constrained_layout=True, figsize=(20, 12))



# Creating a grid



grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)



# Customizing the first grid.



ax1 = fig.add_subplot(grid[0, :2])



# Set the title.



ax1.set_title('Age Distribution by Scan Outcome')



# Plot



ax1.legend()



sns.kdeplot(train[train['target'] == 0]['age'],

            shade=True,

            ax=ax1,

            color='#1A1A1D',

            label='Benign')

sns.kdeplot(train[train['target'] == 1]['age'],

            shade=True,

            ax=ax1,

            color='#C3073F',

            label='Malignant')



# Customizing second grid.



ax2 = fig.add_subplot(grid[0, 2:])



# Set the title.



ax2.set_title('Age Distribution by Train/Test Observations')



# Plot.



sns.kdeplot(train.age, label='Train', shade=True, ax=ax2, color='#1A1A1D')

sns.kdeplot(test.age, label='Test', shade=True, ax=ax2, color='#C3073F')



ax2.legend()



# Customizing third grid.



ax3 = fig.add_subplot(grid[1, :])



# Set the title.



ax3.set_title('Age Distribution by Gender')



# Plot



sns.distplot(train[train.sex == 'female'].age,

             ax=ax3,

             label='Female',

             color='#C3073F')

sns.distplot(train[train.sex == 'male'].age,

             ax=ax3,

             label='Male',

             color='#1A1A1D')

ax3.legend()



plt.show()
# getting temporary resolution feature



train['res']= train['width'].astype(str)+'x'+train['height'].astype(str)

test['res']= test['width'].astype(str)+'x'+test['height'].astype(str)
# Creating a customized chart and giving in figsize etc.



fig = plt.figure(constrained_layout=True, figsize=(20, 12))



# Creating a grid



grid = gridspec.GridSpec(ncols=4, nrows=3, figure=fig)



# Customizing the first grid.



ax1 = fig.add_subplot(grid[0, :2])



# Set the title.



ax1.set_title('Scan Image Resolutions of Train Set')



# Plot.



tres = train.res.value_counts().rename_axis('res').reset_index(name='count')

tres = tres[tres['count'] > 10]

sns.barplot(x='res', y='count', data=tres, palette=black_red, ax=ax1)

plt.xticks(rotation=60)



ax1.legend()



# Customizing the second grid.



ax2 = fig.add_subplot(grid[0, 2:])



# Set the title.



ax2.set_title('Scan Image Resolutions of Test Set')



# Plot



teres = test.res.value_counts().rename_axis('res').reset_index(name='count')

teres = teres[teres['count'] > 10]

sns.barplot(x='res', y='count', data=teres, palette=black_red, ax=ax2)

plt.xticks(rotation=30)

ax2.legend()



# Customizing the third grid.



ax3 = fig.add_subplot(grid[1, :])



# Set the title.



ax3.set_title('Scan Image Resolutions by Target')



# Plot.



sns.countplot(x='res',

              hue='target',

              data=train,

              order=train.res.value_counts().iloc[:15].index,

              palette=black_red[2:4],

              ax=ax3)

ax3.legend()



# Customizing the last grid.



ax4 = fig.add_subplot(grid[2, :])



# Set the title.



ax4.set_title('Malignant Scan Result Frequency by Image Resolution')



# Plot.



res_freq = train.groupby('res')['target'].mean()

res_freq = res_freq[(res_freq > 0) & (res_freq < 1)]

sns.lineplot(x=res_freq.index, y=res_freq, color='#C3073F', ax=ax4)

ax4.legend()

plt.xticks(rotation=60)



plt.show()
# getting rid of temporary features



train.drop(['res'], axis=1, inplace=True)

test.drop(['res'], axis=1, inplace=True)
#creating dummy variables for categorical sex data



sex_dummies = pd.get_dummies(train['sex'], prefix='sex')

train = pd.concat([train, sex_dummies], axis=1)



sex_dummies = pd.get_dummies(test['sex'], prefix='sex')

test = pd.concat([test, sex_dummies], axis=1)



train.drop(['sex'], axis=1, inplace=True)

test.drop(['sex'], axis=1, inplace=True)
# getting dummy variables for location on train set



anatom_dummies = pd.get_dummies(train['location'], prefix='anatom')

train = pd.concat([train, anatom_dummies], axis=1)



# getting dummy variables for location on test set



anatom_dummies = pd.get_dummies(test['location'], prefix='anatom')

test = pd.concat([test, anatom_dummies], axis=1)



# dropping useless columns



train.drop('location', axis=1, inplace=True)

test.drop(['location'], axis=1, inplace=True)
# dropping redundant columns for both dataset



for df in [train, test]:

    df.drop('img_name', axis=1, inplace=True)
# importing basic modelling stuff



import xgboost as xgb



from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, cross_validate

from sklearn.metrics import roc_auc_score, roc_curve
# creating train variables



X = train.drop('target', axis=1)

y = train.target
# taking 15% of the training data as holdout



X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    test_size=0.15,

                                                    stratify=y,

                                                    random_state=42)



# 5 fold stratify for cv



cv = StratifiedKFold(5, shuffle=True, random_state=42)
# setting model hyperparameters, didn't include fine tuning here because of timing reasons...



xg = xgb.XGBClassifier(

    n_estimators=750,

    min_child_weight=0.81,

    learning_rate=0.025,

    max_depth=2,

    subsample=0.80,

    colsample_bytree=0.42,

    gamma=0.10,

    random_state=42,

    n_jobs=-1,

)
estimators = [xg]
# cross validation scheme



def model_check(X_train, y_train, estimators, cv):

    model_table = pd.DataFrame()



    row_index = 0

    for est in estimators:



        MLA_name = est.__class__.__name__

        model_table.loc[row_index, 'Model Name'] = MLA_name



        cv_results = cross_validate(est,

                                    X_train,

                                    y_train,

                                    cv=cv,

                                    scoring='roc_auc',

                                    return_train_score=True,

                                    n_jobs=-1)



        model_table.loc[row_index,

                        'Train roc Mean'] = cv_results['train_score'].mean()

        model_table.loc[row_index,

                        'Test roc Mean'] = cv_results['test_score'].mean()

        model_table.loc[row_index, 'Test Std'] = cv_results['test_score'].std()

        model_table.loc[row_index, 'Time'] = cv_results['fit_time'].mean()



        row_index += 1



    model_table.sort_values(by=['Test roc Mean'],

                            ascending=False,

                            inplace=True)



    return model_table
# display cv results



raw_models = model_check(X_train, y_train, estimators, cv)

display(raw_models.style.background_gradient(cmap='twilight', axis=1))
# fitting train data



xg.fit(X_train, y_train)



# predicting on holdout set

validation = xg.predict_proba(X_test)[:, 1]



# checking results on validation set

roc_auc_score(y_test, validation)
# finding feature importances and creating new dataframe basen on them



feature_importance = xg.get_booster().get_score(importance_type='weight')



keys = list(feature_importance.keys())

values = list(feature_importance.values())



importance = pd.DataFrame(data=values, index=keys,

                          columns=['score']).sort_values(by='score',

                                                         ascending=False)

fig, ax = plt.subplots(figsize=(16, 10))

sns.barplot(x=importance.score.iloc[:20],

            y=importance.index[:20],

            orient='h',

            palette='Reds_r')

ax.set_title('Feature Importances')

plt.show()
# creating adversarial training set



adv_train = train.copy()

adv_train.drop('target', axis=1, inplace=True)

adv_test = test.copy()



adv_train['dataset_label'] = 0

adv_test['dataset_label'] = 1



adv_master = pd.concat([adv_train, adv_test], axis=0)



adv_X = adv_master.drop('dataset_label', axis=1)

adv_y = adv_master['dataset_label']
# holdout set for adv



adv_X_train, adv_X_test, adv_y_train, adv_y_test = train_test_split(adv_X,

                                                    adv_y,

                                                    test_size=0.4,

                                                    stratify=adv_y,

                                                    random_state=42)
xg_adv = xgb.XGBClassifier(

    random_state=42,

    n_jobs=-1,

)



# Fitting train data



xg_adv.fit(adv_X_train, adv_y_train)



# Predicting on holdout set

validation = xg_adv.predict_proba(adv_X_test)[:,1]
def plot_roc_feat(y_trues, y_preds, labels, est, x_max=1.0):

    

    """ A function for displaying roc/auc curve and feature importances. """

    

    fig, ax = plt.subplots(1,2, figsize=(16,6))

    for i, y_pred in enumerate(y_preds):

        y_true = y_trues[i]

        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

        auc = roc_auc_score(y_true, y_pred)

        ax[0].plot(fpr, tpr, label='%s; AUC=%.3f' % (labels[i], auc), marker='o', markersize=1)



    ax[0].legend()

    ax[0].grid()

    ax[0].plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20), linestyle='--')

    ax[0].set_title('ROC curve')

    ax[0].set_xlabel('False Positive Rate')

    ax[0].set_xlim([-0.01, x_max])

    _ = ax[0].set_ylabel('True Positive Rate')

    

    

    feature_importance = est.get_booster().get_score(importance_type='weight')



    keys = list(feature_importance.keys())

    values = list(feature_importance.values())



    importance = pd.DataFrame(data=values, index=keys,

                          columns=['score']).sort_values(by='score',

                                                         ascending=False)

    

    sns.barplot(x=importance.score.iloc[:20],

            y=importance.index[:20],

            orient='h',

            palette=black_red, ax=ax[1])

    ax[1].set_title('Feature Importances')
plot_roc_feat(

    [adv_y_test],

    [validation],

    ['Baseline'],

    xg_adv

)
# dropping features for better randomness



adv_X.drop(['sex_unknown', 'height', 'width'], axis=1, inplace=True)





adv_X_train, adv_X_test, adv_y_train, adv_y_test = train_test_split(adv_X,

                                                    adv_y,

                                                    test_size=0.4,

                                                    stratify=adv_y,

                                                    random_state=42)



# fitting train data



xg_adv.fit(adv_X_train, adv_y_train)



# predicting on holdout set

validation = xg_adv.predict_proba(adv_X_test)[:,1]
plot_roc_feat(

    [adv_y_test],

    [validation],

    ['Baseline'],

    xg_adv

)
# dropping features from original train set



X_train.drop(['sex_unknown', 'width','height'], axis=1, inplace=True)



test.drop(['width','height'], axis=1, inplace=True)
# setting parameters



xg= xgb.XGBClassifier(

    n_estimators=750,

    learning_rate=0.015,

    min_child_weight= 59,

    max_delta_step= 6,

    max_depth= 6,

    subsample= 0.751,

    colsample_bytree= 0.8595,

    gamma= 0,

    reg_lambda= 37,

    random_state=42,

    n_jobs=-1,

)
# display cv results



raw_models = model_check(X_train, y_train, [xg], cv)

display(raw_models.style.background_gradient(cmap='twilight', axis=1))
# fitting and predicting



xg.fit(X_train, y_train)



predictions = xg.predict_proba(test)[:, 1]



meta_df = pd.DataFrame(columns=['image_name', 'target'])



# assigning predictions on submission df



meta_df['image_name'] = sample['image_name']

meta_df['target'] = predictions



# creating submission csv file



meta_df.to_csv('external_tabular_predicts.csv', header=True, index=False)
# loading predictions from csv file and ensemble them



effnet = pd.read_csv('../input/blended-effnets-from-previous-notebook/blended_effnets.csv')

meta = pd.read_csv('./external_tabular_predicts.csv')





sample['target'] = (

                           

                           effnet['target'] * 0.9 +

                           meta['target'] * 0.1 

                          

                          )



sample.to_csv('external_meta_ensembled.csv', header=True, index=False)
# display auc distribution



fig, ax = plt.subplots(figsize=(16,6))

sns.distplot(sample['target'], hist_kws={

                 'rwidth': 0.75,

                 'edgecolor': 'black',

                 'alpha': 0.3

             }, color='#C3073F')

ax.set_title('Final Predictions')

plt.show()