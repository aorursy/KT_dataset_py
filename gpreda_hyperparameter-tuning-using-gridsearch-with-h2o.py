import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.metrics import roc_auc_score

import time

import itertools

import h2o

from h2o.estimators.gbm import H2OGradientBoostingEstimator

from h2o.grid.grid_search import H2OGridSearch

%matplotlib inline
#DATA SPLIT IN TRAIN/VALIDATION/TEST

TRAIN_SIZE = 0.60  

VALID_SIZE = 0.20 

RANDOM_STATE = 2018

IS_LOCAL = False

import os

if(IS_LOCAL):

    PATH="../input/default-of-credit-card-clients-dataset"

else:

    PATH="../input"

print(os.listdir(PATH))
h2o.init()
data_df = h2o.import_file(PATH+"/UCI_Credit_Card.csv", destination_frame="data_df")
data_df.describe()
df_group=data_df.group_by("default.payment.next.month").count()

df_group.get_frame()
features = [f for f in data_df.columns if f not in ['default.payment.next.month']]



i = 0

t0 = data_df[data_df['default.payment.next.month'] == 0].as_data_frame()

t1 = data_df[data_df['default.payment.next.month'] == 1].as_data_frame()



sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(6,4,figsize=(16,24))



for feature in features:

    i += 1

    plt.subplot(6,4,i)

    sns.kdeplot(t0[feature], bw=0.5,label="Not default")

    sns.kdeplot(t1[feature], bw=0.5,label="Default")

    plt.xlabel(feature, fontsize=12)

    locs, labels = plt.xticks()

    plt.tick_params(axis='both', which='major', labelsize=12)

plt.show();
d_df = data_df.as_data_frame()

plt.figure(figsize = (14,6))

plt.title('Amount of credit limit - Density Plot')

sns.set_color_codes("pastel")

sns.distplot(d_df['LIMIT_BAL'],kde=True,bins=200, color="blue")

plt.show()
d_df['LIMIT_BAL'].value_counts().shape
d_df['LIMIT_BAL'].value_counts().head(5)
class_0 = d_df.loc[d_df['default.payment.next.month'] == 0]["LIMIT_BAL"]

class_1 = d_df.loc[d_df['default.payment.next.month'] == 1]["LIMIT_BAL"]

plt.figure(figsize = (14,6))

plt.title('Default amount of credit limit  - grouped by Payment Next Month (Density Plot)')

sns.set_color_codes("pastel")

sns.distplot(class_1,kde=True,bins=200, color="red")

sns.distplot(class_0,kde=True,bins=200, color="green")

plt.show()
var = ['PAY_0',

       'PAY_2', 

       'PAY_3', 

       'PAY_4', 

       'PAY_5',

       'PAY_6']



for v in var:

    fig, ax = plt.subplots(ncols=1, figsize=(14,6))

    s = sns.boxplot(ax = ax, x=v, y="LIMIT_BAL", hue="default.payment.next.month",data=d_df, palette="PRGn",showfliers=False)

    plt.show();
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

s = sns.boxplot(ax = ax1, 

                x="SEX", 

                y="LIMIT_BAL", 

                hue="SEX",data=d_df, 

                palette="PRGn",

                showfliers=True)

s = sns.boxplot(ax = ax2, 

                x="SEX", 

                y="LIMIT_BAL", 

                hue="SEX",data=d_df, 

                palette="PRGn",

                showfliers=False)

plt.show();
var = ['BILL_AMT1',

       'BILL_AMT2',

       'BILL_AMT3',

       'BILL_AMT4',

       'BILL_AMT5',

       'BILL_AMT6']



plt.figure(figsize = (8,8))

plt.title('Amount of bill statement (Apr-Sept) \ncorrelation plot (Pearson)')

corr = data_df[var].cor().as_data_frame()

corr.index = var

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,vmin=-1, vmax=1)

plt.show()
var = ['PAY_AMT1', 

       'PAY_AMT2', 

       'PAY_AMT3', 

       'PAY_AMT4', 

       'PAY_AMT5']



plt.figure(figsize = (8,8))

plt.title('Amount of previous payment (Apr-Sept) \ncorrelation plot (Pearson)')

corr = data_df[var].cor().as_data_frame()

corr.index = var

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,vmin=-1, vmax=1)

plt.show()
var = ['PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']



plt.figure(figsize = (8,8))

plt.title('Repayment status (Apr-Sept) \ncorrelation plot (Pearson)')

corr = data_df[var].cor().as_data_frame()

corr.index = var

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,vmin=-1, vmax=1)

plt.show()
def boxplot_variation(feature1, feature2, feature3, width=16):

    fig, ax1 = plt.subplots(ncols=1, figsize=(width,6))

    s = sns.boxplot(ax = ax1, x=feature1, y=feature2, hue=feature3,

                data=d_df, palette="PRGn",showfliers=False)

    s.set_xticklabels(s.get_xticklabels(),rotation=90)

    plt.show();
boxplot_variation('MARRIAGE','AGE', 'SEX',8)
boxplot_variation('EDUCATION','AGE', 'MARRIAGE',12)
boxplot_variation('AGE','LIMIT_BAL', 'SEX',16)
boxplot_variation('MARRIAGE','LIMIT_BAL', 'EDUCATION',12)
data_df['EDUCATION_SEX'] = data_df['EDUCATION'] + "_" + data_df['SEX']

data_df['SEX_MARRIAGE'] = data_df['SEX'] + "_" + data_df['MARRIAGE']

data_df['EDUCATION_MARRIAGE'] = data_df['EDUCATION'] + "_" + data_df['MARRIAGE']

data_df['EDUCATION_MARRIAGE_SEX'] = data_df['EDUCATION'] + "_" + data_df['MARRIAGE'] + "_" + data_df['SEX']
data_df.shape
train_df, valid_df, test_df = data_df.split_frame(ratios=[TRAIN_SIZE, VALID_SIZE], seed=2018)

target = "default.payment.next.month"

train_df[target] = train_df[target].asfactor()

valid_df[target] = valid_df[target].asfactor()

test_df[target] = test_df[target].asfactor()

print("Number of rows in train, valid and test set : ", train_df.shape[0], valid_df.shape[0], test_df.shape[0])
# define the predictor list - all the features analyzed before (all columns but 'default.payment.next.month')

predictors = features

# initialize the H2O GBM 

gbm = H2OGradientBoostingEstimator()

# train with the initialized model

gbm.train(x=predictors, y=target, training_frame=train_df)
gbm.summary()
print(gbm.model_performance(train_df))
print(gbm.model_performance(valid_df))
tuned_gbm  = H2OGradientBoostingEstimator(

    ntrees = 2000,

    learn_rate = 0.02,

    stopping_rounds = 25,

    stopping_metric = "AUC",

    col_sample_rate = 0.65,

    sample_rate = 0.65,

    seed = RANDOM_STATE

)      

tuned_gbm.train(x=predictors, y=target, training_frame=train_df, validation_frame=valid_df)
tuned_gbm.model_performance(valid_df).auc()
grid_search_gbm = H2OGradientBoostingEstimator(

    stopping_rounds = 25,

    stopping_metric = "AUC",

    col_sample_rate = 0.65,

    sample_rate = 0.65,

    seed = RANDOM_STATE

) 



hyper_params = {

    'learn_rate':[0.01, 0.02, 0.03],

    'max_depth':[4,8,16,24],

    'ntrees':[50, 250, 1000]}



grid = H2OGridSearch(grid_search_gbm, hyper_params,

                         grid_id='depth_grid',

                         search_criteria={'strategy': "Cartesian"})

#Train grid search

grid.train(x=predictors, 

           y=target,

           training_frame=train_df,

           validation_frame=valid_df)
grid_sorted = grid.get_grid(sort_by='auc',decreasing=True)
print(grid_sorted)
best_gbm = grid_sorted.models[0]
print(best_gbm)
best_gbm.varimp_plot()
pred_val = (best_gbm.predict(test_df[predictors])[0]).as_data_frame()

true_val = (test_df[target]).as_data_frame()

prediction_auc = roc_auc_score(pred_val, true_val)

prediction_auc