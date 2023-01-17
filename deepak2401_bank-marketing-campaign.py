import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

from inspect import signature



import os

import sys

from zipfile import ZipFile



import ipywidgets as widgets

from IPython import display



import h2o

from h2o.frame import H2OFrame



from tqdm import tqdm

tqdm.pandas()
pd.set_option('display.width', 1000)

pd.set_option('display.max_columns', 1000)

pd.set_option('display.max_rows', 1000)

pd.set_option('display.max_colwidth', 1000)
PATH = '../input/'

PATH
for curr_dir, subdirs, files in os.walk(PATH):

    print(curr_dir)    

    print(files)
# find number of line in each file

!find ../input/*ank*/ -name '*.csv' | xargs wc -l | sort -nr
# check which seperator symbol (i.e. ',', ':', ';' etc.) is used in 'bank-additional-full.csv'

!head -n 2 $PATH/bank-marketing/bank-additional-full.csv
# check which seperator symbol (i.e. ',', ':', ';' etc.) is used in 'bank-additional.csv'

!head -n 2 $PATH/bank-marketing-full-dataset/bank-additional.csv

 
# find the details of each of columns in the file 'bank-additional-full.csv'

!tail -30 $PATH/bank-marketing/bank-additional-names.txt
feature_columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',

                  'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',

                  'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx',

                   'euribor3m', 'nr_employed']



target_column = 'deposit_subscribed'
h2o.init()
# reading data into h2o frame abbreviated as hf

hf_trn = h2o.import_file(f"{PATH}bank-marketing/bank-additional-full.csv", sep=';')                              

hf_tst = h2o.import_file(f"{PATH}bank-marketing-full-dataset/bank-additional.csv", sep=';')                              
hf_trn.shape, hf_tst.shape
hf_trn.head(5)
hf_tst.head(5)
hf_trn.columns = feature_columns+[target_column]

hf_tst.columns = feature_columns+[target_column]
hf_trn.head(1)
hf_tst.head(1)
from h2o.frame import H2OFrame

def missing_value_info(hf, value, perc=False):      

    missing_hf = H2OFrame({c: hf[c].isin(value).sum() for c in hf.columns})    

    if perc:

        return H2OFrame(missing_hf.as_data_frame().divide(len(hf)).multiply(100).round(2))

    else:

        return missing_hf
missing_value_info(hf_trn, 'unknown')
missing_value_info(hf_trn, 'unknown', True)
# In training data

hf_trn[target_column].table()
# for training data

H2OFrame(hf_trn.types)
# for testing data

H2OFrame(hf_tst.types)
# find which columns are categorical

categ_columns = [hf_trn.names[int(col_index)] 

                              for col_index in hf_trn.columns_by_type(coltype='categorical')]

categ_columns.remove(target_column)

H2OFrame([categ_columns])
df_trn = hf_trn.as_data_frame()

df_trn.shape
sns.set_style("ticks")

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(25, 21))

fig.suptitle(f"Frequency Distribution of Categorical Features\n[{target_column}:`yes`]",

             horizontalalignment='center', y=1.05,

             verticalalignment='center', fontsize=30)



plt.rcParams.update({'font.size': 15})

fig.subplots_adjust(top=0.99, bottom=0.01, hspace=1.5, wspace=0.4)

for ax, c in list(zip(axes.flat, categ_columns)):        

    sns.countplot(c, data=df_trn[df_trn[target_column]=='yes'],                  

                  order= df_trn[df_trn[target_column]=='yes'][c].value_counts().index,

                  ax=ax)

    for p in ax.patches:

        ax.annotate("{}".format(p.get_height()), (p.get_x()+0.1, p.get_height()+50),

                       ha='left', va='bottom', rotation=45)

    ax.tick_params(labelrotation=90)  

    plt.sca(ax)

    plt.yticks(rotation=0)    

    # ax.axis('off')

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_visible(True)

    ax.spines['left'].set_visible(False)

    
# https://stackoverflow.com/questions/19273040/rotating-axis-text-for-each-subplot

sns.set_style("ticks")

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(25, 21))

fig.suptitle(f"Frequency Distribution of Categorical Features\n[{target_column}:`no`]",

             horizontalalignment='center', y=1.05,

             verticalalignment='center', fontsize=30)



plt.rcParams.update({'font.size': 15})

fig.subplots_adjust(top=0.99, bottom=0.01, hspace=1.5, wspace=0.4)



for ax, c in list(zip(axes.flat, categ_columns)):        

    sns.countplot(c, data=df_trn[df_trn[target_column]=='no'],                  

                  order= df_trn[df_trn[target_column]=='no'][c].value_counts().index,

                  ax=ax)

    for p in ax.patches:

        ax.annotate("{}".format(p.get_height()), (p.get_x()+0.1, p.get_height()+50),

                       ha='left', va='bottom', rotation=45)

    ax.tick_params(labelrotation=90)  

    plt.sca(ax)

    plt.yticks(rotation=0)    

    # ax.axis('off')

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_visible(True)

    ax.spines['left'].set_visible(False)

    
widgets_list = []



for c in enumerate(categ_columns):

    widgets_list.append(widgets.Output())   

    

# render in output widgets

for c, wid in list(zip(categ_columns, widgets_list)):

    with wid:

        display.display(hf_trn[c].table().sort(['Count'], ascending=[False]))

    

# create HBox

hbox1 = widgets.HBox(widgets_list[:4])

hbox2 = widgets.HBox(widgets_list[4:])
hbox1
hbox2
# Categorical Unique Count

categ_uc = []

for c in categ_columns:    

    categ_uc.append([c, hf_trn[c].unique().nrows])

h2o.H2OFrame(categ_uc, column_names=['Categ_Column', 'Count']).sort(['Count'], ascending=False)
# find which columns are numerical

numerical_columns = [hf_trn.names[int(col_index)] 

               for col_index in hf_trn.columns_by_type(coltype='numeric')]

h2o.H2OFrame([numerical_columns])
# https://stackoverflow.com/questions/19273040/rotating-axis-text-for-each-subplot

sns.set_style("ticks")

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(25, 21))

fig.suptitle(f"Distribution of Numerical Features\n[{target_column}: `no`]",

             horizontalalignment='center', y=1.1,

             verticalalignment='center', fontsize=40)

plt.rcParams.update({'font.size': 15})

fig.subplots_adjust(top=0.99, bottom=0.05, hspace=1.5, wspace=0.2)



for ax, c in list(zip(axes.flat, numerical_columns)):        

    sns.distplot(a=df_trn[df_trn[target_column] == 'no'][c],

                  kde=False)

    

    ax.tick_params(labelrotation=90)  

    plt.sca(ax)

    plt.yticks(rotation=0)    

    # ax.axis('off')

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_visible(True)

    ax.spines['left'].set_visible(False)

hf_trn_corr = hf_trn[numerical_columns].cor(y=hf_trn[target_column], use='complete.obs')
hf_trn_corr['index'] = h2o.H2OFrame(numerical_columns)
# correlation of each numeric feature column with target column

hf_trn_corr
# correlation matrix of numeric feature columns

plt.figure(figsize=(8, 8))

plt.rcParams.update({'font.size': 10})

hf_trn_corr = hf_trn[numerical_columns].cor()

ax = sns.heatmap(hf_trn_corr.as_data_frame(), annot=True, fmt='.2f')

ax.set_yticklabels(numerical_columns, rotation=0, ha='right', minor=False);
_ = sns.pairplot(df_trn, hue=target_column, palette="husl")
print(categ_columns)
# which Numerical columns contain misisng values in training data

for c in numerical_columns:

    print(c, "=", hf_trn[c].nacnt())
# which categorical columns contain 'unknown'

missing_value_cols = []

for c in categ_columns:

    if 'unknown' in hf_trn[c].categories():

        missing_value_cols.append(c)
missing_value_cols
# https://stackoverflow.com/questions/19273040/rotating-axis-text-for-each-subplot

sns.set_style("ticks")

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(25, 21))

fig.suptitle("Distribution of `Missing` Categorical Features",

             horizontalalignment='center', y=1.05,

             verticalalignment='center', fontsize=40)



plt.rcParams.update({'font.size': 20})

fig.subplots_adjust(top=0.99, bottom=0.4, hspace=1.5, wspace=0.2)



for ax, c in list(zip(axes.flat, missing_value_cols)):        

    sns.countplot(c, data=df_trn,                  

                  order= df_trn[c].value_counts().index,

                  hue=target_column,

                  ax=ax)

    for p in ax.patches:

        ax.annotate("{}".format(p.get_height()), (p.get_x()+0.1, p.get_height()+50),

                       ha='left', va='bottom', rotation=45)

    ax.tick_params(labelrotation=90)  

    plt.sca(ax)

    plt.yticks(rotation=0)    

    # ax.axis('off')

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_visible(True)

    ax.spines['left'].set_visible(False)

    
# number of rows for which all the columns in missing_value_cols have misisng values

from collections import Counter

conditions = ((df_trn['job'] == 'unknown') & 

              (df_trn['marital'] == 'unknown') & 

              (df_trn['education'] == 'unknown') & 

              (df_trn['default'] == 'unknown') &  

              (df_trn['housing'] == 'unknown') & 

              (df_trn['loan'] == 'unknown'))

df_trn[conditions].__len__()
df_trn = hf_trn.as_data_frame()

df_tst = hf_tst.as_data_frame()

df_trn.shape, df_tst.shape
from sklearn.preprocessing import StandardScaler
print(numerical_columns)
scaler = StandardScaler()

%time scaler.fit(df_trn[numerical_columns])
scaler.mean_
df_trn.loc[:, numerical_columns] = pd.DataFrame(scaler.transform(df_trn[numerical_columns]), 

                                                columns=numerical_columns)
df_tst.loc[:, numerical_columns] = pd.DataFrame(scaler.transform(df_tst[numerical_columns]),

                                                columns=numerical_columns)
df_trn.head()
df_tst.head()
hf_trn = H2OFrame(df_trn)

hf_tst = H2OFrame(df_tst)
hf_trn.describe()
hf_tst.describe()
# find numerical column index

for idx in hf_trn.columns_by_type('numeric'):

    hf_tst[int(idx)] = hf_tst[int(idx)].asnumeric()



# find categorical column index

for idx in hf_trn.columns_by_type('categorical'):

    hf_tst[int(idx)] = hf_tst[int(idx)].asfactor()

    
# verify data types

assert hf_trn.columns_by_type('categorical') == hf_tst.columns_by_type('categorical')

assert hf_trn.columns_by_type('numeric') == hf_tst.columns_by_type('numeric')

from h2o.estimators.random_forest import H2ORandomForestEstimator

from h2o.estimators.gbm import H2OGradientBoostingEstimator

from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator

from h2o.grid.grid_search import H2OGridSearch
# Identify predictors and response columns

x_cols = list(hf_trn.columns)

y_col = target_column

x_cols.remove(y_col)

print(x_cols)

print()

print(y_col)
nfolds = 5

gbm_hyper_params = {"ntrees": list(range(10, 251, 20)), 

                    "max_depth": list(range(3, 16)),

                    "min_rows": [2, 3, 4, 5, 6, 7, 8],

                    "learn_rate": [0.1, 0.01, 0.001, 0.0001],

                    "sample_rate": [0.5, 0.6, 0.7, 0.8, 0.9],

                    "col_sample_rate": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

                    "balance_classes": [True, False]

                    }



gbm_search_criteria = {"strategy": "RandomDiscrete", "max_models": 40, 

                       "seed": 42}
gbm = H2OGradientBoostingEstimator(distribution="bernoulli",

                                   nfolds=nfolds,

                                   fold_assignment="Stratified",

                                   keep_cross_validation_predictions=True,

                                   categorical_encoding='auto',

                                   seed=42)
gbm_grid = H2OGridSearch(model=gbm,

                         hyper_params=gbm_hyper_params,

                         search_criteria=gbm_search_criteria,

                         grid_id="gbm_grid_binomial")
%time gbm_grid.train(x=x_cols, y=y_col, training_frame=hf_trn)
# find the trained models based on the descending order of F1 score 

gbm_grid_f1 = gbm_grid.get_grid(sort_by = "F1",

                                decreasing = True)

# F1 score of cross-validated model

gbm_grid_f1
# find the trained models based on the descending order of roc_auc score 

gbm_grid_roc_auc = gbm_grid.get_grid(sort_by = "auc",

                                decreasing = True)

# roc_auc score of cross-validated model

gbm_grid_roc_auc

# save the best model based on f1 score

model_path = h2o.save_model(model=gbm_grid_f1.models[0], 

                            path="../input/saved_models/",

                            force=True)

print(model_path)
!ls -hl ../input/saved_models/
# load the model

gbm_grid_f1 = h2o.load_model(model_path)
metric = [('Train', 'F1', gbm_grid_f1.F1(train=True)[0][0], gbm_grid_f1.F1(train=True)[0][1]),

      ('Train', 'Precision', gbm_grid_f1.precision(train=True)[0][0], gbm_grid_f1.precision(train=True)[0][1]),

      ('Train', 'Recall', gbm_grid_f1.recall(train=True)[0][0], gbm_grid_f1.recall(train=True)[0][1]),

      ('Train', 'ROC_AUC', '-', gbm_grid_f1.auc(train=True)),

      ('Train', 'RR_AUC', '-', gbm_grid_f1.pr_auc(train=True)),           

      

      ('Xval', 'F1', gbm_grid_f1.F1(xval=True)[0][0], gbm_grid_f1.F1(xval=True)[0][1]),

      ('Xval', 'Precision', gbm_grid_f1.precision(xval=True)[0][0], gbm_grid_f1.precision(xval=True)[0][1]),

      ('Xval', 'Recall', gbm_grid_f1.recall(xval=True)[0][0], gbm_grid_f1.recall(xval=True)[0][1]),      

      ('Xval', 'ROC_AUC', '-', gbm_grid_f1.auc(xval=True)),

      ('Xval', 'RR_AUC', '-', gbm_grid_f1.pr_auc(xval=True))]



pd.DataFrame(metric, columns=['Data', 'Metric', 'Threshold', 'Score']).set_index(['Data', 'Metric'])
gbm_grid_f1.plot()
from sklearn.metrics import f1_score, roc_curve, auc

from sklearn.metrics import roc_auc_score, precision_recall_curve 
y_tst_pred_gbm = gbm_grid_f1.model_performance(test_data=hf_tst)    
metric = [('Test', 'F1', y_tst_pred_gbm.F1()[0][0], y_tst_pred_gbm.F1()[0][1]),

      ('Test', 'Precision', y_tst_pred_gbm.precision()[0][0], y_tst_pred_gbm.precision()[0][1]),

      ('Test', 'Recall', y_tst_pred_gbm.recall()[0][0], y_tst_pred_gbm.recall()[0][1]),

      ('Test', 'ROC_AUC', '-', y_tst_pred_gbm.auc()),

      ('Test', 'RR_AUC', '-', y_tst_pred_gbm.pr_auc())]

     



pd.DataFrame(metric, columns=['Data', 'Metric', 'Threshold', 'Score']).set_index(['Data', 'Metric'])
# True Positive Rate

tpr_h2o =y_tst_pred_gbm.tprs

# False Positive Rate

fpr_h2o = y_tst_pred_gbm.fprs
# ROC AUC Curve using Sklearn

plt.title('ROC AUC Curve', color='blue')

plt.plot([0, 1], [0, 1], ls='--', lw=2, alpha=0.2)

ax = sns.lineplot(y=tpr_h2o, x=fpr_h2o, 

                  label="auc={0:.4f}".format(auc(x=fpr_h2o, y=tpr_h2o)))

ax.set_xlabel("False Positive Rate (FPR)", color='r')

ax.set_ylabel("True Positive Rate (TPR)", color='g')

plt.legend(loc='lower right');
# ROC AUC Curve using H2O

y_tst_pred_gbm.plot(type='roc')
# find precision and recall scores at different threshold to plot precision_recall curver

precision = y_tst_pred_gbm.precision(thresholds=[i for i in np.arange(0.02, 1, 0.001)])

recall = y_tst_pred_gbm.recall(thresholds=[i for i in np.arange(0.02, 1, 0.001)])
# take the precision and recall scores for positive class - 'yes'

prec = [p[1] for p in precision]

rec = [r[1] for r in recall]

len(prec), len(rec)
# http://www.chioka.in/differences-between-roc-auc-and-pr-auc/

positive_label = 'yes'

pr_auc = auc(x=rec, y=prec)  



ax = sns.lineplot(y=prec, x=rec, 

                  label="{0}: auc={1:.4f}".format(positive_label, pr_auc))

ax.set_xlabel("Recall", color='g')

ax.set_ylabel("Precision", color='g')

plt.title(f"PR AUC Curve [Positive Class]: `{positive_label}`", color='blue')

plt.plot([0, 1], [0.5, 0.5], linestyle='--', alpha=0.2)

plt.plot([0.5, 0.5], [0, 1], linestyle='--', alpha=0.2)

plt.legend(loc='lower right');