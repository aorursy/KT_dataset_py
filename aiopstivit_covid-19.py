# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# SLIPT DATASET
import sklearn.model_selection as model_selection
# Clustering
from sklearn.cluster import KMeans

from sklearn.metrics import pairwise_distances
import seaborn as sns

## Matplotlib
%matplotlib inline
plt.rc('font', size=14)
# Set the font dictionaries (for plot title and axis titles)
title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
  'verticalalignment':'bottom'} # Bottom vertical alignment for more space
axis_font = {'fontname':'Arial', 'size':'12'}

import matplotlib.ticker as mtick
dr_dark_blue = '#08233F'
dr_blue = '#1F77B4'
dr_orange = '#FF7F0E'
dr_red = '#BE3C28'
!pip install datarobot
import datarobot
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
df.head()
df[df.columns[2]].value_counts().to_frame().style.bar()
def plot_hist_preenchimento(df,fig_filename,log=True):
    fig, ax = plt.subplots()
    fig = plt.gcf()
    fig.set_size_inches(40,15)

    labels_ = df.columns.values
    y_pos = np.arange(len(labels_))
    performance =  100 - (df.isnull().sum().values/df.shape[0]*100)

    ax.bar(y_pos, performance, align='center')
    ax.set_xticks(y_pos)
    ax.set_xticklabels(labels_)
    ax.invert_xaxis()  # labels read top-to-bottom

    ax.set_xlabel('Campos')
    ax.set_ylabel('Preenchimento Porcentagem')
    ax.set_title('Campos mais preenchidos')
    plt.axhline(y=20, ls='--',color='red')
    plt.axhline(y=50, ls='--',color='red')
    plt.axhline(y=70, ls='--',color='red')
    plt.xticks(rotation=45, ha='right')


    #plt.xscale("log")
    if log :
        plt.yscale("log")
    plt.savefig(fig_filename)
    plt.show()

plot_hist_preenchimento(df,'campos_preenchidos_porcentagem.png',log=False)
titulo_ = "(rows,columns) - " + str(df.shape)

df.isnull().sum(axis=1).hist(bins=20)
fig = plt.gcf()
plt.title('Most completed fields : '+titulo_)
fig.set_size_inches(10,5)
plt.xlabel('Missing Features')
plt.ylabel('Row quantity')
#plt.xscale("log")
#plt.yscale("log")
plt.savefig("campos_preenchidos.png")
plt.show()

df.columns = [x.lower().strip().replace(' ','_') for x in df.columns]
df.columns.shape
df.drop(['coronavirusoc43','adenovirus','parainfluenza_3', 'metapneumovirus','chlamydophila_pneumoniae','parainfluenza_2','coronavirus229e','myelocytes','influenza_b','patient_addmited_to_semi-intensive_unit_(1=yes,_0=no)','ionized_calcium'], axis=1, inplace=True)
df.columns.shape
titulo_ = "dataset : (rows,columns) - " + str(df.shape)
df.isnull().sum(axis=1).hist(bins=20)
fig = plt.gcf()
plt.title(titulo_)
fig.set_size_inches(10,5)
plt.xlabel('Null features')
plt.ylabel('Row Count ')
#plt.xscale("log")
#plt.yscale("log")
plt.savefig("campos_preenchidos_apos_remocao.png")
plt.show()
msk_90 = df.isnull().sum(axis=1) > 90
row_idx = msk_90.index[msk_90.values]
df_lt_90 = df.drop(row_idx,axis=0)
df_lt_90.shape
df_lt_90[df_lt_90.columns[2]].value_counts().to_frame().style.bar()
df_lt_90.to_csv('df_lt_90.csv', index=False, encoding='utf-8')
import datarobot as dr
dr.Client(token='_valid_token_', endpoint='https://app.datarobot.com/api/v2')
project = dr.Project.create('df_lt_90.csv',project_name='_1_kaggle_einstein_lt_90')
project.set_target(target='sars-cov-2_exam_result',
                   metric='LogLoss',
                   worker_count = '4',
                   advanced_options=dr.AdvancedOptions(accuracy_optimized_mb = True),
                   partitioning_method = dr.RandomCV(20, 5, seed=0),
                   mode=dr.AUTOPILOT_MODE.FULL_AUTO)
models = project.get_models()
len(models)
models = project.get_models()
roc = models[1].get_roc_curve('validation')
df = pd.DataFrame(roc.roc_points)
df.columns.values

acc_list = []
models_name
for i in np.arange(len(models)):
    roc = models[i].get_roc_curve('validation')
    df = pd.DataFrame(roc.roc_points)
    acc_list.append(df.iloc[df.f1_score.idxmax(),:].to_dict())

df_ = pd.DataFrame(acc_list)
df_.insert(loc=0, column='models', value=models)
df_ = df_.sort_values(by=['accuracy','false_negative_score'],ascending=[False,True])
df_.head()
plt.figure()
plt.scatter(df_.false_negative_score,df_.accuracy)
plt.xlabel('false_negative_score')
plt.ylabel('accuracy')
plt.show()
best_model = models[0]
best_model
roc = best_model.get_roc_curve('validation')
threshold = roc.get_best_f1_threshold()
metrics = roc.estimate_threshold(threshold)
metrics
roc_df = pd.DataFrame({
    'Predicted Negative': [metrics['true_negative_score'],
                           metrics['false_negative_score'],
                           metrics['true_negative_score'] + metrics[
                               'false_negative_score']],
    'Predicted Positive': [metrics['false_positive_score'],
                           metrics['true_positive_score'],
                           metrics['true_positive_score'] + metrics[
                               'false_positive_score']],
    'Total': [len(roc.negative_class_predictions),
              len(roc.positive_class_predictions),
              len(roc.negative_class_predictions) + len(
                  roc.positive_class_predictions)]})
roc_df.index = pd.MultiIndex.from_tuples([
    ('Actual', '-'), ('Actual', '+'), ('Total', '')])
roc_df.columns = pd.MultiIndex.from_tuples([
    ('Predicted', '-'), ('Predicted', '+'), ('Total', '')])
roc_df.style.set_properties(**{'text-align': 'right'})
roc_df
plt.rcParams.update({'font.size': 10}) 

feature_impacts = best_model.get_or_request_feature_impact()


percent_tick_fmt = mtick.PercentFormatter(xmax=1.0)

impact_df = pd.DataFrame(feature_impacts)
impact_df.sort_values(by='impactNormalized', ascending=True, inplace=True)

# Positive values are blue, negative are red
bar_colors = impact_df.impactNormalized.apply(lambda x: dr_red if x < 0 else dr_blue)

ax = impact_df.plot.barh(x='featureName', y='impactNormalized',
                         legend=False,
                         color=bar_colors,
                         figsize=(10, 14))
ax.xaxis.set_major_formatter(percent_tick_fmt)
ax.xaxis.set_tick_params(labeltop=True)
ax.xaxis.grid(True, alpha=0.2)
ax.set_facecolor(dr_dark_blue)

plt.ylabel('')
plt.xlabel('Effect')
plt.xlim((None, 1))  # Allow for negative impact
plt.title('Feature Impact', y=1.04);

dr_roc_green = '#03c75f'
white = '#ffffff'
dr_purple = '#65147D'
dr_dense_green = '#018f4f'
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid': False})

fig = plt.figure(figsize=(8, 8))
axes = fig.add_subplot(1, 1, 1, facecolor=dr_dark_blue)

shared_params = {'shade': True, 'clip': (0, 1), 'bw': 0.2}
sns.kdeplot(np.array(roc.negative_class_predictions),
            color=dr_purple, **shared_params)
sns.kdeplot(np.array(roc.positive_class_predictions),
            color=dr_dense_green, **shared_params)

plt.title('Prediction Distribution')
plt.xlabel('Probability of Event')
plt.xlim([0, 1])
plt.ylabel('Probability Density');