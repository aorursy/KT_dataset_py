import os
from pathlib import Path
import subprocess

# Create the input directory if it doesn't exist
if not os.path.exists('../input'):
    os.makedirs('../input')

file_on_disk = True

# Check if the files are on disk before download
for file in os.listdir('../input'):
    if not Path('../input/' + file).is_file():
        # The file is not on disk
        file_on_disk = False
        break
        
if not file_on_disk:
    # Download the files with your API token in ~/.kaggle
    error = subprocess.call('kaggle competitions download -c titanic -p ../input'.split())
    if not error:
        print('Files downloaded successfully.')
    else:
        print('An error occurred during donwload, check your API token.')
else:
    print('Files are already on disk.')
# Load packages
print('Python packages:')
print('-'*15)

import sys
print('Python version: {}'. format(sys.version))

import pandas as pd
print('pandas version: {}'. format(pd.__version__))

import matplotlib
print('matplotlib version: {}'. format(matplotlib.__version__))

import numpy as np
print('NumPy version: {}'. format(np.__version__))

import scipy as sp
print('SciPy version: {}'. format(sp.__version__)) 

import IPython
from IPython import display
print('IPython version: {}'. format(IPython.__version__)) 

import sklearn
print('scikit-learn version: {}'. format(sklearn.__version__))

# Miscsellaneous libraries
import random
import time

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('')

# Check the input directory
print('Input directory: ')
print('-'*15)
from subprocess import check_output
print(check_output(['ls', '../input']).decode('utf8'))
# Common model algorithms
from sklearn import neighbors, ensemble
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

# Common model helpers
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import model_selection

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import PercentFormatter
import seaborn as sns

# Configure visualization defaults
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
palette = sns.color_palette('Set2', 10)
pylab.rcParams['figure.figsize'] = 18,4
train_df = pd.read_csv('../input/train.csv').set_index(keys='PassengerId', drop=True)
test_df  = pd.read_csv('../input/test.csv').set_index(keys='PassengerId', drop=True)

# Useful for more accurate feature engineering
data_df = train_df.append(test_df)
train_df.sample(10)
train_df.describe(include = 'all')
test_df.describe(include = 'all')
def plot_missing_values(dataset):
    """
        Plots the proportion of missing values per feature of a dataset.
        
        :param dataset: pandas DataFrame
    """
    missing_data_percent = [x / len(dataset) for x in dataset.isnull().sum()]
    data_percent = [1 - x for x in missing_data_percent]

    fig, axs = plt.subplots(1,1,figsize=(18,4))
    plt.bar(dataset.columns.values, data_percent, color='#84B044', linewidth=0)
    plt.bar(dataset.columns.values, missing_data_percent, bottom=data_percent, color='#E76C5D', linewidth=0)

    axs.yaxis.set_major_formatter(PercentFormatter(xmax=1))
train_df.isnull().sum().to_frame('Missing values').transpose()
plot_missing_values(train_df)
test_df.isnull().sum().to_frame('Missing values').transpose()
plot_missing_values(test_df)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,6))

corr_train = train_df[['Age', 'Fare', 'Parch', 'SibSp', 'Survived']].corr()
corr_test = test_df[['Age', 'Fare', 'Parch', 'SibSp']].corr()

# Generate masks for the upper triangles
mask_train = np.zeros_like(corr_train, dtype=np.bool)
mask_train[np.triu_indices_from(mask_train)] = True

mask_test = np.zeros_like(corr_test, dtype=np.bool)
mask_test[np.triu_indices_from(mask_test)] = True

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the train set heatmap with the mask and correct aspect ratio
sns.heatmap(corr_train, ax=ax1, mask=mask_train, cmap=cmap, vmax=.5, center=0, square=True, 
            linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f')
ax1.set_title('Pearson\'s correlation matrix of train set')

# Draw the test heatmap with the mask and correct aspect ratio
sns.heatmap(corr_test, ax=ax2, mask=mask_test, cmap=cmap, vmax=.5, center=0, square=True, 
            linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f')
ax2.set_title('Pearson\'s correlation matrix of test set')
from scipy.stats import entropy
from numpy.linalg import norm

def JSD(P, Q, n_iter=1000):
    """
        Computes the Jensen-Shannon divergence between two probability distributions of different sizes.
        
        :param P: distribution P
        :param Q: distribution Q
        :param n_iter: number of iterations
        :return: Jensen-Shannon divergence
    """
    size = min(len(P),len(Q))
    
    results = []
    for _ in range(n_iter):
        P = np.random.choice(P, size=size, replace=False)
        Q = np.random.choice(Q, size=size, replace=False)

        _P = P / norm(P, ord=1)
        _Q = Q / norm(Q, ord=1)
        _M = 0.5 * (_P + _Q)

        results.append(0.5 * (entropy(_P, _M) + entropy(_Q, _M)))

    return results
# Age vs Survived
g = sns.FacetGrid(train_df, col='Survived', size=4, aspect=2)
g = g.map(sns.distplot, 'Age', color='#D66A84')
# Train set vs Test set
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))

sns.distplot(train_df['Age'].dropna(), ax=ax1, color='#D66A84')
ax1.set_title('Train set')

sns.distplot(test_df['Age'].dropna(), ax=ax2, color='#D66A84')
ax2.set_title('Test set')
age_jsd = JSD(train_df['Age'].dropna().values, test_df['Age'].dropna().values)
print('Jensen-Shannon divergence of Age:', np.mean(age_jsd))
print('Standard deviation:', np.std(age_jsd))
# Fare vs Survived
g = sns.FacetGrid(train_df, col='Survived', palette=palette, size=4, aspect=2)
g = g.map(sns.distplot, 'Fare', color='#25627D')
fig, ax = plt.subplots(figsize=(18,4))

g = sns.distplot(train_df['Fare'], ax=ax, color='#25627D', label='Skewness : %.2f'%(train_df['Fare'].skew()))
g = g.legend(loc='best')
# Train set vs Test set
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))

sns.distplot(train_df['Fare'].dropna(), ax=ax1, color='#25627D')
ax1.set_title('Train set')

sns.distplot(test_df['Fare'].dropna(), ax=ax2, color='#25627D')
ax2.set_title('Test set')
fare_jsd = JSD(train_df['Fare'].dropna().values, test_df['Fare'].dropna().values)
print('Jensen-Shannon divergence of Fare:', np.mean(fare_jsd))
print('Standard deviation:', np.std(fare_jsd))
palette6 = ["#F6B5A4", "#EB7590", "#C8488A", "#872E93", "#581D7F", "#3A1353"]
# Parch vs Survived
g  = sns.catplot(x='Parch', y='Survived', saturation=5, height=4, aspect=4, data=train_df, kind='bar', palette=palette6)
g.despine(left=True)
g = g.set_ylabels("Survival probability")
# Train set vs Test set
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))

sns.distplot(train_df['Parch'], ax=ax1, color='#84B044')
ax1.set_title('Train set')

sns.distplot(test_df['Parch'], ax=ax2, color='#84B044')
ax2.set_title('Test set')
parch_jsd = JSD(train_df['Parch'].values, test_df['Parch'].values)
print('Jensen-Shannon divergence of Parch:', np.mean(parch_jsd))
print('Standard deviation:', np.std(parch_jsd))
palette7 = ["#F7BBA6", "#ED8495", "#E05286", "#A73B8F", "#6F2597", "#511B75", "#37114E"]
# SibSp feature vs Survived
g = sns.catplot(x='SibSp', y='Survived', saturation=5, height=4, aspect=4, data=train_df, kind='bar', palette=palette7)
g.despine(left=True)
g = g.set_ylabels("Survival probability")
# Train set vs Test set
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))

sns.distplot(train_df['SibSp'], ax=ax1, color='#E76C5D')
ax1.set_title('Train set')

sns.distplot(test_df['SibSp'], ax=ax2, color='#E76C5D')
ax2.set_title('Test set')
sibsp_jsd = JSD(train_df['SibSp'].values, test_df['SibSp'].values)
print('Jensen-Shannon divergence of SibSp:', np.mean(sibsp_jsd))
print('Standard deviation:', np.std(sibsp_jsd))
palette4 = ["#F19A9B", "#D54D88", "#7B2A95", "#461765"]
fig, ax = plt.subplots(figsize=(18,4))
jsd = pd.DataFrame(np.column_stack([age_jsd, fare_jsd, parch_jsd, sibsp_jsd]), columns=['Age', 'Fare', 'Parch', 'SibSp'])
sns.boxplot(data=jsd, ax=ax, orient="h", linewidth=1, saturation=5, palette=palette4)
ax.set_title('Jensen-Shannon divergences of numerical features')
plt.figure(figsize=(18, 4))
plt.scatter(train_df['Age'], train_df['Fare'], c=train_df['Survived'].values, cmap='cool')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Age vs Fare')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))

sns.regplot(x='Age', y='Fare', ax=ax1, data=train_df)
ax1.set_title('Train set')
sns.regplot(x='Age', y='Fare', ax=ax2, data=test_df)
ax2.set_title('Test set')
print('PCC for the train set: ', corr_train['Age']['Fare'])
print('PCC for the test set: ', corr_test['Age']['Fare'])
plt.figure(figsize=(18, 4))
plt.scatter(train_df['Age'], train_df['Parch'], c=train_df['Survived'].values, cmap='cool')
plt.xlabel('Age')
plt.ylabel('Parch')
plt.title('Age vs Parch')
palette8 = ["#F8C1A8", "#EF9198", "#E8608A", "#C0458A", "#8F3192", "#63218F", "#4B186C", "#33104A"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))

sns.boxplot(y='Age', x='Parch', ax=ax1, data=train_df, linewidth=1, saturation=5, palette=palette7)
ax1.set_title('Train set')
sns.boxplot(y='Age', x='Parch', ax=ax2, data=test_df, linewidth=1, saturation=5, palette=palette8)
ax2.set_title('Test set')
print('PCC for the train set: ', corr_train['Age']['Parch'])
print('PCC for the test set: ', corr_test['Age']['Parch'])
plt.figure(figsize=(18, 4))
plt.scatter(train_df['Fare'], train_df['Parch'], c=train_df['Survived'].values, cmap='cool')
plt.xlabel('Fare')
plt.ylabel('Parch')
plt.title('Fare vs Parch')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))

sns.boxplot(y='Fare', x='Parch', ax=ax1, data=train_df, linewidth=1, saturation=5, palette=palette7)
ax1.set_title('Train set')
sns.boxplot(y='Fare', x='Parch', ax=ax2, data=test_df, linewidth=1, saturation=5, palette=palette8)
ax2.set_title('Test set')
print('PCC for the train set: ', corr_train['Fare']['Parch'])
print('PCC for the test set: ', corr_test['Fare']['Parch'])
plt.figure(figsize=(18, 4))
plt.scatter(train_df['Fare'], train_df['SibSp'], c=train_df['Survived'].values, cmap='cool')
plt.xlabel('Fare')
plt.ylabel('SibSp')
plt.title('Fare vs SibSp')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))

sns.boxplot(y='Fare', x='SibSp', ax=ax1, data=train_df, linewidth=1, saturation=5, palette=palette7)
ax1.set_title('Train set')
sns.boxplot(y='Fare', x='SibSp', ax=ax2, data=test_df, linewidth=1, saturation=5, palette=palette8)
ax2.set_title('Test set')
print('PCC for the train set: ', corr_train['Fare']['SibSp'])
print('PCC for the test set: ', corr_test['Fare']['SibSp'])
plt.figure(figsize=(18, 4))
plt.scatter(train_df['Parch'], train_df['SibSp'], c=train_df['Survived'].values, cmap='cool')
plt.xlabel('Parch')
plt.ylabel('SibSp')
plt.title('Parch vs SibSp')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))

sns.boxplot(y='Parch', x='SibSp', ax=ax1, data=train_df, linewidth=1, saturation=5, palette=palette7)
ax1.set_title('Train set')
sns.boxplot(y='Parch', x='SibSp', ax=ax2, data=test_df, linewidth=1, saturation=5, palette=palette8)
ax2.set_title('Test set')
print('PCC for the train set: ', corr_train['Parch']['SibSp'])
print('PCC for the test set: ', corr_test['Parch']['SibSp'])
palette3 = ["#EE8695", "#A73B8F", "#501B73"]
# Embarked feature vs Survived
g  = sns.catplot(x='Embarked', y='Survived', saturation=5, height=4, aspect=4, data=train_df, 
                    kind='bar', palette=palette3)
g.despine(left=True)
g = g.set_ylabels('Survival probability')
# Train set vs Test set
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))

train_df['Embarked'].value_counts().plot(kind='barh', ax=ax1)
ax1.set_title('Train set')

test_df['Embarked'].value_counts().plot(kind='barh', ax=ax2)
ax2.set_title('Test set')
palette2 = ["#EE8695", "#A73B8F"]
# Sex feature vs Survived
g  = sns.catplot(x='Sex', y='Survived', saturation=5, height=4, aspect=4, data=train_df, 
                    kind='bar', palette=palette2)
g.despine(left=True)
g = g.set_ylabels('Survival probability')
# Train set vs Test set
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))

train_df['Sex'].value_counts().plot(kind='barh', ax=ax1)
ax1.set_title('Train set')

test_df['Sex'].value_counts().plot(kind='barh', ax=ax2)
ax2.set_title('Test set')
# Pclass feature vs Survived
g  = sns.catplot(x='Pclass', y='Survived', saturation=5, height=4, aspect=4, data=train_df, 
                    kind='bar', palette=palette3)
g.despine(left=True)
g = g.set_ylabels('Survival probability')
# Train set vs Test set
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))

train_df['Pclass'].value_counts().plot(kind='barh', ax=ax1)
ax1.set_title('Train set')

test_df['Pclass'].value_counts().plot(kind='barh', ax=ax2)
ax2.set_title('Test set')
import statsmodels.api as sm
from statsmodels.formula.api import ols

def compute_anova(dataset, group, weight):
    """
        Computes the effect size through ANOVA.
        
        :param dataset: pandas DataFrame
        :param group: categorical feature
        :param weight: continuous feature
        :return: effect size
    """
    mod = ols(weight + ' ~ ' + group, data=dataset).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])
    
    return esq_sm
from scipy.stats import chi2_contingency

def chisq(dataset, c1, c2):
    """
        Performs the Chi squared independence test.
        
        :param dataset: pandas DataFrame
        :param c1: continuous feature 1
        :param c2: continuous feature 2
        :return: array with [Chi^2, p-value]
    """
    groupsizes = dataset.groupby([c1, c2]).size()
    ctsum = groupsizes.unstack(c1)

    result = chi2_contingency(ctsum.fillna(0))
    
    print('Chi^2:', result[0])
    print('p-value:', result[1])
    print('Degrees of freedom:', result[2])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))

sns.boxplot(y='Age', x='Embarked', ax=ax1, data=train_df, linewidth=1, saturation=5, order=['S', 'C', 'Q'], palette=palette3)
ax1.set_title('Train set')
sns.boxplot(y='Age', x='Embarked', ax=ax2, data=test_df, linewidth=1, saturation=5, order=['S', 'C', 'Q'], palette=palette3)
ax2.set_title('Test set')
train_esq_sm = compute_anova(train_df, 'Embarked', 'Age')
test_esq_sm = compute_anova(test_df, 'Embarked', 'Age')

print('ANOVA 1-way for the train set: ', train_esq_sm)
print('ANOVA 1-way for the test set: ', test_esq_sm)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))

sns.boxplot(y='Fare', x='Embarked', ax=ax1, data=train_df, linewidth=1, saturation=5, order=['S', 'C', 'Q'], palette=palette3)
ax1.set_title('Train set')
sns.boxplot(y='Fare', x='Embarked', ax=ax2, data=test_df, linewidth=1, saturation=5, order=['S', 'C', 'Q'], palette=palette3)
ax2.set_title('Test set')
train_esq_sm = compute_anova(train_df, 'Embarked', 'Fare')
test_esq_sm = compute_anova(test_df, 'Embarked', 'Fare')

print('ANOVA 1-way for the train set: ', train_esq_sm)
print('ANOVA 1-way for the test set: ', test_esq_sm)
def plot_embarked_variable(dataset, variable):
    """
        Plots the proportion of variable values per Embarked value of a dataset.
        
        :param dataset: pandas DataFrame
        :param variable: variable to plot
    """
    s_variable_index = dataset.groupby(['Embarked', variable]).size()['S'].index.values
    c_variable_index = dataset.groupby(['Embarked', variable]).size()['C'].index.values
    q_variable_index = dataset.groupby(['Embarked', variable]).size()['Q'].index.values

    index = list(set().union(s_variable_index,c_variable_index,q_variable_index))

    raw_s_variable = dataset.groupby(['Embarked', variable]).size()['S']
    raw_c_variable = dataset.groupby(['Embarked', variable]).size()['C']
    raw_q_variable = dataset.groupby(['Embarked', variable]).size()['Q']

    s_variable = []
    c_variable = []
    q_variable = []

    for i in range(max(index) + 1):
        s_variable.append(raw_s_variable[i] if i in s_variable_index else 0)
        c_variable.append(raw_c_variable[i] if i in c_variable_index else 0)
        q_variable.append(raw_q_variable[i] if i in q_variable_index else 0)

    percent_s_variable = [s_variable[i]/(s_variable[i] + c_variable[i] + q_variable[i]) if i in index else 0 for i in range(max(index) + 1)]
    percent_c_variable = [c_variable[i]/(s_variable[i] + c_variable[i] + q_variable[i]) if i in index else 0 for i in range(max(index) + 1)]
    percent_q_variable = [q_variable[i]/(s_variable[i] + c_variable[i] + q_variable[i]) if i in index else 0 for i in range(max(index) + 1)]

    r = list(range(max(index) + 1))
    bars = [sum(x) for x in zip(percent_s_variable, percent_c_variable)]

    fig, axs = plt.subplots(1,1,figsize=(18,4))
    plt.bar(r, percent_s_variable, color='#08c299')
    plt.bar(r, percent_c_variable, bottom=percent_s_variable, linewidth=0, color='#97de95')
    plt.bar(r, percent_q_variable, bottom=bars, linewidth=0, color='#fce8aa')
    plt.xticks(r, r)
    plt.title('Proportion of Embarked values by ' + variable)
    axs.legend(labels=['S', 'C', 'Q'])
    axs.yaxis.set_major_formatter(PercentFormatter(xmax=1))
plot_embarked_variable(train_df, 'Parch')
chisq(train_df, 'Embarked', 'Parch')
plot_embarked_variable(test_df, 'Parch')
chisq(test_df, 'Embarked', 'Parch')
plot_embarked_variable(train_df, 'SibSp')
chisq(train_df, 'Embarked', 'SibSp')
plot_embarked_variable(test_df, 'SibSp')
chisq(test_df, 'Embarked', 'SibSp')
tmp_train_df = train_df.copy(deep=True)
tmp_train_df['Sex'].replace(['male', 'female'], [0,1], inplace=True)
plot_embarked_variable(tmp_train_df, 'Sex')
chisq(tmp_train_df, 'Embarked', 'Sex')
tmp_test_df = test_df.copy(deep=True)
tmp_test_df['Sex'].replace(['male', 'female'], [0,1], inplace=True)
plot_embarked_variable(tmp_test_df, 'Sex')
chisq(tmp_test_df, 'Embarked', 'Sex')
plot_embarked_variable(train_df, 'Pclass')
chisq(train_df, 'Embarked', 'Pclass')
plot_embarked_variable(test_df, 'Pclass')
chisq(test_df, 'Embarked', 'Pclass')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))

sns.boxplot(y='Age', x='Sex', ax=ax1, data=train_df, linewidth=1, saturation=5, palette=palette2)
ax1.set_title('Train set')
sns.boxplot(y='Age', x='Sex', ax=ax2, data=test_df, linewidth=1, saturation=5, palette=palette2)
ax2.set_title('Test set')
train_esq_sm = compute_anova(train_df, 'Sex', 'Age')
test_esq_sm = compute_anova(test_df, 'Sex', 'Age')

print('ANOVA 1-way for the train set: ', train_esq_sm)
print('ANOVA 1-way for the test set: ', test_esq_sm)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))

sns.boxplot(y='Fare', x='Sex', ax=ax1, data=train_df, linewidth=1, saturation=5, palette=palette2)
ax1.set_title('Train set')
sns.boxplot(y='Fare', x='Sex', ax=ax2, data=test_df, linewidth=1, saturation=5, palette=palette2)
ax2.set_title('Test set')
train_esq_sm = compute_anova(train_df, 'Sex', 'Fare')
test_esq_sm = compute_anova(test_df, 'Sex', 'Fare')

print('ANOVA 1-way for the train set: ', train_esq_sm)
print('ANOVA 1-way for the test set: ', test_esq_sm)
def plot_sex_variable(dataset, variable):
    """
        Plots the proportion of variable values per Sex value of a dataset.
        
        :param dataset: pandas DataFrame
        :param variable: variable to plot
    """
    male_variable_index = dataset.groupby(['Sex', variable]).size()['male'].index.values
    female_variable_index = dataset.groupby(['Sex', variable]).size()['female'].index.values

    index = list(set().union(male_variable_index, female_variable_index))

    raw_male_variable = dataset.groupby(['Sex', variable]).size()['male']
    raw_female_variable = dataset.groupby(['Sex', variable]).size()['female']

    male_variable = []
    female_variable = []

    for i in range(max(index) + 1):
        male_variable.append(raw_male_variable[i] if i in male_variable_index else 0)
        female_variable.append(raw_female_variable[i] if i in female_variable_index else 0)

    percent_male_variable = [male_variable[i]/(male_variable[i] + female_variable[i]) if i in index else 0 for i in range(max(index) + 1)]
    percent_female_variable = [female_variable[i]/(male_variable[i] + female_variable[i]) if i in index else 0 for i in range(max(index) + 1)]

    r = list(range(max(index) + 1))

    fig, axs = plt.subplots(1,1,figsize=(18,4))
    plt.bar(r, percent_male_variable, color='#ce2525')
    plt.bar(r, percent_female_variable, bottom=percent_male_variable, linewidth=0, color='#ff6600')
    plt.xticks(r, r)
    plt.title('Proportion of Sex values by ' + variable)
    axs.legend(labels=['male', 'female'])
    axs.yaxis.set_major_formatter(PercentFormatter(xmax=1))
plot_sex_variable(train_df, 'Parch')
chisq(train_df, 'Sex', 'Parch')
plot_sex_variable(test_df, 'Parch')
chisq(test_df, 'Sex', 'Parch')
plot_sex_variable(train_df, 'SibSp')
chisq(train_df, 'Sex', 'SibSp')
plot_sex_variable(test_df, 'SibSp')
chisq(test_df, 'Sex', 'SibSp')
plot_sex_variable(train_df, 'Pclass')
chisq(train_df, 'Sex', 'Pclass')
plot_sex_variable(test_df, 'Pclass')
chisq(test_df, 'Sex', 'Pclass')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))

sns.boxplot(y='Age', x='Pclass', ax=ax1, data=train_df, linewidth=1, saturation=5, palette=palette3)
ax1.set_title('Train set')
sns.boxplot(y='Age', x='Pclass', ax=ax2, data=test_df, linewidth=1, saturation=5, palette=palette3)
ax2.set_title('Test set')
train_esq_sm = compute_anova(train_df, 'Age', 'Pclass')
test_esq_sm = compute_anova(test_df, 'Age', 'Pclass')

print('ANOVA 1-way for the train set: ', train_esq_sm)
print('ANOVA 1-way for the test set: ', test_esq_sm)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))

sns.boxplot(y='Fare', x='Pclass', ax=ax1, data=train_df, linewidth=1, saturation=5, palette=palette3)
ax1.set_title('Train set')
sns.boxplot(y='Fare', x='Pclass', ax=ax2, data=test_df, linewidth=1, saturation=5, palette=palette3)
ax2.set_title('Test set')
train_esq_sm = compute_anova(train_df, 'Fare', 'Pclass')
test_esq_sm = compute_anova(test_df, 'Fare', 'Pclass')

print('ANOVA 1-way for the train set: ', train_esq_sm)
print('ANOVA 1-way for the test set: ', test_esq_sm)
def plot_pclass_variable(dataset, variable):
    """
        Plots the proportion of variable values per Pclass value of a dataset.
        
        :param dataset: pandas DataFrame
        :param variable: variable to plot
    """
    first_variable_index = dataset.groupby(['Pclass', variable]).size()[1].index.values
    second_variable_index = dataset.groupby(['Pclass', variable]).size()[2].index.values
    third_variable_index = dataset.groupby(['Pclass', variable]).size()[3].index.values

    index = list(set().union(first_variable_index, second_variable_index, third_variable_index))

    raw_first_variable = dataset.groupby(['Pclass', variable]).size()[1]
    raw_second_variable = dataset.groupby(['Pclass', variable]).size()[2]
    raw_third_variable = dataset.groupby(['Pclass', variable]).size()[3]

    first_variable = []
    second_variable = []
    third_variable = []

    for i in range(max(index) + 1):
        first_variable.append(raw_first_variable[i] if i in first_variable_index else 0)
        second_variable.append(raw_second_variable[i] if i in second_variable_index else 0)
        third_variable.append(raw_third_variable[i] if i in third_variable_index else 0)

    percent_first_variable = [first_variable[i]/(first_variable[i] + second_variable[i] + third_variable[i]) if i in index else 0 for i in range(max(index) + 1)]
    percent_second_variable = [second_variable[i]/(first_variable[i] + second_variable[i] + third_variable[i]) if i in index else 0 for i in range(max(index) + 1)]
    percent_third_variable = [third_variable[i]/(first_variable[i] + second_variable[i] + third_variable[i]) if i in index else 0 for i in range(max(index) + 1)]

    r = list(range(max(index) + 1))

    fig, axs = plt.subplots(1,1,figsize=(18,4))
    plt.bar(r, percent_first_variable, color='#264e86')
    plt.bar(r, percent_second_variable, bottom=percent_first_variable, linewidth=0, color='#0074e4')
    plt.bar(r, percent_third_variable, bottom=percent_second_variable, linewidth=0, color='#74dbef')
    plt.xticks(r, r)
    plt.title('Proportion of Pclass values by ' + variable)
    axs.legend(labels=['1', '2', '3'])
    axs.yaxis.set_major_formatter(PercentFormatter(xmax=1))
plot_pclass_variable(train_df, 'Parch')
chisq(train_df, 'Pclass', 'Parch')
plot_pclass_variable(test_df, 'Parch')
chisq(test_df, 'Pclass', 'Parch')
plot_pclass_variable(train_df, 'SibSp')
chisq(train_df, 'Pclass', 'SibSp')
plot_pclass_variable(test_df, 'SibSp')
chisq(test_df, 'Pclass', 'SibSp')
X_train = train_df[['Age', 'Fare', 'Parch', 'SibSp']].copy(deep=True).dropna()

std_scaler = StandardScaler()
X_scaled = std_scaler.fit_transform(X_train)

clf = ensemble.IsolationForest(contamination=0.01)
clf.fit(X_scaled)
y_pred = clf.predict(X_scaled)

X_train['isOutlier'] = y_pred

outliers_list = X_train.index[X_train['isOutlier'] == -1].tolist()

data_df.drop(outliers_list, inplace=True)
train_df.drop(outliers_list, inplace=True)

TRAINING_LENGTH = len(train_df)
X_train[X_train['isOutlier'] == -1]
data_df['Sex'].replace(['male', 'female'], [0,1], inplace=True)
data_df['Fare'].fillna(data_df['Fare'].median(), inplace=True)
# Apply log to Fare to reduce skewness distribution
data_df["Fare"] = data_df["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

fig, ax = plt.subplots(figsize=(16,4))
g = sns.distplot(data_df["Fare"], ax=ax, color='#25627D', label="Skewness : %.2f"%(data_df["Fare"].skew()))
g = g.legend(loc="best")
data_df['FareBin'] = pd.qcut(data_df['Fare'], 6)

label = LabelEncoder()
data_df['FareBin_Code'] = label.fit_transform(data_df['FareBin'])

data_df.drop(['Fare'], 1, inplace=True)
data_df.drop(['FareBin'], 1, inplace=True)
train_df = data_df[:TRAINING_LENGTH]
test_df = data_df[TRAINING_LENGTH:]

logfare_jsd = JSD(train_df['FareBin_Code'].dropna().values, test_df['FareBin_Code'].dropna().values)
print('Jensen-Shannon divergence of Fare:', np.mean(logfare_jsd))
print('Standard deviation:', np.std(logfare_jsd))
fig, ax = plt.subplots(figsize=(16,4))
jsd = pd.DataFrame(np.column_stack([fare_jsd, logfare_jsd]), columns=['Fare', 'LogFare'])
sns.boxplot(data=jsd, ax=ax, orient="h", linewidth=1, saturation=5, palette=palette2)
ax.set_title('Jensen-Shannon divergences of Fare and LogFare')
Ticket = []
for i in data_df['Ticket'].values:
    if not i.isdigit() :
        Ticket.append(i.replace('.', '').replace('', '').strip().split()[0])
    else:
        Ticket.append('X')
        
data_df['Ticket'] = Ticket
data_df = pd.get_dummies(data_df, columns=['Ticket'], drop_first=True)
# Get Title from Name
titles = [i.split(',')[1].split('.')[0].strip() for i in data_df['Name']]
data_df['Title'] = pd.Series(titles, index=data_df.index)

rare_titles = pd.Series(titles).value_counts()
rare_titles = rare_titles[rare_titles < 10].index

data_df['Title'] = data_df['Title'].replace(rare_titles, 'Rare')
data_df['Title'] = data_df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
data_df['Title'] = data_df['Title'].astype(int)
data_df = pd.get_dummies(data_df, columns=['Title'], drop_first=True)
data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp'] + 1
tmp_train_df = data_df[:TRAINING_LENGTH].copy(deep=True)
tmp_test_df = data_df[TRAINING_LENGTH:].copy(deep=True)

fs_jsd = JSD(tmp_train_df['Family_Size'].dropna().values, tmp_test_df['Family_Size'].dropna().values)
print('Jensen-Shannon divergence of Family_Size:', np.mean(fs_jsd))
print('Standard deviation:', np.std(fs_jsd))
fig, ax = plt.subplots(figsize=(16,4))
jsd = pd.DataFrame(np.column_stack([parch_jsd, sibsp_jsd, fs_jsd]), columns=['Fare', 'FareBin', 'Family_Size'])
sns.boxplot(data=jsd, ax=ax, orient="h", linewidth=1, saturation=5, palette=palette3)
ax.set_title('Jensen-Shannon divergences of Parch, SibSp and Family_Size')
print('Train dataset:')
train_df.isnull().sum().to_frame('Missing values').transpose()
print('Test/Validation dataset:')
test_df.isnull().sum().to_frame('Missing values').transpose()
data_df.drop(['Name', 'Parch', 'SibSp'], axis = 1, inplace = True)
data_df['Embarked'].fillna(data_df['Embarked'].mode()[0], inplace=True)
data_df = pd.get_dummies(data_df, columns=['Embarked'], drop_first=True)
data_df['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in data_df['Cabin'] ])
palette9 = ["#F8C7AA", "#F19B9C", "#EA708E", "#D54D88", "#A73B8F", "#7A2995", "#5B1F84", "#451764", "#300F45"]
g = sns.catplot(x='Cabin', y='Survived',saturation=5, aspect=2.5, data=data_df, kind='bar', order=['A','B','C','D','E','F','G','T','X'], palette=palette9)
data_df = pd.get_dummies(data_df, columns=['Cabin'], prefix='Deck', drop_first=True)
tmp_data_df = data_df.copy(deep = True)[['Age']]

imp = SimpleImputer(missing_values=np.nan, strategy='median')
tmp_data_df = pd.DataFrame(data=imp.fit_transform(tmp_data_df),index=tmp_data_df.index.values,columns=tmp_data_df.columns.values)
tmp_data_df['AgeBin'] = pd.qcut(tmp_data_df['Age'], 5, duplicates='drop')
tmp_data_df['AgeBin'].replace(np.NaN, -1, inplace = True)

label = LabelEncoder()
tmp_data_df['AgeBin_Code'] = label.fit_transform(tmp_data_df['AgeBin'])
tmp_data_df.drop(['Age', 'AgeBin'], axis=1, inplace=True)

data_df['AgeBin_Code'] = tmp_data_df['AgeBin_Code']
data_df.drop(['Age'], 1, inplace=True)
# Histogram comparison of Sex, Pclass, and Age by Survival
h = sns.FacetGrid(data_df, row='Sex', col='Pclass', hue='Survived')
h.map(plt.hist, 'AgeBin_Code', alpha=.75)
h.add_legend()
train_df = data_df[:TRAINING_LENGTH]
train_df.Survived = train_df.Survived.astype(int)
test_df = data_df[TRAINING_LENGTH:]
train_df.sample(5)
X = train_df.drop('Survived', 1)
y = train_df['Survived']
X_test = test_df.copy().drop(columns=['Survived'], axis=1)
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_test = std_scaler.transform(X_test)
class CatBoostClassifierCorrected(CatBoostClassifier):
    def fit(self, X, y=None, cat_features=None, sample_weight=None, baseline=None, use_best_model=None, 
            eval_set=None, verbose=None, logging_level=None, plot=False, column_description=None, verbose_eval=None, 
            metric_period=None, silent=None, early_stopping_rounds=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None):
        # Handle different types of label
        self.le_ = LabelEncoder().fit(y)
        transformed_y = self.le_.transform(y)

        self._fit(X=X, y=transformed_y, cat_features=cat_features, pairs=None, sample_weight=sample_weight, group_id=None,
                  group_weight=None, subgroup_id=None, pairs_weight=None, baseline=baseline, use_best_model=use_best_model, 
                  eval_set=eval_set, verbose=verbose, logging_level=logging_level, plot=plot, column_description=column_description,
                  verbose_eval=verbose_eval, metric_period=metric_period, silent=silent, early_stopping_rounds=early_stopping_rounds,
                  save_snapshot=save_snapshot, snapshot_file=snapshot_file, snapshot_interval=None)
        return self
        
    def predict(self, data, prediction_type='Class', ntree_start=0, ntree_end=0, thread_count=1, verbose=None):
        predictions = self._predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose)

        # Return same type as input
        return self.le_.inverse_transform(predictions.astype(np.int64))
# Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    # Ensemble Methods
    ensemble.RandomForestClassifier(),
    
    # Nearest Neighbors
    neighbors.KNeighborsClassifier(),
    
    # XGBoost
    XGBClassifier(),
    
    # LightGBM
    lgb.LGBMClassifier(),
    
    # CatBoost
    CatBoostClassifierCorrected(iterations=100, logging_level='Silent')
    ]

# Split dataset in cross-validation with this splitter class
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)

# Create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

# Create table to compare MLA predictions
MLA_predict = pd.Series()

# Index through MLA and save performance to table
row_index = 0
for alg in MLA:

    # Set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    # Score model with cross validation
    cv_results = model_selection.cross_validate(alg, X, y, cv  = cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    # If this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    
    # Save MLA predictions - see section 6 for usage
    alg.fit(X, y)
    MLA_predict[MLA_name] = alg.predict(X)
    
    row_index+=1
    
# Print and sort table
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
fig, ax = plt.subplots(figsize=(16,6))

# Barplot
sns.barplot(x='MLA Test Accuracy Mean', y='MLA Name', ax=ax, data=MLA_compare, palette=sns.color_palette("coolwarm_r", 5))

# Prettify
plt.title('Machine Learning Algorithm Accuracy Score')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
# Removed models w/o attribute 'predict_proba' required for vote classifier and models with a 1.0 correlation to another model
vote_est = [
    # Ensemble Methods: 
    ('rfc', ensemble.RandomForestClassifier()),
    
    # Nearest Neighbors:
    ('knn', neighbors.KNeighborsClassifier()),
    
    # XGBoost:
    ('xgb', XGBClassifier()),
    
    # LightGBM:
    ('lgb', lgb.LGBMClassifier()),
    
    # CatBoost:
    ('cat', CatBoostClassifierCorrected(iterations=100, logging_level='Silent'))
]

# Hard vote or majority rules
vote_hard = ensemble.VotingClassifier(estimators = vote_est, voting = 'hard')
vote_hard_cv = model_selection.cross_validate(vote_hard, X, y, cv = cv_split)
vote_hard.fit(X, y)

print("Hard Voting Training w/bin score mean: {:.2f}". format(vote_hard_cv['train_score'].mean()*100)) 
print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))
print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))
print('-'*15)

# Soft vote or weighted probabilities
vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
vote_soft_cv = model_selection.cross_validate(vote_soft, X, y, cv  = cv_split)
vote_soft.fit(X, y)

print("Soft Voting Training w/bin score mean: {:.2f}". format(vote_soft_cv['train_score'].mean()*100)) 
print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))
print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))
print('-'*15)
# Hyper-parameter tuning with GridSearchCV:
grid_param = [
            [{
            # RandomForestClassifier
            'criterion': ['gini'], #['gini', 'entropy'],
            'max_depth': [8], #[2, 4, 6, 8, 10, None],
            'n_estimators': [100], #[10, 50, 100, 300],
            'oob_score': [False] #[True, False]
             }],
    
            [{
            # KNeighborsClassifier
            'algorithm': ['auto'], #['auto', 'ball_tree', 'kd_tree', 'brute'],
            'n_neighbors': [7], #[1,2,3,4,5,6,7],
            'weights': ['distance'] #['uniform', 'distance']
            }],
    
            [{
            # XGBClassifier
            'learning_rate': [0.05], #[0.05, 0.1,0.16],
            'max_depth': [10], #[10,30,50],
            'min_child_weight' : [6], #[1,3,6]
            'n_estimators': [200]
             }],
    
            [{
            # LightGBMClassifier
            'learning_rate': [0.01], #[0.01,0.05,0.1],
            'n_estimators': [200],
            'num_leaves': [300], #[300,900,1200],
            'max_depth': [25], #[25,50,75],
             }],
    
            [{
            # CatBoostClassifier
            'depth': [4],
            'learning_rate' : [0.03],
            'l2_leaf_reg': [4],
            'iterations': [300],
            'thread_count': [4]
            }]
        ]

start_total = time.perf_counter()
for clf, param in zip (vote_est, grid_param):
    start = time.perf_counter()        
    best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split, scoring = 'roc_auc')
    best_search.fit(X, y)
    run = time.perf_counter() - start

    best_param = best_search.best_params_
    print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))
    clf[1].set_params(**best_param) 

run_total = time.perf_counter() - start_total
print('Total optimization time was {:.2f} minutes.'.format(run_total/60))

print('-'*15)
# Hard vote or majority rules w/Tuned Hyperparameters
grid_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
grid_hard_cv = model_selection.cross_validate(grid_hard, X, y, cv  = cv_split)
grid_hard.fit(X, y)

print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_hard_cv['train_score'].mean()*100)) 
print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))
print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_hard_cv['test_score'].std()*100*3))
print('-'*15)

# Soft vote or weighted probabilities w/Tuned Hyperparameters
grid_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
grid_soft_cv = model_selection.cross_validate(grid_soft, X, y, cv  = cv_split)
grid_soft.fit(X, y)

print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_soft_cv['train_score'].mean()*100)) 
print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))
print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_soft_cv['test_score'].std()*100*3))
print('-'*15)
test_df['Survived'] = grid_soft.predict(X_test)
test_df['Survived'] = test_df['Survived'].astype(int)
print('Validation Data Distribution: \n', test_df['Survived'].value_counts(normalize = True))
submit = test_df[['Survived']]
#submit.to_csv("../output/submission.csv", index=True)
columns = [c for c in data_df.columns if 'Deck' in c or 'Embarked' in c or 'Ticket' in c]

simple_data_df = data_df.copy(deep=True)
simple_data_df.drop(columns=columns, axis=1, inplace=True)
simple_data_df['Young'] = np.where((simple_data_df['AgeBin_Code']<2), 1, 0)
simple_data_df.drop(columns=['AgeBin_Code'], axis=1, inplace=True)
simple_data_df['P1_Male'] = np.where((simple_data_df['Sex']==0) & (simple_data_df['Pclass']==1), 1, 0)
simple_data_df['P2_Male'] = np.where((simple_data_df['Sex']==0) & (simple_data_df['Pclass']==2), 1, 0)
simple_data_df['P3_Male'] = np.where((simple_data_df['Sex']==0) & (simple_data_df['Pclass']==3), 1, 0)
simple_data_df['P1_Female'] = np.where((simple_data_df['Sex']==1) & (simple_data_df['Pclass']==1), 1, 0)
simple_data_df['P2_Female'] = np.where((simple_data_df['Sex']==1) & (simple_data_df['Pclass']==2), 1, 0)
simple_data_df['P3_Female'] = np.where((simple_data_df['Sex']==1) & (simple_data_df['Pclass']==3), 1, 0)

simple_data_df.drop(columns=['Pclass', 'Sex'], axis=1, inplace=True)
simple_train_df = simple_data_df[:TRAINING_LENGTH]
simple_test_df = simple_data_df[TRAINING_LENGTH:]

simple_data_df.sample(5)
X = simple_train_df.drop('Survived', 1)
y = simple_train_df['Survived']
X_test = simple_test_df.copy().drop(columns=['Survived'], axis=1)

std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_test = std_scaler.transform(X_test)
# Hyper-parameter tuning with GridSearchCV:
grid_param = [
            [{
            # RandomForestClassifier
            'criterion': ['gini'], #['gini', 'entropy'],
            'max_depth': [8], #[2, 4, 6, 8, 10, None],
            'n_estimators': [100], #[10, 50, 100, 300],
            'oob_score': [False] #[True, False]
             }],
    
            [{
            # KNeighborsClassifier
            'algorithm': ['auto'], #['auto', 'ball_tree', 'kd_tree', 'brute'],
            'n_neighbors': [7], #[1,2,3,4,5,6,7],
            'weights': ['distance'] #['uniform', 'distance']
            }],
    
            [{
            # XGBClassifier
            'learning_rate': [0.05], #[0.05, 0.1,0.16],
            'max_depth': [10], #[10,30,50],
            'min_child_weight' : [6], #[1,3,6]
            'n_estimators': [200]
             }],
    
            [{
            # LightGBMClassifier
            'learning_rate': [0.01], #[0.01,0.05,0.1],
            'n_estimators': [200],
            'num_leaves': [300], #[300,900,1200],
            'max_depth': [25], #[25,50,75],
             }],
    
            [{
            # CatBoostClassifier
            'depth': [4],
            'learning_rate' : [0.03],
            'l2_leaf_reg': [4],
            'iterations': [300],
            'thread_count': [4]
            }]
        ]

start_total = time.perf_counter()
for clf, param in zip (vote_est, grid_param):
    start = time.perf_counter()        
    best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split, scoring = 'roc_auc')
    best_search.fit(X, y)
    run = time.perf_counter() - start

    best_param = best_search.best_params_
    print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))
    clf[1].set_params(**best_param) 

run_total = time.perf_counter() - start_total
print('Total optimization time was {:.2f} minutes.'.format(run_total/60))

print('-'*15)
# Hard vote or majority rules w/Tuned Hyperparameters
grid_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
grid_hard_cv = model_selection.cross_validate(grid_hard, X, y, cv  = cv_split)
grid_hard.fit(X, y)

print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_hard_cv['train_score'].mean()*100)) 
print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))
print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_hard_cv['test_score'].std()*100*3))
print('-'*15)

# Soft vote or weighted probabilities w/Tuned Hyperparameters
grid_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
grid_soft_cv = model_selection.cross_validate(grid_soft, X, y, cv  = cv_split)
grid_soft.fit(X, y)

print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_soft_cv['train_score'].mean()*100)) 
print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))
print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_soft_cv['test_score'].std()*100*3))
print('-'*15)
simple_test_df['Survived'] = grid_soft.predict(X_test)
simple_test_df['Survived'] = test_df['Survived'].astype(int)
print('Validation Data Distribution: \n', simple_test_df['Survived'].value_counts(normalize = True))
submit = simple_test_df[['Survived']]
#submit.to_csv("../output/submission.csv", index=True)