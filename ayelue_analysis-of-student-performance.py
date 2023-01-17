import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import scipy.stats as stats

import statsmodels

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

from statsmodels.formula.api import ols

from sklearn.model_selection import train_test_split



data = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
sns.set(style='whitegrid')
data.columns
data.rename(columns={

                    'race/ethnicity':'race',

                    'parental level of education': 'parent_education',

                    'test preparation course': 'pretest'

                    },inplace=True)

data.columns
data.info()
data.isna().sum()
data['avg_score'] = data.loc[:,['math score','reading score','writing score']].apply(np.mean, axis=1).round(4)
sns.distplot(data['avg_score'])
_ = stats.probplot(data['avg_score'], plot=sns.mpl.pyplot)
p = sns.countplot(x='gender', data=data, palette='muted')

_ = plt.setp(p.get_xticklabels(), rotation=90)
sns.boxplot(x='gender',y='avg_score',data=data)
sns.distplot(data[data['gender']=='female']['avg_score'])
sns.distplot(data[data['gender']=='male']['avg_score'])
def prepare_anova_data(column_name):

    list_names_factor_type = list(data[column_name].unique())

    n_sample = data[column_name].value_counts().min()

    groups = [data[data[column_name]==key].sample(n_sample) for key in list_names_factor_type]

    pre_data = pd.concat(groups)

    return pre_data
results = ols('avg_score ~ C(gender)', data=prepare_anova_data('gender')).fit()

results.summary()

aov_table = sm.stats.anova_lm(results, typ=2)

aov_table
p = sns.countplot(x='race', data=data, order=data['race'].value_counts().index, palette='muted')

_ = plt.setp(p.get_xticklabels(), rotation=90)
p = sns.boxplot(x='race',y='avg_score', data=data, palette='muted')

_ = plt.setp(p.get_xticklabels(), rotation=90)
sns.distplot(data[data['race']=='group A']['avg_score'],bins=20)
sns.distplot(data[data['race']=='group B']['avg_score'],bins=20)
sns.distplot(data[data['race']=='group C']['avg_score'],bins=20)
sns.distplot(data[data['race']=='group D']['avg_score'],bins=20)
sns.distplot(data[data['race']=='group E']['avg_score'],bins=15)
results = ols('avg_score ~ C(race)', data=prepare_anova_data('race')).fit()

results.summary()

aov_table = sm.stats.anova_lm(results, typ=2)

aov_table
order=list(data['parent_education'].value_counts().index)

order
p = sns.countplot(x='parent_education', data=data, order=order, palette='muted')

_ = plt.setp(p.get_xticklabels(), rotation=90)
sns.distplot(data[data['parent_education']==order[0]]['avg_score'])
sns.distplot(data[data['parent_education']==order[1]]['avg_score'])
sns.distplot(data[data['parent_education']==order[2]]['avg_score'])
sns.distplot(data[data['parent_education']==order[3]]['avg_score'])
sns.distplot(data[data['parent_education']==order[4]]['avg_score'],bins=15)
sns.distplot(data[data['parent_education']==order[5]]['avg_score'],bins=12)
p = sns.boxplot(data=data, x='parent_education', y='avg_score')

_ = plt.setp(p.get_xticklabels(), rotation=90)
p = sns.countplot(x='parent_education',hue='race', data=data, order=data['parent_education'].value_counts().index, palette='muted')

plt.title('sorted by parent_education')

_ = plt.setp(p.get_xticklabels(), rotation=90)
p = sns.countplot(x='lunch', data=data, palette='muted')
sns.boxplot(x='lunch',y='avg_score',data=data)
sns.distplot(data[data['lunch']=='standard']['avg_score'])
sns.distplot(data[data['lunch']=='free/reduced']['avg_score'])
results = ols('avg_score ~ lunch', data=prepare_anova_data('lunch')).fit()

aov_table = sm.stats.anova_lm(results, typ=2)

aov_table
p = sns.countplot(x='pretest', data=data,palette='muted')

_ = plt.setp(p.get_xticklabels(),rotation=0)
sns.boxplot(x=data['pretest'],y=data['avg_score'])
sns.boxplot(x=data['race'],y=data['avg_score'],hue=data['pretest'])
p = sns.lineplot(x=data['parent_education'],y=data['avg_score'],hue=data['pretest'])

_ = plt.setp(p.get_xticklabels(),rotation=90)
results = ols('avg_score ~ parent_education', data=prepare_anova_data('parent_education')).fit()

results.summary()

aov_table = sm.stats.anova_lm(results, typ=2)

aov_table.round(4)
results = ols('avg_score ~ pretest', data=prepare_anova_data('pretest')).fit()

results.summary()

aov_table = sm.stats.anova_lm(results, typ=2)

aov_table
model = ols('avg_score ~ C(race)*C(lunch)*C(gender)', data).fit()

aov_table = sm.stats.anova_lm(model, typ=2)

aov_table.round(4)
