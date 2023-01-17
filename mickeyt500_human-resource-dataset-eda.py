import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
hr_main = pd.read_csv("../input/HR_comma_sep.csv")
hr_main.head(5)
hr_main.dtypes
hr_main.nunique()
c, n = 'Categorical', 'Continuous'
h = {'satisfaction_level':n, 'last_evaluation':n, 'number_project':c, 'average_montly_hours':n, 

     'time_spend_company':c, 'Work_accident':c, 'left':c, 'promotion_last_5years':c, 'sales':c, 'salary':c}

hr_class = pd.Series(h)

hr_class
new_order = ['sales', 'left', 'salary','number_project', 'Work_accident','promotion_last_5years','time_spend_company'

             ,'satisfaction_level','last_evaluation','average_montly_hours']

hr_main = hr_main[new_order]

hr_main.head()
hr_main.shape
hr_main.isnull().sum()
dupes = hr_main[hr_main.duplicated(keep=False)]

dupes.sort_values(['satisfaction_level','last_evaluation','average_montly_hours']).head(10)
dupes.shape
hr_nondup = hr_main.drop_duplicates(keep='first')

hr_nondup.head()
hr_nondup.shape
hr_main.shape
hr_main.describe()
hr_nondup.describe()
hr_main.describe(include=['object'])
hr_nondup.describe(include=['object'])
hr_main['sales'].value_counts(normalize=True)
hr_nondup['sales'].value_counts(normalize=True)
hr_main['salary'].value_counts(normalize=True)
hr_nondup['salary'].value_counts(normalize=True)
hr_main.time_spend_company.value_counts()
hr_main.sales.value_counts()
sns.boxplot(y='time_spend_company', data=hr_main, color='plum')
sns.violinplot(y='time_spend_company', data=hr_main, color='plum')
hr_main.time_spend_company.value_counts(normalize=True).round(2)
#Lets find our Outlier Boundary (more than 3 standard deviations away from the mean)

std = mean = hr_main.time_spend_company.std()

mean = hr_main.time_spend_company.mean()

mean + (3*std)
outliers = hr_main.time_spend_company > 7.9
hr_main['outliers'] = outliers

hr_main.outliers = hr_main.outliers.astype(int)

hr_main.query('outliers == 1').head()
sns.countplot(x='salary', data=hr_main)
plt.figure(figsize=(10,5))

order = hr_main.sales.value_counts().index

sns.countplot(y='sales', data=hr_main, order=order)

sns.despine()
plt.figure(figsize=(8,5))

sns.countplot(x='number_project', data=hr_main)
plt.figure(figsize=(8,6))

sns.countplot(x='time_spend_company', data=hr_main)
sns.boxplot(y='satisfaction_level',data=hr_main, color='mediumspringgreen')
sns.violinplot(y='satisfaction_level',data=hr_main, color='mediumspringgreen')
sns.distplot(hr_main.satisfaction_level)
hr_main['satisfaction'] = hr_main.satisfaction_level

hr_main.satisfaction = pd.cut(hr_main['satisfaction'], bins=[0,0.2,0.4,0.6,0.8,1.0], 

       include_lowest=True, labels=['VD','D','N','S','VS'])

hr_main.head()
hr_main.satisfaction.value_counts(normalize=True).round(2)
#Lets find out if the number of projects assigned per employee is related to their satisfaction level

proj_sat = hr_main.groupby(['satisfaction','number_project']).size().unstack()

proj_sat
sns.heatmap(proj_sat,cmap='coolwarm',linecolor='white', linewidths=1)
#Now Lets try a Bivariate analysis on the continuos form of satisfaction with monthly hours

sns.jointplot(x='satisfaction_level', y='average_montly_hours', data=hr_main, size=8, kind='hex')
sns.lmplot(x='satisfaction_level',y='average_montly_hours', data=hr_main, hue='satisfaction')
g = sns.factorplot(x='last_evaluation',y='average_montly_hours',

               col='satisfaction',hue='satisfaction',data=hr_main, kind='strip', col_wrap=2, size=6)

g.set(xticks=range(0,1))
#Lets calculate the average monthly hours per project

hr_main['average_hrs_project'] = hr_main.average_montly_hours / hr_main.number_project

hr_main.head()
hr_main.groupby('satisfaction')['average_montly_hours'].mean().reset_index()
hr_main.groupby('number_project')['average_montly_hours'].mean().reset_index()
#Lets compare some categories vs some continuous variables. Due to number of x-variables, we will split these into 2.

g = sns.PairGrid(hr_main,

                 x_vars=["number_project", "time_spend_company"],

                 y_vars=["satisfaction_level", "last_evaluation", "average_montly_hours", 

                         "average_hrs_project"], size=2.5, aspect=3)

g.map(sns.pointplot, ci=0)
h = sns.PairGrid(hr_main,

                 x_vars=["sales",'salary'],

                 y_vars=["satisfaction_level", "last_evaluation", "average_montly_hours", 

                         "average_hrs_project"], size=2.5, aspect=3)

h.map(sns.pointplot, ci=0)
order= ['VD','D','N','S','VS']

sns.factorplot(x='satisfaction', y='average_montly_hours', data=hr_main, kind='bar', palette="coolwarm",

               col='sales', col_wrap=3, order=order)
sns.factorplot(x='satisfaction', y='average_montly_hours', data=hr_main, kind='bar', col='salary', order=order)
#Lets compare some continuous variables with others.

sns.pairplot(hr_main.sample(frac=.3), 

             diag_kind='kde',

             vars=['satisfaction_level', 'last_evaluation', 'average_montly_hours'],

             hue='left', 

             plot_kws={"s": 3}, 

             size=5)
plt.figure(figsize=(10,5))

order = hr_main.sales.value_counts().index

sns.countplot(y='sales', hue='left',data=hr_main, order=order)
hr_main1 = pd.read_csv("../input/HR_comma_sep.csv")

lazy_workers = hr_main1[(hr_main1['last_evaluation']<0.6) 

                        & (hr_main1['average_montly_hours']<170)]

lazy_workers.left.value_counts().reset_index()
hard_workers = hr_main1[(hr_main1['last_evaluation']>0.8) 

                        & (hr_main1['average_montly_hours']>220)]

hard_workers.left.value_counts().reset_index()
hard_workers.mean().reset_index()
lazy_workers.mean().reset_index()
order=['low','medium','high']

sns.countplot(x='salary',hue='left',data=hard_workers,order=order)
sns.countplot(x='salary',hue='left',data=lazy_workers,order=order)
sns.set(style="white")

corr = hr_main1.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)

plt.figure(figsize=(11,9))

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.factorplot(x='number_project',y='average_montly_hours',data=hr_main, hue='left', 

               col='sales', col_wrap=4, kind='point', palette="husl")
sns.factorplot(x='number_project',y='last_evaluation',data=hr_main, hue='left', 

               col='sales', col_wrap=4, kind='point',palette="hls")
from sklearn.model_selection import KFold, cross_val_score

from sklearn.ensemble import RandomForestClassifier
hr_simp=hr_main[['satisfaction_level','average_montly_hours','time_spend_company',

                 'last_evaluation','number_project','left']]
hr_simp_bin = pd.get_dummies(hr_simp)
hr_simp_bin = pd.get_dummies(hr_simp)



y = hr_simp_bin.pop('left')

X = hr_simp_bin.values



kfold = KFold(n_splits=5, random_state=4, shuffle=True)

model = RandomForestClassifier()

results = cross_val_score(model, X, y, cv=kfold, scoring='r2')
results.mean()