import smtplib

from matplotlib import style

import seaborn as sns

sns.set(style='ticks', palette='RdBu')

#sns.set(style='ticks', palette='Set2')

import pandas as pd

import numpy as np

import time

import datetime 

%matplotlib inline

import matplotlib.pyplot as plt

from subprocess import check_output

pd.options.display.max_colwidth = 1000

from time import gmtime, strftime

Time_now = strftime("%Y-%m-%d %H:%M:%S", gmtime())

import timeit

start = timeit.default_timer()

pd.options.display.max_rows = 100

from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import RFECV, SelectKBest

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB, BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import SelectFromModel

from sklearn import svm

from scipy.stats import skew

from scipy.stats.stats import pearsonr

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score
data = pd.read_csv("../input/HospInfo.csv")

df = data
data.columns.values
data.head(n=2).T
data.describe()
categorical_features = (data.select_dtypes(include=['object']).columns.values)

categorical_features
numerical_features = data.select_dtypes(include = ['float64', 'int64']).columns.values

numerical_features
v1 = list(set(df['Emergency Services']))

v2 = list(set(df['Meets criteria for meaningful use of EHRs']))

v3 = list(set(df['Hospital overall rating']))

v4 = list(set(df['Mortality national comparison']))

v5 = list(set(df['Safety of care national comparison']))

v6 = list(set(df['Readmission national comparison']))

v7 = list(set(df['Patient experience national comparison']))

v8 = list(set(df['Effectiveness of care national comparison']))

v9 = list(set(df['Timeliness of care national comparison']))

v10 = list(set(df['Efficient use of medical imaging national comparison']))



print (v1)

print (v2)

print (v3)

print (v4)

print (v5)

print (v6)

print (v7)

print (v8)

print (v9)

print (v10)
mod_df = df 

True_false_map = {'False':0,'True':1}

NAvail_number_map = {'Not Available':0, 

                     'Above the National average':3, 

                     'Same as the National average':2, 

                     'Below the National average':1}

emergency_number_map = {'Not Available':0, 

                       1:1, 

                        3:3, 

                        2:2, 

                        5:5, 

                        4:4}

mod_df['Emergency Services']  = mod_df['Emergency Services'].astype(int)

mod_df['Meets criteria for meaningful use of EHRs'] = mod_df['Meets criteria for meaningful use of EHRs'].fillna('0')

mod_df['Meets criteria for meaningful use of EHRs'] = mod_df['Meets criteria for meaningful use of EHRs'].astype(int)

mod_df['Hospital overall rating'] = pd.to_numeric(mod_df['Hospital overall rating'], errors='coerce')

mod_df['Hospital overall rating'] = mod_df['Hospital overall rating'].fillna('0')

mod_df['Hospital overall rating'] = mod_df['Hospital overall rating'].astype(float)

mod_df['Mortality national comparison']=mod_df['Mortality national comparison'].map(NAvail_number_map)

mod_df['Safety of care national comparison']=mod_df['Safety of care national comparison'].map(NAvail_number_map)

mod_df['Readmission national comparison']=mod_df['Readmission national comparison'].map(NAvail_number_map)

mod_df['Patient experience national comparison']=mod_df['Patient experience national comparison'].map(NAvail_number_map)

mod_df['Effectiveness of care national comparison']=mod_df['Effectiveness of care national comparison'].map(NAvail_number_map)

mod_df['Timeliness of care national comparison']=mod_df['Timeliness of care national comparison'].map(NAvail_number_map)

mod_df['Efficient use of medical imaging national comparison']=mod_df['Efficient use of medical imaging national comparison'].map(NAvail_number_map)

v1 = list(set(mod_df['Emergency Services']))

v2 = list(set(mod_df['Meets criteria for meaningful use of EHRs']))

v3 = list(set(mod_df['Hospital overall rating']))

v4 = list(set(mod_df['Mortality national comparison']))

v5 = list(set(mod_df['Safety of care national comparison']))

v6 = list(set(mod_df['Readmission national comparison']))

v7 = list(set(mod_df['Patient experience national comparison']))

v8 = list(set(mod_df['Effectiveness of care national comparison']))

v9 = list(set(mod_df['Timeliness of care national comparison']))

v10 = list(set(mod_df['Efficient use of medical imaging national comparison']))



print (v1)

print (v2)

print (v3)

print (v4)

print (v5)

print (v6)

print (v7)

print (v8)

print (v9)

print (v10)
data.describe()
numerical_features = data.select_dtypes(include = ['float64', 'int64']).columns.values

numerical_features
categorical_features = (data.select_dtypes(include=['object']).columns.values)

categorical_features
pivot = pd.pivot_table(df,

            values = ['Hospital overall rating', 

                      'Mortality national comparison',

                       'Safety of care national comparison',

                       'Readmission national comparison',

                       'Patient experience national comparison',

                       'Effectiveness of care national comparison',

                       'Timeliness of care national comparison',

                       'Efficient use of medical imaging national comparison'],

            index = ['State'], 

            columns= [],

            aggfunc=[np.mean], 

            margins=True).sort_values(by=('mean', 'Hospital overall rating'), ascending=False).fillna('')

pivot
pivot = pd.pivot_table(df,

            values = ['Hospital overall rating', 

                      'Mortality national comparison',

                       'Safety of care national comparison',

                       'Readmission national comparison',

                       'Patient experience national comparison',

                       'Effectiveness of care national comparison',

                       'Timeliness of care national comparison',

                       'Efficient use of medical imaging national comparison'],

            index = ['State'], 

            columns= [],

            aggfunc=[np.mean], 

            margins=True).sort_values(by=('mean', 'Hospital overall rating'), ascending=False)

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

plt.subplots(figsize = (20, 50))

sns.heatmap(pivot,linewidths=0.2,square=True )
pivot = pd.pivot_table(df,

            values = ['Hospital overall rating', 

#                      'Mortality national comparison',

#                       'Safety of care national comparison',

#                       'Readmission national comparison',

#                       'Patient experience national comparison',

#                       'Effectiveness of care national comparison',

#                       'Timeliness of care national comparison',

#                       'Efficient use of medical imaging national comparison'

                     ],

            index = ['State'], 

            columns= ['Hospital Ownership'],

            aggfunc=[np.mean], 

            margins=True).sort_values(by=('mean', 'Hospital overall rating', 'All'), ascending=False)

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

plt.subplots(figsize = (20, 50))

sns.heatmap(pivot,linewidths=0.2,square=True )
pivot = pd.pivot_table(df,

            values = ['Provider ID'],

            index =  ['Hospital Type'], 

            columns= ['Hospital Ownership'],

            aggfunc=[np.count_nonzero], 

            margins=True).sort_values(by=('count_nonzero', 'Provider ID', 'All'), ascending=False)

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

plt.subplots(figsize = (20, 10))

sns.heatmap(pivot,linewidths=0.2,square=True )
pivot = pd.pivot_table(df,

            values = ['Provider ID'],

            index =  ['State'], 

            columns= ['Hospital Ownership', 'Hospital Type'],

            aggfunc=[np.count_nonzero], 

            margins=True).sort_values(by=('count_nonzero', 'Provider ID', 'All'), ascending=False)

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

plt.subplots(figsize = (30, 20))

sns.heatmap(pivot,linewidths=0.2,square=True )
df.columns.values
def heat_map(corrs_mat):

    sns.set(style="white")

    f, ax = plt.subplots(figsize=(20, 20))

    mask = np.zeros_like(corrs_mat, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True 

    # Generate a custom diverging colormap

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corrs_mat, mask=mask, cmap=cmap, ax=ax)



variable_correlations = df.corr()

#variable_correlations

heat_map(variable_correlations)
df_small = mod_df[[#'State', 

                   'Hospital overall rating', 

                   'Mortality national comparison',

                   'Safety of care national comparison',

                   'Readmission national comparison',

                   #'Patient experience national comparison',

                   #'Effectiveness of care national comparison',

                   #'Timeliness of care national comparison',

                   #'Efficient use of medical imaging national comparison'

    ]]

sns.pairplot(df_small, hue='Hospital overall rating')
df_small = mod_df[[#'State', 

                   'Hospital overall rating', 

                   #'Mortality national comparison',

                   #'Safety of care national comparison',

                   #'Readmission national comparison',

                   'Patient experience national comparison',

                   'Effectiveness of care national comparison',

                   #'Timeliness of care national comparison',

                   #'Efficient use of medical imaging national comparison'

    ]]

sns.pairplot(df_small, hue='Hospital overall rating')
#data = df

sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(2, 4, figsize=(20,20))

sns.despine(left=True)

sns.distplot(df['Hospital overall rating'],                   kde=False, color="r", ax=axes[0, 0])

sns.distplot(df['Mortality national comparison'],             kde=False, color="g", ax=axes[0, 1])

sns.distplot(df['Safety of care national comparison'],        kde=False, color="b", ax=axes[0, 2])

sns.distplot(df['Readmission national comparison'],           kde=False, color="c", ax=axes[0, 3])

sns.distplot(df['Patient experience national comparison'],    kde=False, color="r", ax=axes[1, 0])

sns.distplot(df['Effectiveness of care national comparison'], kde=False, color="g", ax=axes[1, 1])

sns.distplot(df['Timeliness of care national comparison'],    kde=False, color="b", ax=axes[1, 2])

sns.distplot(df['Efficient use of medical imaging national comparison'],  kde=False, color="c", ax=axes[1, 3])



plt.tight_layout()
for i in set(mod_df['Hospital Ownership']):

    aa= mod_df[mod_df['Hospital Ownership'].isin([i])]

    g = sns.factorplot(x='State', 

                       y='Hospital overall rating',

                       data=aa, 

                       saturation=1, 

                       kind="bar", 

                       ci=None, 

                       aspect=3, 

                       linewidth=1, 

                      row = 'Hospital Ownership') 

    locs, labels = plt.xticks()

    plt.setp(labels, rotation=90)
for i in set(mod_df['Hospital Type']):

    aa= mod_df[mod_df['Hospital Type'].isin([i])]

    g = sns.factorplot(x='State', 

                       y='Hospital overall rating',

                       data=aa, 

                       saturation=1, 

                       kind="bar", 

                       ci=None, 

                       aspect=3, 

                       linewidth=1, 

                      row = 'Hospital Type') 

    locs, labels = plt.xticks()

    plt.setp(labels, rotation=90)
for i in set(mod_df['State']):

    aa= mod_df[mod_df['State'].isin([i])]

    g = sns.factorplot(x='City', 

                       y='Hospital overall rating',

                       data=aa, 

                       saturation=1, 

                       kind="bar", 

                       ci=None, 

                       aspect=3, 

                       linewidth=1, 

                      row = 'State') 

    locs, labels = plt.xticks()

    plt.setp(labels, rotation=90)
df_best = df[df['Hospital overall rating']>4]
pivot = pd.pivot_table(df_best,

            values = ['Provider ID'],

            index =  ['State'], 

            columns= ['Hospital Ownership', 'Hospital Type'],

            aggfunc=[np.count_nonzero], 

            margins=True).sort_values(by=('count_nonzero', 'Provider ID', 'All'), ascending=False)

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

plt.subplots(figsize = (30, 20))

sns.heatmap(pivot,linewidths=0.2,square=True )
df_worst = df[df['Hospital overall rating']<2]
pivot = pd.pivot_table(df_worst,

            values = ['Provider ID'],

            index =  ['State'], 

            columns= ['Hospital Ownership', 'Hospital Type'],

            aggfunc=[np.count_nonzero], 

            margins=True).sort_values(by=('count_nonzero', 'Provider ID', 'All'), ascending=False)

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

plt.subplots(figsize = (30, 20))

sns.heatmap(pivot,linewidths=0.2,square=True )
from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.calibration import CalibratedClassifierCV

#import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression

from sklearn import svm



df_small = mod_df[['State', 

                   'Hospital overall rating', 

                   'Mortality national comparison',

                   'Safety of care national comparison',

                   'Readmission national comparison',

                   'Patient experience national comparison',

                   'Effectiveness of care national comparison',

                   'Timeliness of care national comparison',

                   'Efficient use of medical imaging national comparison']]



df_copy = pd.get_dummies(df_small)



df1 = df_copy

y = np.asarray(df1['Hospital overall rating'], dtype="|S6")

df1 = df1.drop(['Hospital overall rating'],axis=1)

X = df1.values

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30)



radm = RandomForestClassifier()

radm.fit(Xtrain, ytrain)



clf = radm

indices = np.argsort(radm.feature_importances_)[::-1]



# Print the feature ranking

print('Feature ranking:')



for f in range(df1.shape[1]):

    print('%d. feature %d %s (%f)' % (f+1 , 

                                      indices[f], 

                                      df1.columns[indices[f]], 

                                      radm.feature_importances_[indices[f]]))