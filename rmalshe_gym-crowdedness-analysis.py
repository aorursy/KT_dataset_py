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
data = pd.read_csv('../input/data.csv')

df = data

df['hour'] = df.timestamp.apply( lambda x: int(np.floor(x/3600))) 

df.head().T
data.columns
data.head(n=2).T
data.describe()
categorical_features = (data.select_dtypes(include=['object']).columns.values)

categorical_features
numerical_features = data.select_dtypes(include = ['float64', 'int64']).columns.values

numerical_features
pivot = pd.pivot_table(df,

            values = ['number_people'],

            index = ['hour'], 

                       columns= ['is_start_of_semester', 'day_of_week'],

                       aggfunc=[np.mean], 

                       margins=True).fillna('')

pivot
pivot = pd.pivot_table(df,

            values = ['number_people'],

            index = ['hour'], 

            columns= ['day_of_week'],

            aggfunc=[np.mean], 

            margins=True)

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = False)

plt.subplots(figsize = (20, 20))

sns.heatmap(pivot,linewidths=0.2,square=True, cmap="coolwarm")
df.columns.values
for i in set(df['day_of_week']):

    aa= df[df['day_of_week'].isin([i])]

    g = sns.factorplot(x='day_of_week', 

                       y="number_people",

                       data=aa, 

                       saturation=1, 

                       kind="bar", 

                       ci=None, 

                       aspect=1.2, 

                       linewidth=1, 

                       hue = 'hour', 

                      col = 'is_start_of_semester') 

    locs, labels = plt.xticks()

    plt.setp(labels, rotation=90)
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
df_small = df[['number_people', 

               'day_of_week', 

#               'is_weekend',

#               'is_holiday', 

               'apparent_temperature', 

               'hour',

#               'temperature',

               'is_start_of_semester']]

sns.pairplot(df_small, hue='day_of_week')
df.columns.values
#data = df

sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(2, 3, figsize=(20,20))

sns.despine(left=True)

sns.distplot(df['number_people'],         kde=False, color="r", ax=axes[0, 0])

sns.distplot(df['timestamp'],             kde=False, color="g", ax=axes[0, 1])

sns.distplot(df['is_holiday'],            kde=False, color="b", ax=axes[0, 2])

sns.distplot(df['apparent_temperature'],  kde=False, color="r", ax=axes[1, 0])

sns.distplot(df['temperature'],           kde=False, color="g", ax=axes[1, 1])

sns.distplot(df['hour'],                  kde=False, color="b", ax=axes[1, 2])

plt.tight_layout()
mod_df = pd.get_dummies(df)

categorical_features = (mod_df.select_dtypes(include=['object']).columns.values)

categorical_features
mod_df_variable_correlations = mod_df.corr()

#variable_correlations

heat_map(mod_df_variable_correlations)
df.columns
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



df_small = df[['number_people', 

               'day_of_week', 

               'is_weekend',

               'is_holiday', 

               'apparent_temperature', 

               #'temperature',

               'hour',

               'is_start_of_semester']]





df_copy = pd.get_dummies(df_small)



df1 = df_copy

y = np.asarray(df1['number_people'], dtype="|S6")

df1 = df1.drop(['number_people'],axis=1)

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
import warnings

warnings.filterwarnings('ignore')



from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import RFECV, SelectKBest

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, Lasso

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB, BernoulliNB

from sklearn.neighbors import KNeighborsClassifier



classifiers = [('RandomForestClassifierG', RandomForestClassifier(n_jobs=-1, criterion='gini')),

               ('RandomForestClassifierE', RandomForestClassifier(n_jobs=-1, criterion='entropy')),

               ('AdaBoostClassifier', AdaBoostClassifier()),

               ('ExtraTreesClassifier', ExtraTreesClassifier(n_jobs=-1)),

               ('KNeighborsClassifier', KNeighborsClassifier(n_jobs=-1)),

               ('DecisionTreeClassifier', DecisionTreeClassifier()),

               ('ExtraTreeClassifier', ExtraTreeClassifier()),

               ('LogisticRegression', LogisticRegression()),

               ('GaussianNB', GaussianNB()),

               ('BernoulliNB', BernoulliNB())

              ]

allscores = []



x, Y = df_copy.drop('number_people', axis=1), df_copy['number_people']



for name, classifier in classifiers:

    scores = []

    for i in range(1): # 3 runs

        roc = cross_val_score(classifier, x, Y)

        scores.extend(list(roc))

    scores = np.array(scores)

    print(name, scores.mean())

    new_data = [(name, score) for score in scores]

    allscores.extend(new_data)
temp = pd.DataFrame(allscores, columns=['classifier', 'score'])

#sns.violinplot('classifier', 'score', data=temp, inner=None, linewidth=0.3)

plt.figure(figsize=(15,10))

sns.factorplot(x='classifier', 

               y="score",

               data=temp, 

               saturation=1, 

               kind="box", 

               ci=None, 

               aspect=1, 

               linewidth=1, 

               size = 10)     

locs, labels = plt.xticks()

plt.setp(labels, rotation=90)