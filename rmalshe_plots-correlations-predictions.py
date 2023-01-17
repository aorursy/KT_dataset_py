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
data1 = pd.read_csv('../input/student-mat.csv')

data2 = pd.read_csv('../input/student-por.csv')



data = [data1,data2]

data=pd.concat(data)

data=data.drop_duplicates(["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"])

data['AvgGrade'] = data[['G1', 'G2', 'G3']].mean(axis=1)
data.columns
data.head(n=10).T
data.head(n=2)
data.describe()
categorical_features = (data.select_dtypes(include=['object']).columns.values)

categorical_features
numerical_features = data.select_dtypes(include = ['float64', 'int64']).columns.values

numerical_features
df = data

pivot = pd.pivot_table(df,

            values = ['AvgGrade', 'G1', 'G2', 'G3'''],

            index = ['school',

                     'sex', 

                    'famsize',

                     'paid',

                    'guardian'], 

                       columns= ['Mjob'],

                       aggfunc=[np.mean], 

                       margins=True).fillna('')

pivot
pivot = pd.pivot_table(df,

            values = ['AvgGrade'],

            index = ['school',

                     'sex', 

                    'famsize',

                     'paid',

                    ],

                           columns= ['Mjob'],

                           aggfunc=[np.mean, np.std], 

                           margins=True)

pivot

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

plt.subplots(figsize = (30, 20))

sns.heatmap(pivot,linewidths=0.2,square=True )
df = data

pivot = pd.pivot_table(df,

            values = ['AvgGrade'],

            index = ['school',

                     'sex',

                     'famsize',

                     'paid'],

                       columns= ['Fjob'],

                       aggfunc=[np.mean], 

                       margins=True).fillna('')

pivot
pivot = pd.pivot_table(df,

            values = ['AvgGrade'],

            index = ['school',

                     'sex',

                     'famsize',

                     'paid'],

                       columns= ['Fjob'],

                       aggfunc=[np.mean, np.std],

                       margins=True)

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

plt.subplots(figsize = (30, 20))

sns.heatmap(pivot,linewidths=0.2,square=True)
df = data

pivot = pd.pivot_table(df,

            values = ['AvgGrade'],

            index = ['school',

                     'sex',

                     'famsize',

                     'paid'],

                       columns= ['internet', 'goout'],

                       aggfunc=[np.mean], 

                       margins=True).fillna('')

pivot
pivot = pd.pivot_table(df,

            values = ['AvgGrade'],

            index = ['school',

                     'sex',

                     'famsize',

                     'paid'],

                       columns= ['internet', 'goout'],

                       aggfunc=[np.mean], 

                       margins=True)

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

plt.subplots(figsize = (30, 20))

sns.heatmap(pivot,linewidths=0.2,square=True)
df.columns
InputFile2_reduced=df

for i in set(InputFile2_reduced['sex']):

    aa= InputFile2_reduced[InputFile2_reduced['sex'].isin([i])]

    g = sns.factorplot(x='Walc', y="AvgGrade",data=aa, 

                   saturation=1, kind="box", 

                   ci=None, aspect=1, linewidth=1, row='Fjob', col = 'Mjob') 

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
df.columns
df_small = df[['AvgGrade', 

               'G1',  

               'G2', 

               'G3',

               'Dalc', 

               'goout',

               'freetime',

               'Medu',

               'Fedu']]

sns.pairplot(df_small)
#data = df

sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(4, 4, figsize=(20,20))

sns.despine(left=True)

sns.distplot(df['AvgGrade'],  kde=False, color="b", ax=axes[0, 0])

sns.distplot(df['G3'],        kde=False, color="b", ax=axes[0, 1])

sns.distplot(df['G2'],        kde=False, color="b", ax=axes[0, 2])

sns.distplot(df['G1'],        kde=False, color="b", ax=axes[0, 3])

sns.distplot(df['studytime'], kde=False, color="b", ax=axes[1, 0])

sns.distplot(df['freetime'],  kde=False, color="b", ax=axes[1, 1])

sns.distplot(df['goout'],     kde=False, color="b", ax=axes[1, 2])

sns.distplot(df['absences'],  kde=False, color="b", ax=axes[1, 3])

sns.distplot(df['Dalc'],      kde=False, color="b", ax=axes[2, 0])

sns.distplot(df['Walc'],      kde=False, color="b", ax=axes[2, 1])

sns.distplot(df['health'],    kde=False, color="b", ax=axes[2, 2])

sns.distplot(df['famrel'],    kde=False, color="b", ax=axes[2, 3])

sns.distplot(df['traveltime'],kde=False, color="b", ax=axes[3, 0])

sns.distplot(df['age'],       kde=False, color="b", ax=axes[3, 1])

sns.distplot(df['Medu'],      kde=False, color="b", ax=axes[3, 2])

sns.distplot(df['Fedu'],      kde=False, color="b", ax=axes[3, 3])

plt.tight_layout()
data.head()
mod_df = df 

binaryYesNo = {'yes': 1, 'no': 0}

school_map  = {'MS': 1, 'GP': 2}

sex_map     = {'M': 1, 'F': 2}

address_map = {'R':1, 'U':2}

famsize_map = {'LE3':1, 'GT3':2}

pstatus_map = {'A':1, 'T':2}

mjob_map    = {'services' : 1, 

            'health' : 2, 

            'other' : 3, 

            'at_home' : 4, 

            'teacher' : 5}



fjob_map    = {'services' : 1, 

            'health' : 2, 

            'other' : 3, 

            'at_home' : 4, 

            'teacher' : 5}



reason_map   = {'course':1, 'other':2, 'reputation':3, 'home':4}

guardian_map = {'other':0, 'father':1, 'mother':1}



mod_df.schoolsup  = mod_df.schoolsup.map(binaryYesNo)

mod_df.famsup     = mod_df.famsup.map(binaryYesNo)

mod_df.paid       = mod_df.paid.map(binaryYesNo)

mod_df.activities = mod_df.activities.map(binaryYesNo)

mod_df.nursery    = mod_df.nursery.map(binaryYesNo)

mod_df.higher     = mod_df.higher.map(binaryYesNo)

mod_df.internet   = mod_df.internet.map(binaryYesNo)

mod_df.romantic   = mod_df.romantic.map(binaryYesNo)



mod_df.school   = mod_df.school.map(school_map)

mod_df.sex      = mod_df.sex.map(sex_map)

mod_df.address  = mod_df.address.map(address_map)

mod_df.famsize  = mod_df.famsize.map(famsize_map)

mod_df.Pstatus  = mod_df.Pstatus.map(pstatus_map)

mod_df.Mjob     = mod_df.Mjob.map(mjob_map)

mod_df.Fjob     = mod_df.Fjob.map(fjob_map)

mod_df.reason   = mod_df.reason.map(reason_map)

mod_df.guardian = mod_df.guardian.map(guardian_map)



#mod_df.to_csv(path + 'modified_df.csv')
categorical_features = (mod_df.select_dtypes(include=['object']).columns.values)

categorical_features
mod_df_variable_correlations = mod_df.corr()

#variable_correlations

heat_map(mod_df_variable_correlations)
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



df_copy = pd.get_dummies(mod_df)



df1 = df_copy

y = np.asarray(df1['AvgGrade'], dtype="|S6")

df1 = df1.drop(['AvgGrade'],axis=1)

X = df1.values

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.50)



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

               ('BernoulliNB', BernoulliNB()), 

              ]

allscores = []



x, Y = mod_df.drop('AvgGrade', axis=1), np.asarray(mod_df['AvgGrade'], dtype="|S6")



for name, classifier in classifiers:

    scores = []

    for i in range(20): # 20 runs

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