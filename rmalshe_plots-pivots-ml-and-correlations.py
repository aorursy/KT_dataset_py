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

path = 'C:/Users/rmalshe/Desktop/CareerDevelopment/PROGRAMMING_LANGUAGES/iPythonNoteBooks/LIBRARY/'



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
data = pd.read_csv("../input/Library_Usage.csv")

"../input/HR_comma_sep.csv"
data.columns
data.head(n=5).T
data.head(n=2)
data.describe()
categorical_features = (data.select_dtypes(include=['object']).columns.values)

categorical_features
numerical_features = data.select_dtypes(include = ['float64', 'int64']).columns.values

numerical_features
df = data

pivot = pd.pivot_table(df,

            values = ['Total Checkouts'],

            index = ['Patron Type Definition'], 

                       columns= ['Age Range'],

                       aggfunc=[np.mean], 

                       margins=True).fillna('')

pivot
pivot = pd.pivot_table(df,

            values = ['Total Checkouts'],

            index = ['Patron Type Definition'], 

                       columns= ['Age Range'],

                       aggfunc=[np.mean], 

                       margins=True)

pivot

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

plt.subplots(figsize = (30, 20))

sns.heatmap(pivot,linewidths=0.2,square=True )
df = data

pivot = pd.pivot_table(df,

            values = ['Total Renewals'],

            index = ['Patron Type Definition'], 

                       columns= ['Age Range'],

                       aggfunc=[np.mean], 

                       margins=True).fillna('').fillna('')

pivot
pivot = pd.pivot_table(df,

            values = ['Total Renewals'],

            index = ['Patron Type Definition'], 

                       columns= ['Age Range'],

                       aggfunc=[np.mean], 

                       margins=True)

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

plt.subplots(figsize = (30, 20))

sns.heatmap(pivot,linewidths=0.2,square=True)
df.columns
df_retired = df[df['Patron Type Definition'].str.contains('RETIRED STAFF')]

pivot = pd.pivot_table(df_retired,

            values = ['Total Renewals'],

            index = ['Patron Type Definition', 

                    'Circulation Active Year', 

                    'Circulation Active Month'], 

                       columns= ['Age Range'],

                       aggfunc=[np.mean], 

                       margins=True).fillna('')

pivot
pivot = pd.pivot_table(df_retired,

            values = ['Total Renewals'],

            index = ['Patron Type Definition', 

                    'Circulation Active Year', 

                    'Circulation Active Month'], 

                       columns= ['Age Range'],

                       aggfunc=[np.mean], 

                       margins=True)

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

plt.subplots(figsize = (30, 20))

sns.heatmap(pivot,linewidths=0.2,square=True)
df.columns
InputFile2_reduced=df

for i in set(InputFile2_reduced['Patron Type Definition']):

    aa= InputFile2_reduced[InputFile2_reduced['Patron Type Definition'].isin([i])]

    g = sns.factorplot(x='Age Range', y="Total Checkouts",data=aa, 

                   saturation=1, kind="bar", 

                   ci=None, aspect=3, linewidth=1, row= 'Patron Type Definition') 

    locs, labels = plt.xticks()

    plt.setp(labels, rotation=90)
InputFile2_reduced=df

for i in set(InputFile2_reduced['Home Library Definition']):

    aa= InputFile2_reduced[InputFile2_reduced['Home Library Definition'].isin([i])]

    g = sns.factorplot(x='Age Range', y="Total Checkouts",data=aa, 

                   saturation=1, kind="bar", 

                   ci=None, aspect=3, linewidth=1, row= 'Home Library Definition') 

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
numerical_features = data.select_dtypes(include = ['float64', 'int64']).columns.values

numerical_features
#data = df

sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(2, 2, figsize=(20,20))

sns.despine(left=True)

sns.distplot(df['Patron Type Code'],  kde=False, color="b", ax=axes[0, 0])

sns.distplot(df['Total Checkouts'],        kde=False, color="b", ax=axes[0, 1])

sns.distplot(df['Total Renewals'],        kde=False, color="b", ax=axes[1, 0])

sns.distplot(df['Total Renewals'],        kde=False, color="b", ax=axes[1, 1])

plt.tight_layout()
df_small = df[['Patron Type Definition', 

               'Total Checkouts',  

               'Total Renewals', 

               'Age Range',

               'Home Library Definition', 

               'Circulation Active Year'

]]

sns.pairplot(df_small, hue='Age Range')
df_small = df_small.sample(n=5000, random_state=20)
df_copy = pd.get_dummies(df_small)
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



df1 = df_copy

y = np.asarray(df1['Total Checkouts'], dtype="|S6")

df1 = df1.drop(['Total Checkouts'],axis=1)

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

               ('BernoulliNB', BernoulliNB())

              ]

allscores = []



x, Y = df_copy.drop('Total Checkouts', axis=1), np.asarray(df_copy['Total Checkouts'], dtype="|S6")



for name, classifier in classifiers:

    scores = []

    for i in range(5): # 5 runs

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
mod_df_variable_correlations = df_copy.corr()

#variable_correlations

heat_map(mod_df_variable_correlations)
def get_redundant_pairs(df):

    '''Get diagonal and lower triangular pairs of correlation matrix'''

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



def get_top_abs_correlations(df, n=5):

    au_corr = df.corr().abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]



print("Top Absolute Correlations")

print(get_top_abs_correlations(df_copy, 20))