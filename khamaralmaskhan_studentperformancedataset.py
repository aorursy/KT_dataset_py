# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import Statment Block 1

pd.options.display.max_colwidth = 100
pd.options.display.max_rows = 100
data = pd.read_csv('/kaggle/input/xAPI-Edu-Data/xAPI-Edu-Data.csv')
df = data
# Observing the columns present in our data
data.columns.values
#Checking information about the data shape and its variables
data.shape
data.head(50)
data.head(n=2).T
# Reviewing Info about the data 
data.info(max_cols = 20)
data.describe()
categorical_features = (data.select_dtypes(include=['object']).columns.values)
categorical_features
numerical_features = data.select_dtypes(include = ['float64', 'int64']).columns.values
numerical_features
pivot = pd.pivot_table(df,
            values = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion'],
            index = ['gender', 'NationalITy', 'PlaceofBirth'], 
                       columns= ['ParentschoolSatisfaction'],
                       aggfunc=[np.mean], 
                       margins=True).fillna('')
pivot
# Import Statments Block 2
from matplotlib import style
import seaborn as sns
sns.set(style='ticks', palette='RdBu')
 
%matplotlib inline
import matplotlib.pyplot as plt
pivot = pd.pivot_table(df,
            values = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion'],
            index = ['gender', 'NationalITy', 'PlaceofBirth'], 
                       columns= ['ParentschoolSatisfaction'],
                       aggfunc=[np.mean, np.std], 
                       margins=True)
cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)
plt.subplots(figsize = (30, 20))
sns.heatmap(pivot,linewidths=0.2,square=True )
def heat_map(corrs_mat):
    sns.set(style="white")
    f, ax = plt.subplots(figsize=(10, 5))
    mask = np.zeros_like(corrs_mat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True 
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corrs_mat, mask=mask, cmap=cmap, ax=ax)

variable_correlations = df.corr()
#variable_correlations
heat_map(variable_correlations)
df_small = df[['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion', 'NationalITy']]
sns.pairplot(df_small, hue='NationalITy')
# data mapping

gender_map = {'M':1, 
              'F':2}

NationalITy_map = {  'Iran': 1,
                     'SaudiArabia': 2,
                     'USA': 3,
                     'Egypt': 4,
                     'Lybia': 5,
                     'lebanon': 6,
                     'Morocco': 7,
                     'Jordan': 8,
                     'Palestine': 9,
                     'Syria': 10,
                     'Tunis': 11,
                     'KW': 12,
                     'KuwaIT': 12,
                     'Iraq': 13,
                     'venzuela': 14}

PlaceofBirth_map =  {'Iran': 1,
                     'SaudiArabia': 2,
                     'USA': 3,
                     'Egypt': 4,
                     'Lybia': 5,
                     'lebanon': 6,
                     'Morocco': 7,
                     'Jordan': 8,
                     'Palestine': 9,
                     'Syria': 10,
                     'Tunis': 11,
                     'KW': 12,
                     'KuwaIT': 12,
                     'Iraq': 13,
                     'venzuela': 14}

StageID_map = {'HighSchool':1, 
               'lowerlevel':2, 
               'MiddleSchool':3}

GradeID_map =   {'G-01':1,
                 'G-02':2,
                 'G-03':3,
                 'G-04':4,
                 'G-05':5,
                 'G-06':6,
                 'G-07':7,
                 'G-08':8,
                 'G-09':9,
                 'G-10':10,
                 'G-11':11,
                 'G-12':12}

SectionID_map = {'A':1, 
                 'C':2, 
                 'B':3}

Topic_map  =    {'Biology' : 1,
                 'Geology' : 2,
                 'Quran' : 3,
                 'Science' : 4,
                 'Spanish' : 5,
                 'IT' : 6,
                 'French' : 7,
                 'English' :8,
                 'Arabic' :9,
                 'Chemistry' :10,
                 'Math' :11,
                 'History' : 12}

Semester_map = {'S':2, 
                'F':1}

Relation_map = {'Mum':2, 
                'Father':1} 

ParentAnsweringSurvey_map = {'Yes':1,
                             'No':0}

ParentschoolSatisfaction_map = {'Bad':0,
                                'Good':1}

StudentAbsenceDays_map = {'Under-7':0,
                          'Above-7':1}

Class_map = {'H':10,
             'M':5,
             'L':2}
mod_df = df
mod_df.gender                 = mod_df.gender.map(gender_map)
mod_df.NationalITy            = mod_df.NationalITy.map(NationalITy_map)
mod_df.PlaceofBirth           = mod_df.PlaceofBirth.map(PlaceofBirth_map)
mod_df.StageID                = mod_df.StageID.map(StageID_map)
mod_df.GradeID                = mod_df.GradeID.map(GradeID_map)
mod_df.SectionID              = mod_df.SectionID.map(SectionID_map)
mod_df.Topic                  = mod_df.Topic.map(Topic_map)
mod_df.Semester               = mod_df.Semester.map(Semester_map)
mod_df.Relation               = mod_df.Relation.map(Relation_map)
mod_df.ParentAnsweringSurvey  = mod_df.ParentAnsweringSurvey.map(ParentAnsweringSurvey_map)
mod_df.ParentschoolSatisfaction   = mod_df.ParentschoolSatisfaction.map(ParentschoolSatisfaction_map)
mod_df.StudentAbsenceDays     = mod_df.StudentAbsenceDays.map(StudentAbsenceDays_map)
mod_df.Class                  = mod_df.Class.map(Class_map)

sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(4, 4, figsize=(20,20))
sns.despine(left=True)
sns.distplot(df['NationalITy'],  kde=False, color="b", ax=axes[0, 0])# kde - Kernel density estimate
sns.distplot(df['PlaceofBirth'],        kde=False, color="b", ax=axes[0, 1])
sns.distplot(df['StageID'],        kde=False, color="b", ax=axes[0, 2])
sns.distplot(df['GradeID'],        kde=False, color="b", ax=axes[0, 3])
sns.distplot(df['SectionID'], kde=False, color="b", ax=axes[1, 0])
sns.distplot(df['Topic'],  kde=False, color="b", ax=axes[1, 1])
sns.distplot(df['Relation'],     kde=False, color="b", ax=axes[1, 2])
sns.distplot(df['raisedhands'],  kde=False, color="b", ax=axes[1, 3])
sns.distplot(df['VisITedResources'],      kde=False, color="b", ax=axes[2, 0])
sns.distplot(df['AnnouncementsView'],      kde=False, color="b", ax=axes[2, 1])
sns.distplot(df['Discussion'],    kde=False, color="b", ax=axes[2, 2])
sns.distplot(df['ParentAnsweringSurvey'],    kde=False, color="b", ax=axes[2, 3])
sns.distplot(df['ParentschoolSatisfaction'],kde=False, color="b", ax=axes[3, 0])
sns.distplot(df['StudentAbsenceDays'],       kde=False, color="b", ax=axes[3, 1])
sns.distplot(df['Class'],      kde=False, color="b", ax=axes[3, 2])

plt.tight_layout()
# Checking if there are any left out categorical columns we havent mapped yet.
categorical_features = (mod_df.select_dtypes(include=['object']).columns.values)
categorical_features

mod_df_variable_correlations = mod_df.corr()
#variable_correlations
heat_map(mod_df_variable_correlations)
# checking original frame before modifications

df.columns
# Import Statments Block 3
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
# Data copy made
df_copy = pd.get_dummies(mod_df)

df1 = df_copy
y = np.asarray(df1['ParentschoolSatisfaction'], dtype="|S6")
df1 = df1.drop(['ParentschoolSatisfaction'],axis=1)
X = df1.values
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.50)

radm = RandomForestClassifier()
radm.fit(Xtrain, ytrain)
# clf is classifier used
clf = radm
indices = np.argsort(radm.feature_importances_)[::-1]

# Print the feature ranking
print('Feature ranking:')

# for f in range(df1.shape[1]):
#     print('%d. feature %d %s (%f)' % (f+1 , 
#                                       indices[f], 
#                                       df1.columns[indices[f]], 
#                                       radm.feature_importances_[indices[f]]))

for f in range(df1.shape[1]):
    print('%d.  %s (%f)' % (f+1 , df1.columns[indices[f]], 
                                      radm.feature_importances_[indices[f]]))
# Import Statments Block 4

# For ensemble models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier

# For KNeighborsClassifier models
from sklearn.neighbors import KNeighborsClassifier

# For Decision Tree
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
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
# Import Statment BLock 5 for accuracy and ROC calculations
from sklearn.model_selection import cross_val_score
x, Y = mod_df.drop('ParentschoolSatisfaction', axis=1), np.asarray(mod_df['ParentschoolSatisfaction'], dtype="|S6")

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
data = pd.read_csv('/kaggle/input/xAPI-Edu-Data/xAPI-Edu-Data.csv')
df_copy = pd.get_dummies(data)
df_copy.head()
df_copy.head().T
Y = df1['ParentschoolSatisfaction_Good'].values
df1 = df1.drop(['ParentschoolSatisfaction_Good'],axis=1)
x = df1.values
Xtrain, Xtest, ytrain, ytest = train_test_split(x, Y, test_size=0.50)
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

#x, Y = mod_df.drop('ParentschoolSatisfaction', axis=1), np.asarray(mod_df['ParentschoolSatisfaction'], dtype="|S6")

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
