# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# csv_location = 'input' # my local

csv_location = '../input' # kaggle

print("csv files:\n" + check_output(["ls", csv_location]).decode("utf8")) # in kaggle use ../input



# Any results you write to the current directory are saved as output.
# Library imports

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



from collections import Counter

from sklearn.metrics import accuracy_score

%matplotlib inline



pd.set_option('display.max_columns', None ) # None will display ALL columns
# Reading in csv data

traindf_path = csv_location + '/' + 'train.csv'

testdf_path = csv_location + '/' + 'test.csv'



traindf = pd.read_csv(traindf_path)

testdf = pd.read_csv(testdf_path)

print("==Whole data set==")

traindf.info()

traindf.columns



# find out counts but leave test.csv alone

testdf.info()



# For submission use later

PassengerId = testdf['PassengerId']
# check it out

traindf.head(3)
# drop PassengerId since no useful info

traindf.drop('PassengerId', axis=1,inplace=True)

testdf.drop('PassengerId', axis=1, inplace=True)
# column wise, how much null?

traindf.isnull().sum()
# Cabin has lots of missing values, rendered not useful

traindf.dropna(axis=1, thresh=len(traindf)/2., inplace=True)

traindf.head() # Cabin is dropped
# drop Cabin in testdf as well

testdf.drop('Cabin', axis = 1, inplace=True)

testdf.head()
# Let's tackle Embarked null values

Counter(traindf['Embarked'].values)



# Find out the 2 rows 

traindf[traindf['Embarked'].isnull()]
# Let's see if Ticket could clue us 

# Assume Ticket could indicate Embarked

# pull rows that has similar Ticket labels

traindf[traindf['Ticket'].str.match(r'^(1135\w{2})$')][['Ticket','Embarked']]



# Match 113xxx and use majority

from collections import Counter

Counter(traindf[traindf['Ticket'].str.match(r'^(113\w{3})$')]['Embarked'])
# Use 'S' for missing Embarked

traindf.fillna({'Embarked':'S'},inplace=True)

testdf.fillna({'Embarked':'S'}, inplace=True)
traindf.isnull().sum()

testdf.isnull().sum()
# Check out the row missing Fare

testdf[testdf['Fare'].isnull()]



# Use Pclass, SibSp, Parch to guess

fareGB = traindf.groupby(['Pclass','SibSp','Parch'])

aggFareGB = fareGB['Fare'].agg(['mean', 'max', 'median','std'])

aggFareGB.loc[(3,0,0), :]



#Most frequent fare in Pclass=3, SibSp=0, Parch=0

freqFareCounter = Counter(traindf[(traindf['Pclass']==3) & (traindf['SibSp']==0) & (traindf['Parch']==0)]['Fare'].values)



# the three most common fares

print(freqFareCounter.most_common(3))

# select stack index

# fareGB['Fare'].describe()
# Let's use the most common in its Pclass, SibSp and Parch: 8.05 

testdf.iloc[152, 7] = 8.05

testdf.iloc[152]
traindf.isnull().sum()

testdf.isnull().sum()
traindf.head()

testdf.head()
# traindf[traindf['Age'].isnull()]

ageGB = traindf.groupby(['Pclass','SibSp','Parch', 'Sex'])



def mostCommonGB_age(s):

    r = Counter(s.values).most_common(1) # first of many most_commons

    return r[0][0]



aggCommonAgeGB_DF = ageGB['Age'].agg([mostCommonGB_age]) # square bracket get you dataframe

# aggCommonAgeGB_DF



aggMeanAgeGB_DF = ageGB['Age'].agg(['mean'])

# aggMeanAgeGB_DF



def getMostCommonAge(pclass, sibsp, parch, sex):

    # query the aggCommonAgeGB_DF

    try:

        r = aggCommonAgeGB_DF.loc[(pclass,sibsp,parch,sex),'mostCommonGB_age']

    except: 

        r = np.nan

        

#     print("r {}".format(r))

    return r



def getAvgAge(pclass, sibsp, parch, sex):

    try:

        r = aggMeanAgeGB_DF.loc[(pclass,sibsp,parch,sex),'mean']

    except:

        r = np.nan

        

#     print("r avg {}".format(r))

    return r



from decimal import Decimal



overallMeanAge = int(Decimal(traindf['Age'].mean()))

# overallMeanAge



import math

def fillBlankAge(r):

    pclass = r['Pclass']

    sibsp = r['SibSp']

    parch = r['Parch']

    sex = r['Sex']

    age = r['Age']

    if (math.isnan(age)):

        ga =  getMostCommonAge(pclass,sibsp,parch,sex)

#         print("getMostCommonAge pclass,sibsp,parch,sex = age{} {} {} {} {}".format(pclass,sibsp,parch,sex,ga))

        if (math.isnan(ga)):

            ga = getAvgAge(pclass,sibsp,parch,sex)

#             print("getAvgAge pclass,sibsp,parch,sex = age{} {} {} {} {}".format(pclass,sibsp,parch,sex,ga))

        if (math.isnan(ga)):

#             print("use overallMeanAge {}".format(overallMeanAge))

            ga = overallMeanAge

            

        return ga

   

#     print("original age{}".format(age))

    return age

    

traindf['Age']=traindf.apply(fillBlankAge, axis=1)
traindf.tail(10)
testdf['Age']=testdf.apply(fillBlankAge, axis=1) # use traindf numbers
# put them in fewer categories: 



import re



name_regex = r'.*(Mrs)|(Mr)|(Dr)|(Miss)|(Master)|(Col)|(Rev)|(Major)|(Ms)|(Capt)|(Countess)|(Mme).*'



def replace_title(a,regex=name_regex):

    import re

    r = re.search(regex, a, re.I)

    if r:

        if r.group(1): return "Mrs"

        if r.group(2): return "Mr"

        if r.group(3): return "Dr"

        if r.group(4): return "Miss"

        if r.group(5): return "Master"

        if r.group(6): return "Col"

        if r.group(7): return "Rev"

        if r.group(8): return "Major"

        if r.group(9): return "Ms"

        if r.group(10): return "Capt"

        if r.group(11): return "Countess"

        if r.group(12): return "Mme"

    else:

        return a



traindf['Title'] = traindf['Name'].apply(replace_title) # another arg name_regex is default in def
traindf['Title'].unique()

traindf[traindf['Name'].str.startswith('Sagess')]

traindf[traindf['Name'].str.startswith('Reuchlin')]
# replace thw two names manually

traindf.iloc[641,-1]='Miss'

traindf.iloc[822, -1]='Mr'
traindf['Title'].unique()
testdf['Title'] = testdf['Name'].apply(replace_title) # another arg name_regex is default in def
testdf['Title'].unique()

testdf[testdf['Name'].str.startswith('Oliva y Ocana')]
# replace thw two names manually

testdf.iloc[414,-1]='Mrs'

testdf['Title'].unique()
# prepare feature for modeling

traindf.tail()

testdf.tail()
# drop Name

traindf.drop('Name', inplace=True, axis=1)

testdf.drop('Name', inplace=True, axis=1)
# drop Ticket

traindf.drop('Ticket', inplace=True, axis=1)

testdf.drop('Ticket', inplace=True, axis=1)
# onehot encode Sex, Pclass, Embarked, Title

# delay to when needed 
# age.ceil

traindf['Age'] = traindf['Age'].transform(lambda x: math.ceil(x))

testdf['Age'] = testdf['Age'].transform(lambda x: math.ceil(x))
gby_survival = traindf.groupby(['Survived', 'Pclass'])



# which class survived best?

pclass_survived = gby_survival['Pclass'].agg(['count'])

pclass_survived
pclass_survived.plot(kind='bar') # quick plot
fg1, ax1 = plt.subplots(figsize=(4,4))

x1 = range(len(pclass_survived.loc[0, 'count']))

c1 = pclass_survived.loc[0,'count'].values

c2 = pclass_survived.loc[1,'count'].values



ax1.bar(x1, c1,label='nonSurvived', alpha=0.5, color='b')

ax1.bar(x1, c2, bottom=c1, label='Survived', alpha=0.5, color='g')



ax1.set_ylabel("Count")

ax1.set_xlabel("Pclass")

ax1.set_title("Count of survivors by Pclass",fontsize=10)

plt.xticks([0,1,2], ['1', '2', '3'] )

plt.legend(loc='upper left');
gby_survivalS = traindf.groupby(['Survived', 'Sex'])

gender_survived = gby_survivalS['Sex'].agg(['count'])

gender_survived
x3 = range(len(gender_survived.loc[0, 'count']))

c3 = gender_survived.loc[0,'count'].values

c4 = gender_survived.loc[1,'count'].values



fg3, ax3 = plt.subplots(figsize=(4,4))



ax3.bar(x3, c3,label='nonSurvived', alpha=0.5, color='b')

ax3.bar(x3, c4, bottom=c3, label='Survived', alpha=0.5, color='g')



ax3.set_ylabel("Count")

ax3.set_xlabel("Sex")

ax3.set_title("Count of survivors by Sex",fontsize=10)

plt.xticks([0,1], ['female', 'male'] )

plt.legend(loc='upper left');
ageNotSurvived = traindf[traindf['Survived']==0][['Survived','Age']]

ageNotSurvived.head()

ageNotSurvivedCount = ageNotSurvived.groupby('Age').agg(['count'])

ageNotSurvivedCount.head()

# ageNotSurvivedCount[('Survived','count')].values # these are counts

ageNotSurvivedCount.index.values # these are Age



ageSurvived = traindf[traindf['Survived']==1][['Survived','Age']]

# ageSurvived.head()

ageSurvivedCount = ageSurvived.groupby('Age').agg(['count'])

# ageSurvivedCount.head()
ageNotSurvivedCount.plot()
fg5, (ax5, ax6) = plt.subplots(nrows=2,ncols=1, sharex=False, sharey=True, figsize=(20,8))

x5 = ageNotSurvivedCount.index.values

x6 = ageSurvivedCount.index.values

c5 = ageNotSurvivedCount[('Survived','count')].values

c6 = ageSurvivedCount[('Survived','count')].values



ax5.bar(x5, c5, label='nonSurvived', alpha=0.5, color='b')

ax6.bar(x6, c6, label='Survived', alpha=0.5, color='g')



ax5.legend(loc=1)

ax6.legend(loc="upper right")

ax5.set_ylabel("Count")

ax6.set_ylabel("Count")

ax6.set_xlabel("Age")

ax5.set_title("Count of survivors by Age",fontsize=10);
traindf.tail(3)

traindf.info() # need convert Sex, Embarked, Title to numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

LEncoder = LabelEncoder()

OHencoder = OneHotEncoder() # may choose not to use 

traindf['Sex'] = LEncoder.fit_transform(traindf['Sex'])

traindf['Title'] = LEncoder.fit_transform(traindf['Title'])

traindf['Embarked'] = LEncoder.fit_transform(traindf['Embarked'])

traindf.head()



# for testdf

testdf['Sex'] = LEncoder.fit_transform(testdf['Sex'])

testdf['Title'] = LEncoder.fit_transform(testdf['Title'])

testdf['Embarked'] = LEncoder.fit_transform(testdf['Embarked'])

testdf.head()
traindf.head(3)

OHencoder.fit(traindf)

OHencoder.n_values_

# OHencoder.transform
# sepeate label from traindf

(X_train, y_train) = (traindf.drop('Survived', axis=1), traindf['Survived'])



# X_test (kaggle use, not for development)

X_test = testdf.copy()
# total survivors vs non-survivors

s = traindf['Survived']

Counter(s)

ss = traindf[['Survived']]

ss.mean()
# random model

from sklearn.dummy import DummyClassifier

from sklearn.model_selection import cross_val_score, ShuffleSplit



for astrategy in ['uniform', 'most_frequent','stratified']:

    randomModel = DummyClassifier(strategy=astrategy, random_state=7)



    # apply cross validation to make full use of data

    cvshuffle = ShuffleSplit(n_splits=5, test_size=0.20, train_size=None, random_state=7) # train_size will complement test_size when None

    scores = cross_val_score(randomModel, X_train, y_train, cv=cvshuffle)

    print("By {} strategy, mean scores are {}".format(astrategy, scores))
# check for outlier

X_train.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean=False)

# scaler.fit

X_train['Title'] = scaler.fit_transform(X_train[['Title']])

X_train['Fare'] = scaler.fit_transform(X_train[['Fare']])

# for testdf:

X_test['Title'] = scaler.fit_transform(X_test[['Title']])

X_test['Fare'] = scaler.fit_transform(X_test[['Fare']])

X_test.head()
# Adjustment to training set

# X_train.drop('Title',axis=1, inplace=True)

# X_train.drop('Fare',axis=1, inplace=True)

# X_train.drop('Embarked',axis=1, inplace=True)



# X_test.drop('Title',axis=1, inplace=True)

# X_test.drop('Fare',axis=1, inplace=True)

# X_test.drop('Embarked',axis=1, inplace=True)
# X_train.drop('SibSp',axis=1, inplace=True)

# X_test.drop('SibSp',axis=1, inplace=True)

# X_train.head()

# X_test.head()
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier



from sklearn.model_selection import ShuffleSplit, GridSearchCV

from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import r2_score



def performance_metric(y_true, y_predict):

    """ Calculates and returns the performance score between 

        true and predicted values based on the metric chosen. """

    

    # TODO: Calculate the performance score between 'y_true' and 'y_predict'

    score = r2_score(y_true, y_predict)

    

    # Return the score

    return score
rf = RandomForestClassifier(criterion='gini', random_state=6)

dt = DecisionTreeClassifier(criterion='gini', random_state=7)

svm = SVC(random_state=8)

gb = GradientBoostingClassifier(random_state=9)



cvSplits = ShuffleSplit(n_splits=5, test_size=0.20, train_size=None, random_state=7)



RFparams = {'min_samples_split':[2,3,4,5,6],'n_estimators':[110, 130],'max_depth':range(3,9), 'max_features':['auto'], 'min_samples_leaf':[1,2,4,3]}  

DTparams = {'criterion':['gini','entropy'],'max_depth':range(3,6), 'min_samples_split':range(2,9),

            'min_samples_leaf':range(2,9), 'max_features':range(1,5), 'splitter':['best', 'random']}

SVMparams = {}

GBparams = {'min_samples_split':range(3,9), 'min_samples_leaf':range(2,5),'max_features':range(1,5),'max_depth':range(4,8)}



scoring = make_scorer(performance_metric)

GS_selected=[]

# participants = [dt,rf]

participants = [rf,dt,svm,gb]



for i, m in enumerate(participants):

    if isinstance(m, RandomForestClassifier):

        gs = GridSearchCV(estimator=m, cv=cvSplits, param_grid={'random_state':[6]}, scoring=scoring) #default mean score #, scoring=scoring)

    if isinstance(m, DecisionTreeClassifier):

        gs = GridSearchCV(estimator=m, cv=cvSplits, param_grid={'random_state':[7]}, scoring=scoring) # 

    if isinstance(m, SVC):

        gs = GridSearchCV(estimator=m, cv=cvSplits, param_grid={'random_state':[8]}, scoring=scoring) 

    if isinstance(m, GradientBoostingClassifier):

        gs = GridSearchCV(estimator=m, cv=cvSplits, param_grid=GBparams, scoring=scoring) 

    

    GS_selected.append(gs)

    

#     GS_selected[i]

# GS_selected # all set up revealed



# run a decision tree fit with all params 

if rf in participants:

    rf_clf = GS_selected[0].fit(X_train, y_train)

    rf_clf.best_estimator_

if dt in participants:

    dt_clf = GS_selected[1].fit(X_train, y_train)

    dt_clf.best_estimator_

if svm in participants:

    svm_clf = GS_selected[2].fit(X_train, y_train)

    svm_clf.best_estimator_

if gb in participants:

    gb_clf = GS_selected[3].fit(X_train, y_train)

    gb_clf.best_estimator_
# predict and write

prediction = gb_clf.predict(X_test)



# create dataframe submission

submission = pd.DataFrame(index=PassengerId)

submission['Survived'] = prediction
# save to submission file

version = 15

submission['Survived'].unique()

submission.head()

submission.to_csv('submittitanictokaggle'+ str(version)+ '.csv')