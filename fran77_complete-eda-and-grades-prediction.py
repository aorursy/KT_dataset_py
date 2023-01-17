# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # graph

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
math = pd.read_csv("../input/student-mat.csv")

por = pd.read_csv("../input/student-por.csv")
math.head()
len(math)
por.head()
len(por)
math.rename(columns={'G1':'G1_Mat', 'G2':'G2_Mat', 'G3':'G3_Mat'}, inplace=True)
por.rename(columns={'G1':'G1_Por', 'G2':'G2_Por', 'G3':'G3_Por'}, inplace=True)
math.columns
# Students with grades in Math and Portuguese

both = pd.merge(math, por, on=['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',

       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',

       'failures', 'schoolsup', 'famsup', 'activities', 'nursery',

       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',

       'Walc', 'health', 'absences'])

both = both.drop_duplicates()
len(both)
# school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)

# sex - student's sex (binary: 'F' - female or 'M' - male)

# age - student's age (numeric: from 15 to 22)

# address - student's home address type (binary: 'U' - urban or 'R' - rural)

# famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)

# Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)

# Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)

# Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)

# Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')

# Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')

# reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')

# guardian - student's guardian (nominal: 'mother', 'father' or 'other')

# traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)

# studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)

# failures - number of past class failures (numeric: n if 1<=n<3, else 4)

# schoolsup - extra educational support (binary: yes or no)

# famsup - family educational support (binary: yes or no)

# paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)

# activities - extra-curricular activities (binary: yes or no)

# nursery - attended nursery school (binary: yes or no)

# higher - wants to take higher education (binary: yes or no)

# internet - Internet access at home (binary: yes or no)

# romantic - with a romantic relationship (binary: yes or no)

# famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)

# freetime - free time after school (numeric: from 1 - very low to 5 - very high)

# goout - going out with friends (numeric: from 1 - very low to 5 - very high)

# Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)

# Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)

# health - current health status (numeric: from 1 - very bad to 5 - very good)

# absences - number of school absences (numeric: from 0 to 93)

# These grades are related with the course subject, Math or Portuguese:



# G1 - first period grade (numeric: from 0 to 20)

# G2 - second period grade (numeric: from 0 to 20)

# G3 - final grade (numeric: from 0 to 20, output target)
por.isna().sum()
por.school.value_counts(normalize=True)
print("%s%% of the students are in Gabriel Pereira school" % (100*round(por.school.value_counts(normalize=True)[0],4)))
por.sex.value_counts(normalize=True)
print("%s%% of the students are girls" % (100*round(por.sex.value_counts(normalize=True)[0],4)))
sns.set(rc={'figure.figsize':(8,6)})

sns.countplot(x="school", hue ="sex", data=por)
sns.countplot(por.age)
por.address.value_counts(normalize=True)
print("%s%% of the students live in an urban area" % (100*round(por.address.value_counts(normalize=True)[0],4)))
por.famsize.value_counts(normalize=True)
print("%s%% of the students live in a family with more than 3 members" % (100*round(por.famsize.value_counts(normalize=True)[0],4)))
sns.countplot(x="school", hue ="address", data=por)
por.Pstatus.value_counts(normalize=True)
print("%s%% of the students' parents live together" % (100*round(por.Pstatus.value_counts(normalize=True)[0],4)))
sns.countplot(x="Pstatus", hue ="famsize", data=por)
sns.countplot(por.Medu)
sns.countplot(por.Fedu)
sns.countplot(x="Medu", hue ="Fedu", data=por)
sns.countplot(por.Mjob)
por.Mjob.value_counts(normalize=True)
print("%s%% of the mothers are at home" % (100*round(por.Mjob.value_counts(normalize=True)[2],4)))
sns.countplot(por.Fjob)
por.Fjob.value_counts(normalize=True)
print("%s%% of the fathers are at home" % (100*round(por.Fjob.value_counts(normalize=True)[2],4)))
sns.countplot(x="Mjob", hue ="Medu", data=por)
sns.countplot(x="Fjob", hue ="Fedu", data=por)
sns.countplot(por.reason)
por.guardian.value_counts(normalize=True)
sns.countplot(x="age", hue ="guardian", data=por)
sns.countplot(por.traveltime)
sns.countplot(x="address", hue ="traveltime", data=por)
sns.countplot(x="school", hue ="traveltime", data=por)
sns.countplot(por.studytime)
sns.countplot(x="school", hue ="studytime", data=por)
sns.countplot(por.failures)
sns.countplot(x="age", hue="failures", data=por)
por.schoolsup.value_counts(normalize=True)
por.famsup.value_counts(normalize=True)
sns.countplot(x="school", hue="famsup", data=por)
por.paid.value_counts(normalize=True)
sns.countplot(por[por.paid == 'yes']['famsup'])
por.activities.value_counts(normalize=True)
sns.countplot(x="sex", hue="activities", data=por)
por.nursery.value_counts(normalize=True)
sns.countplot(x="Medu", hue="nursery", data=por)
por.higher.value_counts(normalize=True)
sns.countplot(x="failures", hue="higher", data=por)
sns.countplot(x="age", hue="higher", data=por)
por.internet.value_counts(normalize=True)
sns.countplot(x="Medu", hue="internet", data=por)
sns.countplot(x="Mjob", hue="internet", data=por)
sns.countplot(x="school", hue="internet", data=por)
por.romantic.value_counts(normalize=True)
sns.countplot(x="age", hue="romantic", data=por)
sns.countplot(por.famrel)
sns.countplot(por.freetime)
por.head()
sns.countplot(por.goout)
sns.countplot(x="freetime", hue="goout", data=por)
sns.countplot(por.Dalc)
sns.countplot(x="goout", hue="Dalc", data=por)
sns.countplot(x="sex", hue="Dalc", data=por)
sns.countplot(por.Walc)
sns.countplot(x="Dalc", hue="Walc", data=por)
sns.countplot(por.health)
sns.countplot(x="sex", hue="health", data=por)
sns.countplot(x="Walc", hue="health", data=por)
sns.distplot(por.absences)
por.columns
por['Total_Grades'] = por['G1_Por'] + por['G2_Por'] + por['G3_Por']
sns.distplot(por.Total_Grades)
GP = por[por.school == 'GP']

MS = por[por.school == 'MS']



sns.distplot(GP.Total_Grades, hist=False, label="GP")

sns.distplot(MS.Total_Grades, hist=False, label="MS")

plt.show()
por['school'] = por['school'].map({'GP': 0, 'MS': 1}).astype(int)

por['sex'] = por['sex'].map({'M': 0, 'F': 1}).astype(int)

por['address'] = por['address'].map({'R': 0, 'U': 1}).astype(int)

por['famsize'] = por['famsize'].map({'LE3': 0, 'GT3': 1}).astype(int)

por['Pstatus'] = por['Pstatus'].map({'A': 0, 'T': 1}).astype(int)

por['Mjob'] = por['Mjob'].map({'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4}).astype(int)

por['Fjob'] = por['Fjob'].map({'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4}).astype(int)

por['reason'] = por['reason'].map({'course': 0, 'other': 1, 'home': 2, 'reputation': 3}).astype(int)

por['guardian'] = por['guardian'].map({'mother': 0, 'father': 1, 'other': 2}).astype(int)

por['schoolsup'] = por['schoolsup'].map({'no': 0, 'yes': 1}).astype(int)

por['famsup'] = por['famsup'].map({'no': 0, 'yes': 1}).astype(int)

por['paid'] = por['paid'].map({'no': 0, 'yes': 1}).astype(int)

por['activities'] = por['activities'].map({'no': 0, 'yes': 1}).astype(int)

por['nursery'] = por['nursery'].map({'no': 0, 'yes': 1}).astype(int)

por['higher'] = por['higher'].map({'no': 0, 'yes': 1}).astype(int)

por['internet'] = por['internet'].map({'no': 0, 'yes': 1}).astype(int)

por['romantic'] = por['romantic'].map({'no': 0, 'yes': 1}).astype(int)
por.head()
por.corr()['Total_Grades'].sort_values(ascending=False)
grades_corr = por.corr()['Total_Grades']
grades_corr = pd.DataFrame({'col':grades_corr.index, 'correlation':grades_corr.values})
no_corr_cols = grades_corr[(grades_corr.correlation < 0.1) & (grades_corr.correlation > -0.1)]

no_corr_cols = list(no_corr_cols.col)
# Droping grades because they are too correlated and can bias the model

X = por.drop(['G1_Por', 'G2_Por', 'G3_Por', 'Total_Grades'], axis=1)

y = por['Total_Grades']
# Droping columns with no correlation

X = X.drop(no_corr_cols, axis=1)
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score



models = [LinearRegression(), Ridge(), Lasso(), DecisionTreeRegressor()]

names = ['LinearRegression', 'Ridge', 'Lasso', 'DecisionTreeRegressor']



for name, clf in zip(names, models):

    cv_model = cross_val_score(clf, X, y, cv=5).mean()

    print(name, ': %s' % cv_model)
dtr = DecisionTreeRegressor()

cvs = range(2,20)

cvs_models = []

for i in cvs:

    cvs_models.append(abs(cross_val_score(clf, X, y, cv=i).mean()))

    

print('Best score with', cvs_models.index(min(cvs_models)), 'subsets : %s' % max(cvs_models))
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
dtr.score(X_test,y_test)