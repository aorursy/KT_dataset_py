# Import basic libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# import visualization libraries

import seaborn as sns

import matplotlib.pyplot as plt

from ggplot import *

%matplotlib inline
# Load the data



df = pd.read_csv('../input/UCI_Credit_Card.csv')

df.sample(5)
df.info()
# Categorical variables description

df[['SEX', 'EDUCATION', 'MARRIAGE']].describe()
# Payment delay description

df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].describe()
# Bill Statement description

df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].describe()
#Previous Payment Description

df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].describe()
df.LIMIT_BAL.describe()
df = df.rename(columns={'default.payment.next.month': 'def_pay', 

                        'PAY_0': 'PAY_1'})

df.head()
# I am interested in having a general idea of the default probability

df.def_pay.sum() / len(df.def_pay)
# Other ways of getting this kind of numbers (as a reference for newbies like myself)

print(df.shape)

print(df.shape[0])

print(df.def_pay.count())

print(len(df.axes[1]))
#importing libraries

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, make_scorer

from sklearn.model_selection import train_test_split
# create the target variable

y = df['def_pay'].copy()

y.sample(5)
# create the features, which now will be everything in the original df

features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',

       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',

       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',

       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

X = df[features].copy()

X.columns
# split the df into train and test, it is important these two do not communicate during the training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# this means we will train on 80% of the data and test on the remaining 20%.
#check that the target is not far off

print(df.def_pay.describe())

print("---------------------------")

print(y_train.describe())

print("---------------------------")

print(y_test.describe())
#create the classifier

classifier = DecisionTreeClassifier(max_depth=10, random_state=14) 

# training the classifier

classifier.fit(X_train, y_train)

# do our predictions on the test

predictions = classifier.predict(X_test)

# see how good we did on the test

accuracy_score(y_true = y_test, y_pred = predictions)
classifier = DecisionTreeClassifier(max_depth=100, random_state=14) 

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy_score(y_true = y_test, y_pred = predictions)
from sklearn.model_selection import GridSearchCV
# define the parameters grid

param_grid = {'max_depth': np.arange(3, 10),

             'criterion' : ['gini','entropy'],

             'max_leaf_nodes': [5,10,20,100],

             'min_samples_split': [2, 5, 10, 20]}



# create the grid

grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 5, scoring= 'accuracy')

# the cv option will be clear in a few cells



#training

grid_tree.fit(X_train, y_train)

#let's see the best estimator

print(grid_tree.best_estimator_)

#with its score

print(np.abs(grid_tree.best_score_))
classifier = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,

            max_features=None, max_leaf_nodes=20,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=20,

            min_weight_fraction_leaf=0.0, presort=False, random_state=None,

            splitter='best')

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy_score(y_true = y_test, y_pred = predictions)
from sklearn.model_selection import KFold
kf = KFold(n_splits=5,random_state=42,shuffle=True)



fold = []

scr = []



for i,(train_index, test_index) in enumerate(kf.split(df)):

    training = df.iloc[train_index,:]

    valid = df.iloc[test_index,:]

    feats = training[features] #defined above

    label = training['def_pay']

    valid_feats = valid[features]

    valid_label = valid['def_pay']

    classifier.fit(feats,label) #it is the last one we run, the best one

    pred = classifier.predict(valid_feats)

    score = accuracy_score(y_true = valid_label, y_pred = pred)

    fold.append(i+1)

    scr.append(score)

    

#create a small df with the scores

performance = pd.DataFrame({'Score':scr,'Fold':fold})

# let's see what we have with ggplot

g = ggplot(performance,aes(x='Fold',y='Score')) + geom_point() + geom_line()

print(g)
def get_feature_importance(clsf, ftrs):

    imp = clsf.feature_importances_.tolist()

    feat = ftrs

    result = pd.DataFrame({'feat':feat,'score':imp})

    result = result.sort_values(by=['score'],ascending=False)

    return result



get_feature_importance(classifier, features)
X = df[['PAY_1']].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

classifier.fit(X_train, y_train) #same classifier as before

predictions = classifier.predict(X_test)

accuracy_score(y_true = y_test, y_pred = predictions)
from sklearn import tree

import graphviz

dot_data = tree.export_graphviz(classifier, out_file=None)  

graph = graphviz.Source(dot_data)  

graph
# import the tool

from sklearn.metrics import f1_score

#recreate the model and evaluate it

X = df[features].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

classifier.fit(X_train, y_train) #same classifier as before

predictions = classifier.predict(X_test)

f1_score(y_true = y_test, y_pred = predictions)
# True Positive (TP): we predict a label of 1 (positive), and the true label is 1.

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

 

# True Negative (TN): we predict a label of 0 (negative), and the true label is 0.

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

 

# False Positive (FP): we predict a label of 1 (positive), but the true label is 0.

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

 

# False Negative (FN): we predict a label of 0 (negative), but the true label is 1.

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

 

print('TP: {}, FP: {}, TN: {}, FN: {}'.format(TP,FP,TN,FN))
param_grid = {'max_depth': np.arange(3, 10),

             'criterion' : ['gini','entropy'],

             'max_leaf_nodes': [5,10,20,100],

             'min_samples_split': [2, 5, 10, 20]}

grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 5, scoring= 'f1')

grid_tree.fit(X_train, y_train)

best = grid_tree.best_estimator_

print(grid_tree.best_estimator_)

print(np.abs(grid_tree.best_score_))

classifier = best

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

print("-------------")

print(f1_score(y_true = y_test, y_pred = predictions))

print(get_feature_importance(classifier, features))

print("-------------")

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

print('TP: {}, FP: {}, TN: {}, FN: {}'.format(TP,FP,TN,FN))
param_grid = {'max_depth': np.arange(3, 10),

             'criterion' : ['gini','entropy'],

             'max_leaf_nodes': [5,10,20,100],

             'min_samples_split': [2, 5, 10, 20],

             'class_weight' : ['balanced']}

grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 5, scoring= 'f1')

grid_tree.fit(X_train, y_train)

best = grid_tree.best_estimator_

print(grid_tree.best_estimator_)

print(np.abs(grid_tree.best_score_))

classifier = best

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

print("-------------")

print(f1_score(y_true = y_test, y_pred = predictions))

print(get_feature_importance(classifier, features))

print("-------------")

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

print('TP: {}, FP: {}, TN: {}, FN: {}'.format(TP,FP,TN,FN))
df.SEX.value_counts() #this is fine, more women than men
df['MARRIAGE'].value_counts()
df.EDUCATION.value_counts() # yes, I am using different ways of calling a column
df.MARRIAGE.value_counts().plot(kind = 'bar')
df.EDUCATION.value_counts().plot(kind = "barh")
def draw_histograms(df, variables, n_rows, n_cols, n_bins):

    fig=plt.figure()

    for i, var_name in enumerate(variables):

        ax=fig.add_subplot(n_rows,n_cols,i+1)

        df[var_name].hist(bins=n_bins,ax=ax)

        ax.set_title(var_name)

    fig.tight_layout()  # Improves appearance a bit.

    plt.show()
bills = df[['BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']]

draw_histograms(bills, bills.columns, 2, 3, 20)
pay = df[['PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]

draw_histograms(pay, pay.columns, 2, 3, 20)
late = df[['PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]

draw_histograms(late, late.columns, 2, 3, 10)



#this is probably more of a category
df.AGE.hist()
df.LIMIT_BAL.hist(bins = 20)
fil = (df.EDUCATION == 5) | (df.EDUCATION == 6) | (df.EDUCATION == 0)

df.loc[fil, 'EDUCATION'] = 4

df.EDUCATION.value_counts()
df.loc[df.MARRIAGE == 0, 'MARRIAGE'] = 3

df.MARRIAGE.value_counts()
fil = (df.PAY_1 == -2) | (df.PAY_1 == -1) | (df.PAY_1 == 0)

df.loc[fil, 'PAY_1'] = 0

fil = (df.PAY_2 == -2) | (df.PAY_2 == -1) | (df.PAY_2 == 0)

df.loc[fil, 'PAY_2'] = 0

fil = (df.PAY_3 == -2) | (df.PAY_3 == -1) | (df.PAY_3 == 0)

df.loc[fil, 'PAY_3'] = 0

fil = (df.PAY_4 == -2) | (df.PAY_4 == -1) | (df.PAY_4 == 0)

df.loc[fil, 'PAY_4'] = 0

fil = (df.PAY_5 == -2) | (df.PAY_5 == -1) | (df.PAY_5 == 0)

df.loc[fil, 'PAY_5'] = 0

fil = (df.PAY_6 == -2) | (df.PAY_6 == -1) | (df.PAY_6 == 0)

df.loc[fil, 'PAY_6'] = 0

late = df[['PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]

draw_histograms(late, late.columns, 2, 3, 10)
df[df.PAY_AMT1 > 300000][['LIMIT_BAL', 'PAY_1', 'PAY_2', 'BILL_AMT2', 

                          'PAY_AMT1', 'BILL_AMT1', 'def_pay']]

# doesn't look weird after all
df[df.PAY_AMT2 > 300000][['LIMIT_BAL', 'PAY_2', 'PAY_3', 'BILL_AMT3', 

                          'PAY_AMT2', 'BILL_AMT2', 'def_pay']]

# doesn't look weird after all
df.groupby(['SEX', 'def_pay']).size()
gender = df.groupby(['SEX', 'def_pay']).size().unstack(1)

# 1 is the default for unstack, but I put it to show explicitly what we are unstacking

gender
# Another, easier, way is to just use crosstab

pd.crosstab(df.SEX, df.def_pay)
gender.plot(kind='bar', stacked = True)
gender['perc'] = (gender[1]/(gender[0] + gender[1])) 

#this creates a new column in our dataset

gender
df[["SEX", "def_pay"]].groupby(['SEX'], 

                                        as_index=False).mean().sort_values(by='def_pay', 

                                                                           ascending=False)
# I like playing with options, so here we go

df[["SEX", "def_pay"]].groupby(['SEX']).mean().sort_values(by='def_pay')
def corr_2_cols(Col1, Col2):

    res = df.groupby([Col1, Col2]).size().unstack()

    res['perc'] = (res[res.columns[1]]/(res[res.columns[0]] + res[res.columns[1]]))

    return res



"""

Side note, you could use res[1] and res[0] and still have a function that 

does what we did before. However, that would mean that you are reffering to the column 

labeled 0 and 1, not the position of it. Thus the function will not work if the unstacked 

variable has different values. 



Moreover, a good exercise is to generalize the function so that the unstacked variable can

have more than 2 values

"""
corr_2_cols('EDUCATION', 'def_pay')
corr_2_cols('MARRIAGE', 'def_pay')
corr_2_cols('MARRIAGE', 'SEX')
corr_2_cols('EDUCATION', 'SEX')
df[['PAY_AMT6', 'BILL_AMT6', 'PAY_AMT5', 

     'BILL_AMT5', 'PAY_AMT4', 'BILL_AMT4', 'PAY_AMT3', 'BILL_AMT3', 

     'PAY_AMT2', 'BILL_AMT2',

     'PAY_AMT1', 'BILL_AMT1',

     'LIMIT_BAL', 'def_pay']].sample(30)
df[df.def_pay == 1][['BILL_AMT2',

     'PAY_AMT1', 'BILL_AMT1', 'PAY_1',

     'LIMIT_BAL']].sample(30)
fil = ((df.PAY_6 == 0) & (df.BILL_AMT6 > 0) & (df.PAY_5 > 0))

df[fil][['BILL_AMT6', 'PAY_AMT5', 'BILL_AMT5', 'PAY_5']].sample(20)
fil = ((df.PAY_6 == 0) & (df.BILL_AMT6 > 0) & (df.PAY_5 > 0) & (df.PAY_AMT5 == 0))

df[fil][['BILL_AMT6', 'PAY_AMT5', 'BILL_AMT5', 'PAY_5']]
fil = ((df.PAY_AMT1 > df.BILL_AMT2) & (df.PAY_1 > 0) & (df.PAY_2 == 0))

df[fil][['BILL_AMT2', 'PAY_2', 'PAY_AMT2', 'BILL_AMT1', 'PAY_1', 'LIMIT_BAL', 'def_pay']].head(15)
g = sns.FacetGrid(df, col = 'def_pay')

g.map(plt.hist, 'AGE')
df.loc[df.PAY_1 > 0, 'PAY_1'] = 1

df.loc[df.PAY_2 > 0, 'PAY_2'] = 1

df.loc[df.PAY_3 > 0, 'PAY_3'] = 1

df.loc[df.PAY_4 > 0, 'PAY_4'] = 1

df.loc[df.PAY_5 > 0, 'PAY_5'] = 1

df.loc[df.PAY_6 > 0, 'PAY_6'] = 1

late = df[['PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]

draw_histograms(late, late.columns, 2, 3, 10)
g = sns.FacetGrid(df, col = 'def_pay', row = 'SEX')

g.map(plt.hist, 'AGE')
g = sns.FacetGrid(df, col='SEX', hue='def_pay')

g.map(plt.hist, 'AGE', alpha=0.6, bins=25) #alpha is for opacity

g.add_legend()
g = sns.FacetGrid(df, col='def_pay', row= "MARRIAGE", hue='SEX')

g.map(plt.hist, 'AGE', alpha=0.3, bins=25) 

g.add_legend()
df['SE_MA'] = df.SEX * df.MARRIAGE

corr_2_cols('SE_MA', 'def_pay')
df['SE_MA_2'] = 0

df.loc[((df.SEX == 1) & (df.MARRIAGE == 1)) , 'SE_MA_2'] = 1 #married man

df.loc[((df.SEX == 1) & (df.MARRIAGE == 2)) , 'SE_MA_2'] = 2 #single man

df.loc[((df.SEX == 1) & (df.MARRIAGE == 3)) , 'SE_MA_2'] = 3 #divorced man

df.loc[((df.SEX == 2) & (df.MARRIAGE == 1)) , 'SE_MA_2'] = 4 #married woman

df.loc[((df.SEX == 2) & (df.MARRIAGE == 2)) , 'SE_MA_2'] = 5 #single woman

df.loc[((df.SEX == 2) & (df.MARRIAGE == 3)) , 'SE_MA_2'] = 6 #divorced woman

corr_2_cols('SE_MA_2', 'def_pay')
del df['SE_MA']

df = df.rename(columns={'SE_MA_2': 'SE_MA'})
df['AgeBin'] = 0 #creates a column of 0

df.loc[((df['AGE'] > 20) & (df['AGE'] < 30)) , 'AgeBin'] = 1

df.loc[((df['AGE'] >= 30) & (df['AGE'] < 40)) , 'AgeBin'] = 2

df.loc[((df['AGE'] >= 40) & (df['AGE'] < 50)) , 'AgeBin'] = 3

df.loc[((df['AGE'] >= 50) & (df['AGE'] < 60)) , 'AgeBin'] = 4

df.loc[((df['AGE'] >= 60) & (df['AGE'] < 70)) , 'AgeBin'] = 5

df.loc[((df['AGE'] >= 70) & (df['AGE'] < 81)) , 'AgeBin'] = 6

df.AgeBin.hist()
bins = [20, 29, 39, 49, 59, 69, 81]

bins_names = [1, 2, 3, 4, 5, 6]

df['AgeBin2'] = pd.cut(df['AGE'], bins, labels=bins_names)

df.AgeBin2.hist()
df['AgeBin3'] = pd.cut(df['AGE'], 6)

df.AgeBin3.value_counts()
df['AgeBin3'] = pd.cut(df['AGE'], 6, labels=bins_names) #just added one option

df.AgeBin3.hist()
df['AgeBin4'] = pd.qcut(df['AGE'], 6)

df.AgeBin4.value_counts()
df['AgeBin4'] = pd.qcut(df['AGE'], 6, labels=bins_names)

df.AgeBin4.hist()
del df['AgeBin2']

del df['AgeBin3']

del df['AgeBin4'] # we don't need these any more

df['AgeBin'] = pd.cut(df['AGE'], 6, labels = [1,2,3,4,5,6])

#because 1 2 3 ecc are "categories" so far and we need numbers

df['AgeBin'] = pd.to_numeric(df['AgeBin'])

df.loc[(df['AgeBin'] == 6) , 'AgeBin'] = 5

df.AgeBin.hist()
corr_2_cols('AgeBin', 'def_pay')
corr_2_cols('AgeBin', 'SEX')
df['SE_AG'] = 0

df.loc[((df.SEX == 1) & (df.AgeBin == 1)) , 'SE_AG'] = 1 #man in 20's

df.loc[((df.SEX == 1) & (df.AgeBin == 2)) , 'SE_AG'] = 2 #man in 30's

df.loc[((df.SEX == 1) & (df.AgeBin == 3)) , 'SE_AG'] = 3 #man in 40's

df.loc[((df.SEX == 1) & (df.AgeBin == 4)) , 'SE_AG'] = 4 #man in 50's

df.loc[((df.SEX == 1) & (df.AgeBin == 5)) , 'SE_AG'] = 5 #man in 60's and above

df.loc[((df.SEX == 2) & (df.AgeBin == 1)) , 'SE_AG'] = 6 #woman in 20's

df.loc[((df.SEX == 2) & (df.AgeBin == 2)) , 'SE_AG'] = 7 #woman in 30's

df.loc[((df.SEX == 2) & (df.AgeBin == 3)) , 'SE_AG'] = 8 #woman in 40's

df.loc[((df.SEX == 2) & (df.AgeBin == 4)) , 'SE_AG'] = 9 #woman in 50's

df.loc[((df.SEX == 2) & (df.AgeBin == 5)) , 'SE_AG'] = 10 #woman in 60's and above

corr_2_cols('SE_AG', 'def_pay')
df['Client_6'] = 1

df['Client_5'] = 1

df['Client_4'] = 1

df['Client_3'] = 1

df['Client_2'] = 1

df['Client_1'] = 1

df.loc[((df.PAY_6 == 0) & (df.BILL_AMT6 == 0) & (df.PAY_AMT6 == 0)) , 'Client_6'] = 0

df.loc[((df.PAY_5 == 0) & (df.BILL_AMT5 == 0) & (df.PAY_AMT5 == 0)) , 'Client_5'] = 0

df.loc[((df.PAY_4 == 0) & (df.BILL_AMT4 == 0) & (df.PAY_AMT4 == 0)) , 'Client_4'] = 0

df.loc[((df.PAY_3 == 0) & (df.BILL_AMT3 == 0) & (df.PAY_AMT3 == 0)) , 'Client_3'] = 0

df.loc[((df.PAY_2 == 0) & (df.BILL_AMT2 == 0) & (df.PAY_AMT2 == 0)) , 'Client_2'] = 0

df.loc[((df.PAY_1 == 0) & (df.BILL_AMT1 == 0) & (df.PAY_AMT1 == 0)) , 'Client_1'] = 0

pd.Series([df[df.Client_6 == 1].def_pay.count(),

          df[df.Client_5 == 1].def_pay.count(),

          df[df.Client_4 == 1].def_pay.count(),

          df[df.Client_3 == 1].def_pay.count(),

          df[df.Client_2 == 1].def_pay.count(),

          df[df.Client_1 == 1].def_pay.count()], [6,5,4,3,2,1])
df['Avg_exp_5'] = ((df['BILL_AMT5'] - (df['BILL_AMT6'] - df['PAY_AMT5']))) / df['LIMIT_BAL']

df['Avg_exp_4'] = (((df['BILL_AMT5'] - (df['BILL_AMT6'] - df['PAY_AMT5'])) +

                 (df['BILL_AMT4'] - (df['BILL_AMT5'] - df['PAY_AMT4']))) / 2) / df['LIMIT_BAL']

df['Avg_exp_3'] = (((df['BILL_AMT5'] - (df['BILL_AMT6'] - df['PAY_AMT5'])) +

                 (df['BILL_AMT4'] - (df['BILL_AMT5'] - df['PAY_AMT4'])) +

                 (df['BILL_AMT3'] - (df['BILL_AMT4'] - df['PAY_AMT3']))) / 3) / df['LIMIT_BAL']

df['Avg_exp_2'] = (((df['BILL_AMT5'] - (df['BILL_AMT6'] - df['PAY_AMT5'])) +

                 (df['BILL_AMT4'] - (df['BILL_AMT5'] - df['PAY_AMT4'])) +

                 (df['BILL_AMT3'] - (df['BILL_AMT4'] - df['PAY_AMT3'])) +

                 (df['BILL_AMT2'] - (df['BILL_AMT3'] - df['PAY_AMT2']))) / 4) / df['LIMIT_BAL']

df['Avg_exp_1'] = (((df['BILL_AMT5'] - (df['BILL_AMT6'] - df['PAY_AMT5'])) +

                 (df['BILL_AMT4'] - (df['BILL_AMT5'] - df['PAY_AMT4'])) +

                 (df['BILL_AMT3'] - (df['BILL_AMT4'] - df['PAY_AMT3'])) +

                 (df['BILL_AMT2'] - (df['BILL_AMT3'] - df['PAY_AMT2'])) +

                 (df['BILL_AMT1'] - (df['BILL_AMT2'] - df['PAY_AMT1']))) / 5) / df['LIMIT_BAL']

df[['LIMIT_BAL', 'Avg_exp_5', 'BILL_AMT5', 'Avg_exp_4', 'BILL_AMT4','Avg_exp_3', 'BILL_AMT3',

    'Avg_exp_2', 'BILL_AMT2', 'Avg_exp_1', 'BILL_AMT1', 'def_pay']].sample(20)
df['Closeness_6'] = (df.LIMIT_BAL - df.BILL_AMT6) / df.LIMIT_BAL

df['Closeness_5'] = (df.LIMIT_BAL - df.BILL_AMT5) / df.LIMIT_BAL

df['Closeness_4'] = (df.LIMIT_BAL - df.BILL_AMT4) / df.LIMIT_BAL

df['Closeness_3'] = (df.LIMIT_BAL - df.BILL_AMT3) / df.LIMIT_BAL

df['Closeness_2'] = (df.LIMIT_BAL - df.BILL_AMT2) / df.LIMIT_BAL

df['Closeness_1'] = (df.LIMIT_BAL - df.BILL_AMT1) / df.LIMIT_BAL

df[['Closeness_6', 'Closeness_5', 'Closeness_4', 'Closeness_3', 'Closeness_2',

   'Closeness_1', 'def_pay']].sample(20)
features = ['LIMIT_BAL', 'EDUCATION', 'MARRIAGE', 'PAY_1','PAY_2', 'PAY_3', 

            'PAY_4', 'PAY_5', 'PAY_6','BILL_AMT1', 'BILL_AMT2',

            'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',

            'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 

            'SE_MA', 'AgeBin', 'SE_AG', 'Avg_exp_5', 'Avg_exp_4',

            'Avg_exp_3', 'Avg_exp_2', 'Avg_exp_1', 'Closeness_5',

            'Closeness_4', 'Closeness_3', 'Closeness_2','Closeness_1']

y = df['def_pay'].copy() # target

X = df[features].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# create the training df by remerging X_train and y_train

df_train = X_train.join(y_train)

df_train.sample(10)
from sklearn.utils import resample
# Separate majority and minority classes

df_majority = df_train[df_train.def_pay==0]

df_minority = df_train[df_train.def_pay==1]



print(df_majority.def_pay.count())

print("-----------")

print(df_minority.def_pay.count())

print("-----------")

print(df_train.def_pay.value_counts())
# Upsample minority class

df_minority_upsampled = resample(df_minority, 

                                 replace=True,     # sample with replacement

                                 n_samples=18677,    # to match majority class

                                 random_state=587) # reproducible results

# Combine majority class with upsampled minority class

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts

df_upsampled.def_pay.value_counts()
# Downsample majority class

df_majority_downsampled = resample(df_majority, 

                                 replace=False,    # sample without replacement

                                 n_samples=5323,     # to match minority class

                                 random_state=587) # reproducible results

# Combine minority class with downsampled majority class

df_downsampled = pd.concat([df_majority_downsampled, df_minority])

# Display new class counts

df_downsampled.def_pay.value_counts()
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=589, ratio = 1.0)

X_SMOTE, y_SMOTE = sm.fit_sample(X_train, y_train)

print(len(y_SMOTE))

print(y_SMOTE.sum())
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier, AdaBoostClassifier

from xgboost import XGBClassifier

from sklearn.metrics import f1_score
# classifier list for the normal training set

clf_list = [DecisionTreeClassifier(max_depth = 3, class_weight = "balanced"), 

            RandomForestClassifier(n_estimators = 100, class_weight = "balanced"), 

            AdaBoostClassifier(DecisionTreeClassifier(max_depth = 3, class_weight = "balanced"),

                               n_estimators = 100), 

            GradientBoostingClassifier(), 

            XGBClassifier()

           ]

# the parameters are set in order to have the same kind of tree every time
# use Kfold to evaluate the normal training set

kf = KFold(n_splits=5,random_state=42,shuffle=True)



mdl = []

fold = []

scr = []



for i,(train_index, test_index) in enumerate(kf.split(df_train)):

    training = df.iloc[train_index,:]

    valid = df.iloc[test_index,:]

    print(i)

    for clf in clf_list:

        model = clf.__class__.__name__

        feats = training[features] #defined above

        label = training['def_pay']

        valid_feats = valid[features]

        valid_label = valid['def_pay']

        clf.fit(feats,label) 

        pred = clf.predict(valid_feats)

        score = f1_score(y_true = valid_label, y_pred = pred)

        fold.append(i+1)

        scr.append(score)

        mdl.append(model)

        print(model)

    

#create a small df with the scores

performance = pd.DataFrame({'Model': mdl, 'Score':scr,'Fold':fold})

g_normal = ggplot(performance,aes(x='Fold',y='Score',group = 'Model',color = 'Model')) + geom_point() + geom_line()

print(g_normal)
# classifier list for the downsampled training set

clf_list = [DecisionTreeClassifier(max_depth = 3), 

            RandomForestClassifier(n_estimators = 100), 

            AdaBoostClassifier(DecisionTreeClassifier(max_depth = 3), n_estimators = 100), 

            GradientBoostingClassifier(), 

            XGBClassifier()

           ]

# the parameters are set in order to have the same kind of tree every time

# use Kfold to evaluate the upsampled training set

kf = KFold(n_splits=5,random_state=42,shuffle=True)



mdl = []

fold = []

scr = []



for i,(train_index, test_index) in enumerate(kf.split(df_downsampled)):

    training = df.iloc[train_index,:]

    valid = df.iloc[test_index,:]

    print(i)

    for clf in clf_list:

        model = clf.__class__.__name__

        feats = training[features] #defined above

        label = training['def_pay']

        valid_feats = valid[features]

        valid_label = valid['def_pay']

        clf.fit(feats,label) 

        pred = clf.predict(valid_feats)

        score = f1_score(y_true = valid_label, y_pred = pred)

        fold.append(i+1)

        scr.append(score)

        mdl.append(model)

        print(model)

    

#create a small df with the scores

performance = pd.DataFrame({'Model': mdl, 'Score':scr,'Fold':fold})

g_downsampled = ggplot(performance,aes(x='Fold',y='Score',group = 'Model',color = 'Model')) + geom_point() + geom_line()

print(g_downsampled)
# normal training set

param_grid = {'n_estimators': [200, 400, 600, 1000], # It is going to be a long search

              'criterion': ['entropy', 'gini'],

              'class_weight' : ['balanced'], 'n_jobs' : [-1]} #use all the computational power you have

acc_scorer = make_scorer(f1_score)

grid_forest = GridSearchCV(RandomForestClassifier(), param_grid, scoring = acc_scorer, cv=5)

%time grid_forest = grid_forest.fit(X_train, y_train)

print(grid_forest.best_estimator_)

print(grid_forest.best_score_)

forest_normal = grid_forest.best_estimator_
#cell added because on Kaggle it takes too much time to run

forest_normal = RandomForestClassifier(bootstrap=True, class_weight='balanced',

            criterion='entropy', max_depth=None, max_features='auto',

            max_leaf_nodes=None, min_impurity_decrease=0.0,

            min_impurity_split=None, min_samples_leaf=1,

            min_samples_split=2, min_weight_fraction_leaf=0.0,

            n_estimators=400, n_jobs=-1, oob_score=False,

            random_state=None, verbose=0, warm_start=False)

print(0.449643478486)
y_upsampled = df_upsampled.def_pay

X_upsampled = df_upsampled.drop(['def_pay'], axis= 1)
# upsampled training set

param_grid = {'n_estimators': [200, 400, 600, 1000],

              'criterion': ['entropy', 'gini'], 'n_jobs' : [-1]}

acc_scorer = make_scorer(f1_score)

grid_forest = GridSearchCV(RandomForestClassifier(), param_grid, scoring = acc_scorer, cv=5)

%time grid_forest = grid_forest.fit(X_upsampled, y_upsampled)

print(grid_forest.best_estimator_)

print(grid_forest.best_score_)

forest_upsampled = grid_forest.best_estimator_
#cell added because on Kaggle it takes too much time to run

forest_upsampled = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=600, n_jobs=-1,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)

print(0.93295704261)
y_downsampled = df_downsampled.def_pay

X_downsampled = df_downsampled.drop(['def_pay'], axis = 1)
# downsampled training set

param_grid = {'n_estimators': [200, 400, 600, 1000],

              'criterion': ['entropy', 'gini'], 'n_jobs' : [-1]}

acc_scorer = make_scorer(f1_score)

grid_forest = GridSearchCV(RandomForestClassifier(), param_grid, scoring = acc_scorer, cv=5)

%time grid_forest = grid_forest.fit(X_downsampled, y_downsampled)

print(grid_forest.best_estimator_)

print(grid_forest.best_score_)

forest_downsampled = grid_forest.best_estimator_
forest_downsampled = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=-1,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)

print(0.686692645307)
# SMOTE training set

param_grid = {'n_estimators': [200, 400, 600, 1000],

              'criterion': ['entropy', 'gini'], 'n_jobs' : [-1]}

acc_scorer = make_scorer(f1_score)

grid_forest = GridSearchCV(RandomForestClassifier(), param_grid, scoring = acc_scorer, cv=5)

%time grid_forest = grid_forest.fit(X_SMOTE, y_SMOTE)

print(grid_forest.best_estimator_)

print(grid_forest.best_score_)

forest_SMOTE = grid_forest.best_estimator_
forest_SMOTE = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=600, n_jobs=-1,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)

print(0.831423121548)
# normal training set

param_grid = {'n_estimators': [200,300],

              'algorithm': ['SAMME', 'SAMME.R'],

              'learning_rate' : [0.5, 0.75, 1.0]}

acc_scorer = make_scorer(f1_score)

grid_ada = GridSearchCV(AdaBoostClassifier(DecisionTreeClassifier(class_weight = "balanced")), 

                        param_grid, scoring = acc_scorer, cv=5)

%time grid_ada = grid_ada.fit(X_train, y_train)

print(grid_ada.best_estimator_)

print(grid_ada.best_score_)

ada_normal = grid_ada.best_estimator_
ada_normal = AdaBoostClassifier(algorithm='SAMME',

          base_estimator=DecisionTreeClassifier(class_weight='balanced', criterion='gini',

            max_depth=None, max_features=None, max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, presort=False, random_state=None,

            splitter='best'),

          learning_rate=0.75, n_estimators=300, random_state=None)

print(0.408435692436)
# upsampled training set

param_grid = {'n_estimators': [200,300],

              'algorithm': ['SAMME', 'SAMME.R'],

              'learning_rate' : [0.5, 0.75, 1.0]}

acc_scorer = make_scorer(f1_score)

grid_ada = GridSearchCV(AdaBoostClassifier(), param_grid, scoring = acc_scorer, cv=5)

%time grid_ada = grid_ada.fit(X_upsampled, y_upsampled)

print(grid_ada.best_estimator_)

print(grid_ada.best_score_)

ada_upsampled = grid_ada.best_estimator_
ada_upsampled = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,

          learning_rate=1.0, n_estimators=300, random_state=None)

print(0.689430792925)
# downsampled training set

param_grid = {'n_estimators': [200,300],

              'algorithm': ['SAMME', 'SAMME.R'],

              'learning_rate' : [0.5, 0.75, 1.0]}

acc_scorer = make_scorer(f1_score)

grid_ada = GridSearchCV(AdaBoostClassifier(), param_grid, scoring = acc_scorer, cv=5)

%time grid_ada = grid_ada.fit(X_downsampled, y_downsampled)

print(grid_ada.best_estimator_)

print(grid_ada.best_score_)

ada_downsampled = grid_ada.best_estimator_
ada_downsampled = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,

          learning_rate=1.0, n_estimators=200, random_state=None)

print(0.673783146613)
# SMOTE training set

param_grid = {'n_estimators': [200,300],

              'algorithm': ['SAMME', 'SAMME.R'],

              'learning_rate' : [0.5, 0.75, 1.0]}

acc_scorer = make_scorer(f1_score)

grid_ada = GridSearchCV(AdaBoostClassifier(), param_grid, scoring = acc_scorer, cv=5)

%time grid_ada = grid_ada.fit(X_SMOTE, y_SMOTE)

print(grid_ada.best_estimator_)

print(grid_ada.best_score_)

ada_SMOTE = grid_ada.best_estimator_
ada_SMOTE = AdaBoostClassifier(algorithm='SAMME', base_estimator=None, learning_rate=0.5,

          n_estimators=200, random_state=None)

print(0.797835003041)
# normal training set

param_grid = {'n_estimators': [200,300],

              'learning_rate' : [0.5, 0.75, 1.0]}

acc_scorer = make_scorer(f1_score)

grid_gbc = GridSearchCV(GradientBoostingClassifier(), param_grid, scoring = acc_scorer, cv=5)

%time grid_gbc = grid_gbc.fit(X_train, y_train)

print(grid_gbc.best_estimator_)

print(grid_gbc.best_score_)

gbc_normal = grid_gbc.best_estimator_
gbc_normal = GradientBoostingClassifier(criterion='friedman_mse', init=None,

              learning_rate=0.5, loss='deviance', max_depth=3,

              max_features=None, max_leaf_nodes=None,

              min_impurity_decrease=0.0, min_impurity_split=None,

              min_samples_leaf=1, min_samples_split=2,

              min_weight_fraction_leaf=0.0, n_estimators=200,

              presort='auto', random_state=None, subsample=1.0, verbose=0,

              warm_start=False)

print(0.443888650557)
#upsampled training set

param_grid = {'n_estimators': [200,300],

              'learning_rate' : [0.5, 0.75, 1.0]}

acc_scorer = make_scorer(f1_score)

grid_gbc = GridSearchCV(GradientBoostingClassifier(), param_grid, scoring = acc_scorer, cv=5)

%time grid_gbc = grid_gbc.fit(X_upsampled, y_upsampled)

print(grid_gbc.best_estimator_)

print(grid_gbc.best_score_)

gbc_upsampled = grid_gbc.best_estimator_
gbc_upsampled = GradientBoostingClassifier(criterion='friedman_mse', init=None,

              learning_rate=1.0, loss='deviance', max_depth=3,

              max_features=None, max_leaf_nodes=None,

              min_impurity_decrease=0.0, min_impurity_split=None,

              min_samples_leaf=1, min_samples_split=2,

              min_weight_fraction_leaf=0.0, n_estimators=300,

              presort='auto', random_state=None, subsample=1.0, verbose=0,

              warm_start=False)

print(0.831025217754)
#downsampled training set

param_grid = {'n_estimators': [200,300],

              'learning_rate' : [0.5, 0.75, 1.0]}

acc_scorer = make_scorer(f1_score)

grid_gbc = GridSearchCV(GradientBoostingClassifier(), param_grid, scoring = acc_scorer, cv=5)

%time grid_gbc = grid_gbc.fit(X_downsampled, y_downsampled)

print(grid_gbc.best_estimator_)

print(grid_gbc.best_score_)

gbc_downsampled = grid_gbc.best_estimator_
gbc_downsampled = GradientBoostingClassifier(criterion='friedman_mse', init=None,

              learning_rate=0.5, loss='deviance', max_depth=3,

              max_features=None, max_leaf_nodes=None,

              min_impurity_decrease=0.0, min_impurity_split=None,

              min_samples_leaf=1, min_samples_split=2,

              min_weight_fraction_leaf=0.0, n_estimators=300,

              presort='auto', random_state=None, subsample=1.0, verbose=0,

              warm_start=False)

print(0.67183972731)
#SMOTE training set

param_grid = {'n_estimators': [200,300],

              'learning_rate' : [0.5, 0.75, 1.0]}

acc_scorer = make_scorer(f1_score)

grid_gbc = GridSearchCV(GradientBoostingClassifier(), param_grid, scoring = acc_scorer, cv=5)

%time grid_gbc = grid_gbc.fit(X_SMOTE, y_SMOTE)

print(grid_gbc.best_estimator_)

print(grid_gbc.best_score_)

gbc_SMOTE = grid_gbc.best_estimator_
gbc_SMOTE = GradientBoostingClassifier(criterion='friedman_mse', init=None,

              learning_rate=0.5, loss='deviance', max_depth=3,

              max_features=None, max_leaf_nodes=None,

              min_impurity_decrease=0.0, min_impurity_split=None,

              min_samples_leaf=1, min_samples_split=2,

              min_weight_fraction_leaf=0.0, n_estimators=200,

              presort='auto', random_state=None, subsample=1.0, verbose=0,

              warm_start=False)

print(0.78976849349)
# Normal sample training

%time forest_normal.fit(X_train, y_train)

predictions = forest_normal.predict(X_test)

print("-------------")

print("f1 score: {}".format(round(f1_score(y_true = y_test, y_pred = predictions),3)))

print("Accuracy: {}".format(round(accuracy_score(y_true = y_test, y_pred = predictions),3)))

print("-------------")

print(get_feature_importance(forest_normal, features))

print("-------------")

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

pred = len(predictions)



print('True Positives: {}'.format(TP))

print('False Positive: {}'.format(FP))

print('True Negative: {}'.format(TN))

print('False Negative: {}'.format(FN))

print('Precision: {}'.format(round(TP/(TP+FP),2)))

print('Recall: {}'.format(round(TP/(TP+FN),2)))

print('Problematic ratio: {}'.format(round(FN/(FN+TP),2)))
# Upsample training

%time forest_upsampled.fit(X_upsampled, y_upsampled)

predictions = forest_upsampled.predict(X_test)

print("-------------")

print("f1 score: {}".format(round(f1_score(y_true = y_test, y_pred = predictions),3)))

print("Accuracy: {}".format(round(accuracy_score(y_true = y_test, y_pred = predictions),3)))

print("-------------")

print(get_feature_importance(forest_upsampled, features))

print("-------------")

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

pred = len(predictions)



print('True Positives: {}'.format(TP))

print('False Positive: {}'.format(FP))

print('True Negative: {}'.format(TN))

print('False Negative: {}'.format(FN))

print('Precision: {}'.format(round(TP/(TP+FP),2)))

print('Recall: {}'.format(round(TP/(TP+FN),2)))

print('Problematic ratio: {}'.format(round(FN/(FN+TP),2)))
# Downsample training

%time forest_downsampled.fit(X_downsampled, y_downsampled)

predictions = forest_downsampled.predict(X_test)

print("-------------")

print("f1 score: {}".format(round(f1_score(y_true = y_test, y_pred = predictions),3)))

print("Accuracy: {}".format(round(accuracy_score(y_true = y_test, y_pred = predictions),3)))

print("-------------")

print(get_feature_importance(forest_downsampled, features))

print("-------------")

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

pred = len(predictions)



print('True Positives: {}'.format(TP))

print('False Positive: {}'.format(FP))

print('True Negative: {}'.format(TN))

print('False Negative: {}'.format(FN))

print('Precision: {}'.format(round(TP/(TP+FP),2)))

print('Recall: {}'.format(round(TP/(TP+FN),2)))

print('Problematic ratio: {}'.format(round(FN/(FN+TP),2)))
# SMOTE training

%time forest_SMOTE.fit(X_SMOTE, y_SMOTE)

predictions = forest_SMOTE.predict(X_test)

print("-------------")

print("f1 score: {}".format(round(f1_score(y_true = y_test, y_pred = predictions),3)))

print("Accuracy: {}".format(round(accuracy_score(y_true = y_test, y_pred = predictions),3)))

print("-------------")

print(get_feature_importance(forest_SMOTE, features))

print("-------------")

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

pred = len(predictions)



print('True Positives: {}'.format(TP))

print('False Positive: {}'.format(FP))

print('True Negative: {}'.format(TN))

print('False Negative: {}'.format(FN))

print('Precision: {}'.format(round(TP/(TP+FP),2)))

print('Recall: {}'.format(round(TP/(TP+FN),2)))

print('Problematic ratio: {}'.format(round(FN/(FN+TP),2)))
# Normal sample training

%time ada_normal.fit(X_train, y_train)

predictions = ada_normal.predict(X_test)

print("-------------")

print("f1 score: {}".format(round(f1_score(y_true = y_test, y_pred = predictions),3)))

print("Accuracy: {}".format(round(accuracy_score(y_true = y_test, y_pred = predictions),3)))

print("-------------")

print(get_feature_importance(ada_normal, features))

print("-------------")

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

pred = len(predictions)



print('True Positives: {}'.format(TP))

print('False Positive: {}'.format(FP))

print('True Negative: {}'.format(TN))

print('False Negative: {}'.format(FN))

print('Precision: {}'.format(round(TP/(TP+FP),2)))

print('Recall: {}'.format(round(TP/(TP+FN),2)))

print('Problematic ratio: {}'.format(round(FN/(FN+TP),2)))
# Upsample training

%time ada_upsampled.fit(X_upsampled, y_upsampled)

predictions = ada_upsampled.predict(X_test)

print("-------------")

print("f1 score: {}".format(round(f1_score(y_true = y_test, y_pred = predictions),3)))

print("Accuracy: {}".format(round(accuracy_score(y_true = y_test, y_pred = predictions),3)))

print("-------------")

print(get_feature_importance(ada_upsampled, features))

print("-------------")

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

pred = len(predictions)



print('True Positives: {}'.format(TP))

print('False Positive: {}'.format(FP))

print('True Negative: {}'.format(TN))

print('False Negative: {}'.format(FN))

print('Precision: {}'.format(round(TP/(TP+FP),2)))

print('Recall: {}'.format(round(TP/(TP+FN),2)))

print('Problematic ratio: {}'.format(round(FN/(FN+TP),2)))
# Downsample training

%time ada_downsampled.fit(X_downsampled, y_downsampled)

predictions = ada_downsampled.predict(X_test)

print("-------------")

print("f1 score: {}".format(round(f1_score(y_true = y_test, y_pred = predictions),3)))

print("Accuracy: {}".format(round(accuracy_score(y_true = y_test, y_pred = predictions),3)))

print("-------------")

print(get_feature_importance(ada_downsampled, features))

print("-------------")

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

pred = len(predictions)



print('True Positives: {}'.format(TP))

print('False Positive: {}'.format(FP))

print('True Negative: {}'.format(TN))

print('False Negative: {}'.format(FN))

print('Precision: {}'.format(round(TP/(TP+FP),2)))

print('Recall: {}'.format(round(TP/(TP+FN),2)))

print('Problematic ratio: {}'.format(round(FN/(FN+TP),2)))
# SMOTE training

%time ada_SMOTE.fit(X_SMOTE, y_SMOTE)

predictions = ada_SMOTE.predict(X_test)

print("-------------")

print("f1 score: {}".format(round(f1_score(y_true = y_test, y_pred = predictions),3)))

print("Accuracy: {}".format(round(accuracy_score(y_true = y_test, y_pred = predictions),3)))

print("-------------")

print(get_feature_importance(ada_SMOTE, features))

print("-------------")

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

pred = len(predictions)



print('True Positives: {}'.format(TP))

print('False Positive: {}'.format(FP))

print('True Negative: {}'.format(TN))

print('False Negative: {}'.format(FN))

print('Precision: {}'.format(round(TP/(TP+FP),2)))

print('Recall: {}'.format(round(TP/(TP+FN),2)))

print('Problematic ratio: {}'.format(round(FN/(FN+TP),2)))
# Normal sample training

%time gbc_normal.fit(X_train, y_train)

predictions = gbc_normal.predict(X_test)

print("-------------")

print("f1 score: {}".format(round(f1_score(y_true = y_test, y_pred = predictions),3)))

print("Accuracy: {}".format(round(accuracy_score(y_true = y_test, y_pred = predictions),3)))

print("-------------")

print(get_feature_importance(gbc_normal, features))

print("-------------")

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

pred = len(predictions)



print('True Positives: {}'.format(TP))

print('False Positive: {}'.format(FP))

print('True Negative: {}'.format(TN))

print('False Negative: {}'.format(FN))

print('Precision: {}'.format(round(TP/(TP+FP),2)))

print('Recall: {}'.format(round(TP/(TP+FN),2)))

print('Problematic ratio: {}'.format(round(FN/(FN+TP),2)))
# Upsample training

%time gbc_upsampled.fit(X_upsampled, y_upsampled)

predictions = gbc_upsampled.predict(X_test)

print("-------------")

print("f1 score: {}".format(round(f1_score(y_true = y_test, y_pred = predictions),3)))

print("Accuracy: {}".format(round(accuracy_score(y_true = y_test, y_pred = predictions),3)))

print("-------------")

print(get_feature_importance(gbc_upsampled, features))

print("-------------")

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

pred = len(predictions)



print('True Positives: {}'.format(TP))

print('False Positive: {}'.format(FP))

print('True Negative: {}'.format(TN))

print('False Negative: {}'.format(FN))

print('Precision: {}'.format(round(TP/(TP+FP),2)))

print('Recall: {}'.format(round(TP/(TP+FN),2)))

print('Problematic ratio: {}'.format(round(FN/(FN+TP),2)))
# Downsample training

%time gbc_downsampled.fit(X_downsampled, y_downsampled)

predictions = gbc_downsampled.predict(X_test)

print("-------------")

print("f1 score: {}".format(round(f1_score(y_true = y_test, y_pred = predictions),3)))

print("Accuracy: {}".format(round(accuracy_score(y_true = y_test, y_pred = predictions),3)))

print("-------------")

print(get_feature_importance(gbc_downsampled, features))

print("-------------")

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

pred = len(predictions)



print('True Positives: {}'.format(TP))

print('False Positive: {}'.format(FP))

print('True Negative: {}'.format(TN))

print('False Negative: {}'.format(FN))

print('Precision: {}'.format(round(TP/(TP+FP),2)))

print('Recall: {}'.format(round(TP/(TP+FN),2)))

print('Problematic ratio: {}'.format(round(FN/(FN+TP),2)))
# SMOTE training

%time gbc_SMOTE.fit(X_SMOTE, y_SMOTE)

predictions = gbc_SMOTE.predict(X_test)

print("-------------")

print("f1 score: {}".format(round(f1_score(y_true = y_test, y_pred = predictions),3)))

print("Accuracy: {}".format(round(accuracy_score(y_true = y_test, y_pred = predictions),3)))

print("-------------")

print(get_feature_importance(gbc_SMOTE, features))

print("-------------")

TP = np.sum(np.logical_and(predictions == 1, y_test == 1))

TN = np.sum(np.logical_and(predictions == 0, y_test == 0))

FP = np.sum(np.logical_and(predictions == 1, y_test == 0))

FN = np.sum(np.logical_and(predictions == 0, y_test == 1))

pred = len(predictions)



print('True Positives: {}'.format(TP))

print('False Positive: {}'.format(FP))

print('True Negative: {}'.format(TN))

print('False Negative: {}'.format(FN))

print('Precision: {}'.format(round(TP/(TP+FP),2)))

print('Recall: {}'.format(round(TP/(TP+FN),2)))

print('Problematic ratio: {}'.format(round(FN/(FN+TP),2)))