import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb

import os



from pandas import Series, DataFrame

from pylab import rcParams

from sklearn import preprocessing



from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split 

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_score



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
train_file = '/kaggle/input/titanic/train.csv'

train_df = pd.read_csv(train_file)

train_df.head()
train_df.describe(include = 'all')
train_df.isnull().sum()
Embarked_mode = train_df.Embarked.mode()[0]

train_df.Embarked.fillna(Embarked_mode, inplace = True)
Age_median = train_df.Age.median()

train_df.Age.fillna(Age_median, inplace = True)
train_df.isnull().sum()
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

train_df.head()
from sklearn.preprocessing import LabelEncoder

Sex_encode = LabelEncoder().fit_transform(train_df.Sex)

Embarked_encode = LabelEncoder().fit_transform(train_df.Embarked)



print(Sex_encode[:10], Embarked_encode[:10])
from sklearn.preprocessing import OneHotEncoder



Embarked_1hot = OneHotEncoder().fit_transform(Embarked_encode.reshape(-1, 1))

Embarked_1hot = Embarked_1hot.toarray()



print(Embarked_1hot[:5])
train_df.drop(['Sex', 'Embarked'], inplace = True, axis=1)
Sex_df      = pd.DataFrame(Sex_encode, columns = ['Sex_encded'])

Embarked_df = pd.DataFrame(Embarked_1hot, columns = ['Embrk_C', 'Embrk_Q', 'Embrk_S'])

train_cl_df = pd.concat([train_df, Sex_df, Embarked_df], axis=1)
family_members = train_df.Parch + train_df.SibSp

lonely = family_members + 1

lonely[family_members == 0] = 1

lonely[family_members > 0] = 0

lonely.describe()
Lonely_df   = pd.DataFrame(lonely, columns = ['Lonely'])

train_cl_df = pd.concat([train_cl_df, Lonely_df], axis=1)

print(train_cl_df.describe())

print()

print(train_cl_df.head())
%matplotlib inline

rcParams['figure.figsize'] = 10, 8

sb.set_style('whitegrid')



sb.pairplot(train_cl_df, vars = ['Survived','Pclass','Age','SibSp','Parch','Fare','Sex_encded','Lonely'], palette = 'husl')

plt.show()
# Using heatmap to see the correlation between each pair of variables:



sb.heatmap(train_cl_df.corr(), vmin=-1, vmax=1, annot=True, cmap = 'RdBu_r')

plt.show()
fig, axis = plt.subplots(2, 2,figsize=(10,8))

sb.barplot(x = 'Sex_encded', y ='Survived', data = train_cl_df, ax = axis[0,0])

sb.barplot(x = 'Pclass',     y ='Survived', data = train_cl_df, ax = axis[0,1], palette = 'hls')

sb.barplot(x = 'Embrk_C',    y ='Survived', data = train_cl_df, ax = axis[1,0], palette = 'RdBu_r')

sb.barplot(x = 'Embrk_S',    y ='Survived', data = train_cl_df, ax = axis[1,1], palette = 'RdBu_r')

plt.show()
rcParams['figure.figsize'] = 5, 4

sb.barplot(x = 'Lonely', y ='Survived', data = train_cl_df)

plt.show()
fig, axis = plt.subplots(1, 2,figsize=(12,5))

sb.boxplot(x = 'Pclass', y ='Fare', data = train_cl_df, hue = 'Survived', showfliers = False, ax = axis[0], palette = 'hls')

sb.boxplot(x = 'Sex_encded', y ='Age', data = train_cl_df, hue = 'Survived', showfliers = False, ax = axis[1])

plt.show()
fig, axis = plt.subplots(1, 3,figsize=(17,4))

sb.boxplot(x = 'Sex_encded', y ='Fare', data = train_cl_df, ax = axis[0], hue = 'Survived', showfliers = False)

sb.barplot(x = 'Sex_encded', y ='Survived', data = train_cl_df, ax = axis[1], hue = 'Lonely', palette = 'RdBu_r')

sb.barplot(x = 'Sex_encded', y ='Survived', data = train_cl_df, ax = axis[2], hue = 'Pclass', palette = 'hls')

plt.show()
train_cl_df.drop(['Fare'], inplace = True, axis=1)

train_cl_df.head()
print(train_cl_df.info())

train_cl_df.describe()
X_train, X_test, Y_train, Y_test = train_test_split(train_cl_df.drop('Survived', axis=1),

                                                   train_cl_df['Survived'], test_size=0.2, random_state=10)                             

all_classifiers = {'Ada Boost': AdaBoostClassifier(),

                 'Random Forest': RandomForestClassifier(),

                 'Decision Tree' : DecisionTreeClassifier(),

                 'Logistic Regression': LogisticRegression(solver='liblinear',fit_intercept=True),

                 'KNN': KNeighborsClassifier(),

                 'Gaussian NB': GaussianNB(),

                 'Beroulli  NB': BernoulliNB(),

                  'SVC': SVC(probability = False )}  
ML_name = []

ML_accuracy = []

for Name,classifier in all_classifiers.items():

    classifier.fit(X_train,Y_train)

    Y_pred = classifier.predict(X_test)

    ML_accuracy.append(metrics.accuracy_score(Y_test,Y_pred)) 

    ML_name.append(Name) 
rcParams['figure.figsize'] = 8, 4

plt.barh(ML_name, ML_accuracy, color = 'g')

plt.xlabel('Accuracy Score', fontsize = '14')

plt.ylabel('Machine Learning Algorithms', fontsize = '14')

plt.xlim([0.65, 0.86])

plt.show()
n_estim = [30, 50, 100, 200, 500]

learn_r = [1., 2., 3., 5., 10.]



max_score = 0



for n in n_estim:

    for lr in learn_r:

        MLA = AdaBoostClassifier(n_estimators = n, learning_rate = lr)

        MLA.fit(X_train,Y_train)

        Y_pred = MLA.predict(X_test)

        if metrics.accuracy_score(Y_test,Y_pred) > max_score:

            max_score, n_best, r_best = metrics.accuracy_score(Y_test,Y_pred), n, lr



print('maximum accuracy score, n_estimators, learning_rate:')

print(max_score, n_best, r_best)
#n_estimators=100, min_samples_leaf=10, min_samples_split=20, max_depth=6),

n_estim       = [50, 100, 200]

min_samp_lf   = [1, 2, 5, 10]

min_samp_splt = [2, 4, 8, 12]

maxim_depth   = [2, 4, 8, 12, None]



max_score = 0



for n in n_estim:

    for ml in min_samp_lf:

        for ms in min_samp_splt:

            for md in maxim_depth:

                MLA = RandomForestClassifier(n_estimators=n, min_samples_leaf=ml, min_samples_split=ms, max_depth=md)

                MLA.fit(X_train,Y_train)

                Y_pred = MLA.predict(X_test)

                if metrics.accuracy_score(Y_test,Y_pred) > max_score:

                    max_score, n_best, l_best, s_best, d_best = metrics.accuracy_score(Y_test,Y_pred), n, ml, ms, md



print('maximum accuracy score, n_estimators, min_samples_leaf, min_samples_split, max_depth:')

print(max_score, n_best, l_best, s_best, d_best)
criteri       = ['gini', 'entropy']

min_samp_lf   = [1, 2, 5, 10]

min_samp_splt = [2, 4, 8, 12]

maxim_depth   = [2, 4, 8, 12, None]



max_score = 0



for c in criteri:

    for ml in min_samp_lf:

        for ms in min_samp_splt:

            for md in maxim_depth:

                MLA = DecisionTreeClassifier(criterion=c, min_samples_leaf=ml, min_samples_split=ms, max_depth=md)

                MLA.fit(X_train,Y_train)

                Y_pred = MLA.predict(X_test)

                if metrics.accuracy_score(Y_test,Y_pred) > max_score:

                    max_score, c_best, l_best, s_best, d_best = metrics.accuracy_score(Y_test,Y_pred), c, ml, ms, md





print('maximum accuracy score, criterion, min_samples_leaf, min_samples_split, max_depth:')

print(max_score, c_best, l_best, s_best, d_best)
MLA = DecisionTreeClassifier(criterion='gini', min_samples_leaf=1, min_samples_split=8, max_depth=None)

MLA.fit(X_train,Y_train)

Y_pred = MLA.predict(X_test)

print(metrics.classification_report(Y_test, Y_pred))
CV_scores = cross_val_score(MLA.fit(X_train,Y_train), X_train,Y_train, cv=5)

print ('5-fold cross-validation: scores = ')

print(CV_scores)
Y_pred_CV = cross_val_predict(MLA.fit(X_train, Y_train), X_train, Y_train, cv=6)



rcParams['figure.figsize'] = 5, 4

sb.heatmap(confusion_matrix(Y_train, Y_pred_CV), annot=True, cmap='Greens')

plt.show()
test_file = '/kaggle/input/titanic/test.csv'

test_df = pd.read_csv(test_file)

passengerID_test = test_df.PassengerId

print(test_df.head())

test_df.describe(include = 'all')
Age_median = test_df.Age.median()

test_df.Age.fillna(Age_median, inplace = True)



Fare_median = test_df.Fare.median()

test_df.Fare.fillna(Fare_median, inplace = True)
Sex_encode = LabelEncoder().fit_transform(test_df.Sex)

Embarked_encode = LabelEncoder().fit_transform(test_df.Embarked)



Embarked_1hot = OneHotEncoder().fit_transform(Embarked_encode.reshape(-1, 1))

Embarked_1hot = Embarked_1hot.toarray()



print(Sex_encode[:10], Embarked_encode[:10])
test_df.drop(['Sex', 'Embarked'], inplace = True, axis=1)



Sex_df      = pd.DataFrame(Sex_encode, columns = ['Sex_encded'])

Embarked_df = pd.DataFrame(Embarked_1hot, columns = ['Embrk_C', 'Embrk_Q', 'Embrk_S'])

test_cl_df  = pd.concat([test_df, Sex_df, Embarked_df], axis=1)
family_members = test_df.Parch + test_df.SibSp

lonely = family_members + 1

lonely[family_members == 0] = 1

lonely[family_members > 0] = 0

lonely.describe()
Lonely_df      = pd.DataFrame(lonely, columns = ['Lonely'])

test_cl_df = pd.concat([test_cl_df, Lonely_df], axis=1)



test_cl_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare'], inplace = True, axis=1) 



print(test_cl_df.describe())

print()

test_cl_df.head()
X_eval  = test_cl_df 

#classifier = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, min_samples_split=2, max_depth=None) # 2nd best RF

classifier = RandomForestClassifier(n_estimators=50, min_samples_leaf=5, min_samples_split=4, max_depth=4) # best RF

#classifier = DecisionTreeClassifier(min_samples_leaf=1, min_samples_split=8, max_depth=12) # 2nd best DT

#classifier = DecisionTreeClassifier(criterion= 'entropy', min_samples_leaf=1, min_samples_split=2, max_depth=4) # best DT

classifier.fit(X_train,Y_train)



Y_pred_eval_RF = classifier.predict(X_eval)
Y_pred_eval_RF_df = pd.DataFrame({'PassengerId': passengerID_test,'Survived' :Y_pred_eval_RF})

Y_pred_eval_RF_df.to_csv('evaluation.csv',index = False)
from IPython.display import FileLink

FileLink(r'evaluation.csv')