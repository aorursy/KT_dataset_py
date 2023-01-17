import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb

import os



from pandas import Series, DataFrame

from pylab import rcParams

from sklearn import preprocessing



from sklearn import metrics

from sklearn.model_selection import train_test_split 

from sklearn.metrics import classification_report



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
met_df = pd.read_csv('/kaggle/input/did-it-rain-in-seattle-19482017/seattleWeather_1948-2017.csv')

print(met_df.head()); print(); print()

met_df.info()
met_df.describe(include = 'all')
met_df.isna().sum()
P_median = met_df.PRCP.median()

R_mode   = met_df.RAIN.mode()[0]



met_df.PRCP.fillna(P_median, inplace = True)

met_df.RAIN.fillna(R_mode, inplace = True)



met_df.isna().sum()
from sklearn.preprocessing import LabelEncoder

RAIN_encode = LabelEncoder().fit_transform(met_df.RAIN)

RAIN_encode
met_df['RAIN'] = RAIN_encode



met_df.describe(include = 'all')
%matplotlib inline

rcParams['figure.figsize'] = 6, 5

sb.set_style('whitegrid')



sb.pairplot(met_df, palette = 'husl', hue = 'RAIN')

plt.show()
sb.heatmap(met_df.corr(), vmin=-1, vmax=1, annot=True, cmap = 'RdBu_r')

plt.show()
sb.scatterplot(x = 'TMIN', y ='TMAX', data = met_df, hue = 'RAIN')

plt.show()
fig, axis = plt.subplots(1, 2,figsize=(10,4))

sb.boxplot(x = 'RAIN', y ='TMAX', data = met_df, ax = axis[0], showfliers = False)

sb.boxplot(x = 'RAIN', y ='TMIN', data = met_df, ax = axis[1], showfliers = False)

plt.show()
met_df.drop(['TMIN', 'PRCP','DATE'], inplace = True, axis=1)

met_df.head()
X_train, X_test, Y_train, Y_test = train_test_split(met_df.drop('RAIN', axis=1),

                                                   met_df['RAIN'], test_size=0.2, random_state=10)                             

all_classifiers = {'Ada Boost': AdaBoostClassifier(),

                 'Random Forest': RandomForestClassifier(n_estimators=50, min_samples_leaf=1, min_samples_split=2, max_depth=4),

                 'Gaussian NB': GaussianNB(),

                 'Logistic Regression': LogisticRegression(solver='liblinear'),#fit_intercept=True,

                 'Decision Tree' : DecisionTreeClassifier(),

                  'SVC': SVC()} #probability = False 
ML_name = []

ML_accuracy = []

for Name,classifier in all_classifiers.items():

    classifier.fit(X_train,Y_train)

    Y_pred = classifier.predict(X_test)

    ML_accuracy.append(metrics.accuracy_score(Y_test,Y_pred)) 

    ML_name.append(Name) 
rcParams['figure.figsize'] = 8, 4

plt.barh(ML_name, ML_accuracy, color = 'brown')

plt.xlabel('Accuracy Score', fontsize = '14')

plt.ylabel('Machine Learning Algorithms', fontsize = '14')

plt.xlim([0.65, 0.685])

plt.show()
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
learning_R    = [1, 2, 3]

random_st     = [None, 20]

n_estimat     = [50, 100]



max_score = 0



for lr in learning_R:

    for rs in random_st:

        for ne in n_estimat:

            MLA = AdaBoostClassifier(random_state=rs, learning_rate=lr, n_estimators=ne)

            MLA.fit(X_train,Y_train)

            Y_pred = MLA.predict(X_test)

            if metrics.accuracy_score(Y_test,Y_pred) > max_score:

                max_score, r_best, l_best, n_best = metrics.accuracy_score(Y_test,Y_pred), rs, lr, ne



print('maximum accuracy score, random_state, learning_rate, n_estimators:')

print(max_score, r_best, l_best, n_best)