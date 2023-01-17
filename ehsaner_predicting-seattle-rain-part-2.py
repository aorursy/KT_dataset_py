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

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

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
date = pd.to_datetime(met_df.DATE, format="%Y-%m-%d")

met_df['DATE'] = date

met_df.head()
## T_RATIO

T_RATIO =  met_df.TMIN / met_df.TMAX

T_RATIO[T_RATIO > 1]  = 1 #correcting an error in initial data where TMIN > TMAX



## PRCP in the past n days (here, n = 3)

n = 3

PRCP_n = np.empty(len(met_df.PRCP))

PRCP_n[:] = 0

PRCP_n = [sum(met_df.PRCP.values[i-n:i]) / n for i in range(n, len(met_df.PRCP.values))]



## daily PRCP

from datetime import datetime

DoY_str = met_df.DATE.dt.strftime('%j')

Day_of_Year = [int(a) for a in DoY_str]

#group data based on "day of year"

groupD     = met_df.groupby(Day_of_Year)

DoY_PRCP   = groupD['PRCP'].mean()  # daily climatological mean PRCP

daily_PRCP = [DoY_PRCP[a] for a in Day_of_Year]
# Add all these new variables to DataFrame:

T_RATIO_df     = pd.DataFrame(T_RATIO,     columns = ['T_RATIO'])

PRCP_n_df      = pd.DataFrame(PRCP_n,      columns = ['PRCP_n'])

daily_PRCP_df  = pd.DataFrame(daily_PRCP,  columns = ['daily_PRCP'])



met_new_df = pd.concat([met_df['DATE'],met_df['PRCP'],PRCP_n_df,daily_PRCP_df,

                        met_df['TMAX'],met_df['TMIN'],T_RATIO_df,met_df['RAIN']], axis = 1)



met_new_df.PRCP_n.fillna(met_new_df.PRCP.median(), inplace = True)



met_new_df.describe(include = 'all')
%matplotlib inline

rcParams['figure.figsize'] = 10, 8

sb.set_style('whitegrid')



sb.pairplot(met_new_df, palette = 'husl', hue = 'RAIN')

plt.show()
rcParams['figure.figsize'] = 10, 8

sb.heatmap(met_new_df.corr(), vmin=-1, vmax=1, annot=True, cmap = 'RdBu_r')

plt.show()
fig, axis = plt.subplots(2, 2,figsize=(12,10))

sb.scatterplot(x = 'TMIN', y ='TMAX', data = met_new_df, hue = 'RAIN', ax = axis[0,0])

sb.scatterplot(x = 'daily_PRCP', y ='T_RATIO', data = met_new_df, hue = 'RAIN', ax = axis[0,1])

sb.scatterplot(x = 'daily_PRCP', y ='TMAX', data = met_new_df, hue = 'RAIN', ax = axis[1,0])

sb.scatterplot(x = 'daily_PRCP', y ='TMIN', data = met_new_df, hue = 'RAIN', ax = axis[1,1])

plt.show()
fig, axis = plt.subplots(1, 3,figsize=(15,5))

sb.boxplot(x = 'RAIN', y ='TMAX', data = met_new_df, ax = axis[0], showfliers = False, palette = 'hls')

sb.boxplot(x = 'RAIN', y ='TMIN', data = met_new_df, ax = axis[1], showfliers = False, palette = 'hls')

sb.boxplot(x = 'RAIN', y ='T_RATIO', data = met_new_df, ax = axis[2], showfliers = False)

plt.show()
fig, axis = plt.subplots(1, 3,figsize=(15,5))

sb.boxplot(x = 'RAIN', y ='PRCP', data = met_new_df, ax = axis[0], showfliers = False, palette = 'husl')

sb.boxplot(x = 'RAIN', y ='daily_PRCP', data = met_new_df, ax = axis[1], showfliers = False)

sb.boxplot(x = 'RAIN', y ='PRCP_n', data = met_new_df, ax = axis[2], showfliers = False)

plt.show()
met_new_df.drop(['DATE','PRCP','TMIN','TMAX'], inplace = True, axis=1)

met_new_df.head()
X_train, X_test, Y_train, Y_test = train_test_split(met_new_df.drop('RAIN', axis=1),

                                                   met_new_df['RAIN'], test_size=0.2, random_state=10)                             

all_classifiers = {'Gradient Boost': GradientBoostingClassifier(),

                 'Ada Boost': AdaBoostClassifier(),

                 'Random Forest': RandomForestClassifier(n_estimators=50, min_samples_leaf=2, min_samples_split=4, max_depth=6),

                 'Logistic Regression': LogisticRegression(),

                 'Decision Tree' : DecisionTreeClassifier(),

                 'KNN': KNeighborsClassifier(),

                 'Gaussian NB': GaussianNB(),

                 'Beroulli  NB': BernoulliNB(),

                  'SVC': SVC(probability = True)} 
ML_name = []

ML_accuracy = []

for Name,classifier in all_classifiers.items():

    classifier.fit(X_train,Y_train)

    Y_pred = classifier.predict(X_test)

    ML_accuracy.append(metrics.accuracy_score(Y_test,Y_pred)) 

    ML_name.append(Name) 
rcParams['figure.figsize'] = 8, 4

plt.barh(ML_name, ML_accuracy, color = 'purple')

plt.xlabel('Accuracy Score', fontsize = '14')

plt.ylabel('Machine Learning Algorithms', fontsize = '14')

plt.xlim([0.7, 0.84])

plt.show()
solve       = ['liblinear', 'sag', 'lbfgs']

fit_interc  = [True, False]

interc_scal = [1, 2, 3]

ccc         = [1, 2, 3]



max_score = 0



for so in solve:

    for fi in fit_interc:

        for ins in interc_scal:

            for c in ccc:

                MLA = LogisticRegression(solver=so, fit_intercept=fi, intercept_scaling=ins, C=c)

                MLA.fit(X_train,Y_train)

                Y_pred = MLA.predict(X_test)

                if metrics.accuracy_score(Y_test,Y_pred) > max_score:

                    max_score, so_best, fi_best, ins_best, c_best = metrics.accuracy_score(Y_test,Y_pred), so, fi, ins, c



print('maximum accuracy score, solver, fit_intercept, intercept_scaling, C:')

print(max_score, so_best, fi_best, ins_best, c_best)
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
learn_r       = [1, 2, 3]

min_samp_lf   = [1, 2, 5]

min_samp_splt = [2, 4, 8]

maxim_depth   = [3, 5, 10]



max_score = 0



for c in learn_r:

    for ml in min_samp_lf:

        for ms in min_samp_splt:

            for md in maxim_depth:

                MLA = GradientBoostingClassifier(learning_rate=c, min_samples_leaf=ml, min_samples_split=ms, max_depth=md)

                MLA.fit(X_train,Y_train)

                Y_pred = MLA.predict(X_test)

                if metrics.accuracy_score(Y_test,Y_pred) > max_score:

                    max_score, c_best, l_best, s_best, d_best = metrics.accuracy_score(Y_test,Y_pred), c, ml, ms, md





print('maximum accuracy score, learning_rate, min_samples_leaf, min_samples_split, max_depth:')

print(max_score, c_best, l_best, s_best, d_best)
MLA = DecisionTreeClassifier(criterion='gini', min_samples_leaf=10, min_samples_split=2, max_depth=8)

MLA.fit(X_train,Y_train)

Y_pred = MLA.predict(X_test)

print(metrics.classification_report(Y_test, Y_pred))
CV_scores = cross_val_score(MLA, X_train, Y_train, cv=5)

print ('5-fold cross-validation: scores = ')

print(CV_scores)
Y_train_pred = cross_val_predict(MLA, X_train, Y_train)

rcParams['figure.figsize'] = 5, 4

sb.heatmap(confusion_matrix(Y_train, Y_train_pred), annot=True, cmap='Purples')

plt.show()