import kaggle

import os

import pandas as pd, numpy as np

from datetime import datetime

import time



df = pd.read_csv('../input/ATP.csv', low_memory=False)
# what does it look like?

print(df.shape)

df.head()
df.info()
# these variables do not seem relevant to me. might be assessed in a further work

df = df.drop(columns=['tourney_id','tourney_name','tourney_date','match_num','winner_entry','loser_entry','winner_id','winner_name','score','loser_id','loser_name'])



# convert numeric varibales to the correct type (csv_read fct does not make auto convert)

col_names_to_convert = ['winner_seed','draw_size','winner_ht','winner_age','winner_rank','winner_rank_points',

                       'loser_seed','loser_ht','loser_age','loser_rank','loser_rank_points','best_of','minutes',

                       'w_ace','w_df','w_svpt','w_1stIn','w_1stWon','w_2ndWon','w_SvGms','w_bpSaved','w_bpFaced',

                       'l_ace','l_df','l_svpt','l_1stIn','l_1stWon','l_2ndWon','l_SvGms','l_bpSaved','l_bpFaced'

                       ]

for col_name in col_names_to_convert:

    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
df.describe().transpose()
# append a new target variable with the code assigned to winner player (0 when P1 | 1 when P2)

# For this set of data, the winner is always P1, so append 0s to the target variable

df['target'] = np.zeros(df.shape[0], dtype = int)
# Now we'll generate the second batch of data, ie, by switching P1 and P2. The winner this time will be P2, and the target variable =1

# generate data by switching among P1 and P2 (target will be P2)

df2 = df.copy()

# switch between variables from P1 and those from P2

df2[['winner_seed','winner_hand','winner_ht','winner_ioc','winner_age','winner_rank','winner_rank_points']] = df[['loser_seed','loser_hand','loser_ht','loser_ioc','loser_age','loser_rank','loser_rank_points']]

df2[['loser_seed','loser_hand','loser_ht','loser_ioc','loser_age','loser_rank','loser_rank_points']] = df[['winner_seed','winner_hand','winner_ht','winner_ioc','winner_age','winner_rank','winner_rank_points']]

df2[['w_ace','w_df','w_svpt','w_1stIn','w_1stWon','w_2ndWon','w_SvGms','w_bpSaved','w_bpFaced']] = df[['l_ace','l_df','l_svpt','l_1stIn','l_1stWon','l_2ndWon','l_SvGms','l_bpSaved','l_bpFaced']]

df2[['l_ace','l_df','l_svpt','l_1stIn','l_1stWon','l_2ndWon','l_SvGms','l_bpSaved','l_bpFaced']] = df[['w_ace','w_df','w_svpt','w_1stIn','w_1stWon','w_2ndWon','w_SvGms','w_bpSaved','w_bpFaced']]

df2['target'] = np.ones(df2.shape[0], dtype = int)



df = df.append(df2)
df.head(2).append(df.tail(2))
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

df['surface'] = lb.fit_transform(df['surface'].astype(str))

df['tourney_level'] = lb.fit_transform(df['tourney_level'].astype(str))

df['winner_hand'] = lb.fit_transform(df['winner_hand'].astype(str))

df['loser_hand'] = lb.fit_transform(df['loser_hand'].astype(str))

df['round'] = lb.fit_transform(df['round'].astype(str))

df['winner_ioc'] = lb.fit_transform(df['winner_ioc'].astype(str))

df['loser_ioc'] = lb.fit_transform(df['loser_ioc'].astype(str))
# replace nan with 0 and infinity with large values

df = df.fillna(df.median())
# subsample for test purpose : TODO: REMOVE FOR FINAL RUN

df = df.sample(100000)



# split train/test subsets (80% train, 20% test)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.target, test_size=.2, random_state=0)
# import classifiers from sklearn

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.linear_model import RidgeClassifier

from sklearn.linear_model import PassiveAggressiveClassifier



# set names and prepare the benchmark list

names = ["K Near. Neighb.", "Decision Tree", "Random Forest", "Naive Bayes", "Quad. Dis. Analys", "AdaBoost", 

         "Neural Net" #, "RBF SVM", "Linear SVM", "Ridge Classifier"

        ]



classifiers = [

    KNeighborsClassifier(10),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    GaussianNB(),

    QuadraticDiscriminantAnalysis(),

    AdaBoostClassifier(),

    MLPClassifier(alpha=1, max_iter=1000)

    # too long run for the test

    #SVC(gamma=2, C=1),

    #SVC(kernel="linear", C=.025),

    #RidgeClassifier(tol=.01, solver="lsqr")

]
# init time 

tim = time.time()

print('Learn. model\t\t score\t\t\ttime')

scores = []



for name, clf in zip(names, classifiers):

        print(name, end='')

        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test)

        print('\t\t', round(score, 3), '%', '\t\t', round(time.time() - tim, 3))

        scores.append(score)

        tim = time.time()

# plot results

import matplotlib.pyplot as plt



plt.rcdefaults()



y_pos = np.arange(len(names))



plt.bar(y_pos, scores, align='center', alpha=0.5)

plt.xticks(y_pos, names, rotation='vertical')

plt.ylabel('Accuracy')

plt.title('Model comparison for ATP prediction')



plt.show()