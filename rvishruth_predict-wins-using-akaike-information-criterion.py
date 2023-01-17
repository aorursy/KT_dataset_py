import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns
df = pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')

df.head()
df.info()
cols = ['gameId', 'blueEliteMonsters',  'blueAvgLevel',

        'blueTotalMinionsKilled',  'blueCSPerMin',

       'blueGoldPerMin', 'redFirstBlood', 'blueDeaths', 'redDeaths', 'redEliteMonsters', 'redTotalGold', 'redAvgLevel', 'redTotalExperience',

       'redTotalMinionsKilled',  'redGoldDiff',

       'redExperienceDiff', 'redCSPerMin', 'redGoldPerMin']



df_new = df.copy()

df_new = df_new.drop(cols, axis = 1)



df_new.head()
y = df_new['blueWins']

X = sm.add_constant(df_new.loc[:, 'blueWardsPlaced' :])



model = sm.Logit(y, X).fit()

results = model.summary()

results
from sklearn.model_selection import train_test_split 



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
def minAIC(X,y):

    variables = X.columns

    model = sm.Logit(y,X[variables]).fit()

    while True:

        print("Old AIC = ", model.aic)

        maxp = np.max(model.pvalues)

        newvariables = variables[model.pvalues < maxp]

        newmodel = sm.Logit(y,X[newvariables]).fit()

        print("New AIC = ", newmodel.aic)

        if newmodel.aic < model.aic:

            model = newmodel

            variables = newvariables

        else:

            break

    return model,variables



new_model, logit_variables = minAIC(X_train, y_train)

new_model = sm.Logit(y_train, X_train[logit_variables]).fit()

results = new_model.summary()

results
colu = ['blueDragons',

                  'blueGoldDiff', 'blueExperienceDiff', 'redDragons']

data = X_test.loc[:, colu  ]

data.head()
answers = new_model.predict(data)

answers
from sklearn.metrics import confusion_matrix, roc_curve



FPR, TPR, thresh = roc_curve(y_test, answers)

plt.scatter(FPR,TPR)

plt.xlabel("FPR")

plt.ylabel("TPR")

plt.show()
from sklearn.metrics import accuracy_score



answers_2 =[]

for x in answers:

  if x >=0.50:

    answers_2.append(1)

  else:

    answers_2.append(0)

    

acc_knn = accuracy_score(answers_2, y_test)

print("accuracy score = " ,acc_knn)
