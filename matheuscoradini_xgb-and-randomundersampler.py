import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RepeatedKFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

import seaborn as sns

import xgboost as xgb

from imblearn.under_sampling import RandomUnderSampler
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.head()
df['Class'].value_counts()
X = df.drop('Class', axis = 1)

y = df['Class']



Xtrn, Xval, ytrn, yval = train_test_split(X, y, test_size=0.33, random_state=1)



mdl = RandomForestClassifier(n_jobs = -1, n_estimators = 20)

mdl.fit(Xtrn, ytrn)

pred = mdl.predict(Xval)

print(confusion_matrix(yval,pred))

print('False Negative MEAN: {}'.format(confusion_matrix(yval,pred)[1,0]/(confusion_matrix(yval,pred)[1,1] + confusion_matrix(yval,pred)[1,0])))
X = df.drop('Class', axis = 1)

y = df['Class']

mdl = RandomForestClassifier(n_jobs = -1)



Xtrn, Xval, ytrn, yval = train_test_split(X, y, test_size=0.33, random_state=0)



ru = RandomUnderSampler()

Xres, yres = ru.fit_sample(Xtrn, ytrn)



mdl.fit(Xres, yres)

pred = mdl.predict(Xval)

print(confusion_matrix(yval,pred))

print('False Negative MEAN: {}'.format(confusion_matrix(yval,pred)[1,0]/(confusion_matrix(yval,pred)[1,1] + confusion_matrix(yval,pred)[1,0])))


Xtrn, Xval, ytrn, yval = train_test_split(X, y, test_size=0.33, random_state=0)

ru = RandomUnderSampler()

Xres, yres = ru.fit_sample(Xtrn, ytrn)



mdl = xgb.XGBClassifier()

mdl.fit(Xres, yres)

pred = mdl.predict(Xval)

print(confusion_matrix(yval,pred))

print('False Negative MEAN: {}'.format(confusion_matrix(yval,pred)[1,0]/(confusion_matrix(yval,pred)[1,1] + confusion_matrix(yval,pred)[1,0])))
def cross_val(X, y, mdl):

    

    rkfold = RepeatedKFold(n_splits = 3, n_repeats = 5, random_state = 0)

    result1 = []

    result0 = []

    for treino, teste in rkfold.split(X):

        Xtrn, Xval = X.iloc[treino], X.iloc[teste]

        ytrn, yval = y.iloc[treino], y.iloc[teste]

        

        ru = RandomUnderSampler()

        Xres, yres = ru.fit_sample(Xtrn, ytrn)

        

        mdl.fit(Xres, yres)

        pred = mdl.predict(Xval)

        erro_1 = confusion_matrix(yval,pred)[1,0]/(confusion_matrix(yval,pred)[1,1] + confusion_matrix(yval,pred)[1,0])

        

        erro_0 = confusion_matrix(yval,pred)[0,1]/(confusion_matrix(yval,pred)[0,0] + confusion_matrix(yval,pred)[0,1])

        

        result1.append(erro_1)

        result0.append(erro_0)

        

    print ('False Negatives MEAN:',np.mean(result1))

    print ('False Positives MEAN:',np.mean(result0))
X = df.drop('Class', axis = 1)

y = df['Class']

mdl = RandomForestClassifier(n_jobs = -1)



cross_val(X, y, mdl)
X = df.drop('Class', axis = 1)

y = df['Class']

mdl = xgb.XGBClassifier()



cross_val(X, y, mdl)