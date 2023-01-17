# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn.cross_validation import train_test_split





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv("../input/EEG data.csv")

#print(df)



#X_train subject 0 all videos except for 0

subject0 = df['subject ID']==0

video0 = df['Video ID']!=0            

dff = df[subject0 & video0]

#print(np.array(dff['Meditation']))

Y_train = np.array(dff['Meditation'])

X_train = np.array(dff[['Delta', 'Theta', 'Alpha 1', 'Alpha 2', 'Beta 1', 'Beta 2', 'Gamma1', 'Gamma2']])

#print(X_train)



#clf = linear_model.SGDRegressor(alpha=0.0001, average=False, epsilon=0.1, eta0=0.01,

#            fit_intercept=False, l1_ratio=0.15, learning_rate='invscaling',

#            loss='squared_loss', n_iter=5, penalty='l2', power_t=0.25,

#            random_state=None, shuffle=True, verbose=0, warm_start=False)





X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=32)





clf = linear_model.LinearRegression()



clf.fit(X_train, Y_train)

print(clf.coef_)



#X_test subject 0 and only video 0

video0 = df['Video ID']==0            

dff = df[subject0 & video0]

Y_test = np.array(dff['Meditation'])

X_test = np.array(dff[['Delta', 'Theta', 'Alpha 1', 'Alpha 2', 'Beta 1', 'Beta 2', 'Gamma1', 'Gamma2']])

#print(X_test)



print(clf.predict(X_test))

print(Y_test)



print("Residual sum of squares: %.2f" % np.mean((clf.predict(X_test)-Y_test)**2))



print("Score: %.2f" % clf.score(X_test, Y_test))





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn.cross_validation import train_test_split

from sklearn.cross_validation import cross_val_score





df = pd.read_csv("../input/EEG data.csv")

#print(df)



#X_train subject 0 all videos except for 0

#subject0 = df['subject ID']==0

#video0 = df['Video ID']!=0            

#dff = df[subject0 & video0]

#print(np.array(dff['Meditation']))

Y_train = np.array(df['Meditation'])

X_train = np.array(df[['Alpha 1', 'Alpha 2']])

#print(X_train)



clf = linear_model.LinearRegression()



X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=22)



#scores = cross_val_score(clf, X_train, Y_train, cv=6, scoring='r2')

#print(scores)



clf.fit(X_train, Y_train)

print(clf.coef_)



#print("Residual sum of squares: %.2f" % np.mean((clf.predict(X_test)-Y_test)**2))



print("Score: %.2f" % clf.score(X_test, Y_test))



#X_test subject 0 video 0

#video0 = df['Video ID']==0            

#dff = df[subject0 & video0]

#Y_test = np.array(dff['Meditation'])

#X_test = np.array(dff[['Theta', 'Alpha 1', 'Alpha 2']])

#print(X_test)



#print(clf.predict(X_test))

#print(Y_test)



#print("Residual sum of squares: %.2f" % np.mean((clf.predict(X_test)-Y_test)**2))



#print("Score: %.2f" % clf.score(X_test, Y_test))


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn.cross_validation import train_test_split

from sklearn.cross_validation import cross_val_score

from sklearn.linear_model import RidgeClassifierCV



df = pd.read_csv("../input/EEG data.csv")

#print(df)



#X_train subject 0 all videos except for 0

#subject0 = df['subject ID']==0

#video0 = df['Video ID']!=0            

#dff = df[subject0 & video0]

#print(np.array(dff['Meditation']))

Y = np.array(df['Self-defined label'])

X = np.array(df[['Delta', 'Theta', 'Alpha 1', 'Alpha 2', 'Beta 1', 'Beta 2', 'Gamma1', 'Gamma2']])

#print(X_train)

#print(Y[9400])





X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=22)



clf = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), fit_intercept=False, normalize=True, scoring=None, cv=None )



clf.fit(X_train, Y_train)



print(clf.score(X_test, Y_test))



print(clf.coef_)