#import libraries

import pandas as pd

import numpy as np
#import beers

beers = pd.read_csv('../input/beers.csv', index_col='name')
#import breweries

breweries = pd.read_csv('../input/breweries.csv', index_col='name')
sbeers = beers[

    ['abv','ibu','style']

]

sbeers = sbeers.dropna()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(sbeers[['ibu','abv']], sbeers[['style']], random_state=0)

from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler()

#X_train = scaler.fit_transform(X_train)

#X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='liblinear', multi_class='ovr')

logreg.fit(X_train, np.ravel(y_train), )

print('Accuracy of Logistic regression classifier on training set: {:.2f}'

     .format(logreg.score(X_train, y_train)))

print('Accuracy of Logistic regression classifier on test set: {:.2f}'

     .format(logreg.score(X_test, y_test)))
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

fvals = le.fit_transform(np.ravel(sbeers[['style']]))

fvals = np.unique(fvals)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=1001)

rf.fit(X_train, le.transform(np.ravel(y_train)));
predictions = rf.predict(X_test)
def find_nearest(array, value):

    array = np.asarray(array)

    idx = (np.abs(array - value)).argmin()

    return array[idx]
for i in range (len(predictions)):

    predictions[i] = find_nearest(fvals, predictions[i])
predictions = predictions.astype(int)

predictions = le.inverse_transform(predictions)
foo = pd.DataFrame({'actual_values' : y_test['style'], 'predictions':predictions})

foo['comp'] = np.where(foo['actual_values']==foo['predictions'], 1, 0)

print('Accuracy of Logistic regression classifier on test set: {:.2f}'

     .format((len(foo[foo['comp']==1])/len(foo))))
tr_sc = pd.concat([X_train,y_train], axis=1)

tr_sc = tr_sc.sort_values(by='abv')
tr_sc.tail(20)
from sklearn.svm import SVC

svclassifier = SVC(kernel='rbf')

#svclassifier = SVC(kernel='Gaussian')

svclassifier.fit(X_train, np.ravel(y_train));
y_pred = svclassifier.predict(X_test)
foo = pd.DataFrame({'actual_values' : y_test['style'], 'predictions':y_pred})

foo['comp'] = np.where(foo['actual_values']==foo['predictions'], 1, 0)

print('Accuracy of Logistic regression classifier on test set: {:.2f}'

     .format((len(foo[foo['comp']==1])/len(foo))))
beers.head()
breweries = pd.read_csv('../input/breweries.csv')

breweries = breweries.rename(columns ={'Unnamed: 0':'brewery_id'})
cdata = beers.merge(breweries, on='brewery_id')

cdata['state'] = cdata['state'].str.strip()
cdata[cdata['state']=="ND"]