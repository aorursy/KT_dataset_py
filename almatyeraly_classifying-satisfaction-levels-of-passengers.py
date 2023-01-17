import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest, chi2
train = pd.read_csv('/kaggle/input/airline-passenger-satisfaction/train.csv')

test = pd.read_csv('/kaggle/input/airline-passenger-satisfaction/test.csv')

d = pd.concat([train, test])

d = d.drop(columns=['Unnamed: 0', 'id', 'Arrival Delay in Minutes', 'Gender'])
y = d.satisfaction

ydict = {'neutral or dissatisfied':0,

        'satisfied':1}

y = y.map(ydict)



d = d.drop(columns='satisfaction')

X = d



X = pd.get_dummies(X, columns=['Customer Type', 'Type of Travel', 'Class'])
bestfeatures = SelectKBest(score_func=chi2, k=X.columns.size)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Feature','Score'] 

print(featureScores.nlargest(X.columns.size,'Score')) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

print('Train', X_train.shape, y_train.shape)

print('Test', X_test.shape, y_test.shape)
model = RandomForestClassifier()

model.fit(X_train, y_train)

pred = model.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, pred))