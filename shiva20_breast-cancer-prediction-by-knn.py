import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score 

from sklearn.metrics import classification_report 

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
data = pd.read_csv('../input/data.csv', index_col=False)

data.head(5)
print(data.shape)
print(data.columns)
print(data.describe)
data['diagnosis']=data['diagnosis'].apply(lambda x: '1' if x=='M' else '0')
data.head(10)
unique_diagnosis=data.groupby('diagnosis').size()

print(unique_diagnosis)
unique_id=data.groupby('id').size()

print(unique_id)
data.drop(['id','Unnamed: 32'],axis=1)
data.plot(kind='density',layout=(5,7),subplots=True,sharex=False, legend=False, fontsize=1)

plt.show()
data.plot(kind='density', subplots=True, layout=(5,7), sharex=False, legend=False, fontsize=1)

plt.show()
data.hist(sharex=False,layout=(5,7) ,sharey=False, xlabelsize=1, ylabelsize=1)

plt.show()
y=data.diagnosis

x=data.iloc[:,1:32]
import seaborn as sns

r = x.corr()

fig, ax = plt.subplots(figsize=(20,20))         # Sample figsize in inches

sns.heatmap(r, annot=True, linewidths=.5, ax=ax)

#sns.heatmap(r, annot = True)

plt.show()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=7)
seed = 7

scoring = 'accuracy'

models=[]

models.append(('KNN', KNeighborsClassifier()))

results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=5, random_state=seed)

    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
pipelines=[]

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',

KNeighborsClassifier())])))

results = []

names = []

for name, model in pipelines:

    kfold = KFold(n_splits=5, random_state=seed)

    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
scaler = StandardScaler().fit(x_train)

print(scaler)

rescaledX = scaler.transform(x_train)

neighbors = [1,3,5,7,9,11,13,15,17,19,21]

param_grid = dict(n_neighbors=neighbors)

model = KNeighborsClassifier(weights='uniform', algorithm='auto',p=2, metric='minkowski')

kfold = KFold(n_splits=5, random_state=seed)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)

grid_result = grid.fit(rescaledX, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
scaler = StandardScaler().fit(x_train)

rescaledX = scaler.transform(x_train)

model = KNeighborsClassifier(n_neighbors=13,weights='distance', algorithm='auto',p=2, metric='minkowski')

kfold = KFold(n_splits=5, random_state=seed)

model.fit(rescaledX, y_train)

rescaledValidationX = scaler.transform(x_test)

predictions = model.predict(rescaledValidationX)

print(accuracy_score(y_test, predictions))    

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))