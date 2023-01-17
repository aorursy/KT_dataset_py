# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy  # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.model_selection import GridSearchCV





df = pd.read_csv("../input/diabetes.csv")
df.head()
print(df.shape)
print(df.describe())


df.info()
fields = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

for field in fields:

    print('field %s: num 0-entries: %d' %(field,len(df.loc[df[field]==0,field])))
def replace_zero_field(df,field):

    nonzero_vals = df.loc[df[field]!=0,field]

    avg = nonzero_vals.median()

    length = len(df.loc[df[field]==0,field])

    df.loc[df[field]==0,field] = avg

    print('field %s fixed %d entries with value %.3f' %(field,length,avg))

    

for field in fields:

    replace_zero_field(df,field)

    

print()



for field in fields:

       print('field %s: num 0-entries: %d' %(field,len(df.loc[df[field]==0,field])))




df.describe()
data_arr = df.values

X = data_arr[:,0:8]

Y = data_arr[:,8]

print(data_arr.shape)

print(X.shape)

print(Y.shape)
test_size = 0.25

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size,random_state=99)


kfold = KFold(n_splits=10,random_state=96)

model = LogisticRegression()

results = cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')

print(round(100*results.mean(),2),"%","(+/- ", round(100*results.std(),2),"% )")
kfold = KFold(n_splits=10,random_state=99)

model = LogisticRegression()

results = cross_val_score(model,X_train,Y_train,cv=kfold,scoring='neg_log_loss')

print(results.mean())

print(results.std())
kfold = KFold(n_splits=10,random_state=99)

model = KNeighborsRegressor()

result = cross_val_score(model,X_train,Y_train,cv=kfold,scoring='neg_mean_squared_error')

print(result.mean())
kfold = KFold(n_splits=10,random_state=99)

model = RandomForestClassifier(n_estimators=120,max_features=7)

result = cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')

print(result.mean())
kfold = KFold(n_splits=10,random_state=99)

model = svm.SVC(kernel='linear')

result = cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')

print(result.mean())
mdl = svm.SVC()



# prepare a range of values to test

param_grid = [

  {'C': [0.95,0.1,1,10,15,20], 'kernel': ['linear']}

 ]



grid = GridSearchCV(estimator=mdl, param_grid=param_grid,cv=5,scoring='precision')

grid.fit(X_train, Y_train)

# summarize the results of the grid search

print("Best score SVC : ",round(100*grid.best_score_,2),"%")

print("Best estimator for SVC parameter C : ",grid.best_estimator_.C)



mdl = LogisticRegression()



# prepare a range of values to test

param_grid = [

  {'C': [0.95,0.1,1,10,15,20]}

 ]



grid1 = GridSearchCV(estimator=mdl, param_grid=param_grid,cv=5)

grid1.fit(X_train, Y_train)



# summarize the results of the grid search

print("Best score linear regression : ",round(100*grid1.best_score_,2),"%")

print("Best estimator for linear regression parameter C : ",grid.best_estimator_.C)

def cross_valid(model, X_test, y_test, nb_folds):

    fold_size = X_test.shape[0] // nb_folds

    scores = []

    for i in range(nb_folds):

        beg = i * fold_size

        end = (i + 1) * fold_size

        scores.append(model.score(X_test[beg:end],y_test[beg:end]))

    return 'Score : {}% (+/- {}%)'.format(round(numpy.mean(scores)*100,2),round(numpy.std(scores)*100,2))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=99)



kfold = KFold(n_splits=10, random_state=99)

log = LogisticRegression(C=grid1.best_estimator_.C)

log.fit(X_train,Y_train)

print("Cross validation train data : ",cross_valid(log,X_train,Y_train,5))

print("Accuracy test data : ",cross_valid(log,X_test,Y_test,5))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=99)



kfold = KFold(n_splits=10, random_state=99)

svc = svm.SVC(kernel='linear',C=grid.best_estimator_.C)

svc.fit(X_train,Y_train)

print("Cross validation train data : ",cross_valid(svc,X_train,Y_train,5))

print("Accuracy test data : ",cross_valid(svc,X_test,Y_test,5))