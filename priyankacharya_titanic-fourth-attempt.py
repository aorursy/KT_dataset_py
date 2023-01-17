# Import all the necessary modules

import pandas as pd

import numpy as np

from sklearn import preprocessing, model_selection, svm, metrics
# Read training data and prepare predictor and target variables

df_train = pd.read_csv("train.csv")

X_train = df_train[['Sex','Pclass','Age']]

y_train = df_train['Survived']
# Preprocess predictor data to better fit the model

# Preprocess data.index,'Age'

X_train.loc[X_train.loc[X_train.Sex=="female"].loc[X_train.Age.isna()].index,'Age']=27.92

X_train.loc[X_train.loc[X_train.Sex=="male"].loc[X_train.Age.isna()].index,'Age']=30.73

X_train['Sex'] = np.where(X_train.Sex=="female",1,-1)
# Apply linear SVM to predict survival rate

for C in [1e-3,0.1,1]:

    clf = svm.LinearSVC(C=C,loss='hinge')

    clf.fit(X_train,y_train)

    print(clf.score(X_train,y_train))

    y_pred = clf.predict(X_train)
# Apply linear SVM on normalized data to predict survival rate

X_train_norm = preprocessing.normalize(X_train)



for C in [1e-3,0.1,1]:

    clf = svm.LinearSVC(C=C,loss='hinge')

    clf.fit(X_train,y_train)

    print(clf.score(X_train,y_train))

    y_pred = clf.predict(X_train)
for C in [1e-3,0.1,1]:

    for kernel in ['linear','rbf','sigmoid']:

        print("Predicting for ",kernel," with ",C)

        clf = svm.SVC(C=C,gamma="auto",kernel=kernel)

        clf.fit(X_train,y_train)

        print(clf.score(X_train,y_train))
# Read test data and preprocess

df_test = pd.read_csv("test.csv")

X_test =  df_test[['Sex','Pclass','Age']]

X_test.loc[X_test.loc[X_test.Sex=="female"].loc[X_test.Age.isna()].index,'Age']=30.27

X_test.loc[X_test.loc[X_test.Sex=="male"].loc[X_test.Age.isna()].index,'Age']=30.27

X_test['Sex'] = np.where(X_test.Sex=="female",1,-1)
clf = svm.SVC(C=1,kernel="rbf",gamma="auto")

clf.fit(X_train,y_train)

print(clf.score(X_train,y_train))

y_pred = clf.predict(X_test)

pd.DataFrame(data={'PassengerId':df_test['PassengerId'],'Survived':y_pred}).to_csv("my_submission_5.csv",index=False)
clf = svm.SVC(C=1e7,kernel="rbf",gamma="auto")

clf.fit(X_train_norm,y_train)

print(clf.score(X_train_norm,y_train))

y_pred = clf.predict(X_test)

pd.DataFrame(data={'PassengerId':df_test['PassengerId'],'Survived':y_pred}).to_csv("my_submission_5.csv",index=False)