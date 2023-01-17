from pandas import Series,DataFrame

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt
heart=pd.read_csv('../input/heart.csv',header=0)

heart.info()
heart.describe().T
fig,axes=plt.subplots(4,4,figsize=(15,15))

feature_names=heart.columns

for f_name,ax in zip(feature_names,axes.ravel()):

    ax.hist(heart[f_name])

    ax.set_title(f_name)

plt.tight_layout()

plt.show()
heart_df=heart.copy()



num_features=['age','trestbps','chol','thalach','oldpeak']



for i in num_features:

    if i=='oldpeak':

        cut=3

        labels=['low','med','high']

    else:

        cut=4

        labels=['low','med-low','med-high','high']

    heart_df[i]=pd.qcut(heart[i],cut,labels=labels)

    new_col=pd.get_dummies(heart_df[i],prefix=i)

    heart_df=pd.concat([new_col,heart_df],axis=1)

    heart_df.drop(i,axis=1,inplace=True)



print(heart_df.head())

heart_df.columns
cat_features=['sex','cp','fbs','restecg','exang','slope','ca','thal']



for j in cat_features:

    new_col=pd.get_dummies(heart_df[j],prefix=j)

    heart_df=pd.concat([new_col,heart_df],axis=1)

    heart_df.drop(j,axis=1,inplace=True)



print(heart_df.head())

heart_df.columns
print('Number of features in new dataframe: {}'.format(len(heart_df.columns)))
X=heart_df.iloc[:,:44].values

y=heart_df.iloc[:,44].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y,random_state=101)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(C=0.1)

logmodel.fit(X_train, y_train)

print('Train accuracy: {}'.format(logmodel.score(X_train,y_train)))

print('Test accuracy: {}'.format(logmodel.score(X_test,y_test)))
from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=2)

forest.fit(X_train,y_train)

print(forest.score(X_train,y_train))

print(forest.score(X_test,y_test))
from sklearn.neighbors import KNeighborsClassifier

for i in range(1,10):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    train_score=knn.score(X_train,y_train)

    test_score=knn.score(X_test,y_test)

    print('k={} train score: {:.2f}  test score {:.2f}'.format(i,train_score,test_score))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=10,shuffle=True,random_state=7)

for i in range(1,10):

    knn=KNeighborsClassifier(n_neighbors=i)

    results = cross_val_score(knn, X, y, cv=kfold)

    print('k={} mean score: {:.2f}'.format(i,results.mean()))
X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(

    heart.iloc[:,:13].values, heart.iloc[:,13].values, test_size=0.30, stratify=heart.iloc[:,13].values,random_state=101)

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler().fit(X_train_org)

X_train_org_scaled=scaler.transform(X_train_org)

X_test_org_scaled=scaler.transform(X_test_org)



logmodel = LogisticRegression()

logmodel.fit(X_train_org_scaled, y_train_org)

print('Train accuracy: {}'.format(logmodel.score(X_train_org_scaled,y_train_org)))

print('Test accuracy: {}'.format(logmodel.score(X_test_org_scaled,y_test_org)))
from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=2)

forest.fit(X_train_org,y_train_org)

print('Train accuracy: {}'.format(forest.score(X_train_org,y_train_org)))

print('Test accuracy: {}'.format(forest.score(X_test_org,y_test_org)))
from sklearn.svm import LinearSVC

svc=LinearSVC(C=0.1).fit(X_train,y_train)

print('Train accuracy: {}'.format(svc.score(X_train,y_train)))

print('Test accuracy: {}'.format(svc.score(X_test,y_test)))
knn=KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train,y_train)

score=knn.score(X_test,y_test)

print('Best model so far:\n\n KNeighbors with k=6, data transformed into One-Hot code\n accuracy: {:.2f}'.format(score))