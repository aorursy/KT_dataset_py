import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
datapath = "../input/kc1_data.txt"
df=pd.read_csv(datapath,sep=",",header=None,
                 names=['log','v(g)','ev(g)','iv(g)'
                        ,'n','v','l','d','i','e','b'
                        ,'t','10Code','10Comment','10Blank'
                        ,'10CodeAndComment','uniq_op','uniq_Opnd'
                        ,'total_op','total_Opnd','branchCount'
                        ,'problems'],encoding = 'latin')
df.head()
df.columns
df.dtypes
df.shape
df.isnull().sum()
X =df.drop(["problems"],axis=1)
X.head()
y = df[["problems"]]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
x_train.shape  , y_train.shape
x_test.shape , y_test.shape
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svc_model = SVC()
svc_model.fit(x_train,y_train)
svc_pred = svc_model.predict(x_test)
svc_score = accuracy_score(svc_pred,y_test)*100
svc_score
from sklearn.naive_bayes import GaussianNB
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(x_train,y_train)
naive_bayes_pred = naive_bayes_model.predict(x_test)
naive_bayes_score = accuracy_score(naive_bayes_pred,y_test)*100
naive_bayes_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
k_fold = KFold(len(df), n_folds=10, shuffle=True, random_state=0)
svc_cv_model = SVC()
svc_cv_score = cross_val_score(svc_cv_model,X,y,cv=k_fold,scoring = 'accuracy')*100
svc_cv_score
svc_cv_score.mean()
naive_bayes_cv_model = GaussianNB()
naive_bayes_cv_score = cross_val_score(naive_bayes_cv_model,X,y,cv=k_fold,scoring = 'accuracy')*100
naive_bayes_cv_score
naive_bayes_cv_score.mean()
naive_bayes_cv_model.fit(X,y)
naive_bayes_cv_pred = naive_bayes_cv_model.predict(X)
naive_bayes_cv_score = accuracy_score(naive_bayes_cv_pred,y)*100
naive_bayes_cv_score
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier()
tree_cv_score = cross_val_score(tree_model,X,y,cv=k_fold,scoring = 'accuracy')*100
tree_cv_score
tree_cv_score.mean()
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_cv_score = cross_val_score(logistic_model,X,y,cv=k_fold,scoring = 'accuracy')*100
logistic_cv_score
logistic_cv_score.mean()
from sklearn.neighbors import KNeighborsClassifier
k_range = range(1,26)
scores = []
for k in k_range :
    KNN = KNeighborsClassifier(n_neighbors=k)
    KNN.fit(x_train,y_train)
    pred = KNN.predict(x_test)
    scores.append(accuracy_score(pred,y_test)*100)
    
print(pd.DataFrame(scores))
plt.plot(k_range,scores)
plt.xlabel("K for KNN")
plt.ylabel("Testing scores")
plt.show()
k_range = range(1,26)
scores = []
for k in k_range :
    KNN = KNeighborsClassifier(n_neighbors=k)
    KNN_cv_score = cross_val_score(KNN,X,y,cv=k_fold,scoring = 'accuracy')*100
    cv_score = scores.append(KNN_cv_score)
    
print(pd.DataFrame(scores))
KNN_cv_score.mean()
