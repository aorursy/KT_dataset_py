import numpy as np
import pandas as pd
pd.options.display.max_rows = 40000
pd.options.display.max_columns = 40000
ds=pd.read_csv('train.csv')

ds=ds.drop(columns=['Cabin','Name','PassengerId','Ticket'])
ds.corr(method ='pearson')
dlist=[]
for items in ds.isnull().sum().iteritems():
    if items[1]>0:
        dlist.append(items[0])
for j in dlist:
    count=0
    for i in ds[j].value_counts().iteritems():
        count=count+1
        if count==1:
            ds[j].fillna(value=i[1],inplace=True)
ds.isnull().sum()
ds.Embarked.value_counts()
ds = ds[ds.Embarked != 644]
sur=ds[['Survived']]
ds=ds.drop(columns=['Survived'])
ds.Embarked.value_counts()
ds=pd.get_dummies(ds,columns=['Sex','Embarked'],drop_first=True)
ds
dx=ds.copy()
list(ds)
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
dx=sc_X.fit_transform(dx)
dx
dx1=pd.DataFrame(dx, columns=['Pclass',
 'Age',
 'SibSp',
 'Parch',
 'Fare',
 'Sex_male',
 'Embarked_Q',
 'Embarked_S'])
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
sel_model_train=SelectFromModel(Lasso(alpha=0.005,random_state=0,tol=0.007,max_iter=100000))
sel_model_train.fit(dx1,sur)
sel_feat=dx1.columns[(sel_model_train.get_support())]
sel_feat
dx1=dx1[sel_feat]
dx1=dx1.iloc[:,:].values
sur=sur.iloc[:,:].values
from sklearn.ensemble import GradientBoostingClassifier
classifier=GradientBoostingClassifier(learning_rate= 0.001,max_depth=7,n_estimators= 1000,subsample= 0.7)
classifier.fit(dx1,sur)

from sklearn.svm import SVC
classifier=SVC()
classifier.fit(dx1,sur)
from sklearn.model_selection import GridSearchCV
n_estimators = [10, 100, 1000]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]
grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
grid_search = GridSearchCV(estimator=classifier, param_grid=grid, n_jobs=-1, cv=10, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(dx1, sur)
grid_result.best_params_
parameters={"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
             "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
             "min_child_weight" : [ 1, 3, 5, 7 ],
             "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
             "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
              "booster"         : ['gblinear','gbtree'],
              "n_estimator"      :[50,100,300,500]}
random_search=RandomizedSearchCV(estimator= classifier,
                        param_distributions=parameters,
                        scoring = 'neg_mean_absolute_error',
                        cv = 10,
                        n_iter=50,
                        n_jobs= 4,
                        verbose=5,
                        return_train_score=True,
                        random_state=42)
best_acc=random_search.fit(dx1,sur)
random_search.best_estimator_
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=dx1,y=sur,cv=10)
print(accuracies.mean())

tds=pd.read_csv('test.csv')
tds=tds.drop(columns=['Cabin','Name','PassengerId','Ticket'])
elist=[]
for items in tds.isnull().sum().iteritems():
    if items[1]>0:
        elist.append(items[0])
for j in elist:
    count=0
    for i in tds[j].value_counts().iteritems():
        count=count+1
        if count==1:
            tds[j].fillna(value=i[1],inplace=True)
tds.isnull().sum()
tds.Embarked.unique()
tds=pd.get_dummies(tds,columns=['Sex','Embarked'],drop_first=True)
tdx=tds.copy()
tdx=tdx.drop(columns=['Embarked_Q'])
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
tdx=sc_X.fit_transform(tdx)
ypred=classifier.predict(tdx)
ca=pd.read_csv("gender_submission.csv")
ca=pd.DataFrame(ca)
cdf = pd.DataFrame(ypred)
xs=pd.concat([ca['PassengerId'],cdf],axis=1)
xs.columns=['PassengerId','Survived']
xs.to_csv('Xtunedgraboost.csv',index=False)