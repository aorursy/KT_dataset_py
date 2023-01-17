import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
data=pd.read_csv("../input/employee_data.csv")
data.head()
data.shape
data.describe(include='all')
data.isna().sum()
sns.countplot(data.status)

plt.title('No of levels in status')
sns.countplot(data.department)

plt.xticks(rotation=45)

plt.title('No of categories in department')
sns.boxplot(x='status',y='satisfaction',data=data)
sns.boxplot(x='status',y='tenure',data=data)
sns.countplot(x='status',hue='department',data=data)

plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.5)
y=data.status

X=data

X.drop('status',axis=1,inplace=True)
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.3,random_state=1)

print(train_X.shape)

print(test_X.shape)

print(train_y.shape)

print(test_y.shape)
cat_cols=['department','salary']

num_cols=data.columns.difference(cat_cols)
#preprocessing on train data

train_X['department'].fillna('others',inplace=True)

train_X['filed_complaint'].fillna('0',inplace=True)

train_X['recently_promoted'].fillna('0',inplace=True)
train_X.isna().sum()
imp_num=SimpleImputer(missing_values=np.nan,strategy='mean')

train_X[num_cols]=imp_num.fit_transform(train_X[num_cols])
train_X[cat_cols]=train_X[cat_cols].apply(lambda x:x.astype("category"))

train_X[num_cols]=train_X[num_cols].apply(lambda x:x.astype("float"))
train_num_data=train_X.loc[:,num_cols]

train_cat_data=train_X.loc[:,cat_cols]
stand=StandardScaler()

stand.fit(train_num_data[train_num_data.columns])

train_num_data[train_num_data.columns]=stand.transform(train_num_data[train_num_data.columns])
train_X=pd.concat([train_num_data,train_cat_data],axis=1)
train_X=pd.get_dummies(train_X,columns=cat_cols)
#preprocessing on validation data

test_X['department'].fillna('others',inplace=True)

test_X['filed_complaint'].fillna('0',inplace=True)

test_X['recently_promoted'].fillna('0',inplace=True)
test_X[num_cols]=imp_num.transform(test_X[num_cols])
test_X[cat_cols]=test_X[cat_cols].apply(lambda x:x.astype("category"))

test_X[num_cols]=test_X[num_cols].apply(lambda x:x.astype("float"))
test_num_data=test_X.loc[:,num_cols]

test_cat_data=test_X.loc[:,cat_cols]
test_num_data[test_num_data.columns]=stand.transform(test_num_data[test_num_data.columns])



test_X=pd.concat([test_num_data,test_cat_data],axis=1)
test_X=pd.get_dummies(test_X,columns=cat_cols)
print(train_X.shape)

print(test_X.shape)
#MODEL 1

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report



log=LogisticRegression()

log.fit(train_X,train_y)



train_pred1=log.predict(train_X)

test_pred1=log.predict(test_X)
print("clasifiacation report on train:",classification_report(train_y,train_pred1))

print("clasifiacation report on test:",classification_report(test_y,test_pred1))
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):

    

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")

    plt.legend(loc="best")

    return plt
plot_learning_curve(estimator=log,title='logistic_regression_learning_curve',X=train_X,y=train_y,cv=5,ylim=(0.25,1.00))
#MODEL 2

from sklearn.tree  import DecisionTreeClassifier

dtc=DecisionTreeClassifier()

dtc.fit(train_X,train_y)

train_preds2=dtc.predict(train_X)

test_preds2=dtc.predict(test_X)
print("clasifiacation report on train:",classification_report(train_y,train_preds2))

print("clasifiacation report on test:",classification_report(test_y,test_preds2))
plot_learning_curve(estimator=dtc,title='Decesion_tree_learning_curve',X=train_X,y=train_y,cv=5,ylim=(0.50,1.50))
#MODEL 3

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

dt=DecisionTreeClassifier()

params={

    'criterion':['gini','entropy'],

    'max_depth':np.arange(4,20,1),

    'min_samples_split':np.arange(0.001,0.1,0.01),

    'max_features':['log2','sqrt','auto'],

    'min_weight_fraction_leaf':np.arange(0.001,0.25,0.05)

}

random=RandomizedSearchCV(dt,param_distributions=params,n_iter=10,verbose=1)

random.fit(train_X,train_y)
random.best_estimator_
params2={

    'criterion':['gini','entropy'],

    'max_depth':np.arange(10,16,1),

    'min_samples_split':np.arange(0.05,0.1,0.01),

    'max_features':['log2','sqrt','auto'],

    'min_weight_fraction_leaf':np.arange(0.001,0.25,0.05)

}

grid=GridSearchCV(estimator=dt,param_grid=params2,cv=5,verbose=1,n_jobs=-1)

grid.fit(train_X,train_y)
grid.best_estimator_.fit(train_X,train_y)
train_preds3=grid.best_estimator_.predict(train_X)

test_preds3=grid.best_estimator_.predict(test_X)
print("clasifiacation report on train:",classification_report(train_y,train_preds3))

print("clasifiacation report on test:",classification_report(test_y,test_preds3))
plot_learning_curve(estimator=grid.best_estimator_,title='Decision_tree_learning_curve',X=train_X,y=train_y,cv=5,ylim=(0.50,1.50))
#MODEL4

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()

param_knn={

    'n_neighbors':np.arange(3,18,1),

    'weights':['uniform','distance'],

    'algorithm':['auto','brute'],

}

random2=RandomizedSearchCV(knn,param_distributions=param_knn,n_iter=10,verbose=1,cv=5)

random2.fit(train_X,train_y)

random2.best_estimator_
param_knn2={

    'n_neighbors':np.arange(3,10,1),

    'weights':['uniform','distance'],

    'algorithm':['auto','brute'],

}

grid2=GridSearchCV(estimator=knn,param_grid=param_knn2,cv=5,verbose=1,n_jobs=-1)

grid2.fit(train_X,train_y)
grid2.best_estimator_.fit(train_X,train_y)

train_preds4=grid2.best_estimator_.predict(train_X)

test_preds4=grid2.best_estimator_.predict(test_X)

print("clasifiacation report on train:",classification_report(train_y,train_preds4))

print("clasifiacation report on test:",classification_report(test_y,test_preds4))
plot_learning_curve(estimator=grid2.best_estimator_,title='knn_learning_curve',X=train_X,y=train_y,cv=5,ylim=(0.50,1.50))
#MODEL 5

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(train_X,train_y)

train_preds5=rfc.predict(train_X)

test_preds5=rfc.predict(test_X)
print("clasifiacation report on train:",classification_report(train_y,train_preds5))

print("clasifiacation report on test:",classification_report(test_y,test_preds5))
rfc2=RandomForestClassifier(n_jobs=-1,max_features="sqrt",class_weight="balanced_subsample")

param_grid = {"n_estimators" : np.arange(10,100,1),

           "max_depth" : np.arange(8,20,1),

           "min_samples_leaf" : np.arange(5,20,1),

           "class_weight" : ['balanced','balanced_subsample']}

random3=RandomizedSearchCV(estimator=rfc2,param_distributions=param_grid,n_iter=5,cv=5,verbose=1)
random3.fit(train_X,train_y)
random3.best_estimator_
param_grid2 = {"n_estimators" : np.arange(75,100,1),

           "max_depth" : np.arange(12,18,1),

           "min_samples_leaf" : np.arange(10,16,1),

           "class_weight" : ['balanced','balanced_subsample']}

grid3=GridSearchCV(estimator=rfc2,param_grid=param_grid2,cv=2,verbose=1,n_jobs=-1)

grid3.fit(train_X,train_y)

grid3.best_estimator_.fit(train_X,train_y)

train_preds6=grid3.best_estimator_.predict(train_X)

test_preds6=grid3.best_estimator_.predict(test_X)
print("clasifiacation report on train:",classification_report(train_y,train_preds6))

print("clasifiacation report on test:",classification_report(test_y,test_preds6))
plot_learning_curve(estimator=grid3.best_estimator_,title='knn_learning_curve',X=train_X,y=train_y,cv=5,ylim=(0.50,1.50))
results=pd.DataFrame({' ':['Employeed','Left'],'LOG-f1_score':[0.86,0.41],'Decision Tree 1-f1_score':[0.97,0.91],'Decision Tree 2-f1_score':[0.91,0.72],'Knn-f1_score':[0.97,0.91],'Random Forest 1':[0.98,0.94],'Random Forest 2':[0.98,0.92]})
results
# "Rondom forest 1", is the best model for prediction,since it has high f1-score