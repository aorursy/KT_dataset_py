import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
df=pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')

df.describe()

df.isnull().sum()
y=df['target']

x=df.drop(['target'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier()

clf.fit(x_train,y_train)

clf.score(x_test,y_test)
predict=clf.predict(x_test)

y_preds=clf.predict_proba(x_test)

y_preds_positive=y_preds[:,1]
from sklearn.metrics import confusion_matrix

conf=confusion_matrix(y_test,predict)

conf
def plot_matrix(conf):

    

    f,ax1=plt.subplots(figsize=(3,3))

    ax1=sns.heatmap(conf,annot=True,cbar=False)

    

plot_matrix(conf)
from sklearn.metrics import roc_auc_score

roc_auc=roc_auc_score(y_test,y_preds_positive)

roc_auc
from sklearn.metrics import roc_curve

fpr,tpr,thresholds=roc_curve(y_test,y_preds_positive)
def plot_roc_curve(fpr,tpr):

    plt.plot(fpr,tpr,color='orange',label='ROC')

    plt.plot([0,1],[0,1],color='darkblue',label='Guessing',linestyle='--')

    plt.legend()

    plt.show()

    

plot_roc_curve(fpr,tpr)
from sklearn.metrics import classification_report

c_report=classification_report(y_test,predict)

print(c_report)
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
def model_score(y_test,predict):

    

    accuracy=accuracy_score(y_test,predict)

    recall=recall_score(y_test,predict)

    precision=precision_score(y_test,predict)

    f1=f1_score(y_test,predict)

    

    metrics_dict={'accuracy':accuracy,

                  'recall':recall,

                  'precision':precision,

                   'f1':f1}

    

    print(f'accuracy:{accuracy}')

    print(f'recall:{recall}')

    print(f'precision:{precision}')

    print(f'f1:{f1}')

    

    return metrics_dict

    
base_line=model_score(y_test,predict)
from sklearn.model_selection import RandomizedSearchCV
grid={'n_estimators':[10,100,200,500,1000,1200],'max_depth':[None,5,10,20,30],'max_features':['auto','sqrt'],'min_samples_split':[2,3,4],'min_samples_leaf':[1,2,4]}
rs_clf=RandomizedSearchCV(estimator=clf,param_distributions=grid,n_iter=5,cv=5,verbose=2)
rs_clf.fit(x_train,y_train)
rs_clf.best_params_
rs_clf_predcit=rs_clf.predict(x_test)
auto_model=model_score(y_test,rs_clf_predcit)
base_line