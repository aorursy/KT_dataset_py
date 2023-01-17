import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
df=pd.read_csv('../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')

df.describe()

df.isnull().sum()
y=df['default.payment.next.month']

x=df.drop(['default.payment.next.month'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier()

clf.fit(x_train,y_train)
clf.score(x_test,y_test)
from sklearn.model_selection import cross_val_score

accuracy_validation=cross_val_score(clf,x_test,y_test,cv=5)

precision_validation=cross_val_score(clf,x_test,y_test,cv=5,scoring='precision')

recall_validation=cross_val_score(clf,x_test,y_test,cv=5,scoring='recall')

f1_validation=cross_val_score(clf,x_test,y_test,scoring='f1')



print(f'accuracy_validation:{accuracy_validation}')

print(f'precision_validation:{precision_validation}')

print(f'recall_validation:{recall_validation}')

print(f'f1_validation:{f1_validation}')
predictions=clf.predict(x_test)

y_preds=clf.predict_proba(x_test)

y_preds_positive=y_preds[:,1]
from sklearn.metrics import roc_auc_score

auc_score=roc_auc_score(y_test,y_preds_positive)

auc_score
from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(y_test,y_preds_positive)
def plot_roc_curve(fpr,tpr):

    plt.plot(fpr,tpr,color='orange',label='ROC')

    plt.plot([0,1],[0,1],color='darkblue',label='Guessing',linestyle='--')

    plt.legend()

    plt.show()

    

plot_roc_curve(fpr,tpr)
from sklearn.metrics import confusion_matrix

conf=confusion_matrix(y_test,predictions)

conf
def plot_matrix(conf):

    f,ax1=plt.subplots(figsize=(3,3))

    ax1=sns.heatmap(conf,annot=True,cbar=False)

    

plot_matrix(conf)
cf=np.array(conf)

matrix_accuracy=(cf[1,1]+cf[0,0])/cf.sum()

matrix_accuracy
from sklearn.metrics import classification_report

report_cls=classification_report(y_test,predictions)

report_cls
from sklearn.metrics import recall_score,precision_score,recall_score,f1_score,accuracy_score
def evaluate_model(y_test,predictions):

    accuracy=accuracy_score(y_test,predictions)

    precision=precision_score(y_test,predictions)

    recall=recall_score(y_test,predictions)

    f1=f1_score(y_test,predictions)

    

    metrics_dict={'accuracy':accuracy,

                  'precision':precision,

                  'recall':recall,

                  'f1':f1}

    print(f'accuracy:{round(accuracy,2)}')

    print(f'precision:{round(precision,2)}')

    print(f'recall:{round(recall,2)}')

    print(f'f1:{round(recall,2)}')

    

    return metrics_dict
base_line=evaluate_model(y_test,predictions)
from sklearn.model_selection import RandomizedSearchCV
grid={'n_estimators':[10,100,150],

      'max_depth':[None,5,10],

      'max_features':['sqrt','auto'],

      'min_samples_split':[2,4,6],

      'min_samples_leaf':[1,2,3]}
rs_clf=RandomizedSearchCV(estimator=clf,param_distributions=grid,n_iter=5,cv=5,verbose=2)
rs_clf.fit(y_train,x_train)
rs_clf.best_params_
rs_predictions=rs_clf.fit(x_test,y_test)
y_clf=rs_clf.predict(x_test)
rs_metrics=evaluate_model(y_test,y_clf)
base_line