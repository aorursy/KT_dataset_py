import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
df=pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.describe()
df.isnull().sum()
y=df['Class']
x=df.drop(['Class'],axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(x_train,y_train);
clf.score(x_test,y_test)
from sklearn.model_selection import cross_val_score
validation=cross_val_score(clf,x_test,y_test,cv=5)
validation
y_preds=clf.predict_proba(x_test)
y_positive_preds=y_preds[:,1]
from sklearn.metrics import roc_auc_score
roc_score=roc_auc_score(y_test,y_positive_preds)
roc_score
from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(y_test,y_positive_preds)
def plot_roc_curve(fpr,tpr):
    plt.plot(fpr,tpr,color='orange',label='ROC')
    plt.plot([0,1],[0,1],color='darkblue',label='Guessing',linestyle='--')
    plt.legend()
    plt.show()
    
plot_roc_curve(fpr,tpr)
from sklearn.metrics import confusion_matrix
conf=confusion_matrix(y_test,clf.predict(x_test))
conf
def plot_matrix(conf):
    f,ax1=plt.subplots(figsize=(3,3))
    ax1=sns.heatmap(conf,annot=True,cbar=False)
    
plot_matrix(conf)
from sklearn.metrics import classification_report
report=classification_report(y_test,clf.predict(x_test))
report
predictions=clf.predict(x_test)
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
def evaluate_model(y_test,predictions):
    
    accuracy=accuracy_score(y_test,predictions)
    precision=precision_score(y_test,predictions)
    recall=recall_score(y_test,predictions)
    f1=f1_score(y_test,predictions)
    
    metric_dict={'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1}
    
    print(f'accuracy:{accuracy}')
    print(f'precision:{precision}')
    print(f'recall:{recall}')
    print(f'f1:{f1}')
    
    return metric_dict
baseline_model=evaluate_model(y_test,predictions)
from sklearn.model_selection import RandomizedSearchCV
grid={'n_estimators':[10,100,150],
      'max_depth':[None,5,10,15],
      'max_features':['auto','sqrt'],
      'min_samples_split':[2,4],
       'min_samples_leaf':[1,2]}
rs_clf=RandomizedSearchCV(estimator=clf,param_distributions=grid,n_iter=5,verbose=2)
rs_clf.fit(x_train,y_train)
rs_clf.best_params_
rs_y_preds=rs_clf.predict(x_test)
rs_metrics=evaluate_model(y_test,rs_y_preds)
