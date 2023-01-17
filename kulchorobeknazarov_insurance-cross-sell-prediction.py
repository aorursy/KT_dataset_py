import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
df=pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
df.describe()
df.isnull().sum()
df.info()
data_w_d=pd.get_dummies(df,drop_first=True)
y=data_w_d['Response']
x=data_w_d.drop(['Response'],axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
predicted_x=clf.predict(x_test)
y_preds=clf.predict_proba(x_test)
y_preds_positive=y_preds[:,1]
from sklearn.metrics import confusion_matrix
conf=confusion_matrix(y_test,predicted_x)
conf
def plot_matrix(conf):
    f,ax1=plt.subplots(figsize=(3,3))
    ax1=sns.heatmap(conf,annot=True,cbar=False)
    
plot_matrix(conf)
conf_accuracy=(conf[0,0]+conf[1,1])/conf.sum()
conf_accuracy
from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(y_test,y_preds_positive)
def plot_roc_curve(fpr,tpr):
    plt.plot(fpr,tpr,color='orange',label='ROC')
    plt.plot([0,1],[0,1],color='darkblue',label='Guesiing',linestyle='--')
    plt.legend()
    plt.show
    
plot_roc_curve(fpr,tpr)
from sklearn.metrics import roc_auc_score
roc_score=roc_auc_score(y_test,y_preds_positive)
roc_score
from sklearn.metrics import classification_report
c_report=classification_report(y_test,predicted_x)
print(c_report)
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
def model_evaluation(y_test,predicted_x):
    
    accuracy=round(accuracy_score(y_test,predicted_x),2)
    precision=round(precision_score(y_test,predicted_x),2)
    recall=round(recall_score(y_test,predicted_x),2)
    f1=round(f1_score(y_test,predicted_x),2)
    
    metrics_dict={"accuracy":accuracy,"precision":precision,"recall":recall,'f1':f1}
    
    print(f'accuracy :{accuracy}')
    print(f'precision:{precision}')
    print(f'recall:{recall}')
    print(f'f1:{f1}')
    
    return metrics_dict
base_line_model=model_evaluation(y_test,predicted_x)
from sklearn.model_selection import RandomizedSearchCV
grid={'n_estimators':[100,200],'max_depth':[None,5,10,20,30],'max_features':['auto','sqrt'],'min_samples_split':[2,4,6],'min_samples_leaf':[2,3,4]}
rs_clf=RandomizedSearchCV(estimator=clf,param_distributions=grid,n_iter=5,cv=5,verbose=2)
auto_model=rs_clf.fit(x_train,y_train);
rs_clf.best_params_
auto_predicted=rs_clf.predict(x_test)
model_evaluation(y_test,auto_predicted)
base_line_model
