import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import confusion_matrix,roc_curve,classification_report
df=pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.describe()
df.isnull().sum()
data_cleaned=df.drop(['Unnamed: 32'],axis=1)
data_w_d=pd.get_dummies(data_cleaned,drop_first=True)
y=data_w_d['diagnosis_M']
x=data_w_d.drop(['diagnosis_M'],axis=1)
models={'LogisticRegression':LogisticRegression(),'KNN':KNeighborsClassifier(),'RandomForest':RandomForestClassifier()}
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
def fit_and_score(models,x_train,x_test,y_train,y_test):
    model_scores={}
    for name ,model in models.items():
        model.fit(x_train,y_train)
        model_scores[name]=model.score(x_test,y_test)
    return model_scores
model_scores=fit_and_score(models,x_train,x_test,y_train,y_test)
model_scores
clf=RandomForestClassifier()
clf.fit(x_train,y_train)
predicted_x=clf.predict(x_test)
y_preds=clf.predict_proba(x_test)
y_preds_positive=y_preds[:,1]
conf=confusion_matrix(y_test,predicted_x)
conf
def plot_matrix(conf):
    f,ax1=plt.subplots(figsize=(4,4))
    ax1=sns.heatmap(conf,annot=True,cbar=False)
    
plot_matrix(conf)
fpr,tpr,thresholds=roc_curve(y_test,y_preds_positive)
def plot_roc_curve(fpr,tpr):
    plt.plot(fpr,tpr,color='orange',label='ROC')
    plt.plot([0,1],[0,1],color='darkblue',label='Guessing',linestyle='--')
    plt.legend()
    plt.show()
    
plot_roc_curve(fpr,tpr)
c_rep=classification_report(y_test,predicted_x)
print(c_rep)
def evaluate_model(y_test,predicted_x):
    accuracy=accuracy_score(y_test,predicted_x)
    recall=recall_score(y_test,predicted_x)
    precision=precision_score(y_test,predicted_x)
    f1=f1_score(y_test,predicted_x)
    
    models_dict={'accuracy':accuracy,'recall':recall,'precision':precision,'f1':f1}
    
    print(f'accuracy:{accuracy}')
    print(f'recall:{recall}')
    print(f'precision:{precision}')
    print(f'f1:{f1}')
base_line=evaluate_model(y_test,predicted_x)
grid={'n_estimators':[100,150,200],'max_depth':[None,5,10],'max_features':['auto','sqrt'],'min_samples_split':[2,4,5],'min_samples_leaf':[2,3,4]}
rs_clf=RandomizedSearchCV(estimator=clf,param_distributions=grid,n_iter=10,cv=5,verbose=2)
rs_clf.fit(x_train,y_train)
auto=rs_clf.predict(x_test)
auto=evaluate_model(y_test,auto)
base_line=evaluate_model(y_test,predicted_x)
