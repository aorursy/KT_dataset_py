import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.display import display, HTML
pd.options.display.max_rows = 50

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train_processed.csv',index_col=0)
test_df = pd.read_csv('../input/test_processed.csv',index_col=0)
def fix_data(df):
    df['PassengerId']=df['PassengerId'].astype(int)
    df['SibSp']=df['SibSp'].astype(int)
    df['Parch']=df['Parch'].astype(int)
    df['Surviving family members']=df['Surviving family members'].replace(np.NaN,0)
    df['Known family members']=df['Known family members'].replace(np.NaN,0)
    df['Surviving family members']=df['Surviving family members'].astype(int)
    df['Known family members']=df['Known family members'].astype(int)
    df['Cabin letter']=df['Cabin letter'].replace(['FG','FE'],['F','F'])
    df['Pclass']=df['Pclass'].replace([1.0,2.0,3.0],['first','second','third'])
    
    return df.copy()
train_df=fix_data(train_df)
test_df=fix_data(test_df)
def generate_submission(test_df,pred):
    submission=pd.DataFrame()
    submission['Survived']=pred
    submission['PassengerId']=test_df['PassengerId']
    submission=submission.set_index('PassengerId')
    submission.to_csv('my_submission.csv')
    #display(submission)
avg_survival_prob=train_df['Survived'].sum()/train_df['Survived'].count()
def regress_family_survival_rate(df,avg_survival_prob):
    c=0.5
    q=(df['Known family members'])
    res=(c**q)*avg_survival_prob+(1-c**q)*df['Family survival rate'].replace(np.NaN,0)
    #res[df['Family survival rate'].isnull()]=avg_survival_prob
    df['Regressed family survival rate']=res
    return df.copy()
train_df=regress_family_survival_rate(train_df,avg_survival_prob)
test_df=regress_family_survival_rate(test_df,avg_survival_prob)
#display(train_df[['Family survival rate','Known family members','Regressed family survival rate']])
#display(test_df[['Family survival rate','Known family members','Regressed family survival rate']])
from sklearn.preprocessing import StandardScaler
#display(train_df.head())

def prep_data(train_df,test_df,training_columns):
    numeric_columns=train_df[training_columns].select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
    #categorical_columns=list(set(training_columns)-set(numeric_columns))
    ytrain=(train_df['Survived']*1).astype(int)   
    X=pd.concat([train_df[training_columns],test_df[training_columns]])
    X=(pd.get_dummies(X)).astype('float64')   
    X[numeric_columns]=StandardScaler().fit_transform(X[numeric_columns])
    Xtrain=X.iloc[:train_df.shape[0],:]
    Xtest=X.iloc[train_df.shape[0]:,:]
    return Xtrain,ytrain,Xtest
from sklearn.model_selection import train_test_split
training_columns=['Pclass', 'Sex','Fare','Title','Cabin letter','Embarked','Regressed family survival rate','Known family members'] #'Parch','SibSp','
X,y,X_test=prep_data(train_df,test_df,training_columns)
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
display(X_test.head())
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
def cross_val_neighbors(X,y):
    for k in range(1,20):
        knn=KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=10)
        print("Cross-validation accuracy with k=%d: %.5f"%(k,scores.mean()))
#cross_val_neighbors(X,y)    
k=9
knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(X,y)
scores = cross_val_score(knn, X, y, cv=10)
knn_pred_train=cross_val_predict(knn,X,y,cv=10)
print("Cross-validation accuracy with LNN and k=%d: %.5f"%(k,scores.mean()))

knn_pred=knn.predict(X_test)
knn_prob=knn.predict_proba(X_test)
print("Predictions saved as knn_pred")
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score, cross_val_predict
gnb=GaussianNB()
gnb.fit(X,y)
scores = cross_val_score(gnb, X, y, cv=10)
gnb_pred_train=cross_val_predict(gnb,X,y,cv=10)
print("Cross-validation accuracy with naive Bayes: %.5f"% scores.mean())

gnb_pred=gnb.predict(X_test)
gnb_prob=gnb.predict_proba(X_test)
print("Predictions saved as gnb_pred")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score, cross_val_predict

def cross_val_lr(X,y):
    clist=np.logspace(np.log10(0.6),np.log10(1.2),10)
    for pen in ['l2','l1']:
        for cval in clist:
            lr=LogisticRegression(C=cval,penalty=pen)
            scores = cross_val_score(lr, X, y, cv=10)
            print("Cross-validation accuracy with C=%f with penalty %s: %.5f"%(cval,pen,scores.mean()))
#cross_val_lr(X,y)    
lr=LogisticRegression(C=1,penalty='l2')
lr.fit(X,y)
scores = cross_val_score(lr, X, y, cv=10)
lr_pred_train=cross_val_predict(lr,X,y,cv=10)
print("Cross-validation accuracy with logistic regression: %.5f"% scores.mean())

lr_pred=lr.predict(X_test)
lr_prob=lr.predict_proba(X_test)
print("Predictions saved as lr_pred")
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score, cross_val_predict

def cross_val_svm(X,y):
    clist=np.logspace(np.log10(1.5),np.log10(3.5),10)
    glist=list(np.logspace(np.log10(0.01),np.log10(0.05),10))
    for kernel in ['rbf']:#,'linear']:
        for cval in clist:
            for gamma in glist:
                svm=SVC(C=cval,kernel=kernel,gamma=gamma)
                scores = cross_val_score(svm, X, y, cv=10)
                print("Cross-validation accuracy with C=%f with kernel %s and gamma %s: %.5f"%(cval,kernel,str(gamma),scores.mean()))
#cross_val_svm(X,y)    
svm=SVC(C=2.5,kernel='rbf',gamma='auto',probability=True)
svm.fit(X,y)
scores = cross_val_score(svm, X, y, cv=10)
svm_pred_train=cross_val_predict(svm,X,y,cv=10)
print("Cross-validation accuracy with SVM: %.5f"% scores.mean())

svm_pred=svm.predict(X_test)
svm_prob=svm.predict_proba(X_test)
print("Predictions saved as svm_pred")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score, cross_val_predict

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def acc_model(params):
    rf = RandomForestClassifier(**params)
    return cross_val_score(rf, X, y).mean()

param_space = {
    'max_depth': hp.choice('max_depth', range(3,12)),
    'max_features': hp.choice('max_features', range(10,25)), #(1,len(X.columns))
    'n_estimators': hp.choice('n_estimators', range(100,300)), #(100,500)
    'criterion': hp.choice('criterion', ["gini"])}#, "entropy"]

best = 0
def f(params):
    global best
    acc = acc_model(params)
    if acc > best:
        best = acc
        print('')
        print ('new best:', best, params)
    else:
        print('.', end='')
    return {'loss': -acc, 'status': STATUS_OK}

def hyperparameter_optimization(param_space,f):
    global best
    trials = Trials()
    best = fmin(f, param_space, algo=tpe.suggest, max_evals=100, trials=trials)
    return best
    
#best=hyperparameter_optimization(param_space,f)
best={'criterion': 'gini', 'max_depth': 4, 'max_features': 22, 'n_estimators': 224}
print ('best:')
print (best)

rf=RandomForestClassifier(**best)
rf.fit(X,y)
scores = cross_val_score(rf, X, y, cv=10)
rf_pred_train=cross_val_predict(rf,X,y,cv=10)
print("Cross-validation accuracy with Random Forest: %.5f"% scores.mean())
display(pd.DataFrame(index=X.columns,data=rf.feature_importances_,columns=["Importances"]))
rf_pred=rf.predict(X_test)
rf_prob=rf.predict_proba(X_test)
print("Predictions saved as rf_pred")
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score, cross_val_predict

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def acc_model(params):
    xgb = XGBClassifier(**params)
    return cross_val_score(xgb, X, y).mean()

param_space = {
    'max_depth': hp.randint('max_depth', 20),
    'min_child_weight': hp.choice('min_child_weight', [1]),
    'gamma': hp.loguniform('gamma', -2, 1),
    'lambda':hp.loguniform('lambda_reg', -2, 1),
    'subsample':hp.uniform('subsample', 0.5, 1),
    'eta':hp.loguniform('learning_rate', -1, 0)
}

best = 0
def f(params):
    global best
    acc = acc_model(params)
    if acc > best:
        best = acc
        print('')
        print ('new best:', best, params)
    else:
        print('.', end='')
    return {'loss': -acc, 'status': STATUS_OK}

def hyperparameter_optimization(param_space,f):
    trials = Trials()
    best = fmin(f, param_space, algo=tpe.suggest, max_evals=100, trials=trials)
    return best
#best=hyperparameter_optimization(param_space,f)
best={'gamma': 2.126, 'lambda': 0.2, 'learning_rate': 0.48308, 'max_depth': 3, 'min_child_weight': 1, 'subsample': 0.991}
print ('best:')
print (best)

xgb=XGBClassifier(**best)
xgb.fit(X,y)
scores = cross_val_score(xgb, X, y, cv=10)
xgb_pred_train=cross_val_predict(xgb,X,y,cv=10)
print("Cross-validation accuracy with XGB: %.5f"% scores.mean())
display(pd.DataFrame(index=X.columns,data=xgb.feature_importances_,columns=["Importances"]))
xgb_pred=xgb.predict(X_test)
xgb_prob=xgb.predict_proba(X_test)
print("Predictions saved as xgb_pred")
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score, cross_val_predict

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def acc_model(params):
    ada = AdaBoostClassifier(**params)
    return cross_val_score(ada, X, y).mean()

param_space = {
    'learning_rate': hp.loguniform('learning_rate', -1, 0),
    'n_estimators': (hp.choice('n_estimators', range(10,200,1)))}

best = 0
def f(params):
    global best
    acc = acc_model(params)
    if acc > best:
        best = acc
        print('')
        print ('new best:', best, params)
    else:
        print('.', end='')
    return {'loss': -acc, 'status': STATUS_OK}

def hyperparameter_optimization(param_space,f):
    global best
    trials = Trials()
    best = fmin(f, param_space, algo=tpe.suggest, max_evals=100, trials=trials)
    return best
#best=hyperparameter_optimization(param_space,f)
best={'learning_rate': 0.87, 'n_estimators': 200}#0.8338945005611672 {'learning_rate': 0.8722548459537658, 'n_estimators': 186}
print('')
print ('best:')
print (best)

ada=AdaBoostClassifier(**best)
ada.fit(X,y)
scores = cross_val_score(ada, X, y, cv=10)
ada_pred_train=cross_val_predict(ada,X,y,cv=10)
print("Cross-validation accuracy with AdaBoost: %.5f"% scores.mean())
display(pd.DataFrame(index=X.columns,data=ada.feature_importances_,columns=["Importances"]))
ada_pred=ada.predict(X_test)
ada_prob=ada.predict_proba(X_test)
print("Predictions saved as ada_pred")
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score, cross_val_predict

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def acc_model(params):
    mlp =MLPClassifier(**params)
    return cross_val_score(mlp, X, y).mean()

param_space = {
    'alpha': hp.loguniform('lambda_reg', -2, -1),
    'hidden_layer_sizes': hp.choice('hidden',[(10,10),]),
    'activation' : hp.choice('activation',['identity', 'logistic', 'tanh', 'relu']),
    'max_iter':hp.choice('max_iter',[1000]),
    'tol':hp.choice('tol',[1e-4])}

best = 0
def f(params):
    global best
    acc = acc_model(params)
    if acc > best:
        best = acc
        print('')
        print ('new best:', best, params)
    else:
        print('.', end='')
    return {'loss': -acc, 'status': STATUS_OK}

def hyperparameter_optimization(param_space,f):
    global best
    trials = Trials()
    best = fmin(f, param_space, algo=tpe.suggest, max_evals=10, trials=trials)
    return best
#best=hyperparameter_optimization(param_space,f)
best={'alpha': 0.07,'hidden_layer_sizes': (10,10),'activation' : 'relu','max_iter':2000,'tol':1e-4}
print('')
print ('best:')
print (best)

#mlp=MLPClassifier(**best)
mlp=MLPClassifier(**best)
mlp.fit(X,y)
scores = cross_val_score(mlp, X, y, cv=10)
mlp_pred_train=cross_val_predict(mlp,X,y,cv=10)
print("Cross-validation accuracy with MLP: %.5f"% scores.mean())
mlp_pred=mlp.predict(X_test)
mlp_prob=mlp.predict_proba(X_test)
print("Predictions saved as mlp_pred")
from sklearn.ensemble import VotingClassifier
eclf = VotingClassifier(estimators=[ ('knn', knn),  ('gnb', gnb),('lr', lr),('svm', svm),
                                   ('rf', rf),('xgb',xgb),('ada',ada),('mlp',mlp)],
                        voting='soft')
eclf.fit(X, y)
ypred=eclf.predict(X)
ypred_test=eclf.predict(X_test)
#print(eclf.get_params)
print("Voting accuracy:",accuracy_score(y,ypred))
generate_submission(test_df,ypred_test)
merged_res=pd.DataFrame()
merged_res['knn']=knn_pred
merged_res['gnb']=gnb_pred
merged_res['lr']=lr_pred
merged_res['svm']=svm_pred
merged_res['rf']=rf_pred
merged_res['xgb']=xgb_pred
merged_res['ada']=ada_pred
merged_res['mlp']=mlp_pred

merged_res_prob=pd.DataFrame()
merged_res_prob['knn']=knn_prob[:,1] 
merged_res_prob['gnb']=gnb_prob[:,1] 
merged_res_prob['lr']=lr_prob[:,1] 
merged_res_prob['svm']=svm_prob[:,1] 
merged_res_prob['rf']=rf_prob[:,1] 
merged_res_prob['xgb']=xgb_prob[:,1] 
merged_res_prob['ada']=ada_prob[:,1]
merged_res_prob['mlp']=mlp_prob[:,1] 


agreement_df=pd.DataFrame(data=0,index=merged_res.columns,columns=merged_res.columns)
for col in list(merged_res.columns):
    for row in list(merged_res.columns):
        agreement_df.at[row,col]=(merged_res[row]==merged_res[col]).sum()
agreement_df=agreement_df/merged_res.shape[0]
import seaborn as sns
print("% Agreement between classifiers:")
sns.heatmap(agreement_df,annot=True, fmt = ".2f")
plt.show()
print("Probability correlations between classifers:")
sns.heatmap(merged_res_prob.corr(),annot=True, fmt = ".2f")
plt.show()
#display(agreement_df)
X_prob_train=pd.DataFrame()
X_pred_train['knn']=knn_pred_train
X_pred_train['gnb']=gnb_pred_train
X_pred_train['lr']=lr_pred_train
X_pred_train['svm']=svm_pred_train
X_pred_train['rf']=rf_pred_train
X_pred_train['xgb']=xgb_pred_train
X_pred_train['ada']=ada_pred_train
X_pred_train['mlp']=mlp_pred_train
def cross_val_lr(X,y):
    clist=np.logspace(np.log10(0.01),np.log10(0.05),10)
    for pen in ['l1']:#,'l1']:
        for cval in clist:
            lr_merge=LogisticRegression(C=cval,penalty=pen)
            scores = cross_val_score(lr_merge, X_pred_train, y, cv=10)
            print("Cross-validation accuracy with C=%f with penalty %s: %.5f"%(cval,pen,scores.mean()))
cross_val_lr(X_pred_train,y)    
lr_merge=LogisticRegression(C=0.02,penalty='l1')
lr_merge.fit(X_pred_train,y)
scores = cross_val_score(lr_merge, X_pred_train, y, cv=10)
merged_pred = cross_val_predict(lr_merge, X_pred_train, y, cv=10)
pred_2_submit=lr_merge.predict(merged_res)
print("Merged accuracy with logistic regression: %.5f"% scores.mean())
display(pd.DataFrame(data=lr_merge.coef_[0],index=X_pred_train.columns))
generate_submission(test_df,pred_2_submit)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

def display_confusion(y,ypred):
    cm=pd.DataFrame(confusion_matrix(y, ypred, labels=[0,1]))
    cm.columns=["Prediction: Died","Prediction: Survived"]
    cm.index=["Died","Survived"]
    display(cm)
display_confusion(y,merged_pred)
print(classification_report(y,merged_pred,target_names=["Died","Survived"]))
X_prob_train=pd.DataFrame()
for classifier in ['knn','gnb','lr','svm','rf','xgb','ada','mlp']:
    string2run="X_prob_train['{}']={}.predict_proba(X)[:,1]".format(classifier,classifier)
    #print(string2run)
    exec(string2run)
#display(X_prob_train.head())


def cross_val_lr(X,y):
    clist=np.logspace(np.log10(0.001),np.log10(0.02),10)
    for pen in ['l1']:#,'l1']:
        for cval in clist:
            lr_merge2=LogisticRegression(C=cval,penalty=pen)
            scores = cross_val_score(lr_merge2, X_prob_train, y, cv=10)
            print("Cross-validation accuracy with C=%f with penalty %s: %.5f"%(cval,pen,scores.mean()))
        
#cross_val_lr(X_prob_train,y)    
lr_merge2=LogisticRegression(C=0.02,penalty='l1')
lr_merge2.fit(X_pred_train,y)
scores = cross_val_score(lr_merge2, X_prob_train, y, cv=10)
merged_pred_prob = cross_val_predict(lr_merge, X_prob_train, y, cv=10)
pred_2_submit_prob=lr_merge.predict(merged_res_prob)
print("Merged accuracy with logistic regression: %.5f"% scores.mean())
print("Coefficients:", lr_merge2.coef_[0])
generate_submission(test_df,pred_2_submit_prob)
