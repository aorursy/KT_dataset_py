import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
data = pd.read_csv('/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv',sep=';')
data
#Check for missing values
data.isnull().sum()
#Outlier Detection
#Systolic BP(ap_hi) not greater than 370 and not less than 70
#Dialstolic BP(ap_lo) not greater than 360 and not less than 50
outlier = (data['ap_hi'] >= 370)| (data['ap_hi'] <= 70)|(data['ap_lo']>= 360)|(data['ap_lo']<=50)
data[outlier].count()
data = data[~outlier]
# BMI is an important feature in predicting the Cardio vascular disease. As this feature is not available, we can calculate 
# it using Height and Weight.
#First convert Height in metres and square it
def bmi_conversion(x):
    x = (x/100)**2
    return x
data['height1'] = data['height'].apply(lambda x: bmi_conversion(x))
data
data['BMI'] = data['weight']/data['height1']
data['age'] = data['age'].apply(lambda x: round(x/365))
data
data
plt.scatter(data['cardio'],pd.to_numeric(data['ap_hi']),label = 'Systolic')
plt.scatter(data['cardio'],data['BMI'],label = 'BMI')            
plt.legend()
plt.show()
cardio = data['cardio']
cardio
#Dropping columns ID and Height1 as they are not important and cardio as it is dependent variable
data.drop(['id','height1','cardio'],axis = 1, inplace = True)
data
#Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X_tr,X_ts,Y_tr,Y_ts = train_test_split(data,cardio,test_size = 0.2,random_state = 0)
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_tr,Y_tr)
Y_pred = lr.predict(X_ts)
fpr1,tpr1,thresholds = roc_curve(Y_ts,lr.predict_proba(X_ts)[:,1])
lr_a = auc(fpr1,tpr1)
lr_acc = lr.score(X_ts,Y_ts)
print('AUC of Logistic Regression:',lr_a)
print('Accuracy of Logistic Regression:',lr_acc)
plt.figure()
plt.plot(fpr1,tpr1)
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Curve')
plt.show()
#Confusion Matrix
from sklearn.metrics import confusion_matrix
Y_pred = lr.predict(X_ts)
lr_cm = confusion_matrix(Y_ts,Y_pred)
lr_cm
#sensitivity calculation
lr_sen = lr_cm[0,0]/(lr_cm[0,0]+lr_cm[0,1])
print('Sensitivity of Logistic regression:',lr_sen)
#GridSearch cross validation for logistic regresssion
from sklearn.model_selection import GridSearchCV
log_reg = LogisticRegression(solver="liblinear")
grid = {"penalty" : ["l1","l2"], "C" : np.arange(10,50,5)}
log_reg_cv = GridSearchCV(log_reg, grid, cv = 3)
log_reg_cv.fit(X_tr,Y_tr)
print("Tuned hyperparameter: {}".format(log_reg_cv.best_params_))
#Best model
logreg_best = LogisticRegression(C = 25, penalty = "l1",solver = "liblinear")
logreg_best.fit(X_tr,Y_tr)
probs = logreg_best.predict_proba(X_ts)[:,1]
fpr2,tpr2,thresholds = roc_curve(Y_ts,probs)
lr_best_a = auc(fpr2,tpr2)
lr_best_acc = logreg_best.score(X_ts,Y_ts)
print('AUC of Logistic Regression(tuned):',lr_best_a)
print('Accuracy of Logistic Regression(tuned):',lr_best_acc)
plt.figure()
plt.plot(fpr2,tpr2,label = 'With Tuning')
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Curve')
plt.legend()
plt.show()
#Confusion Matrix for best result and sentivity calculation
from sklearn.metrics import confusion_matrix
Y_pred = logreg_best.predict(X_ts)
log_cm = confusion_matrix(Y_ts,Y_pred)
log_cm
log_sen = log_cm[0,0]/(log_cm[0,0]+log_cm[0,1])
print('Sensitivity of Logistic regression:',log_sen)
#Calibration
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
lr = LogisticRegression(solver = 'liblinear',C=30)
lr.fit(X_tr,Y_tr)
probs_tr = lr.predict_proba(X_tr)[:,1]
probs_ts = lr.predict_proba(X_ts)[:,1]
cur = calibration_curve(Y_ts,probs_ts,n_bins=10)
cur
from matplotlib import pyplot as plt
plt.plot(cur[1],cur[0])
#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_tr,Y_tr)
probs = rf.predict_proba(X_ts)[:,1]
fpr1,tpr1,thresholds = roc_curve(Y_ts,probs)
rf_a = auc(fpr1,tpr1)
rf_acc = rf.score(X_ts,Y_ts)
print('AUC of Random Forest:',rf_a)
print('Accuracy of Random Forest:',rf_acc)

plt.figure()
plt.plot(fpr1,tpr1)
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Curve - Random Forest')
plt.show()
rf.feature_importances_
res = pd.DataFrame()
res['columns'] = data.columns.tolist()
res['vals'] = rf.feature_importances_
#Order of features which are contributing most to the prediction
res = res.sort_values('vals',ascending = False)
res
#Finding the AUC on increasing the variables according to their importance. Considering number of trees as 100
r1 = []
for i in range(1,len(res.index)):
    c = list(res[:i]['columns'])
    print(c)
    rf = RandomForestClassifier(n_estimators = 100)
    rf.fit(X_tr[c],Y_tr)
    probs = rf.predict_proba(X_ts[c])[:,1]
    fpr,tpr,thresholds = roc_curve(Y_ts,probs)
    a=auc(fpr,tpr)
    print(a)
    r1.append(a)
plt.figure()
plt.plot(list(range(len(r1))),r1)
plt.xlabel('Num of Variables')
plt.ylabel('AUC')
plt.title('Increase in Area Under Curve on addition of variables')
plt.show()
#AUC for train set on changing the number of trees
tr_results = []
ts_results = []
for i in range(10,200,10):
    rf = RandomForestClassifier(n_estimators = i,n_jobs = -1)
    rf.fit(X_tr,Y_tr)
    train_pred = rf.predict(X_tr)
    fpr,tpr,thresholds = roc_curve(Y_tr,train_pred)
    roc_auc = auc(fpr,tpr)
    tr_results.append(roc_auc)
#AUC for test set on changing number of trees    
    Y_proba = rf.predict_proba(X_ts)[:,1]
    fpr,tpr,thresholds = roc_curve(Y_ts,Y_proba)
    roc_auc = auc(fpr,tpr)
    ts_results.append(roc_auc)
line1 = plt.plot(list(range(10,200,10)),tr_results,'b',label = "Train AUC")
line2 = plt.plot(list(range(10,200,10)),ts_results,'r',label = "Test AUC")
plt.xlabel('n-estimators')
plt.ylabel('AUC')
plt.title('AUC for Random Forest on Train and Test set')
plt.legend()
plt.show()
#ROC on changing the size of train data
r = []
for i in range(1,1000,10):
    s = int(i/1000*len(X_tr.index))
    print(s)
    rf = RandomForestClassifier(n_estimators = 30)
    rf.fit(X_tr[:s],Y_tr[:s])
    probs = rf.predict_proba(X_ts)[:,1]
    fpr,tpr,thresholds = roc_curve(Y_ts,probs)
    a=auc(fpr,tpr)
    print(a)
    r.append(a)
plt.plot(list(range(1,1000,10)),r)
plt.xlabel('Length of train data')
plt.ylabel('AUC')
plt.title('AUC on increasing the size of train data')
parameter_optimizationR={'criterion':('gini','entropy'),
                       'max_depth':(1,2,3,4,5,6), 'max_features':('auto','log2'),'n_estimators':(10,20,30,50,70)}
randomforest_gridcv=GridSearchCV(RandomForestClassifier(),parameter_optimizationR)
randomforest_gridcv.fit(X_tr,Y_tr)
randomforest_gridcv.best_params_
rf_best = RandomForestClassifier(n_estimators = 10,max_depth = 6,max_features = 'auto000',criterion = 'entropy')
rf_best.fit(X_tr,Y_tr)
probs = rf_best.predict_proba(X_ts)[:,1]
fpr2,tpr2,thresholds = roc_curve(Y_ts,probs)
rf_best_a = auc(fpr2,tpr2)
rf_best_acc = rf_best.score(X_ts,Y_ts)
print('AUC of Random Forest(tuned):',rf_best_a)
print('Accuracy of Random Forest(tuned):',rf_best_acc)
plt.figure()
plt.plot(fpr1,tpr1,label = 'Without tuning')
plt.plot(fpr2,tpr2,label = 'With tuning')
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Curve - Random Forest')
plt.legend()
plt.show()
Y_pred=rf_best.predict(X_ts)
from sklearn.metrics import classification_report,confusion_matrix
cr = classification_report(Y_ts,Y_pred)
rf_cm = confusion_matrix(Y_ts,Y_pred)
rf_sen = rf_cm[0,0]/(rf_cm[0,0]+rf_cm[0,1])
print('Sensitivity of Random Forest:',rf_sen)
print("Test Accuracy for best model:",rf_best.score(X_ts,Y_ts))
rf_best.feature_importances_
res = pd.DataFrame()
res['columns'] = data.columns.tolist()
res['vals'] = rf_best.feature_importances_
#Order of features which are contributing most to the prediction
res = res.sort_values('vals',ascending = False)
res
r2 = []
accuracy = []
for i in range(1,len(res.index)):
    c = list(res[:i]['columns'])
    print(c)
    rf_best.fit(X_tr[c],Y_tr)
    probs = rf_best.predict_proba(X_ts[c])[:,1]
    fpr,tpr,thresholds = roc_curve(Y_ts,probs)
    a=auc(fpr,tpr)
    acc = rf_best.score(X_ts[c],Y_ts)
    print(a,acc)
    r2.append(a)
    accuracy.append(acc)
plt.figure()
plt.plot(list(range(len(r1))),r1,label = 'Without tuning')
plt.plot(list(range(len(r2))),r2,label = 'With tuning')
plt.plot(list(range(len(r2))),accuracy,label = 'Accuracy')
plt.xlabel('Num of Variables')
plt.ylabel('AUC')
plt.title('Increase in Area Under Curve on addition of variables')
plt.legend()
plt.show()
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_tr,Y_tr)
probs = nb.predict_proba(X_ts)[:,1]
fpr,tpr,thresholds = roc_curve(Y_ts,probs)
nb_a = auc(fpr,tpr)
nb_acc = nb.score(X_ts,Y_ts)
print('AUC of Naive Bayes:',nb_a)
print('Accuracy of Naive Bayes:',nb_acc)
plt.figure()
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Curve - Naive Bayes')
plt.show()
Y_pred = nb.predict(X_ts)
cr = classification_report(Y_ts,Y_pred)
nb_cm = confusion_matrix(Y_ts,Y_pred)
nb_sen = nb_cm[0,0]/(nb_cm[0,0]+nb_cm[0,1])
print('Sensitivity of Naive Bayes:',nb_sen)
r = []
accuracy = []
for i in range(1,len(res.index)):
    c = list(res[:i]['columns'])
    print(c)
    nb.fit(X_tr[c],Y_tr)
    probs = nb.predict_proba(X_ts[c])[:,1]
    fpr,tpr,thresholds = roc_curve(Y_ts,probs)
    a=auc(fpr,tpr)
    acc = nb.score(X_ts[c],Y_ts)
    print(a,acc)
    r.append(a)
    accuracy.append(acc)
plt.figure()
plt.plot(list(range(len(r))),r,label = 'AUC')
plt.plot(list(range(len(r))),accuracy,label = 'Accuracy')
plt.xlabel('Num of Variables')
plt.ylabel('AUC')
plt.title('Naive Bayes - change in AUC and Accuracy on addition of variables')
plt.legend()
plt.show()
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_tr,Y_tr)
probs = dt.predict_proba(X_ts)[:,1]
fpr1,tpr1,thresholds = roc_curve(Y_ts,probs)
dt_a = auc(fpr1,tpr1)
dt_acc = dt.score(X_ts,Y_ts)
print('AUC of Decision Tree:',dt_a)
print('Accuracy of Decision Tree:',dt_acc)
plt.figure()
plt.plot(fpr1,tpr1,label = 'Without tuning')
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Curve - Decision Tree')
plt.legend()
plt.show()
parameters_optimization={'criterion':('gini','entropy'),'max_depth':(1,2,4,5,6,7),
                       'max_features':(1,2,3,4,5,6),'max_leaf_nodes':(2,3,4,5,6)}
dt_gridsearch=GridSearchCV(DecisionTreeClassifier(),parameters_optimization)
dt_gridsearch.fit(X_tr,Y_tr)
dt_gridsearch.best_params_

dt_best = DecisionTreeClassifier(criterion = 'entropy',max_depth = 5,max_features = 6,max_leaf_nodes = 6)
dt_best.fit(X_tr,Y_tr)
probs = dt_best.predict_proba(X_ts)[:,1]
fpr2,tpr2,thresholds = roc_curve(Y_ts,probs)
dt_best_a = auc(fpr2,tpr2)
dt_best_acc = dt_best.score(X_ts,Y_ts)
print('AUC of Decision Tree(Tuned):',dt_best_a)
print('Accuracy of Decision Tree(Tuned):',dt_best_acc)
plt.figure()
plt.plot(fpr1,tpr1,label = 'Without tuning')
plt.plot(fpr2,tpr2,label = 'With tuning')
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Curve - Decision Tree')
plt.legend()
plt.show()
Y_pred = dt_best.predict(X_ts)
cr = classification_report(Y_ts,Y_pred)
dt_cm = confusion_matrix(Y_ts,Y_pred)
dt_sen = dt_cm[0,0]/(dt_cm[0,0]+dt_cm[0,1])
print('Sensitivity of Decision Tree:',dt_sen)
#XGBoosting
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_tr,Y_tr)
probs = xgb.predict_proba(X_ts)[:,1]
fpr1,tpr1,thresholds = roc_curve(Y_ts,probs)
xgb_a = auc(fpr1,tpr1)
xgb_acc = xgb.score(X_ts,Y_ts)
print('AUC of XGBoost:',xgb_a)
print('Accuracy of XGBoost:',xgb_acc)
from sklearn.model_selection import StratifiedKFold
learning_rate = [0.0001,0.001,0.01,0.1,0.2,0.3]
n_estimators = [100,200,300,400,500]
param_grid = dict(learning_rate = learning_rate, n_estimators = n_estimators)
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 7)
grid_search = GridSearchCV(XGBClassifier(),param_grid,scoring = "neg_log_loss",n_jobs = -1,cv = kfold)
grid_search.fit(X_tr,Y_tr)
grid_search.best_params_
#Finding the best Learning rate and n_estimators 
from sklearn.model_selection import StratifiedKFold
learning_rate= [0.01]
n_estimators = [500]
max_depth = np.arange(3,10)
gamma = [0]
param_grid = dict(learning_rate = learning_rate, n_estimators = n_estimators, max_depth = max_depth,gamma=gamma)
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 7)
grid_search = GridSearchCV(XGBClassifier(),param_grid,scoring = "neg_log_loss",n_jobs = -1,cv = kfold)
grid_search.fit(X_tr,Y_tr)
grid_search.best_params_
#Best model
xgb_best = XGBClassifier(learning_rate = 0.01,n_estimators = 500,max_depth = 5,gamma = 0)
xgb_best.fit(X_tr,Y_tr)
probs = xgb_best.predict_proba(X_ts)[:,1]
fpr2,tpr2,thresholds = roc_curve(Y_ts,probs)
xgb_best_a = auc(fpr2,tpr2)
xgb_best_acc = xgb_best.score(X_ts,Y_ts)
print('AUC of XGBoost(tuned):',xgb_best_a)
print('Accuracy of XGBoost(tuned):',xgb_best_acc)
plt.figure()
plt.plot(fpr1,tpr1,label = 'Without tuning')
plt.plot(fpr2,tpr2,label = 'With tuning')
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Curve - XGBClassifier')
plt.legend()
plt.show()
Y_pred = xgb_best.predict(X_ts)
cr = classification_report(Y_ts,Y_pred)
xgb_cm = confusion_matrix(Y_ts,Y_pred)
xgb_sen = xgb_cm[0,0]/(xgb_cm[0,0]+xgb_cm[0,1])
print('Sensitivity of XGBoost:',xgb_sen)
models = pd.DataFrame({'Model':['Logistic Regression','Random Forest','Naive Bayes','Decision Tree','XGBoost'],
                       'AUC(Without tuning)':[lr_a, rf_a, nb_a,dt_a,xgb_a],
                       'AUC(With tuning)':[lr_best_a,rf_best_a,nb_a,dt_best_a,xgb_best_a],
                       'Accuracy(without tuning)':[lr_acc, rf_acc, nb_acc,dt_acc,xgb_acc],
                       'Accuracy(with tuning)':[lr_best_acc,rf_best_acc,nb_acc,dt_best_acc,xgb_best_acc],
                       'Sensitivity':[log_sen,rf_sen,nb_sen,dt_sen,xgb_sen]})
models.sort_values(by=['Accuracy(with tuning)'],ascending=False)
