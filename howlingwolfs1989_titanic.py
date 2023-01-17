#Panda & Visualiztion Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import re
sns.set()
%matplotlib inline
#Algorithem Imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, precision_score, accuracy_score
from sklearn.metrics import recall_score, classification_report, f1_score, roc_curve, auc
data = pd.read_csv('../input/train.csv')
df = data.copy()
df.head()
df.info()
df.isnull().sum()
sns.countplot(x='Survived', data=df);
sns.countplot(x='Sex', data=df);
sns.countplot(x='Sex', hue='Survived', data=df);
sns.countplot(x='Pclass', data=df);
sns.catplot(x='Sex', col='Pclass', hue='Survived', data=df, kind='count');
df['Name'].head(10)
df['Title'] = df['Name'].apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
df['Title'].value_counts()
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
i = df[~df['Title'].isin(['Mr', 'Mrs', 'Miss', 'Master'])].index
df.loc[i, 'Title'] = 'Rare Title'
df['Title'].unique()
sns.countplot(x='Title', hue='Survived', data=df);
df['Fsize'] = df['SibSp'] + df['Parch']+1
sns.countplot(x='Fsize', hue='Survived', data=df);
temp = df.groupby('Fsize')['Survived'].value_counts(normalize=True).reset_index(name='Perc')
plt.figure(figsize=(15,6));
plt.subplot(121)
sns.barplot(x='Fsize', y='Perc', hue='Survived', data=temp, dodge=True);
plt.subplot(122)
sns.barplot(x='Fsize', y='Perc', hue='Survived', data=temp, dodge=False);
temp = df['Ticket'].value_counts().reset_index(name='Tsize')
df = df.merge(temp, left_on='Ticket', right_on='index').drop('index', axis=1)
df.head()
sns.countplot(x='Tsize', hue='Survived', data=df);
temp = df.groupby('Tsize')['Survived'].value_counts(normalize=True).reset_index(name='Perc')
plt.figure(figsize=(15,6));
plt.subplot(121)
sns.barplot(x='Tsize', y='Perc', hue='Survived', data=temp, dodge=True);
plt.subplot(122)
sns.barplot(x='Tsize', y='Perc', hue='Survived', data=temp, dodge=False);
sns.lmplot(x='Tsize',y='Perc',hue='Survived',data=temp);
df['Group'] = df[['Tsize', 'Fsize']].max(axis=1)
df['GrpSize'] = ''
df.loc[df['Group']==1, 'GrpSize'] = df.loc[df['Group']==1, 'GrpSize'].replace('', 'solo')
df.loc[df['Group']==2, 'GrpSize'] = df.loc[df['Group']==2, 'GrpSize'].replace('', 'couple')
df.loc[(df['Group']<=4) & (df['Group']>=3), 'GrpSize'] = df.loc[(df['Group']<=4) & (df['Group']>=3), 'GrpSize'].replace('', 'group')
df.loc[df['Group']>4, 'GrpSize'] = df.loc[df['Group']>4, 'GrpSize'].replace('', 'large group')
df
sns.countplot(x='GrpSize', order=['solo', 'couple', 'group', 'large group'], hue='Survived', data=df);
df['Fare'].isnull().sum()
plt.subplots(figsize=(15,6))
sns.distplot(df['Fare']);
df[df['Fare'] < 0]
df[df['Fare'] == 0]
df.loc[(df['Fare'] == 0) & (df['Pclass'] == 1), 'Fare'] = df[df['Pclass'] == 1]['Fare'].mean()
df.loc[(df['Fare'] == 0) & (df['Pclass'] == 2), 'Fare'] = df[df['Pclass'] == 2]['Fare'].mean()
df.loc[(df['Fare'] == 0) & (df['Pclass'] == 3), 'Fare'] = df[df['Pclass'] == 3]['Fare'].mean()
df['FareCat'] = ''
df.loc[df['Fare']<=10, 'FareCat'] = '0-10'
df.loc[(df['Fare']>10) & (df['Fare']<=25), 'FareCat'] = '10-25'
df.loc[(df['Fare']>25) & (df['Fare']<=40), 'FareCat'] = '25-40'
df.loc[(df['Fare']>40) & (df['Fare']<=70), 'FareCat'] = '40-70'
df.loc[(df['Fare']>70) & (df['Fare']<=100), 'FareCat'] = '70-100'
df.loc[df['Fare']>100, 'FareCat'] = '100+'
df[['Fare', 'FareCat']].head(10)
sns.countplot(x='FareCat', order=['0-10', '10-25', '25-40', '40-70', '70-100', '100+'], hue='Survived', data=df);
df['FarePP'] = df['Fare']/df['Group']
sns.distplot(df['FarePP']);
sns.countplot(x='Embarked', hue='Survived', data=df);
sns.barplot(x='Embarked', y='Age', hue='Survived', data=df);
temp = df.groupby('Embarked')['Survived'].value_counts(normalize=True).reset_index(name='Perc')
plt.figure(figsize=(15,6));
plt.subplot(121)
sns.barplot(x='Embarked', y='Perc', hue='Survived', data=temp, dodge=True);
plt.subplot(122)
sns.barplot(x='Embarked', y='Perc', hue='Survived', data=temp, dodge=False);
sns.kdeplot(df[df['Survived'] == 0]['Age'].dropna(), shade=True);
sns.kdeplot(df[df['Survived'] == 1]['Age'].dropna(), shade=True);
sns.kdeplot(df['Age'], label="bw: Default")
sns.kdeplot(df['Age'], bw=.1, label="bw: 0.1")
sns.kdeplot(df['Age'], bw=2, label="bw: 2")
plt.legend();
df.isnull().sum()
df['Age'].fillna(0, inplace=True)
df['Embarked'].fillna('O', inplace=True)
df['Cabin'].fillna('O', inplace=True)
numerical = df[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title']]
def to_binary(col):
    if col == 'male':
        return 1
    else:
        return 0
numerical['Sex'] = 0
numerical.loc[:, ('Sex')] = df['Sex'].apply(to_binary)
def to_title(col):
    if col == 'Mr':
        return 0
    elif col == 'Mrs':
        return 1
    elif col == 'Miss':
        return 2
    elif col == 'Master':
        return 3
    else:
        return 4
df['Title'].unique()
numerical['Title'] = 0
numerical.loc[:,'Title'] = df['Title'].apply(to_title)
def to_embark(col):
    if col == 'O':
        return 0
    elif col == 'S':
        return 1
    elif col == 'C':
        return 2
    else:
        return 3
numerical['Embarked'] = 0
numerical.loc[:, 'Embarked'] = df['Embarked'].apply(to_embark)
numerical.head()
X = numerical.drop('Survived', axis=1)
Y = numerical['Survived']
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3,random_state=0)
print(xtrain.shape, ytrain.shape)
print(xtest.shape, ytest.shape)
clf_lr = LogisticRegression()
clf_lr.fit(xtrain, ytrain)
lr_pred = clf_lr.predict(xtest)
lr_pred_prb = clf_lr.predict_proba(xtest)
lr_pred_prb[0:5,0:5]
lr_pred_prb = clf_lr.predict_proba(xtest)[:,1]
xtest.head()
xt = xtest.copy()
xt['pred'] = lr_pred
xt['pred_probability'] = lr_pred_prb
xt['actual'] = ytest
xt.head()
tn, fp, fn, tp = confusion_matrix(xt['actual'], xt['pred']).ravel()
conf_matrix=pd.DataFrame({"pred_Survived":[tp,fp],"pred_Not Survived":[fn,tn]},index=["Survived","Not Survived"])
conf_matrix
accuracy_lr = accuracy_score(ytest,lr_pred)
print("Accuracy: {}".format(accuracy_lr))
precision_lr = precision_score(ytest,lr_pred)
print("Precision: {}".format(precision_lr))
recall_lr = recall_score(ytest,lr_pred)
print("Recall: {}".format(recall_lr))
f1_lr = f1_score(ytest,lr_pred)
print("F1 Score: {}".format(f1_lr))
ytrain.value_counts()
print(classification_report(ytest,lr_pred))
tpr = recall_lr
fpr = fp / (fp + tn)
tpr, fpr
def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(8,6))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='best')
fpr,tpr,threshold=roc_curve(ytest,lr_pred_prb)
auc = roc_auc_score(ytest,lr_pred_prb)
auc
sns.set_context('poster')
plot_roc_curve(fpr,tpr,label='AUC = %0.3f'% auc)
print (auc)
def adjusted_classes(pred_prob, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in pred_prob]
def precision_recall_threshold(p, r, thresholds, t=0.5):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """
    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    pred_adj = adjusted_classes(lr_pred_prb, t)
    tn, fp, fn, tp = confusion_matrix(ytest, pred_adj).ravel()
    print(pd.DataFrame({"pred_Survived":[tp,fp],"pred_Not Survived":[fn,tn]},index=["Survived","Not Survived"]))
    
    print("\n Accuracy: ",(tp+tn)/(tn+fp+fn+tp)*100)
    
    # plot the curve
    plt.figure(figsize=(8,6))
    plt.title("Precision and Recall curve at current threshold")
    plt.step(r, p, color='b', alpha=0.2,
             where='post')
    plt.fill_between(r, p, step='post', alpha=0.2,
                     color='b')
    plt.ylim([-0.01, 1.01]);
    plt.xlim([-0.01, 1.01]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    
    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k',
            markersize=15)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds,line=0.5):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 6))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.axvline(x=line)
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
p , r , thresholds = precision_recall_curve(ytest,lr_pred_prb)
plot_precision_recall_vs_threshold(p,r,thresholds,line=0.5)
scaler = StandardScaler()  
scaler.fit(xtrain)
X_train_=scaler.transform(xtrain)
X_test_=scaler.transform(xtest)
X_train=pd.DataFrame(data=X_train_, columns=xtrain.columns)
X_test=pd.DataFrame(data=X_test_, columns=xtest.columns)
clf_knn = KNeighborsClassifier(n_neighbors=6)
clf_knn.fit(X_train,ytrain)
knn_pred=clf_knn.predict(X_test)
knn_pred_prb=clf_knn.predict_proba(X_test)[:,1]
accuracy_knn = accuracy_score(ytest,knn_pred)
print("Accuracy : {}".format(accuracy_knn))
print(classification_report(ytest,knn_pred))
fpr,tpr,threshold=roc_curve(ytest,knn_pred_prb)
xt['Pred_1'] = knn_pred
xt['Pred_probability_1'] = knn_pred_prb
xt['actual'] = ytest
xt.head(10)
sns.set_context('poster')
auc_knn=roc_auc_score(ytest,knn_pred_prb)
plot_roc_curve(fpr,tpr,label='AUC = %0.3f'% auc_knn)
clf_dt = DecisionTreeClassifier(criterion='gini',max_depth=3)
clf_dt.fit(xtrain, ytrain)
dt_pred = clf_dt.predict(xtest)
dt_pred_prb=clf_dt.predict_proba(xtest)[:,1]
accuracy_dt = accuracy_score(ytest,dt_pred)
print("Accuracy: {}".format(accuracy_dt))
print(classification_report(ytest,dt_pred))
auc_dt=roc_auc_score(ytest,dt_pred_prb)
fpr,tpr,threshold=roc_curve(ytest,dt_pred_prb)
plot_roc_curve(fpr,tpr,label='AUC = %0.3f'% auc_dt)
features_tuple=list(zip(X.columns,clf_dt.feature_importances_))
feature_imp=pd.DataFrame(features_tuple,columns=["Feature Names","Importance"])
feature_imp=feature_imp.sort_values("Importance",ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x="Feature Names",y="Importance", data=feature_imp, color='r')
plt.xlabel("Titanic Features")
plt.ylabel("Importance")
plt.title("Decision Classifier - Features Importance")
clf_rf = RandomForestClassifier(max_depth=4)
clf_rf.fit(xtrain, ytrain)
rf_pred = clf_rf.predict(xtest)
rf_pred_prb=clf_rf.predict_proba(xtest)[:,1]
accuracy_rf = accuracy_score(ytest,rf_pred)
print("Accuracy: {}".format(accuracy_rf))
print(classification_report(ytest,rf_pred))
auc_rf=roc_auc_score(ytest,rf_pred_prb)
fpr,tpr,threshold=roc_curve(ytest,rf_pred_prb)
plot_roc_curve(fpr,tpr,label='AUC = %0.3f'% auc_rf)
features_tuple=list(zip(X.columns,clf_rf.feature_importances_))
feature_imp=pd.DataFrame(features_tuple,columns=["Feature Names","Importance"])
feature_imp=feature_imp.sort_values("Importance",ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x="Feature Names",y="Importance", data=feature_imp, color='b')
plt.xlabel("Titanic Features")
plt.ylabel("Importance")
plt.title("Random Forest Classifier - Features Importance")
param_grid1 = {"n_estimators" : [9, 18, 27, 36, 45, 54, 63],
           "max_depth" : [1, 5, 10, 15, 20, 25, 30],
           "min_samples_leaf" : [1, 2, 4, 6, 8, 10]}

RF = RandomForestClassifier()
# Instantiate the GridSearchCV object: logreg_cv
RF_cv1 = GridSearchCV(RF, param_grid1, cv=5,scoring='roc_auc',n_jobs=4)

# Fit it to the data
RF_cv1.fit(X,Y)

RF_cv1.cv_results_, RF_cv1.best_params_, RF_cv1.best_score_
param_grid2 = {"n_estimators" : [25,26,27,28,29],
           "max_depth" : [3,4,5,6,7],
           "min_samples_leaf" : [0.5,1,2]}

RF = RandomForestClassifier()
# Instantiate the GridSearchCV object: logreg_cv
RF_cv2 = GridSearchCV(RF, param_grid2, cv=5,scoring='roc_auc',n_jobs=4)

# Fit it to the data
RF_cv2.fit(X,Y)

RF_cv2.cv_results_, RF_cv2.best_params_, RF_cv2.best_score_
RF_tuned = RandomForestClassifier(max_depth=4, min_samples_leaf=1, n_estimators=29)
RF_tuned.fit(xtrain, ytrain)
rf_pred_t = RF_tuned.predict(xtest)
rf_pred_prb_t=RF_tuned.predict_proba(xtest)[:,1]
accuracy_rf_t = accuracy_score(ytest,rf_pred_t)
print("Accuracy affter tuning: {}".format(accuracy_rf_t))
print(classification_report(ytest,rf_pred_t))
auc_rf_t=roc_auc_score(ytest,rf_pred_prb_t)
fpr,tpr,threshold=roc_curve(ytest,rf_pred_prb_t)
plot_roc_curve(fpr,tpr,label='AUC Tuned = %0.3f'% auc_rf_t)
RF_dict = {"Algorithm":["Random Forest","Random Forest"],
           "Action":["First Run","Tuned"],
           "Accuracy":[accuracy_rf,accuracy_rf_t],
           "AUC":[auc_rf,auc_rf_t]}
comparison=pd.DataFrame(RF_dict)
comparison
print("Difference in Accuracy is: %0.3f"%((comparison.loc[1,'Accuracy']-comparison.loc[0,'Accuracy'])*100))
print("Difference in AUC is: %0.3f"%((comparison.loc[1,'AUC']-comparison.loc[0,'AUC'])*100))
clf_xgb = xgb.XGBClassifier(seed=42,nthread=1)
clf_xgb.fit(xtrain, ytrain)
xgb_pred = clf_xgb.predict(xtest)
xgb_pred_prb=clf_xgb.predict_proba(xtest)[:,1]
accuracy_xgb = accuracy_score(ytest,xgb_pred)
print("Accuracy: {}".format(accuracy_xgb))
print(classification_report(ytest,xgb_pred))
auc_xgb=roc_auc_score(ytest,xgb_pred_prb)
fpr,tpr,threshold=roc_curve(ytest,xgb_pred_prb)
plot_roc_curve(fpr,tpr,label='AUC = %0.3f'% auc_xgb)
dtrain = xgb.DMatrix(data=xtrain, label=ytrain)
xgb_param=clf_xgb.get_params()
cv_result=xgb.cv(xgb_param,dtrain,num_boost_round=100,nfold=5,metrics={'auc'},early_stopping_rounds=10,seed=42)
cv_result
def modelfit(alg, dtrain, xtrain, xtest, ytrain, ytest,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param=alg.get_params()
        cv_result=xgb.cv(xgb_param,dtrain,num_boost_round=alg.get_params()['n_estimators'],nfold=cv_folds,metrics={'auc'},
                     early_stopping_rounds=early_stopping_rounds,seed=42)
        alg.set_params(n_estimators=cv_result.shape[0])
        print("n_estimators : ",alg.get_params()['n_estimators'])
    
    #fit algorithm on data
    alg.fit(xtrain,ytrain)
    pred=alg.predict(xtest)
    predprob=alg.predict_proba(xtest)[:,1]
    
    print ("\nModel Report")
    print ("Accuracy(Test) : %.4g" % accuracy_score(ytest, pred))
    print ("AUC Score(Test) : %f" % roc_auc_score(ytest, predprob))
xgb1=xgb.XGBClassifier(learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=42)
modelfit(xgb1,dtrain, xtrain, xtest,ytrain,ytest)
param_test1 = {
 'max_depth':range(3,11,2),
 'min_child_weight':range(0,9,2)
}

gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=9, max_depth=5,
 min_child_weight=1, gamma=0.0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=42), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch1.fit(xtrain,ytrain)

gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_
param_test2 = {
 'max_depth':[6,7,8],
 'min_child_weight':[1,2,3,4]
}

gsearch2 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=9, max_depth=5,
 min_child_weight=0.5, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=42), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch2.fit(xtrain,ytrain)

gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_
param_test3 = {
 'gamma':[i/10.0 for i in range(0,7)]
}

gsearch3 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=76, max_depth=8,
 min_child_weight=4, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=42), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch3.fit(xtrain,ytrain)

gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_
xgb2=xgb.XGBClassifier(learning_rate =0.1,
 n_estimators=1000,
 max_depth=8,
 min_child_weight=1,
 gamma=0.0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=42)
modelfit(xgb1,dtrain, xtrain, xtest,ytrain,ytest)
param_test4 = {
 'subsample':[i/10.0 for i in range(6,11)],
 'colsample_bytree':[i/10.0 for i in range(6,11)]
}

gsearch4 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=9, max_depth=8,
 min_child_weight=1, gamma=0.0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=42), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch4.fit(xtrain,ytrain)

gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_
param_test5 = {
 'subsample':[i/100.0 for i in range(10,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}

gsearch5 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=76, max_depth=8,
 min_child_weight=4, gamma=0.0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=42), 
 param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch5.fit(xtrain,ytrain)

gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_
xgb3=xgb.XGBClassifier(learning_rate =0.05,
 n_estimators=1000,
 max_depth=9,
 min_child_weight=1,
 gamma=0.0,
 subsample=0.8,
 colsample_bytree=0.75,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=42)
modelfit(xgb3,dtrain, xtrain, xtest, ytrain, ytest)
xgb4=xgb.XGBClassifier(learning_rate =0.05,
 n_estimators=12,
 max_depth=9,
 min_child_weight=1,
 gamma=0.0,
 subsample=0.8,
 colsample_bytree=0.75,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=42)
xgb4.fit(xtrain, ytrain)
xgb_pred_t = xgb4.predict(xtest)
xgb_pred_prb_t = xgb4.predict_proba(xtest)[:,1]
accuracy_xgb_t = accuracy_score(ytest,xgb_pred_t)
print("Accuracy: {}".format(accuracy_xgb_t))
print(classification_report(ytest,xgb_pred_t))
auc_xgb_t=roc_auc_score(ytest,xgb_pred_prb_t)
fpr,tpr,threshold=roc_curve(ytest,xgb_pred_prb_t)
plot_roc_curve(fpr,tpr,label='AUC = %0.3f'% auc_xgb_t)
F_dict = {
    "Algorithms":["Logistic Regression","KNN","Decision Tree Classifier","Random Forest","Hyperparameter Tuned Random Forest","XGBoost","Tuned XGBoost"],
    "Accuracy":[accuracy_lr,accuracy_knn,accuracy_dt,accuracy_rf,accuracy_rf_t,accuracy_xgb,accuracy_xgb_t],
    "AUC":[auc,auc_knn,auc_dt,auc_rf,auc_rf_t,auc_xgb,auc_xgb_t]
}
final_result=pd.DataFrame(F_dict)
final_result
