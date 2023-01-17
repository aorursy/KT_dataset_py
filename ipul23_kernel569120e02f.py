#import Library
%matplotlib inline
import math
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import copy
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC,SVR
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsClassifier
import datetime
from sklearn.metrics import confusion_matrix
from fastai.structured import add_datepart
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.model_selection import RepeatedKFold

# ignore Deprecation Warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
df_train = pd.read_csv("../input/ugm1234/Data_training.csv")
df_test = pd.read_csv("../input/ugm1234/Data_testing.csv")
df_train.head()
df_train.describe()
df_train.info()
#Transform boolean
df_train['y'].replace({'yes':1,'no':0},inplace=True)
numerik = df_train.dtypes[df_train.dtypes != 'object'].index
numerik = df_train.dtypes[df_train.dtypes != 'object'].index
kategorik = df_train.dtypes[df_train.dtypes == 'object'].index
print("Terdapat ",len(numerik),"fitur numerik")
print("Terdapat ",len(kategorik),"fitur kategorik")
sns.catplot(x="y", y="euribor3m", data=df_train)
sns.catplot(x="y", y="cons.price.idx", data=df_train)
sns.catplot(x="y", y="cons.conf.idx", data=df_train)
sns.catplot(x="y", y="duration", data=df_train)
sns.catplot(x="y", y="nr.employed", data=df_train)
#Output Matriks Korelasi
plt.figure(figsize = (10,7))
corr = df_train[numerik].corr()
sns.heatmap(corr,fmt = '.2f',annot = True)
print(numerik)
select  = df_train[['emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']]
sns.pairplot(select)
f,ax = plt.subplots(1,2,figsize=(15,7))
df_train['y'].value_counts().plot.pie(ax=ax[0],labels =["Tidak","Ya"],autopct='%1.1f%%')
sns.countplot(df_train['y'],ax=ax[1])
ax[1].set_xticklabels(['Tidak','Ya'])
#Binning Age
df_train['age_binning'] = df_train['job']
loc = df_train[df_train['age'] <=20].index
df_train['age_binning'].iloc[loc] = "muda"
loc = df_train[(df_train['age'] >20) & (df_train['age'] <=40)].index
df_train['age_binning'].iloc[loc] = "dewasa"
loc = df_train[(df_train['age'] >40)].index
df_train['age_binning'].iloc[loc] = "tua"
plt.figure(figsize = (10,7))
sns.countplot('age_binning',hue='y',data=df_train)
plt.figure(figsize = (10,7))
sns.violinplot(data=df_train,x='previous',y='campaign')
plt.figure(figsize = (10,7))
sns.violinplot(data=df_train,x='pdays',y='campaign')
sns.distplot(df_train['duration'])
plt.figure(figsize = (10,7))
sns.boxplot(data=df_train,x='y',y='duration')
print(kategorik)
fig, axs = plt.subplots(ncols=3,figsize=(20,6))
sns.countplot(df_train['job'], ax=axs[0])
sns.countplot(df_train['marital'], ax=axs[1])
sns.countplot(df_train['education'], ax=axs[2])
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
plt.show()
fig, axs = plt.subplots(ncols=3,figsize=(20,6))
sns.countplot(df_train['default'], ax=axs[0])
sns.countplot(df_train['housing'], ax=axs[1])
sns.countplot(df_train['loan'], ax=axs[2])
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
plt.show()
fig, axs = plt.subplots(ncols=3,figsize=(20,6))
sns.countplot(df_train['contact'], ax=axs[0])
sns.countplot(df_train['month'], ax=axs[1])
sns.countplot(df_train['poutcome'], ax=axs[2])
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
plt.show()
sns.countplot(df_train['day_of_week'])
plt.show()
plt.figure(figsize = (10,7))
sns.violinplot(y='duration',x='poutcome',data=df_train)
df_train_drop = df_train.drop(['emp.var.rate','nr.employed','default','age'],axis=1)
#Transform data
df_train_transform = pd.get_dummies(df_train_drop)
df_train_transform.head()
df_train_transform.shape
#Standardize data
X_train,y_train=df_train_transform.drop('y',axis=1),df_train_transform['y']
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
from imblearn.combine import SMOTEENN
sm = SMOTEENN(random_state=7)
X_train_scaled, y_train = sm.fit_sample(X_train_scaled, y_train)
model = linear_model.LogisticRegression()
kfold = KFold(n_splits=10, random_state=7)
cvLR  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
print(cvLR)
print(cvLR.mean())
def ConfusionMatrixCV(model,X,y):
    kf = KFold(n_splits=10, random_state=7)
    conf_mat = []
    for train_index, test_index in kf.split(X):
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       model.fit(X_train, y_train)
       conf_mat.append(confusion_matrix(y_test, model.predict(X_test)))
    return conf_mat
#Confusion Matrix Logistic Regression
model = linear_model.LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr',
          n_jobs=None, penalty='l2', random_state=None, solver='liblinear',
          tol=0.0001, verbose=0, warm_start=False) 
conf_mat_logit = ConfusionMatrixCV(model,X_train_scaled,y_train)
len(conf_mat_logit)
#Confusion-Matrix RegLog
print("============================================Confusion-Matrix RegLog===================================================")
f,axs = plt.subplots(ncols=5,figsize=(20,3))
sns.heatmap(conf_mat_logit[0],ax=axs[0],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_logit[1],ax=axs[1],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_logit[2],ax=axs[2],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_logit[3],ax=axs[3],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_logit[4],ax=axs[4],fmt = '.2f',annot = True)
i=1
for ax in axs:
    plt.sca(ax)
    plt.title(f'Cross-Validation-{i}')
    i+=1
plt.show()
f,axs = plt.subplots(ncols=5,figsize=(20,3))
sns.heatmap(conf_mat_logit[5],ax=axs[0],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_logit[6],ax=axs[1],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_logit[7],ax=axs[2],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_logit[8],ax=axs[3],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_logit[9],ax=axs[4],fmt = '.2f',annot = True)
for ax in axs:
    plt.sca(ax)
    plt.title(f'Cross-Validation-{i}')
    i+=1
plt.show()
len(df_train)
#kategori_cek = ['poutcome','contact'] #nonexistent, telephone
drop_kategorik = ['default','month']
drop_numerik = ['euribor3m']
#default
model = tree.DecisionTreeClassifier()
kfold = KFold(n_splits=10, random_state=7)
cvDT  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
model = RandomForestClassifier()
kfold = KFold(n_splits=10, random_state=7)
cvRF  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
model = linear_model.LogisticRegression()
kfold = KFold(n_splits=10, random_state=7)
cvLR  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
model = KNeighborsClassifier()
kfold = KFold(n_splits=10, random_state=7)
cvKNN  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
model = SGDClassifier()
kfold = KFold(n_splits=10, random_state=7)
cvSGD  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
model = LinearSVC()
kfold = KFold(n_splits=10, random_state=7)
cvSVC  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
name = ['Decision Tree','Random Forest','Logistic Regression','KNN','SGDClassifier','LinearSVC']
data = pd.DataFrame(columns=name)
data['Decision Tree'] = cvDT
data['Random Forest'] = cvRF
data['Logistic Regression'] = cvLR
data['KNN'] = cvKNN
data['SGDClassifier'] = cvSGD
data['LinearSVC'] = cvSVC
data
data.plot(style='.-',figsize=(10,7))
plt.title("Cross-Validation Plot")
plt.show()
for i in data.columns:
    print(i,": ",data[i].mean())
df_train_copy = copy.copy(df_train_drop)
for i in df_train_copy.columns:
    if (df_train_copy[i].dtypes == 'object'):
            df_train_copy[i].replace({'unknown':df_train_copy[i].mode()[0]},inplace=True)
            print(df_train_copy[i].unique())
#Transform data
df_train_transform = pd.get_dummies(df_train_copy)
df_train_transform.head()
#Standardize data
X_train,y_train=df_train_transform.drop('y',axis=1),df_train_transform['y']
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
#default
model = tree.DecisionTreeClassifier()
kfold = KFold(n_splits=10, random_state=7)
cvDT  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
model = RandomForestClassifier()
kfold = KFold(n_splits=10, random_state=7)
cvRF  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
model = linear_model.LogisticRegression()
kfold = KFold(n_splits=10, random_state=7)
cvLR  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
model = KNeighborsClassifier()
kfold = KFold(n_splits=10, random_state=7)
cvKNN  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
model = SGDClassifier()
kfold = KFold(n_splits=10, random_state=7)
cvSGD  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
model = LinearSVC()
kfold = KFold(n_splits=10, random_state=7)
cvSVC  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
name = ['Decision Tree','Random Forest','Logistic Regression','KNN','SGDClassifier','LinearSVC']
data = pd.DataFrame(columns=name)
data['Decision Tree'] = cvDT
data['Random Forest'] = cvRF
data['Logistic Regression'] = cvLR
data['KNN'] = cvKNN
data['SGDClassifier'] = cvSGD
data['LinearSVC'] = cvSVC
data
for i in data.columns:
    print(i,": ",data[i].mean())
data.plot(style='.-',figsize=(10,7))
plt.title("Cross-Validation Plot")
plt.show()
df_train_copy = copy.copy(df_train_drop)
df_train_copy.drop(['month','day_of_week'],axis=1,inplace=True)
#Transform data
df_train_transform = pd.get_dummies(df_train_copy)
df_train_transform.head()
#Standardize data
X_train,y_train=df_train_transform.drop('y',axis=1),df_train_transform['y']
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
#default
model = tree.DecisionTreeClassifier()
kfold = KFold(n_splits=10, random_state=7)
cvDT  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
model = RandomForestClassifier()
kfold = KFold(n_splits=10, random_state=7)
cvRF  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
model = linear_model.LogisticRegression()
kfold = KFold(n_splits=10, random_state=7)
cvLR  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
model = KNeighborsClassifier()
kfold = KFold(n_splits=10, random_state=7)
cvKNN  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
model = SGDClassifier()
kfold = KFold(n_splits=10, random_state=7)
cvSGD  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
model = LinearSVC()
kfold = KFold(n_splits=10, random_state=7)
cvSVC  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
name = ['Decision Tree','Random Forest','Logistic Regression','KNN','SGDClassifier','LinearSVC']
data = pd.DataFrame(columns=name)
data['Decision Tree'] = cvDT
data['Random Forest'] = cvRF
data['Logistic Regression'] = cvLR
data['KNN'] = cvKNN
data['SGDClassifier'] = cvSGD
data['LinearSVC'] = cvSVC
data
for i in data.columns:
    print(i,": ",data[i].mean())
data.plot(style='.-',figsize=(10,7))
plt.title("Cross-Validation Plot")
plt.show()
#Drop fitur date
df_train_copy = copy.copy(df_train_drop)
df_train_copy.drop(['month','day_of_week'],axis=1,inplace=True)
#Standardize data
X_train,y_train=df_train_transform.drop('y',axis=1),df_train_transform['y']
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
#Modelling
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
kfold = KFold(n_splits=10, random_state=7)
model = GridSearchCV(linear_model.LogisticRegression(penalty='l2'), param_grid,cv=kfold)
model.fit(X_train_scaled,y_train)
print("best parameter : ",model.best_params_)
model = linear_model.LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr',
          n_jobs=None, penalty='l2', random_state=None, solver='liblinear',
          tol=0.0001, verbose=0, warm_start=False)
kfold = KFold(n_splits=10, random_state=7)
cvLR  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
print(cvLR)
print(cvLR.mean())
#Drop fitur date
"""
df_train_copy = copy.copy(df_train_drop)
df_train_copy.drop(['month','day_of_week'],axis=1,inplace=True)
#Standardize data
X_train,y_train=df_train_transform.drop('y',axis=1),df_train_transform['y']
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
#Modelling
random_grid = {'bootstrap': [True, False],#Drop fitur date
 'max_depth': [10, 20, 30, 40,None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600]}
kfold = KFold(n_splits=10, random_state=7)
model = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
model.fit(X_train_scaled,y_train)
print("best parameter : ",model.best_params_)
"""
#Drop fitur date
df_train_copy = copy.copy(df_train_drop)
df_train_copy.drop(['month','day_of_week'],axis=1,inplace=True)
df_train_transform = pd.get_dummies(df_train_copy)
df_train_transform.head()
#Standardize data
X_train,y_train=df_train_transform.drop('y',axis=1),df_train_transform['y']
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
model = RandomForestClassifier(n_estimators=200,min_samples_split=2,min_samples_leaf=2,max_features='auto',max_depth=10,bootstrap=True)
kfold = KFold(n_splits=10, random_state=7)
cvLR  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
print(cvLR)
print(cvLR.mean())
featimp = pd.DataFrame()
model.fit(X_train_scaled,y_train)
featimp['name'] = X_train.columns
featimp['values'] = model.feature_importances_
featimp.sort_values(by='values',ascending=True,inplace=True)
featimp
drop = featimp[:20].name
X_copy = X_train.drop(drop,axis=1)
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_copy)
kfold = KFold(n_splits=10, random_state=7)
cvLR  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
print(cvLR)
print(cvLR.mean())
#RegLog
drop = featimp[:15].name
X_copy = X_train.drop(drop,axis=1)
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_copy)
kfold = KFold(n_splits=10, random_state=7)
model = linear_model.LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr',
          n_jobs=None, penalty='l2', random_state=None, solver='liblinear',
          tol=0.0001, verbose=0, warm_start=False) 
cvLR  = cross_val_score(model, X_train_scaled,y_train, cv=kfold)
print(cvLR)
print(cvLR.mean())
def ConfusionMatrixCV(model,X,y):
    kf = KFold(n_splits=10, random_state=7)
    conf_mat = []
    for train_index, test_index in kf.split(X):
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       model.fit(X_train, y_train)
       conf_mat.append(confusion_matrix(y_test, model.predict(X_test)))
    return conf_mat
#Confusion Matrix Logistic Regression
model = linear_model.LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr',
          n_jobs=None, penalty='l2', random_state=None, solver='liblinear',
          tol=0.0001, verbose=0, warm_start=False) 
conf_mat_logit = ConfusionMatrixCV(model,X_train_scaled,y_train)
len(conf_mat_logit)
#Confusion Matrix Random Forest
model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf='deprecated', min_samples_split=2,
            min_weight_fraction_leaf='deprecated', n_estimators=500,
            n_jobs=None, oob_score=False, random_state=None, verbose=0,
            warm_start=False) 
conf_mat_rf = ConfusionMatrixCV(model,X_train_scaled,y_train)
len(conf_mat_rf)
#Confusion-Matrix RegLog
print("============================================Confusion-Matrix RegLog===================================================")
f,axs = plt.subplots(ncols=5,figsize=(20,3))
sns.heatmap(conf_mat_logit[0],ax=axs[0],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_logit[1],ax=axs[1],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_logit[2],ax=axs[2],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_logit[3],ax=axs[3],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_logit[4],ax=axs[4],fmt = '.2f',annot = True)
i=1
for ax in axs:
    plt.sca(ax)
    plt.title(f'Cross-Validation-{i}')
    i+=1
plt.show()
f,axs = plt.subplots(ncols=5,figsize=(20,3))
sns.heatmap(conf_mat_logit[5],ax=axs[0],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_logit[6],ax=axs[1],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_logit[7],ax=axs[2],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_logit[8],ax=axs[3],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_logit[9],ax=axs[4],fmt = '.2f',annot = True)
for ax in axs:
    plt.sca(ax)
    plt.title(f'Cross-Validation-{i}')
    i+=1
plt.show()
print("============================================Confusion-Matrix RandomForest===================================================")
#Confusion-Matrix RandomForest
f,axs = plt.subplots(ncols=5,figsize=(20,3))
sns.heatmap(conf_mat_rf[0],ax=axs[0],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_rf[1],ax=axs[1],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_rf[2],ax=axs[2],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_rf[3],ax=axs[3],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_rf[4],ax=axs[4],fmt = '.2f',annot = True)
i=1
for ax in axs:
    plt.sca(ax)
    plt.title(f'Cross-Validation-{i}')
    i+=1
plt.show()
f,axs = plt.subplots(ncols=5,figsize=(20,3))
sns.heatmap(conf_mat_rf[5],ax=axs[0],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_rf[6],ax=axs[1],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_rf[7],ax=axs[2],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_rf[8],ax=axs[3],fmt = '.2f',annot = True)
sns.heatmap(conf_mat_rf[9],ax=axs[4],fmt = '.2f',annot = True)
for ax in axs:
    plt.sca(ax)
    plt.title(f'Cross-Validation-{i}')
    i+=1
plt.show()
#Logit-ROC Curve
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

def ROCAUC(model,X,y):
    # Run classifier with cross-validation and plot ROC curves
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    kf = KFold(n_splits=10, random_state=7)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        probas_ = model.fit(X_train, y_train).predict_proba(X_test)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic logistic-regression')
    plt.legend(loc="lower right")
    plt.show()
model = linear_model.LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr',
              n_jobs=None, penalty='l2', random_state=None, solver='liblinear',
              tol=0.0001, verbose=0, warm_start=False) 
ROCAUC(model,X_train_scaled,y_train)
#random forest-ROC Curve
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

def ROCAUC(model,X,y):
    # Run classifier with cross-validation and plot ROC curves
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    aucs = []
    i = 0
    kf = KFold(n_splits=10, random_state=7)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        probas_ = model.fit(X_train, y_train).predict_proba(X_test)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic random-forest')
    plt.legend(loc="lower right")
    plt.show()
model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf='deprecated', min_samples_split=2,
            min_weight_fraction_leaf='deprecated', n_estimators=500,
            n_jobs=None, oob_score=False, random_state=None, verbose=0,
            warm_start=False)  
ROCAUC(model,X_train_scaled,y_train)
#Eksplor varian
for i in df_test.columns:
    print(df_test[i].value_counts())
print("before :",df_train_drop.shape)
df_train_drop = df_train_drop[df_train_drop['contact'] == 'telephone']
df_train_drop = df_train_drop[df_train_drop['pdays'] == 999]
df_train_drop = df_train_drop[df_train_drop['previous'] == 0]
df_train_drop = df_train_drop[df_train_drop['poutcome'] == 'nonexistent']
print("after :",df_train_drop.shape)
df_train_drop.head()
for i in df_train_drop.columns:
    if (len(df_train_drop[i].unique()) == 1):
        df_train_drop.drop(i,axis=1,inplace=True)
df_train_drop.head()
df_train_drop.drop(['month','day_of_week'],axis=1,inplace=True)
df_train_drop.head()
#Binning Age
df_test['age_binning'] = df_test['job']
loc = df_test[df_test['age'] <=20].index
df_test['age_binning'].iloc[loc] = "muda"
loc = df_test[(df_test['age'] >20) & (df_test['age'] <=40)].index
df_test['age_binning'].iloc[loc] = "dewasa"
loc = df_test[(df_test['age'] >40)].index
df_test['age_binning'].iloc[loc] = "tua"

#Transform data
df_train_transform = pd.get_dummies(df_train_drop)
df_train_transform.head()
#Standardize data
X_train,y_train=df_train_transform.drop('y',axis=1),df_train_transform['y']
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
model = model = linear_model.LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr',
              n_jobs=None, penalty='l2', random_state=None, solver='liblinear',
              tol=0.0001, verbose=0, warm_start=False)
model.fit(X_train_scaled,y_train)
for i in df_test.columns:
    if (i not in df_train_drop.columns):
        df_test.drop(i,axis=1,inplace=True)
df_train_transform = pd.get_dummies(df_test)
df_train_transform.head()
#Standardize data
X_test=df_train_transform
sc = StandardScaler()
X_test_scaled = sc.fit_transform(X_test)
predict = model.predict(X_test_scaled)
predict
cc = pd.DataFrame()
cc['rest'] = predict
cc['rest'].value_counts()
submit = pd.DataFrame()
submit['y'] = predict
submit.to_csv("../working/submit.csv", index=False)
