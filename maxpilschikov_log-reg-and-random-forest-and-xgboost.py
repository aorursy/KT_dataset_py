import pandas as pd
import numpy as np
data=pd.read_csv('../input/bank-customer-churn-modeling/Churn_Modelling.csv', sep=',')
data.head(10)
data[data.Surname=='Hill']
len(pd.unique(data.Surname))
data=data.drop(['RowNumber','CustomerId', 'Surname'], axis='columns')
data.describe()
data.info()
def summary(data):
    print('Shape: ' , data.shape)
    return( pd.DataFrame({ "Dtypes ":data.dtypes , 
                           "NAs":data.isnull().sum() ,
                           "uniques":data.nunique() ,
                            "Levels":[ data[i].unique() for i in data.columns]}))
summary(data)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,10))
ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)

g=ax1.hist(data['CreditScore'], bins=500, color='y', alpha=0.9)
g=ax2.boxplot(data['CreditScore'])
data=data.drop(data[data['CreditScore']<385].index)
len(data)
fig = plt.figure(figsize=(15,10))
ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)

g=ax1.hist(data['CreditScore'], bins=500, color='y', alpha=0.9)
g=ax2.boxplot(data['CreditScore'])
fig = plt.figure(figsize=(15,10))
ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)

g=ax1.hist(data['Age'], bins=500, color='y', alpha=0.9)
g=ax2.boxplot(data['Age'])
data=data.drop(data[data['Age']>60].index)
len(data)
fig = plt.figure(figsize=(15,10))
ax2=fig.add_subplot(221)
ax3=fig.add_subplot(222)

g2=ax2.hist(data['Age'], bins=500, color='y', alpha=0.9)
g3=ax3.boxplot(data['Age'])
fig = plt.figure(figsize=(15,10))
ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)

g=ax1.hist(data['Balance'], bins=500, color='y', alpha=0.9)
g=ax2.boxplot(data['Balance'])
len(data[data.Balance==0])
fig = plt.figure(figsize=(15,10))
ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)

g=ax1.hist(data['EstimatedSalary'], bins=1000, color='y', alpha=0.9)
g=ax2.boxplot(data['EstimatedSalary'])
from scipy import stats
W, p = stats.shapiro(data.CreditScore.iloc[:5000])
print(W, p)
W, p = stats.shapiro(data.Age.iloc[:5000])
print(W, p)
W, p = stats.shapiro(data.EstimatedSalary.iloc[:5000])
print(W, p)
import seaborn as sns
corr = data.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
data.head(10)
from sklearn import preprocessing

norm = preprocessing.StandardScaler()
norm.fit(data[['CreditScore','Age','Balance','EstimatedSalary','Tenure','NumOfProducts']])
N=norm.transform(data[['CreditScore','Age','Balance','EstimatedSalary','Tenure','NumOfProducts']])
N
data[['CreditScore','Age','Balance','EstimatedSalary','Tenure','NumOfProducts']]=N
data.head()
data1 = pd.get_dummies(data, columns =['Gender', 'Geography'], drop_first=True)
data1.head()
X = data1.iloc[:, 2:].drop(['Exited'], axis='columns')

Y = data1.iloc[:, 8]
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, train_size=(0.7), test_size=(0.3))
len(Y_test[Y_test==1])
len(Y_test[Y_test==0])
classifier=LogisticRegression()
classifier.fit(X_train, Y_train)
predicted_y = classifier.predict(X_test)
print('predicted_y:', predicted_y)
print('coef_:', classifier.coef_)
print('accuracy_score:',classifier.score(X_test, Y_test))
len(predicted_y[np.where(predicted_y==0)])
len(predicted_y[np.where(predicted_y==1)])
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, predicted_y)
tn, fp, fn, tp=cm.ravel()
print(cm)
print(tn, fp, fn, tp)
Re=tp/(tp+fn)
Pr=tp/(tp+fp)
Sp=tn/(tn+fp)
Bac=(Re+Sp)/2
F=2*Re*Pr/(Re+Pr)
print('Accuracy (log_reg):',classifier.score(X_test, Y_test))
print('Recall (log_reg):', Re)
print('Precision (log_reg):', Pr)
print('F-measure (log_reg):', F)
b=2   # приоритет у Recall # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*Re*Pr/(Re+Pr*b**2)
print('F_2-measure (log_reg):', F_b)
print('Balanced accuracy (log_reg):', Bac)
print('Specificity_ (log_reg):', Sp)
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, cross_validate
clf_log = LogisticRegression()

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_validate(clf_log, X, Y, cv=cv, n_jobs=-1, scoring=['accuracy','precision','recall'])

print("Accuracy_test (log_reg): {}".format(scores['test_accuracy'].mean()), 
      "Recall_test (log_reg): {}".format(scores['test_recall'].mean()),
      "Precision_test (log_reg): {}".format(scores['test_precision'].mean()), sep='\n')

b=2   # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*scores['test_recall'].mean()*scores['test_precision'].mean()/(scores['test_recall'].mean()+scores['test_precision'].mean()*b**2)
print('F_2-measure_test (log_reg):', F_b)
from sklearn.ensemble import RandomForestClassifier
clf_forest = RandomForestClassifier(random_state=1, n_estimators=500, min_samples_split=10, min_samples_leaf=2)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_validate(clf_forest, X, Y, cv=cv, n_jobs=-1, scoring=['accuracy','precision','recall'], return_train_score=True)

print("Accuracy_test (Forest): {}".format(scores['test_accuracy'].mean()), 
      "Recall_test (Forest): {}".format(scores['test_recall'].mean()),
      "Precision_test (Forest): {}".format(scores['test_precision'].mean()), sep='\n')

b=2   # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*scores['test_recall'].mean()*scores['test_precision'].mean()/(scores['test_recall'].mean()+scores['test_precision'].mean()*b**2)
print('F_2-measure_test (Forest):', F_b)
clf_forest.fit(X_train, Y_train)
predicted_y = clf_forest.predict(X_test)
predicted_y
cm = confusion_matrix(Y_test, predicted_y)
tn, fp, fn, tp=cm.ravel()
print(cm)
# Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
#parameters
params = {
    "n_estimators": [350, 400, 450],
    "min_samples_split": [6, 8, 10],
    "min_samples_leaf": [1, 2, 4]
}
random_search=RandomizedSearchCV(clf_forest, param_distributions=params, n_iter=5, scoring='roc_auc',n_jobs=-1, cv=cv,verbose=3)
random_search.fit(X_train,Y_train)
random_search.best_estimator_
random_search.best_params_
random_forest = RandomForestClassifier(min_samples_leaf=4, min_samples_split=10,
                       n_estimators=350, random_state=1)
from sklearn.model_selection import cross_val_score
score = cross_val_score(random_forest,X,Y,cv=10)
score.mean()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
random_forest.fit(X_train, Y_train)
Y_test_preds=random_forest.predict(X_test)
print('Accuracy (Forest): {0:.2f}'.format(accuracy_score(Y_test, Y_test_preds)))
print('Precision (Forest): {0:.2f}'.format(precision_score(Y_test, Y_test_preds)))
print('Recall (Forest): {0:.2f}'.format(recall_score(Y_test, Y_test_preds)))
print('F2 (Forest): {0:.2f}'.format(fbeta_score(Y_test, Y_test_preds, 2)))
from xgboost.sklearn import XGBClassifier
#parameters
params = {
    "learning_rate"    :[0.05,0.10,0.15,0.20,0.25,0.30],
    "max_depth"        :[ 3,4,5,6,8,10,12,15 ],
    "min_child_weight" :[ 1,3,5,7 ],
    "gamma"            :[ 0.0,0.1,0.2,0.3,0.4 ],
    "colsample_bytree" :[ 0.3, 0.4, 0.5, 0.7 ]
}
classifier = XGBClassifier()
random_search=RandomizedSearchCV(classifier, param_distributions=params, n_iter=5, scoring='roc_auc',n_jobs=-1, cv=cv,verbose=3)
random_search.fit(X_train, Y_train)
random_search.best_estimator_
random_search.best_params_
classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0.1,
              learning_rate=0.25, max_delta_step=0, max_depth=4,
              min_child_weight=5, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
score = cross_val_score(classifier,X,Y,cv=10)
score.mean()
classifier.fit(X_train, Y_train)
Y_test_preds=classifier.predict(X_test)
print('Accuracy (XGboost): {0:.2f}'.format(accuracy_score(Y_test, Y_test_preds)))
print('Precision (XGboost): {0:.2f}'.format(precision_score(Y_test, Y_test_preds)))
print('Recall (XGboost): {0:.2f}'.format(recall_score(Y_test, Y_test_preds)))
print('F2 (XGboost): {0:.2f}'.format(fbeta_score(Y_test, Y_test_preds, 2)))
data.head()
data.Exited[data.Exited==0].count()
data.Exited[data.Exited==1].count()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, train_size=(0.7), test_size=(0.3))
len(Y_test[Y_test==1])
len(Y_test[Y_test==0])
classifier=LogisticRegression(class_weight='balanced')
classifier.fit(X_train, Y_train)
predicted_y = classifier.predict(X_test)
print('predicted_y:', predicted_y)
print('coef_:', classifier.coef_)
print('accuracy_score:',classifier.score(X_test, Y_test))
len(predicted_y[np.where(predicted_y==0)])
len(predicted_y[np.where(predicted_y==1)])
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, predicted_y)
tn, fp, fn, tp=cm.ravel()
print(cm)
print(tn, fp, fn, tp)
Re=tp/(tp+fn)
Pr=tp/(tp+fp)
Sp=tn/(tn+fp)
Bac=(Re+Sp)/2
F=2*Re*Pr/(Re+Pr)
print('Accuracy (log_reg):',classifier.score(X_test, Y_test))
print('Recall (log_reg):', Re)
print('Precision (log_reg):', Pr)
print('F-measure (log_reg):', F)
b=2   # приоритет у Recall # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*Re*Pr/(Re+Pr*b**2)
print('F_2-measure (log_reg):', F_b)
print('Balanced accuracy (log_reg):', Bac)
print('Specificity_ (log_reg):', Sp)
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, cross_validate
clf_log = LogisticRegression(class_weight='balanced')

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_validate(clf_log, X, Y, cv=cv, n_jobs=-1, scoring=['accuracy','precision','recall'])

print("Accuracy_test (log_reg): {}".format(scores['test_accuracy'].mean()), 
      "Recall_test (log_reg): {}".format(scores['test_recall'].mean()),
      "Precision_test (log_reg): {}".format(scores['test_precision'].mean()), sep='\n')

b=2   # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*scores['test_recall'].mean()*scores['test_precision'].mean()/(scores['test_recall'].mean()+scores['test_precision'].mean()*b**2)
print('F_2-measure_test (log_reg):', F_b)
clf_forest = RandomForestClassifier(class_weight='balanced', random_state=1, n_estimators=500, min_samples_split=10, min_samples_leaf=2)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_validate(clf_forest, X, Y, cv=cv, n_jobs=-1, scoring=['accuracy','precision','recall'], return_train_score=True)

print("Accuracy_test (Forest): {}".format(scores['test_accuracy'].mean()), 
      "Recall_test (Forest): {}".format(scores['test_recall'].mean()),
      "Precision_test (Forest): {}".format(scores['test_precision'].mean()), sep='\n')

b=2   # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*scores['test_recall'].mean()*scores['test_precision'].mean()/(scores['test_recall'].mean()+scores['test_precision'].mean()*b**2)
print('F_2-measure_test (Forest):', F_b)
classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0.1,
              learning_rate=0.25, max_delta_step=0, max_depth=4,
              min_child_weight=5, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=5, seed=None,
              silent=None, subsample=1, verbosity=1)
score = cross_val_score(classifier,X,Y,cv=10)
score.mean()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
classifier.fit(X_train, Y_train)
Y_test_preds=classifier.predict(X_test)
print('Accuracy (XGBoost): {0:.2f}'.format(accuracy_score(Y_test, Y_test_preds)))
print('Precision (XGBoost): {0:.2f}'.format(precision_score(Y_test, Y_test_preds)))
print('Recall (XGBoost): {0:.2f}'.format(recall_score(Y_test, Y_test_preds)))
print('F2 (XGBoost): {0:.2f}'.format(fbeta_score(Y_test, Y_test_preds, 2)))
num_0 = len(data1[data1['Exited']==0])
num_1 = len(data1[data1['Exited']==1])
print(num_0,num_1)
# oversampling
oversampled_data = pd.concat([ data1[data1['Exited']==0] , data1[data1['Exited']==1].sample(num_0, replace=True) ])
print(len(oversampled_data))
# undersampling
undersampled_data = pd.concat([data1[data1['Exited']==0].sample(num_1) , data1[data1['Exited']==1] ])
print(len(undersampled_data))
X_o = oversampled_data.iloc[:, 2:].drop(['Exited'], axis='columns')
Y_o = oversampled_data.iloc[:, 8]
X_train_o, X_test_o, Y_train_o, Y_test_o = train_test_split(X_o, Y_o, stratify=Y_o, train_size=(0.7), test_size=(0.3))
classifier_o=LogisticRegression()
classifier_o.fit(X_train_o, Y_train_o)
predicted_y_o = classifier_o.predict(X_test_o)
print('predicted_y:', predicted_y_o)
print('coef_:', classifier_o.coef_)
print('accuracy_score:',classifier_o.score(X_test_o, Y_test_o))
cm_o = confusion_matrix(Y_test_o, predicted_y_o)
tn, fp, fn, tp=cm_o.ravel()
print(cm_o)
Re=tp/(tp+fn)
Pr=tp/(tp+fp)
Sp=tn/(tn+fp)
Bac=(Re+Sp)/2
F=2*Re*Pr/(Re+Pr)
print('Accuracy (log_reg):',classifier_o.score(X_test_o, Y_test_o))
print('Recall (log_reg):', Re)
print('Precision (log_reg):', Pr)
print('F-measure (log_reg):', F)
b=2   # приоритет у Recall # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*Re*Pr/(Re+Pr*b**2)
print('F_2-measure (log_reg):', F_b)
print('Balanced accuracy (log_reg):', Bac)
print('Specificity_ (log_reg):', Sp)
predicted_y_o = classifier_o.predict(X_test)
print('predicted_y:', predicted_y_o)
print('coef_:', classifier_o.coef_)
print('accuracy_score:',classifier_o.score(X_test, Y_test))
cm_o = confusion_matrix(Y_test, predicted_y_o)
tn, fp, fn, tp=cm_o.ravel()
print(cm_o)
Re=tp/(tp+fn)
Pr=tp/(tp+fp)
Sp=tn/(tn+fp)
Bac=(Re+Sp)/2
F=2*Re*Pr/(Re+Pr)
print('Accuracy (log_reg):',classifier_o.score(X_test, Y_test))
print('Recall (log_reg):', Re)
print('Precision (log_reg):', Pr)
print('F-measure (log_reg):', F)
b=2   # приоритет у Recall # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*Re*Pr/(Re+Pr*b**2)
print('F_2-measure (log_reg):', F_b)
print('Balanced accuracy (log_reg):', Bac)
print('Specificity_ (log_reg):', Sp)
X_u = undersampled_data.iloc[:, 2:].drop(['Exited'], axis='columns')
Y_u = undersampled_data.iloc[:, 8]
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train_u, X_test_u, Y_train_u, Y_test_u = train_test_split(X_u, Y_u, stratify=Y_u, train_size=(0.7), test_size=(0.3))
classifier_u=LogisticRegression()
classifier_u.fit(X_train_u, Y_train_u)
predicted_y_u = classifier_u.predict(X_test_u)
print('predicted_y:', predicted_y_u)
print('coef_:', classifier_u.coef_)
print('accuracy_score:',classifier_u.score(X_test_u, Y_test_u))
cm_u = confusion_matrix(Y_test_u, predicted_y_u)
tn, fp, fn, tp=cm_u.ravel()
print(cm_u)
Re=tp/(tp+fn)
Pr=tp/(tp+fp)
Sp=tn/(tn+fp)
Bac=(Re+Sp)/2
F=2*Re*Pr/(Re+Pr)
print('Accuracy (log_reg):',classifier_u.score(X_test_u, Y_test_u))
print('Recall (log_reg):', Re)
print('Precision (log_reg):', Pr)
print('F-measure (log_reg):', F)
b=2   # приоритет у Recall # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*Re*Pr/(Re+Pr*b**2)
print('F_2-measure (log_reg):', F_b)
print('Balanced accuracy (log_reg):', Bac)
print('Specificity_ (log_reg):', Sp)
predicted_y_u = classifier_u.predict(X_test)
print('predicted_y:', predicted_y_u)
print('coef_:', classifier_u.coef_)
print('accuracy_score:',classifier_u.score(X_test, Y_test))
cm_u = confusion_matrix(Y_test, predicted_y_u)
tn, fp, fn, tp=cm_u.ravel()
print(cm_u)
Re=tp/(tp+fn)
Pr=tp/(tp+fp)
Sp=tn/(tn+fp)
Bac=(Re+Sp)/2
F=2*Re*Pr/(Re+Pr)
print('Accuracy (log_reg):',classifier_u.score(X_test, Y_test))
print('Recall (log_reg):', Re)
print('Precision (log_reg):', Pr)
print('F-measure (log_reg):', F)
b=2   # приоритет у Recall # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*Re*Pr/(Re+Pr*b**2)
print('F_2-measure (log_reg):', F_b)
print('Balanced accuracy (log_reg):', Bac)
print('Specificity_ (log_reg):', Sp)
clf_forest_o = RandomForestClassifier(random_state=1, n_estimators=500, min_samples_split=10, min_samples_leaf=2)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_validate(clf_forest_o, X_o, Y_o, cv=cv, n_jobs=-1, scoring=['accuracy','precision','recall'], return_train_score=True)

print("Accuracy_test (Forest): {}".format(scores['test_accuracy'].mean()), 
      "Recall_test (Forest): {}".format(scores['test_recall'].mean()),
      "Precision_test (Forest): {}".format(scores['test_precision'].mean()), sep='\n')

b=2   # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*scores['test_recall'].mean()*scores['test_precision'].mean()/(scores['test_recall'].mean()+scores['test_precision'].mean()*b**2)
print('F_2-measure_test (Forest):', F_b)
clf_forest_o.fit(X_o, Y_o)
predicted_y_o = clf_forest_o.predict(X_test)
print('predicted_y:', predicted_y_o)
print('accuracy_score:', clf_forest_o.score(X_test, Y_test))
from sklearn.metrics import confusion_matrix

cm_o = confusion_matrix(Y_test, predicted_y_o)
tn, fp, fn, tp=cm_o.ravel()
print(cm_o)
Re=tp/(tp+fn)
Pr=tp/(tp+fp)
Sp=tn/(tn+fp)
Bac=(Re+Sp)/2
F=2*Re*Pr/(Re+Pr)
print('Accuracy (Forest):',clf_forest_o.score(X_test, Y_test))
print('Recall (Forest):', Re)
print('Precision (Forest):', Pr)
print('F-measure (Forest):', F)
b=2   # приоритет у Recall # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*Re*Pr/(Re+Pr*b**2)
print('F_2-measure (Forest):', F_b)
print('Balanced accuracy (Forest):', Bac)
print('Specificity_ (Forest):', Sp)
clf_forest_u = RandomForestClassifier(random_state=1, n_estimators=500, min_samples_split=10, min_samples_leaf=2)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_validate(clf_forest_u, X_u, Y_u, cv=cv, n_jobs=-1, scoring=['accuracy','precision','recall'], return_train_score=True)

print("Accuracy_test (Forest): {}".format(scores['test_accuracy'].mean()), 
      "Recall_test (Forest): {}".format(scores['test_recall'].mean()),
      "Precision_test (Forest): {}".format(scores['test_precision'].mean()), sep='\n')

b=2   # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*scores['test_recall'].mean()*scores['test_precision'].mean()/(scores['test_recall'].mean()+scores['test_precision'].mean()*b**2)
print('F_2-measure_test (Forest):', F_b)
clf_forest_u.fit(X_u, Y_u)
predicted_y_u = clf_forest_u.predict(X_test)
print('predicted_y:', predicted_y_u)
print('accuracy_score:', clf_forest_u.score(X_test, Y_test))
cm_u = confusion_matrix(Y_test, predicted_y_u)
tn, fp, fn, tp=cm_u.ravel()
print(cm_u)
Re=tp/(tp+fn)
Pr=tp/(tp+fp)
Sp=tn/(tn+fp)
Bac=(Re+Sp)/2
F=2*Re*Pr/(Re+Pr)
print('Accuracy (Forest):',clf_forest_u.score(X_test, Y_test))
print('Recall (Forest):', Re)
print('Precision (Forest):', Pr)
print('F-measure (Forest):', F)
b=2   # приоритет у Recall # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*Re*Pr/(Re+Pr*b**2)
print('F_2-measure (Forest):', F_b)
print('Balanced accuracy (Forest):', Bac)
print('Specificity_ (Forest):', Sp)
#parameters
params = {
    "learning_rate"    :[0.05,0.10,0.15,0.20,0.25,0.30],
    "max_depth"        :[ 3,4,5,6,8,10,12,15 ],
    "min_child_weight" :[ 1,3,5,7 ],
    "gamma"            :[ 0.0,0.1,0.2,0.3,0.4 ],
    "colsample_bytree" :[ 0.3, 0.4, 0.5, 0.7 ]
}
classifier = XGBClassifier()
random_search=RandomizedSearchCV(classifier, param_distributions=params, n_iter=5, scoring='roc_auc',n_jobs=-1, cv=cv,verbose=3)
random_search.fit(X_o,Y_o)
random_search.best_estimator_
random_search.best_params_
classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0.3,
              learning_rate=0.15, max_delta_step=0, max_depth=10,
              min_child_weight=5, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
from sklearn.model_selection import cross_val_score
score = cross_val_score(classifier,X_o,Y_o,cv=10)
score.mean()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
classifier.fit(X_train_o, Y_train_o)
Y_test_preds=classifier.predict(X_test_o)
print('Accuracy (XCBoost): {0:.2f}'.format(accuracy_score(Y_test_o, Y_test_preds)))
print('Precision (XCBoost): {0:.2f}'.format(precision_score(Y_test_o, Y_test_preds)))
print('Recall (XCBoost): {0:.2f}'.format(recall_score(Y_test_o, Y_test_preds)))
print('F2 (XCBoost): {0:.2f}'.format(fbeta_score(Y_test_o, Y_test_preds, 2)))
classifier.fit(X_train_o, Y_train_o)
Y_test_preds=classifier.predict(X_test)
print('Accuracy (XCBoost): {0:.2f}'.format(accuracy_score(Y_test, Y_test_preds)))
print('Precision (XCBoost): {0:.2f}'.format(precision_score(Y_test, Y_test_preds)))
print('Recall (XCBoost): {0:.2f}'.format(recall_score(Y_test, Y_test_preds)))
print('F2 (XCBoost): {0:.2f}'.format(fbeta_score(Y_test, Y_test_preds, 2)))
#parameters
params = {
    "learning_rate"    :[0.05,0.10,0.15,0.20,0.25,0.30],
    "max_depth"        :[ 3,4,5,6,8,10,12,15 ],
    "min_child_weight" :[ 1,3,5,7 ],
    "gamma"            :[ 0.0,0.1,0.2,0.3,0.4 ],
    "colsample_bytree" :[ 0.3, 0.4, 0.5, 0.7 ]
}
random_search.fit(X_u,Y_u)
random_search.best_estimator_
random_search.best_params_
classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7, gamma=0.3,
              learning_rate=0.1, max_delta_step=0, max_depth=4,
              min_child_weight=7, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
score = cross_val_score(classifier,X_u,Y_u,cv=10)
score.mean()
classifier.fit(X_train_u, Y_train_u)
Y_test_preds=classifier.predict(X_test_u)
print('Accuracy (XCBoost): {0:.2f}'.format(accuracy_score(Y_test_u, Y_test_preds)))
print('Precision (XCBoost): {0:.2f}'.format(precision_score(Y_test_u, Y_test_preds)))
print('Recall (XCBoost): {0:.2f}'.format(recall_score(Y_test_u, Y_test_preds)))
print('F2 (XCBoost): {0:.2f}'.format(fbeta_score(Y_test_u, Y_test_preds, 2)))
classifier.fit(X_train_u, Y_train_u)
Y_test_preds=classifier.predict(X_test)
print('Accuracy (XCBoost): {0:.2f}'.format(accuracy_score(Y_test, Y_test_preds)))
print('Precision (XCBoost): {0:.2f}'.format(precision_score(Y_test, Y_test_preds)))
print('Recall (XCBoost): {0:.2f}'.format(recall_score(Y_test, Y_test_preds)))
print('F2 (XCBoost): {0:.2f}'.format(fbeta_score(Y_test, Y_test_preds, 2)))
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')
X_sm, Y_sm = smote.fit_sample(X, Y)
len(Y_sm[Y_sm==0])
len(X_sm)
X_train_sm, X_test_sm, Y_train_sm, Y_test_sm = train_test_split(X_sm, Y_sm, random_state=0, stratify=Y_sm, train_size=(0.7), test_size=(0.3))
len(Y_test_sm[Y_test_sm==1])
len(Y_test_sm[Y_test_sm==0])
classifier=LogisticRegression()
classifier.fit(X_train_sm, Y_train_sm)
predicted_y = classifier.predict(X_test_sm)
print('predicted_y:', predicted_y)
print('coef_:', classifier.coef_)
print('accuracy_score:',classifier.score(X_test_sm,Y_test_sm))
cm = confusion_matrix(Y_test_sm, predicted_y)
print(cm)
tn, fp, fn, tp=cm.ravel()
Re=tp/(tp+fn)
Pr=tp/(tp+fp)
Sp=tn/(tn+fp)
F=2*Re*Pr/(Re+Pr)
print(tn, fp, fn, tp)
print('Accuracy_score (log_reg):',classifier.score(X_test_sm,Y_test_sm))
print('Recall (log_reg):', Re)
print('Precision (log_reg):', Pr)
print('F-measure (log_reg):', F)
b=2   # приоритет у Recall # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*Re*Pr/(Re+Pr*b**2)
print('F_2-measure (Forest):', F_b)
print('Balanced accuracy (Forest):', Bac)
print('Specificity_ (log_reg):', Sp)
from imblearn.under_sampling import TomekLinks
tl = TomekLinks(sampling_strategy ='majority')
X_tl, Y_tl = tl.fit_sample(X, Y)
len(Y_tl[Y_tl==0])
len(Y_tl[Y_tl==1])
X_train_tl, X_test_tl, Y_train_tl, Y_test_tl = train_test_split(X_tl, Y_tl, random_state=0, stratify=Y_tl, train_size=(0.7), test_size=(0.3))
len(Y_test_tl[Y_test_tl==1])
len(Y_test_tl[Y_test_tl==0])
classifier=LogisticRegression()
classifier.fit(X_train_tl, Y_train_tl)
predicted_y = classifier.predict(X_test_tl)
print('predicted_y:', predicted_y)
print('coef_:', classifier.coef_)
print('accuracy_score:',classifier.score(X_test_tl,Y_test_tl))
len(predicted_y[np.where(predicted_y==0)])
len(predicted_y[np.where(predicted_y==1)])
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test_tl, predicted_y)
tn, fp, fn, tp=cm.ravel()
print(cm, tn, fp, fn, tp)
tn, fp, fn, tp=cm.ravel()
Re=tp/(tp+fn)
Pr=tp/(tp+fp)
Sp=tn/(tn+fp)
F=2*Re*Pr/(Re+Pr)
print(tn, fp, fn, tp)
print('accuracy_score (log_reg):',classifier.score(X_test_tl,Y_test_tl))
print('Recall (log_reg):', Re)
print('Precision (log_reg):', Pr)
print('F-measure (log_reg):', F)
b=2   # приоритет у Recall # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*Re*Pr/(Re+Pr*b**2)
print('F_2-measure (log_reg):', F_b)
print('Balanced accuracy (log_reg):', Bac)
print('Specificity_ (log_reg):', Sp)
clf_forest_sm = RandomForestClassifier(random_state=1, n_estimators=500, min_samples_split=10, min_samples_leaf=2)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_validate(clf_forest_sm, X_sm, Y_sm, cv=cv, n_jobs=-1, scoring=['accuracy','precision','recall'], return_train_score=True)

print("Accuracy_test (Forest): {}".format(scores['test_accuracy'].mean()), 
      "Recall_test (Forest): {}".format(scores['test_recall'].mean()),
      "Precision_test (Forest): {}".format(scores['test_precision'].mean()), sep='\n')

b=2   # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*scores['test_recall'].mean()*scores['test_precision'].mean()/(scores['test_recall'].mean()+scores['test_precision'].mean()*b**2)
print('F_2-measure_test (Forest):', F_b)
clf_forest_sm.fit(X_sm, Y_sm)
predicted_y_sm = clf_forest_sm.predict(X_test)
print('predicted_y:', predicted_y_sm)
print('accuracy_score:', clf_forest_sm.score(X_test, Y_test))
cm_sm = confusion_matrix(Y_test, predicted_y_sm)
tn, fp, fn, tp=cm_sm.ravel()
print(cm_u)
Re=tp/(tp+fn)
Pr=tp/(tp+fp)
Sp=tn/(tn+fp)
Bac=(Re+Sp)/2
F=2*Re*Pr/(Re+Pr)
print('Accuracy (Forest):',clf_forest_u.score(X_test, Y_test))
print('Recall (Forest):', Re)
print('Precision (Forest):', Pr)
print('F-measure (Forest):', F)
b=2   # приоритет у Recall # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*Re*Pr/(Re+Pr*b**2)
print('F_2-measure (Forest):', F_b)
print('Balanced accuracy (Forest):', Bac)
print('Specificity_ (Forest):', Sp)
clf_forest_tl = RandomForestClassifier(random_state=1, n_estimators=500, min_samples_split=10, min_samples_leaf=2)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_validate(clf_forest_tl, X_tl, Y_tl, cv=cv, n_jobs=-1, scoring=['accuracy','precision','recall'], return_train_score=True)

print("Accuracy_test (Forest): {}".format(scores['test_accuracy'].mean()), 
      "Recall_test (Forest): {}".format(scores['test_recall'].mean()),
      "Precision_test (Forest): {}".format(scores['test_precision'].mean()), sep='\n')

b=2   # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*scores['test_recall'].mean()*scores['test_precision'].mean()/(scores['test_recall'].mean()+scores['test_precision'].mean()*b**2)
print('F_2-measure_test (Forest):', F_b)
clf_forest_tl.fit(X_tl, Y_tl)
predicted_y_tl = clf_forest_tl.predict(X_test)
print('predicted_y:', predicted_y_tl)
print('accuracy_score:', clf_forest_sm.score(X_test, Y_test))
cm_tl = confusion_matrix(Y_test, predicted_y_tl)
tn, fp, fn, tp=cm_tl.ravel()
print(cm_u)
Re=tp/(tp+fn)
Pr=tp/(tp+fp)
Sp=tn/(tn+fp)
Bac=(Re+Sp)/2
F=2*Re*Pr/(Re+Pr)
print('Accuracy (Forest):',clf_forest_tl.score(X_test, Y_test))
print('Recall (Forest):', Re)
print('Precision (Forest):', Pr)
print('F-measure (Forest):', F)
b=2   # приоритет у Recall # b>1(Recall), 0<b<1(Precision)
F_b=(1+b**2)*Re*Pr/(Re+Pr*b**2)
print('F_2-measure (Forest):', F_b)
print('Balanced accuracy (Forest):', Bac)
print('Specificity_ (Forest):', Sp)
#parameters
params = {
    "learning_rate"    :[0.05,0.10,0.15,0.20,0.25,0.30],
    "max_depth"        :[ 3,4,5,6,8,10,12,15 ],
    "min_child_weight" :[ 1,3,5,7 ],
    "gamma"            :[ 0.0,0.1,0.2,0.3,0.4 ],
    "colsample_bytree" :[ 0.3, 0.4, 0.5, 0.7 ]
}
classifier = XGBClassifier()
random_search=RandomizedSearchCV(classifier, param_distributions=params, n_iter=5, scoring='roc_auc',n_jobs=-1, cv=cv,verbose=3)
random_search.fit(X_sm,Y_sm)
random_search.best_estimator_
random_search.best_params_
classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0.3,
              learning_rate=0.15, max_delta_step=0, max_depth=10,
              min_child_weight=5, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
score = cross_val_score(classifier,X_sm,Y_sm,cv=10)
score.mean()
classifier.fit(X_train_sm, Y_train_sm)
Y_test_preds=classifier.predict(X_test_sm)
print('Accuracy (XGBoost): {0:.2f}'.format(accuracy_score(Y_test_sm, Y_test_preds)))
print('Precision (XGBoost): {0:.2f}'.format(precision_score(Y_test_sm, Y_test_preds)))
print('Recall (XGBoost): {0:.2f}'.format(recall_score(Y_test_sm, Y_test_preds)))
print('F2 (XGBoost): {0:.2f}'.format(fbeta_score(Y_test_sm, Y_test_preds, 2)))
classifier.fit(X_train_sm, Y_train_sm)
Y_test_preds=classifier.predict(X_test)
print('Accuracy (XGBoost): {0:.2f}'.format(accuracy_score(Y_test, Y_test_preds)))
print('Precision (XGBoost): {0:.2f}'.format(precision_score(Y_test, Y_test_preds)))
print('Recall (XGBoost): {0:.2f}'.format(recall_score(Y_test, Y_test_preds)))
print('F2 (XGBoost): {0:.2f}'.format(fbeta_score(Y_test, Y_test_preds, 2)))
#parameters
params = {
    "learning_rate"    :[0.05,0.10,0.15,0.20,0.25,0.30],
    "max_depth"        :[ 3,4,5,6,8,10,12,15 ],
    "min_child_weight" :[ 1,3,5,7 ],
    "gamma"            :[ 0.0,0.1,0.2,0.3,0.4 ],
    "colsample_bytree" :[ 0.3, 0.4, 0.5, 0.7 ]
}
classifier = XGBClassifier()
random_search=RandomizedSearchCV(classifier, param_distributions=params, n_iter=5, scoring='roc_auc',n_jobs=-1, cv=cv,verbose=3)
random_search.fit(X_tl,Y_tl)
random_search.best_estimator_
random_search.best_params_
classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0.3,
              learning_rate=0.15, max_delta_step=0, max_depth=10,
              min_child_weight=5, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
score = cross_val_score(classifier,X_tl,Y_tl,cv=10)
score.mean()
classifier.fit(X_train_tl, Y_train_tl)
Y_test_preds=classifier.predict(X_test_tl)
print('Accuracy (XGBoost): {0:.2f}'.format(accuracy_score(Y_test_tl, Y_test_preds)))
print('Precision (XGBoost): {0:.2f}'.format(precision_score(Y_test_tl, Y_test_preds)))
print('Recall (XGBoost): {0:.2f}'.format(recall_score(Y_test_tl, Y_test_preds)))
print('F2 (XGBoost): {0:.2f}'.format(fbeta_score(Y_test_tl, Y_test_preds, 2)))
classifier.fit(X_train_tl, Y_train_tl)
Y_test_preds=classifier.predict(X_test)
print('Accuracy: {0:.2f}'.format(accuracy_score(Y_test, Y_test_preds)))
print('Precision: {0:.2f}'.format(precision_score(Y_test, Y_test_preds)))
print('Recall: {0:.2f}'.format(recall_score(Y_test, Y_test_preds)))
print('F2: {0:.2f}'.format(fbeta_score(Y_test, Y_test_preds, 2)))
tbl = {'Par.':[2, 3.1, 3.2, 3.2, 3.3, 3.3], 
         'Kind':['-', 'Balanced weight', 'Random oversampling', 'Random undersampling','Oversampling(SMOTE)','Undersampling(Tomek Links)'], 
         'Log_Reg':[0.02, 0.51, 0.50, 0.51, 0.63, 0.13], 
         'Forest':[0.39, 0.52, 0.92, 0.76, 0.83, 0.65], 
         'XGBoost':[0.39, 0.61, 0.79, 0.67, 0.67, 0.57]}
table=pd.DataFrame(tbl)
table
