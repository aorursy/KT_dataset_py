import numpy as np # linear algebra

import pandas as pd # to import csv and for data manipulation

import matplotlib.pyplot as plt # to plot graph

import seaborn as sns # for intractve graphs

import numpy as np # for linear algebra

import datetime # to dela with date and time

%matplotlib inline

from sklearn.preprocessing import StandardScaler # for preprocessing the data

from sklearn.ensemble import RandomForestClassifier # Random forest classifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier # for Decision Tree classifier

from sklearn.svm import SVC # for SVM classification

from sklearn.naive_bayes import  ComplementNB

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV # for tunnig hyper parameter it will use all combination of given parameters

from sklearn.model_selection import RandomizedSearchCV # same for tunning hyper parameter but will use random combinations of parameters

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,recall_score,precision_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import learning_curve, validation_curve
data = pd.read_csv('../input/creditcard.csv', sep=",")
pd.set_option('display.max_columns', None)
data.head()
data.info()
data.describe()
sns.countplot("Class",data=data)
Fraud_transacation = data[data["Class"]==1]

Normal_transacation= data[data["Class"]==0]

plt.figure(figsize=(10,6))

plt.subplot(121)

Fraud_transacation.Amount.plot.hist(title="Fraud Transacation")

plt.subplot(122)

Normal_transacation.Amount.plot.hist(title="Normal Transaction")
data.hist(figsize=(20,20))

plt.show()
sns.set(rc={'figure.figsize':(40,40)}, font_scale=1.5, style='whitegrid')

sns.heatmap(data.corr(), annot=True, fmt='.3f')
data.corr()
x_train, x_test, y_train, y_test = train_test_split(

    data.drop('Class', axis=1), data['Class'], test_size=0.3, random_state=1)
def metrics(target):

    print('recall: ',recall_score(y_test, target))

    print('precision: ', precision_score(y_test, target))

    print('roc auc: ', roc_auc_score(y_test, target))
from sklearn.preprocessing import MinMaxScaler 

scale_features_mm = MinMaxScaler() 

norm_x_train = scale_features_mm.fit_transform(x_train) 

norm_x_test = scale_features_mm.transform(x_test) 
log_reg_start = LogisticRegression()

log_reg_start.fit(norm_x_train, y_train)

target_log_reg_start = log_reg_start.predict(norm_x_test)
metrics(target_log_reg_start)
cnb_start = ComplementNB()

cnb_start.fit(norm_x_train, y_train)

target_cnb_start = cnb_start.predict(norm_x_test)
metrics(target_cnb_start)
rf_start = RandomForestClassifier()

rf_start.fit(x_train, y_train)

target_rf_start = rf_start.predict(x_test)
metrics(target_rf_start)
gb_start = GradientBoostingClassifier()

gb_start.fit(x_train, y_train)

target_gb_start = gb_start.predict(x_test)
metrics(target_gb_start)
br_start = BaggingClassifier()

br_start.fit(x_train, y_train)

target_br_start = br_start.predict(x_test)
metrics(target_br_start)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
def plot_validation_curve(estimator, title, X, y, 

                          param_name, param_range, cv, 

                          scoring="accuracy"):

                                                   

    train_scores, test_scores = validation_curve(

        estimator, X, y, param_name=param_name, param_range=param_range,

        cv=cv, scoring=scoring, n_jobs=1)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)



    plt.title(title)

    plt.xlabel(param_name)

    plt.ylabel("Score")

    #plt.ylim(-31.1, 0.0)

    lw = 2

    plt.plot(param_range, train_scores_mean, label="Training score",

                 color="darkorange", lw=lw)

    plt.fill_between(param_range, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.2,

                     color="darkorange", lw=lw)

    plt.plot(param_range, test_scores_mean, label="Cross-validation score",

                 color="navy", lw=lw)

    plt.fill_between(param_range, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.2,

                     color="navy", lw=lw)

    plt.legend(loc="best")

    return plt
def best_param(classificator,param_name, m_x_train, m_x_test):

    print("best params: ", classificator.best_params_)

    print("best score: ", classificator.best_score_)

    best_classificator = classificator.best_estimator_.fit(m_x_train, y_train)

    target = best_classificator.predict(m_x_test)

    metrics(target)

    print("Построение кривой обучения")

    plot_learning_curve(classificator.best_estimator_, "Learning curve", m_x_test, y_test, cv=3)

    return best_classificator
def valid_curve(best_classificator,param_name, param_range, x_train):

    print("Построение кривой валидации")

    plot_validation_curve(best_classificator, 'Validation curve', x_train, y_train, param_name=param_name, 

                          param_range=param_range, cv=3,  scoring="roc_auc")
parameters_lr = {'solver':['lbfgs', 'newton-cg'],

                 'max_iter': range(100, 300, 100)}

best_log_reg_model = GridSearchCV(LogisticRegression(tol=0.001), parameters_lr, cv=3, scoring='roc_auc')

best_log_reg_model.fit(norm_x_train, y_train)
best_lr = best_param(best_log_reg_model,'solver',x_train,x_test)
valid_curve(best_lr,'max_iter',[100, 200, 300], x_train)
parameters_cnb = {'alpha':[0,1],

                 'fit_prior':[True, False]}

best_cnb = GridSearchCV(ComplementNB(), parameters_cnb, cv=3, scoring='roc_auc')

best_cnb.fit(norm_x_train, y_train)
plot_learning_curve(best_cnb.best_estimator_, 'Learning Curve', norm_x_train, y_train)
plot_validation_curve(best_cnb.best_estimator_, 'Validation Curve', norm_x_train, y_train, cv=3, param_name='alpha', param_range=[1.0, 0])
parameters_random_forest = {'n_estimators':[1, 3, 5, 7, 10], 

                            'max_depth':[1, 3, 5, 7, 10],

                           'max_features':['auto', 'sqrt']}

best_random_forest = GridSearchCV(RandomForestClassifier(), parameters_random_forest, cv=3, scoring='roc_auc')

best_random_forest.fit(x_train, y_train)
best_rf = best_param(best_random_forest,'n_estimators',x_train,x_test)
valid_curve(best_rf,'max_depth',[1, 3, 5, 7, 10],x_train)
parameters_gradient_boosting = {'n_estimators':[1, 3, 5, 7, 10], 

                            'max_depth':[1, 3, 5, 7, 10]}

best_gradient_boosting = GridSearchCV(GradientBoostingClassifier(), parameters_gradient_boosting, cv=3, scoring='roc_auc')

best_gradient_boosting.fit(x_train, y_train)
best_gb = best_param(best_gradient_boosting,'n_estimators',x_train,x_test)
valid_curve(best_gb,'n_estimators', [1, 3, 5, 7, 10],x_train)
tuned_parameters = {'n_estimators': [5, 10, 15],

                    'max_samples' : [0.6, 0.8, 1.0]}

clf_bg = GridSearchCV(BaggingClassifier(), tuned_parameters, cv=5, scoring='roc_auc')

clf_bg.fit(x_train, y_train)
best_bg = best_param(best_gradient_boosting,'n_estimators',x_train,x_test)
valid_curve(best_bg,'n_estimators', [5, 10, 15],x_train)
sns.barplot(data=norm_x_test)