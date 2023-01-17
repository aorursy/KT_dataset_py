# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import scikitplot as skplt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
def basic_eda(df):
    print('----------Top5 records-----------')
    print(df.head())
    print('----------Information of data-----------')
    print(df.info())
    print('----------Shape of data-----------')
    print(df.shape)
    print('----------Columns of dataset-----------')
    print(df.columns)
    print('----------Statistics of dataset-----------')
    print(df.describe())
    print('----------Missing values-----------')
    print(df.isnull().sum())
#EDA of Training dataset
basic_eda(train)
#EDA of Test dataset
basic_eda(test)
#Checking & Plotting missing values of training dataset on heatmap
missing_count_train = train.isnull().sum()
missing_prcnt_train = train.isnull().sum()/600000*100
missing_train = pd.DataFrame({'missing_count': missing_count_train, 'missing%': missing_prcnt_train}).sort_values(by='missing%', ascending=False)
print(missing_train)

sns.heatmap(train.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title('Missing values in Training dataset')
plt.show()
#Checking & Plotting missing values of test dataset on heatmap
missing_count_test = test.isnull().sum()
missing_prcnt_test = test.isnull().sum()/400000*100
missing_test = pd.DataFrame({'missing_count': missing_count_test, 'missing%': missing_prcnt_test}).sort_values(by='missing%', ascending=False)
print(missing_test)

sns.heatmap(test.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title('Missing values in Test dataset')
plt.show()
# Data imputation in training dataset

for col in train.columns:
    if train[col].isnull().sum() > 0:
        train[col].fillna(train[col].mode()[0], inplace=True)
# Data imputation in test dataset

for col in test.columns:
    if test[col].isnull().sum() > 0:
        test[col].fillna(test[col].mode()[0], inplace=True)
# Checking again if any missing value present in the datasets
sns.heatmap(train.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title('Missing values in Training dataset')
plt.show()
sns.heatmap(test.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title('Missing values in Test dataset')
plt.show()
# Filter categorical columns in Training dataset & print
cat_cols_train = train.columns[train.dtypes==object]
print(cat_cols_train)
# Print categorical value counts
for i in cat_cols_train:
    print(train[i].value_counts())
train.describe(include=['O']) # Categorical columns details of Training dataset
# Filter categorical columns in Test dataset & print
cat_cols_test = test.columns[test.dtypes==object]
print(cat_cols_test)
# Print categorical value counts
for i in cat_cols_test:
    print(test[i].value_counts())
test.describe(include=['O']) # Categorical columns details of Test dataset
# Encoding binary objects (bin_3 & bin_4)
bin_encoding = {'F':0, 'T':1, 'N':0, 'Y':1}
train['bin_3'] = train['bin_3'].map(bin_encoding)
train['bin_4'] = train['bin_4'].map(bin_encoding)

test['bin_3'] = test['bin_3'].map(bin_encoding)
test['bin_4'] = test['bin_4'].map(bin_encoding)
# Encoding nominal objects (nom_0,nom_1,nom_2,nom_3,nom_4,nom_5,nom_6,nom_7,nom_8,nom_9)
from category_encoders.target_encoder import TargetEncoder

ce_target = TargetEncoder()
train[['nom_0','nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']] = ce_target.fit_transform(train[['nom_0','nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']], train['target'])
test[['nom_0','nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']] = ce_target.transform(test[['nom_0','nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']])
# Encoding ordinal objects (ord_1,ord_2,ord_3,ord_4,ord_5)
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
train[['ord_1','ord_2', 'ord_3', 'ord_4', 'ord_5']] = enc.fit_transform(train[['ord_1','ord_2', 'ord_3', 'ord_4', 'ord_5']], train['target'])
test[['ord_1','ord_2', 'ord_3', 'ord_4', 'ord_5']] = enc.transform(test[['ord_1','ord_2', 'ord_3', 'ord_4', 'ord_5']])
train.sample(10)  # Checking sample training dataset after encoding
test.sample(10)  # Checking sample test dataset after encoding
# Feature Selection using SelectKBest

X_feat_sel = train.drop(columns=['target', 'id', 'day', 'month'], axis=1)
y_feat_sel = train['target']

#Applying SelectKBest class to extract top 10 best features
from sklearn.feature_selection import SelectKBest, chi2
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X_feat_sel,y_feat_sel)
score = fit.scores_
columns = X_feat_sel.columns
featureScores = pd.DataFrame({'Feature': columns, 'Score': score})
print(featureScores.nlargest(15,'Score'))
# Feature Selection using Yellobrick

# One dimensional
from yellowbrick.features import Rank1D
visualizer = Rank1D(algorithm='shapiro')
visualizer.fit(X_feat_sel,y_feat_sel)
visualizer.transform(X_feat_sel)
visualizer.show()

# Two domensional
from yellowbrick.features import Rank2D
visualizer2 = Rank2D(algorithm='pearson', colormap='RdBu_r')
visualizer2.fit(X_feat_sel,y_feat_sel)
visualizer2.transform(X_feat_sel)
visualizer2.show()
#Selecting dependent and independent variables based on SelectKBest feature
X = train[['ord_0','ord_1','ord_3','ord_4','ord_5', 'bin_0','bin_1','bin_2','bin_4', 'nom_8','nom_9']]
y = train['target']
#Splitting Training dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)
#Choosing best parameters of Logistic regression using Grid search
'''grid = {'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100]}

CV_lr = GridSearchCV(estimator=LogisticRegression(), param_grid=grid, cv= 5)
CV_lr.fit(X_train, y_train)
print("tuned hyperparameters :",CV_lr.best_params_)
print("tuned parameter accuracy (best score):",CV_lr.best_score_)'''
#Using Logistic Regression with best parameters as per Grid search

lr = LogisticRegression(penalty='l2', C=0.001, max_iter=100)
acc_lr_cv = cross_val_score(estimator=lr,X=X_train,y=y_train,cv=10)  #K=10
print("Average accuracy of Logistic Regression using K-fold cross validation is :",np.mean(acc_lr_cv))
    
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
acc_lr = metrics.accuracy_score(y_pred_lr, y_test)
print('Accuracy of Logistic Regression is: ', metrics.accuracy_score(y_pred_lr, y_test))
print('Classification report: ', classification_report(y_test, y_pred_lr))
print('Confusion matrix: ', confusion_matrix(y_test, y_pred_lr))
#Choosing best parameters of Decision Tree using Grid search
'''grid = {'criterion' : ['gini', 'entropy'],
       'max_depth' : np.arange(1,10),
       'min_samples_split' : np.arange(2,10),
       'max_features' : ['auto', 'sqrt', 'log2']}

CV_dtc = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=grid, cv=5)
CV_dtc.fit(X_train, y_train)
print("tuned hyperparameters :",CV_dtc.best_params_)
print("tuned parameter accuracy (best score):",CV_dtc.best_score_)'''
#Using Decision tree Classifier with Best parameter from Grid Search

dtc = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=7, max_features='log2',min_samples_leaf=1)
acc_dtc_cv = cross_val_score(estimator=dtc,X=X_train,y=y_train,cv=10)  #K=10
print("Average accuracy of Decision Classifier using K-fold cross validation is :",np.mean(acc_dtc_cv))

dtc.fit(X_train, y_train)
y_pred_dtc = dtc.predict(X_test)
acc_dtc = metrics.accuracy_score(y_pred_dtc, y_test)
print('Accuracy of test Decision Tree Classifier is: ', metrics.accuracy_score(y_pred_dtc, y_test))
print('Classification report: ', classification_report(y_test, y_pred_dtc))
print('Confusion matrix: ', confusion_matrix(y_test, y_pred_dtc))
#Using Grid search to get best parameters for Random Forest classifier
'''param_grid = { 
    'n_estimators': [10, 20, 30, 40, 50],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [2, 3, 4, 5, 6],
    'bootstrap': [True, False],
    'criterion' :['gini', 'entropy'],
    'min_samples_leaf' : [5, 10, 15, 20]
}

CV_rfc = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)
print("tuned hyperparameters :",CV_rfc.best_params_)
print("tuned parameter accuracy (best score):",CV_rfc.best_score_)'''
#Using Random Forest Classifier with Gridsearch best parameters

rfc = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, min_samples_split=2, min_samples_leaf=5, max_features='log2', bootstrap=False)
acc_rfc_cv=cross_val_score(estimator=rfc,X=X_train,y=y_train,cv=10)  #K=10
print("Average accuracy of Random Forest Classifier using K-fold cross validation is :",np.mean(acc_rfc_cv))

rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)
acc_rfc = metrics.accuracy_score(y_pred_rfc, y_test)
print('Accuracy of test Random Forest Classifier is: ', metrics.accuracy_score(y_pred_rfc, y_test))
print('Classification report: ', classification_report(y_test, y_pred_rfc))
print('Confusion matrix: ', confusion_matrix(y_test, y_pred_rfc))
#Choosing best parameters of Bagging classifier using Grid search
'''grid = {'n_estimators' : np.arange(10,100),
       'bootstrap' : ['True', 'False'],
       'bootstrap_features' : ['True', 'False']}

CV_bagclf = GridSearchCV(estimator=BaggingClassifier(), param_grid=grid, cv=5)
CV_bagclf.fit(X_train, y_train)
print("tuned hyperparameters :",CV_bagclf.best_params_)
print("tuned parameter accuracy (best score):",CV_bagclf.best_score_)'''
#Using Bagging classifier with best parameters from Grid search

bagclf = BaggingClassifier(n_estimators=27, bootstrap=True, bootstrap_features=True)
acc_bagclf_cv = cross_val_score(estimator=bagclf,X=X_train,y=y_train,cv=10)  #K=10
print("Average accuracy of Bagging classifier using K-fold cross validation is :",np.mean(acc_bagclf_cv))

bagclf.fit(X_train, y_train)
y_pred_bagclf = bagclf.predict(X_test)
acc_bagclf = accuracy_score(y_test, y_pred_bagclf)
print('Accuracy of Bagging classifier is: ', accuracy_score(y_test, y_pred_bagclf))
print('Classification report: ', classification_report(y_test, y_pred_bagclf))
print('Confusion matrix: ', confusion_matrix(y_test, y_pred_bagclf))
#Choosing best parameters of AdaBoost classifier using Grid search
'''grid = {'n_estimators' : np.arange(10,100)}
CV_abc = GridSearchCV(estimator=AdaBoostClassifier(),param_grid=grid, cv=5)
CV_abc.fit(X_train, y_train)
print("tuned hyperparameters :",CV_abc.best_params_)
print("tuned parameter accuracy (best score):",CV_abc.best_score_)'''
#Using AdaBoost classifier with best parameters from Grid search

abc = AdaBoostClassifier(n_estimators=18)
acc_abc_cv = cross_val_score(estimator=abc,X=X_train,y=y_train,cv=10)  #K=10
print("Average accuracy of AdaBoost classifier using K-fold cross validation is :",np.mean(acc_abc_cv))

abc.fit(X_train, y_train)
y_pred_abc = abc.predict(X_test)
acc_abc = accuracy_score(y_test, y_pred_abc)
print('Accuracy of AdaBoost classifier is: ', accuracy_score(y_test, y_pred_abc))
print('Classification report: ', classification_report(y_test, y_pred_abc))
print('Confusion matrix: ', confusion_matrix(y_test, y_pred_abc))
#Choosing best parameters of Gradient Boosting classifier using Grid search
'''grid = {'n_estimators' : np.arange(10,100),
       'loss': ['deviance', 'exponential'],
       'learning_rate' : [0.001, 0.01, 0.1]}
CV_gbc = GridSearchCV(estimator=GradientBoostingClassifier(),param_grid=grid, cv=5)
CV_gbc.fit(X_train, y_train)
print("tuned hyperparameters :",CV_gbc.best_params_)
print("tuned parameter accuracy (best score):",CV_gbc.best_score_)'''
#Using Gradient Boosting classifier with best parameters from Grid search

gbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=41, loss='exponential')
acc_gbc_cv = cross_val_score(estimator=gbc,X=X_train,y=y_train,cv=10)  #K=10
print("Average accuracy of  Gradient Boosting classifier using K-fold cross validation is :",np.mean(acc_gbc_cv))

gbc.fit(X_train, y_train)
y_pred_gbc = gbc.predict(X_test)
acc_gbc = accuracy_score(y_test, y_pred_gbc)
print('Accuracy of Gradient Boosting classifier is: ', accuracy_score(y_test, y_pred_gbc))
print('Classification report: ', classification_report(y_test, y_pred_gbc))
print('Confusion matrix: ', confusion_matrix(y_test, y_pred_gbc))
#Using XGBoost classifier

xbc = xgb.XGBClassifier(random_state=1,learning_rate=0.01)
acc_xbc_cv = cross_val_score(estimator=xbc,X=X_train,y=y_train,cv=10)  #K=10
print("Average accuracy of XGBoost classifier using K-fold cross validation is :",np.mean(acc_xbc_cv))

xbc.fit(X_train, y_train)
y_pred_xbc = xbc.predict(X_test)
acc_xbc = accuracy_score(y_test, y_pred_xbc)
print('Accuracy of XGBoost classifier is: ', accuracy_score(y_test, y_pred_xbc))
print('Classification report: ', classification_report(y_test, y_pred_xbc))
print('Confusion matrix: ', confusion_matrix(y_test, y_pred_xbc))
#Comparing Accuracy of each model
models = pd.DataFrame({'Model' : ['RandomForest', 'DecisionTree', 'LogisticRegression', 'BaggingClassifier', 'AdaBoost', 'GradientBoost', 'XgBoost'], 
                      'Score' : [acc_rfc, acc_dtc, acc_lr, acc_bagclf, acc_abc, acc_gbc, acc_xbc]})
models.sort_values(by='Score', ascending=False)
fig, ax=plt.subplots(figsize=(10,6))
sns.barplot(x='Model', y='Score', data=models)
ax.set_xlabel('Classifiers')
ax.set_ylabel('Accuracy Score')
ax.set_title('Classifiers Vs Accuracy score')
ax.set_ylim([0.7, 0.9])
plt.show()
from yellowbrick.classifier import ClassificationReport

#Classification report of Logistic Regression classifier
visualizer = ClassificationReport(lr, classes=[0, 1],cmap="YlGn", size=(600, 360))
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

#Classification report of Decision tree classifier
visualizer = ClassificationReport(dtc, classes=[0, 1],cmap="YlGn", size=(600, 360))
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

#Classification report of Random Forest classifier
visualizer = ClassificationReport(rfc, classes=[0, 1],cmap="YlGn", size=(600, 360))
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

#Classification report of Bagging classifier
visualizer = ClassificationReport(bagclf, classes=[0, 1],cmap="YlGn", size=(600, 360))
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

#Classification report of AdaBoost classifier
visualizer = ClassificationReport(abc, classes=[0, 1],cmap="YlGn", size=(600, 360))
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

#Classification report of GradientBoost classifier
visualizer = ClassificationReport(gbc, classes=[0, 1],cmap="YlGn", size=(600, 360))
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

#Classification report of XGBoost classifier
visualizer = ClassificationReport(xbc, classes=[0, 1],cmap="YlGn", size=(600, 360))
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
from yellowbrick.classifier import ROCAUC

# ROC-AUC curve  of Logistic Regression classifier
visualizer = ROCAUC(lr, classes=[0, 1])
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

# ROC-AUC curve  of Decision Tree classifier
visualizer = ROCAUC(dtc, classes=[0, 1])
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

# ROC-AUC curve  of Random Forest classifier
visualizer = ROCAUC(rfc, classes=[0, 1])
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

# ROC-AUC curve  of Bagging classifier
visualizer = ROCAUC(bagclf, classes=[0, 1])
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

# ROC-AUC curve  of AdaBoost classifier
visualizer = ROCAUC(abc, classes=[0, 1])
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

# ROC-AUC curve  of GradientBoost classifier
visualizer = ROCAUC(gbc, classes=[0, 1])
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

# ROC-AUC curve  of XGBoost classifier
visualizer = ROCAUC(xbc, classes=[0, 1])
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
# Submission of prediction file

predictions = lr.predict(test[['ord_0','ord_1','ord_3','ord_4','ord_5', 'bin_0','bin_1','bin_2','bin_4', 'nom_8','nom_9']])

id = test['id']

output = pd.DataFrame({'id': id, 'target': predictions})
output.to_csv('submission.csv', index=False)
