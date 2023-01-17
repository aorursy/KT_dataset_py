# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

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

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_titanic = pd.read_csv("/kaggle/input/titanic/train.csv") # Importing training dataset

test = pd.read_csv("/kaggle/input/titanic/test.csv") # Importing test datset
df_titanic.columns   # Columns of training dataset
df_titanic.head()   #Top 5 results of training Dataset
df_titanic.info()   #Information about training Dataset
df_titanic.shape   #Checking rows and columns of training dataset
df_titanic.describe()   #Statistics of training Dataframe
#Checking missing values & percentage in training Dataset

missing_percent=df_titanic.isnull().sum()/891*100

missing_count=df_titanic.isnull().sum()

df_null_train= pd.DataFrame(data={'missing_percent_train': missing_percent, 'missing_count_train': missing_count},

                            index=test.columns) 

df_null_train.sort_values(by='missing_percent_train', ascending=False)
test.head()  # Columns of test dataset
test.shape   # No. of Rows and Columns of test dataset
test.info()  # Information about test dataset
test.isnull().sum()   # Missing values in test dataset
#Checking missing values & percentage in test Dataset

missing_percent_test=test.isnull().sum()/418*100

missing_count_test=test.isnull().sum()

df_null_test= pd.DataFrame(data={'missing_percent_test': missing_percent_test, 'missing_count_test': missing_count_test},

                      index=test.columns) 

df_null_test.sort_values(by='missing_percent_test', ascending=False)
#Replacing missing values in training dataset

df_titanic['Embarked'].fillna(df_titanic['Embarked'].dropna().mode(), inplace=True)

df_titanic['Embarked'].fillna(df_titanic['Embarked'].value_counts()[0], inplace=True)

df_titanic['Age'].fillna(df_titanic['Age'].dropna().mean(), inplace=True)





#from sklearn.impute import SimpleImputer

#imp = SimpleImputer(strategy='most_frequent')

#temp = imp.fit_transform(df_titanic)
#Replacing missing values in test dataset

test['Age'].fillna(test['Age'].dropna().mean(), inplace=True)

test['Fare'].fillna(test['Fare'].dropna().mean(), inplace=True)
df_titanic.isnull().sum()  #Checking missing value in training dataset again
test.isnull().sum()  #Checking missing value in test dataset again
df_titanic.head()
#Creating new feature from SibSp and Parch

df = df_titanic.copy()   #Copying training dataset into new dataframe

df_test = test.copy()    #Copying test dataset into new dataframe



df['TotalFamily'] = df['SibSp'] + df['Parch']

df['FamilyBucket'] = 'FamilyBucket'

df.loc[df['TotalFamily'] == 0, 'FamilyBucket'] = 'Single'

df.loc[(df['TotalFamily']>=1) & (df['TotalFamily']<=3), 'FamilyBucket'] = 'SmallFamily'

df.loc[df['TotalFamily']>3, 'FamilyBucket'] = 'LargeFamily'



df_test['TotalFamily'] = df_test['SibSp'] + df_test['Parch']

df_test['FamilyBucket'] = 'FamilyBucket'

df_test.loc[df_test['TotalFamily'] == 0, 'FamilyBucket'] = 'Single'

df_test.loc[(df_test['TotalFamily']>=1) & (df_test['TotalFamily']<=3), 'FamilyBucket'] = 'SmallFamily'

df_test.loc[df_test['TotalFamily']>3, 'FamilyBucket'] = 'LargeFamily'





#Creating new feature from Age

df['AgeGroup'] = 'agegroup'

df.loc[df['Age']<=1, 'AgeGroup'] = 'Infant'

df.loc[(df['Age']>1) & (df['Age']<=5), 'AgeGroup'] = 'Child'

df.loc[(df['Age']>5) & (df['Age']<=10), 'AgeGroup'] = 'YoungChild'

df.loc[(df['Age']>10) & (df['Age']<=50), 'AgeGroup'] = 'Adult'

df.loc[df['Age']>50, 'AgeGroup'] = 'SeniorCitizen'



df_test['AgeGroup'] = 'agegroup'

df_test.loc[df_test['Age']<=1, 'AgeGroup'] = 'Infant'

df_test.loc[(df_test['Age']>1) & (df_test['Age']<=5), 'AgeGroup'] = 'Child'

df_test.loc[(df_test['Age']>5) & (df_test['Age']<=10), 'AgeGroup'] = 'YoungChild'

df_test.loc[(df_test['Age']>10) & (df_test['Age']<=50), 'AgeGroup'] = 'Adult'

df_test.loc[df_test['Age']>50, 'AgeGroup'] = 'SeniorCitizen'

#Encoding object values in training dataset

DAgeGroup = pd.get_dummies(df['AgeGroup'], prefix='AgeGroup')

df = pd.concat([df, DAgeGroup], axis=1)

DFamilyBucket = pd.get_dummies(df['FamilyBucket'], prefix='FamilyBucket')

df = pd.concat([df, DFamilyBucket], axis=1)

DEmbarked = pd.get_dummies(df['Embarked'], prefix='Embarked')

df = pd.concat([df, DEmbarked], axis=1)

DSex = pd.get_dummies(df['Sex'], prefix = 'Sex')

df = pd.concat([df, DSex], axis=1)



#Encoding object values in test dataset

DAgeGroup_test = pd.get_dummies(df_test['AgeGroup'], prefix='AgeGroup')

df_test = pd.concat([df_test, DAgeGroup_test], axis=1)

DFamilyBucket_test = pd.get_dummies(df_test['FamilyBucket'], prefix='FamilyBucket')

df_test = pd.concat([df_test, DFamilyBucket_test], axis=1)

DEmbarked_test = pd.get_dummies(df_test['Embarked'], prefix='Embarked')

df_test = pd.concat([df_test, DEmbarked_test], axis=1)

DSex_test = pd.get_dummies(df_test['Sex'], prefix = 'Sex')

df_test = pd.concat([df_test, DSex_test], axis=1)
#Removing Name, Ticket & Cabin from training dataset & test dataset

df = df.drop(columns=['Name', 'Ticket', 'Cabin'], axis=1)

df_test = df_test.drop(columns=['Name', 'Ticket', 'Cabin'], axis=1)
df.head()
df_test.head()
#Checking Correlation matrix with Heatmap in training dataset

plt.figure(figsize=(20,10))

sns.heatmap(df.corr(), annot=True,cmap='cubehelix_r')

plt.show()
X_feat_sel = df.drop(columns=['PassengerId', 'Survived', 'FamilyBucket', 'AgeGroup', 'Embarked', 'Sex'], axis=1)

y_feat_sel = df['Survived']



#Applying SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X_feat_sel,y_feat_sel)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X_feat_sel.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Feature','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features
#Using inbuilt class feature_importances of tree based classifiers



from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()

model.fit(X_feat_sel,y_feat_sel)

print(model.feature_importances_) 

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X_feat_sel.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
temp1=df.groupby(['Survived'])['AgeGroup'].value_counts()

plt.figure(figsize=(20,6))

sns.barplot(temp1.index, temp1.values)

plt.title('Survived Vs Age group count')

plt.xlabel('Survived, Age group')

plt.ylabel('Age group count')

plt.show()
#Checking distribution of Fare

fig,ax=plt.subplots(figsize=(8,4))

sns.distplot(df['Fare'],hist=True,kde=True,color='g')

plt.show()
#Checking distribution of Gender between Survived and Fare using strip plot

fig,ax=plt.subplots(figsize=(8,4))

sns.stripplot(x='Survived', y='Fare', hue='Sex', data=df)

plt.show()
#Checking distribution of Age Group between Survived and Fare using swarm plot

fig, ax=plt.subplots(figsize=(10,6))

sns.swarmplot(x='Survived', y='Fare', hue='AgeGroup', data=df)

plt.show()
#Checking distribution of Family Group between Survived and Fare using swarm plot

fig, ax=plt.subplots(figsize=(10,6))

sns.swarmplot(x='Survived', y='Fare', hue='FamilyBucket', data=df)

plt.show()
#Pair plot of Survived, Pclass, Age, Fare, TotalFamily & Sex

sns.pairplot(df[['Sex','Survived','Pclass','Age', 'TotalFamily','Fare']],hue='Sex')

plt.show()
#Plotting Bar plot of Survived Vs Avg. Fare

temp11=df.groupby(['Survived'])['Fare'].mean()

sns.barplot(x=temp11.index,y=temp11.values, data=df_titanic, color='c')

plt.ylabel('Avg. Fare')

plt.title('Survived Vs Avg. Fare')

plt.show()
#Selecting dependent and independent variables

#X = df.drop(columns=['PassengerId', 'Survived', 'FamilyBucket', 'AgeGroup', 'Embarked', 'Sex'], axis=1)

X = df[['Fare', 'Sex_female', 'Sex_male', 'FamilyBucket_SmallFamily', 'Pclass', 'Age', 'Embarked_C', 'FamilyBucket_Single', 'AgeGroup_Infant', 'FamilyBucket_LargeFamily']]

y = df['Survived']
#Splitting Training dataset



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)
#Using Random Forest Classifier



rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=2, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True)

acc_rfc_cv=cross_val_score(estimator=rfc,X=X_train,y=y_train,cv=10)  #K=10

print("Average accuracy of Random Forest Classifier using K-fold cross validation is :",np.mean(acc_rfc_cv))



rfc.fit(X_train, y_train)

y_pred_rfc = rfc.predict(X_test)

acc_rfc = metrics.accuracy_score(y_pred_rfc, y_test)

print('Accuracy of test Random Forest Classifier is: ', metrics.accuracy_score(y_pred_rfc, y_test))

print('Classification report: ', classification_report(y_test, y_pred_rfc))

print('Confusion matrix: ', confusion_matrix(y_test, y_pred_rfc))
#Using Grid search to get best parameters for Random Forest classifier

param_grid = { 

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

print("tuned parameter accuracy (best score):",CV_rfc.best_score_)
#Using Random Forest Classifier with Gridsearch best parameters



rfc2 = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, min_samples_split=2, min_samples_leaf=5, max_features='log2', bootstrap=False)

acc_rfc_cv2=cross_val_score(estimator=rfc2,X=X_train,y=y_train,cv=10)  #K=10

print("Average accuracy of Random Forest Classifier using K-fold cross validation is :",np.mean(acc_rfc_cv2))



rfc2.fit(X_train, y_train)

y_pred_rfc2 = rfc2.predict(X_test)

acc_rfc2 = metrics.accuracy_score(y_pred_rfc2, y_test)

print('Accuracy of test Random Forest Classifier is: ', metrics.accuracy_score(y_pred_rfc2, y_test))

print('Classification report: ', classification_report(y_test, y_pred_rfc2))

print('Confusion matrix: ', confusion_matrix(y_test, y_pred_rfc2))
#Using Decision Tree classifier



dtc = DecisionTreeClassifier(criterion='entropy', max_depth=2, min_samples_split=2, min_samples_leaf=1)

acc_dtc_cv=cross_val_score(estimator=dtc,X=X_train,y=y_train,cv=10)  #K=10

print("Average accuracy of Decision Classifier using K-fold cross validation is :",np.mean(acc_dtc_cv))



dtc.fit(X_train, y_train)

y_pred_dtc = dtc.predict(X_test)

acc_dtc = metrics.accuracy_score(y_pred_dtc, y_test)

print('Accuracy of test Decision Tree Classifier is: ', metrics.accuracy_score(y_pred_dtc, y_test))

print('Classification report: ', classification_report(y_test, y_pred_dtc))

print('Confusion matrix: ', confusion_matrix(y_test, y_pred_dtc))

#Choosing best parameters of Decision Tree using Grid search

grid = {'criterion' : ['gini', 'entropy'],

       'max_depth' : np.arange(1,10),

       'min_samples_split' : np.arange(2,10),

       'max_features' : ['auto', 'sqrt', 'log2']}



CV_dtc = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=grid, cv=5)

CV_dtc.fit(X_train, y_train)

print("tuned hyperparameters :",CV_dtc.best_params_)

print("tuned parameter accuracy (best score):",CV_dtc.best_score_)
#Using Decision tree Classifier with Best parameter from Grid Search



dtc2 = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_split=6, max_features='log2',min_samples_leaf=1)

acc_dtc_cv2 = cross_val_score(estimator=dtc2,X=X_train,y=y_train,cv=10)  #K=10

print("Average accuracy of Decision Classifier using K-fold cross validation is :",np.mean(acc_dtc_cv2))



dtc2.fit(X_train, y_train)

y_pred_dtc2 = dtc2.predict(X_test)

acc_dtc2 = metrics.accuracy_score(y_pred_dtc2, y_test)

print('Accuracy of test Decision Tree Classifier is: ', metrics.accuracy_score(y_pred_dtc2, y_test))

print('Classification report: ', classification_report(y_test, y_pred_dtc2))

print('Confusion matrix: ', confusion_matrix(y_test, y_pred_dtc2))
#Using Logistic Regression



lr = LogisticRegression(penalty='l2', C=1.0, max_iter=100)

acc_lr_cv=cross_val_score(estimator=lr,X=X_train,y=y_train,cv=10)  #K=10

print("Average accuracy of Logistic Regression using K-fold cross validation is :",np.mean(acc_lr_cv))



lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

acc_lr = metrics.accuracy_score(y_pred_lr, y_test)

print('Accuracy of Logistic Regression is: ', metrics.accuracy_score(y_pred_lr, y_test))

print('Classification report: ', classification_report(y_test, y_pred_lr))

print('Confusion matrix: ', confusion_matrix(y_test, y_pred_lr))
#Choosing best parameters of Logistic egression using Grid search

grid = {'penalty': ['l1', 'l2'],

    'C': [0.001, 0.01, 0.1, 1, 10, 100]}



CV_lr = GridSearchCV(estimator=LogisticRegression(), param_grid=grid, cv= 5)

CV_lr.fit(X_train, y_train)

print("tuned hyperparameters :",CV_lr.best_params_)

print("tuned parameter accuracy (best score):",CV_lr.best_score_)
#Using Logistic Regression with best parameters as per Grid search



lr2 = LogisticRegression(penalty='l1', C=1, max_iter=100)

acc_lr_cv2=cross_val_score(estimator=lr2,X=X_train,y=y_train,cv=10)  #K=10

print("Average accuracy of Logistic Regression using K-fold cross validation is :",np.mean(acc_lr_cv2))



lr2.fit(X_train, y_train)

y_pred_lr2 = lr2.predict(X_test)

acc_lr2 = metrics.accuracy_score(y_pred_lr2, y_test)

print('Accuracy of Logistic Regression is: ', metrics.accuracy_score(y_pred_lr2, y_test))

print('Classification report: ', classification_report(y_test, y_pred_lr2))

print('Confusion matrix: ', confusion_matrix(y_test, y_pred_lr2))
#Using KNN classifier



knc = KNeighborsClassifier(n_neighbors=5)

acc_knc_cv=cross_val_score(estimator=knc,X=X_train,y=y_train,cv=10)  #K=10

print("Average accuracy of KNN classifier using K-fold cross validation is :",np.mean(acc_knc_cv))



knc.fit(X_train, y_train)

y_pred_knc = knc.predict(X_test)

acc_knc = metrics.accuracy_score(y_pred_knc, y_test)

print('Accuracy of KNN classifier is: ', metrics.accuracy_score(y_pred_knc, y_test))

print('Classification report: ', classification_report(y_test, y_pred_knc))

print('Confusion matrix: ', confusion_matrix(y_test, y_pred_knc))
#Choosing best parameters of KNN using Grid search

grid ={"n_neighbors":np.arange(1,50)}

CV_knc=GridSearchCV(KNeighborsClassifier(),grid,cv=10)#K=10 

CV_knc.fit(X_train,y_train)

print("tuned hyperparameter K:",CV_knc.best_params_)

print("tuned parameter accuracy (best score):",CV_knc.best_score_)
#Using KNN classifier again with Gridsearch best parameters



knc2 = KNeighborsClassifier(n_neighbors=7)

acc_knc_cv2 = cross_val_score(estimator=knc2,X=X_train,y=y_train,cv=10)  #K=10

print("Average accuracy of KNN classifier using K-fold cross validation is :",np.mean(acc_knc_cv2))



knc2.fit(X_train, y_train)

y_pred_knc2 = knc2.predict(X_test)

acc_knc2 = metrics.accuracy_score(y_pred_knc2, y_test)

print('Accuracy of KNN classifier is: ', metrics.accuracy_score(y_pred_knc2, y_test))

print('Classification report: ', classification_report(y_test, y_pred_knc2))

print('Confusion matrix: ', confusion_matrix(y_test, y_pred_knc2))
#Using SVM classifier



svc=SVC(C=1,kernel='linear',degree=3,gamma=1)

acc_svc_cv=cross_val_score(estimator=svc,X=X_train,y=y_train,cv=10)  #K=10

print("Average accuracy of SVM classifier using K-fold cross validation is :",np.mean(acc_svc_cv))



svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)

acc_svm = metrics.accuracy_score(y_pred_knc, y_test)

print('Accuracy of SVM classifier is: ', metrics.accuracy_score(y_pred_svc, y_test))

print('Classification report: ', classification_report(y_test, y_pred_svc))

print('Confusion matrix: ', confusion_matrix(y_test, y_pred_svc))
#Using Bagging classifier



bagclf=BaggingClassifier(n_estimators=100, bootstrap_features=True)

acc_bagclf_cv=cross_val_score(estimator=bagclf,X=X_train,y=y_train,cv=10)  #K=10

print("Average accuracy of Bagging classifier using K-fold cross validation is :",np.mean(acc_bagclf_cv))



bagclf.fit(X_train, y_train)

y_pred_bagclf = bagclf.predict(X_test)

acc_bagclf = accuracy_score(y_test, y_pred_bagclf)

print('Accuracy of Bagging classifier is: ', accuracy_score(y_test, y_pred_bagclf))

print('Classification report: ', classification_report(y_test, y_pred_bagclf))

print('Confusion matrix: ', confusion_matrix(y_test, y_pred_bagclf))

#Choosing best parameters of Bagging classifier using Grid search

grid = {'n_estimators' : np.arange(10,100),

       'bootstrap' : ['True', 'False'],

       'bootstrap_features' : ['True', 'False']}



CV_bagclf = GridSearchCV(estimator=BaggingClassifier(), param_grid=grid, cv=5)

CV_bagclf.fit(X_train, y_train)

print("tuned hyperparameters :",CV_bagclf.best_params_)

print("tuned parameter accuracy (best score):",CV_bagclf.best_score_)
#Using Bagging classifier with best parameters from Grid search



bagclf2 = BaggingClassifier(n_estimators=27, bootstrap=True, bootstrap_features=True)

acc_bagclf_cv2 = cross_val_score(estimator=bagclf2,X=X_train,y=y_train,cv=10)  #K=10

print("Average accuracy of Bagging classifier using K-fold cross validation is :",np.mean(acc_bagclf_cv2))



bagclf2.fit(X_train, y_train)

y_pred_bagclf2 = bagclf2.predict(X_test)

acc_bagclf2 = accuracy_score(y_test, y_pred_bagclf2)

print('Accuracy of Bagging classifier is: ', accuracy_score(y_test, y_pred_bagclf2))

print('Classification report: ', classification_report(y_test, y_pred_bagclf2))

print('Confusion matrix: ', confusion_matrix(y_test, y_pred_bagclf2))
#Using AdaBoost classifier



abc = AdaBoostClassifier()

acc_abc_cv = cross_val_score(estimator=abc,X=X_train,y=y_train,cv=10)  #K=10

print("Average accuracy of AdaBoost classifier using K-fold cross validation is :",np.mean(acc_abc_cv))



abc.fit(X_train, y_train)

y_pred_abc = abc.predict(X_test)

acc_abc = accuracy_score(y_test, y_pred_abc)

print('Accuracy of AdaBoost classifier is: ', accuracy_score(y_test, y_pred_abc))

print('Classification report: ', classification_report(y_test, y_pred_abc))

print('Confusion matrix: ', confusion_matrix(y_test, y_pred_abc))
#Choosing best parameters of AdaBoost classifier using Grid search

grid = {'n_estimators' : np.arange(10,100)}

CV_abc = GridSearchCV(estimator=AdaBoostClassifier(),param_grid=grid, cv=5)

CV_abc.fit(X_train, y_train)

print("tuned hyperparameters :",CV_abc.best_params_)

print("tuned parameter accuracy (best score):",CV_abc.best_score_)
#Using AdaBoost classifier with best parameters from Grid search



abc2 = AdaBoostClassifier(n_estimators=18)

acc_abc_cv2 = cross_val_score(estimator=abc2,X=X_train,y=y_train,cv=10)  #K=10

print("Average accuracy of AdaBoost classifier using K-fold cross validation is :",np.mean(acc_abc_cv2))



abc2.fit(X_train, y_train)

y_pred_abc2 = abc2.predict(X_test)

acc_abc2 = accuracy_score(y_test, y_pred_abc2)

print('Accuracy of AdaBoost classifier is: ', accuracy_score(y_test, y_pred_abc2))

print('Classification report: ', classification_report(y_test, y_pred_abc2))

print('Confusion matrix: ', confusion_matrix(y_test, y_pred_abc2))
#Using Gradient Boosting classifier



gbc = GradientBoostingClassifier()

acc_gbc_cv = cross_val_score(estimator=gbc,X=X_train,y=y_train,cv=10)  #K=10

print("Average accuracy of  Gradient Boosting classifier using K-fold cross validation is :",np.mean(acc_gbc_cv))



gbc.fit(X_train, y_train)

y_pred_gbc = gbc.predict(X_test)

acc_gbc = accuracy_score(y_test, y_pred_gbc)

print('Accuracy of Gradient Boosting classifier is: ', accuracy_score(y_test, y_pred_gbc))

print('Classification report: ', classification_report(y_test, y_pred_gbc))

print('Confusion matrix: ', confusion_matrix(y_test, y_pred_gbc))
#Choosing best parameters of Gradient Boosting classifier using Grid search

grid = {'n_estimators' : np.arange(10,100),

       'loss': ['deviance', 'exponential'],

       'learning_rate' : [0.001, 0.01, 0.1]}

CV_gbc = GridSearchCV(estimator=GradientBoostingClassifier(),param_grid=grid, cv=5)

CV_gbc.fit(X_train, y_train)

print("tuned hyperparameters :",CV_gbc.best_params_)

print("tuned parameter accuracy (best score):",CV_gbc.best_score_)
#Using Gradient Boosting classifier with best parameters from Grid search



gbc2 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=41, loss='exponential')

acc_gbc_cv2 = cross_val_score(estimator=gbc2,X=X_train,y=y_train,cv=10)  #K=10

print("Average accuracy of  Gradient Boosting classifier using K-fold cross validation is :",np.mean(acc_gbc_cv2))



gbc2.fit(X_train, y_train)

y_pred_gbc2 = gbc2.predict(X_test)

acc_gbc2 = accuracy_score(y_test, y_pred_gbc2)

print('Accuracy of Gradient Boosting classifier is: ', accuracy_score(y_test, y_pred_gbc2))

print('Classification report: ', classification_report(y_test, y_pred_gbc2))

print('Confusion matrix: ', confusion_matrix(y_test, y_pred_gbc2))
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

models = pd.DataFrame({'Model' : ['RandomForest', 'DecisionTree', 'LogisticRegression', 'KNN', 'SVM', 'BaggingClassifier', 'AdaBoost', 'GradientBoost', 'XgBoost'], 

                      'Score' : [acc_rfc, acc_dtc, acc_lr, acc_knc, acc_svm, acc_bagclf, acc_abc, acc_gbc, acc_xbc]})

models.sort_values(by='Score', ascending=False)

fig, ax=plt.subplots(figsize=(15,6))

sns.barplot(x='Model', y='Score', data=models)

ax.set_xlabel('Classifiers')

ax.set_ylabel('Accuracy Score')

ax.set_title('Classifiers Vs Accuracy score')

ax.set_ylim([0.6, 0.9])

plt.show()
#Comparing Accuracy of each model post Grid search

models2 = pd.DataFrame({'Model' : ['RandomForest', 'DecisionTree', 'LogisticRegression', 'KNN', 'SVM', 'BaggingClassifier', 'AdaBoost', 'GradientBoost'], 

                      'Score' : [acc_rfc2, acc_dtc2, acc_lr2, acc_knc2, acc_svm, acc_bagclf2, acc_abc2, acc_gbc2]})

models2.sort_values(by='Score', ascending=False)

fig, ax=plt.subplots(figsize=(15,6))

sns.barplot(x='Model', y='Score', data=models)

ax.set_xlabel('Classifiers')

ax.set_ylabel('Accuracy Score')

ax.set_title('Classifiers Vs Accuracy score post Grid search')

ax.set_ylim([0.6, 0.9])

plt.show()
#Plotting ROC curve using Random Forest



y_pred_rfc_prob = rfc.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, y_pred_rfc_prob)

plt.title('ROC curve for test samples using Random Forest')

plt.show()

#Plotting ROC curve using Decision Tree



y_pred_dtc_prob = dtc.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, y_pred_dtc_prob)

plt.title('ROC curve for test samples using Decision Tree')

plt.show()
#Plotting ROC curve using Logistic Regression



y_pred_lr_prob = lr.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, y_pred_lr_prob)

plt.title('ROC curve for test samples using Logistic regressor')

plt.show()
#Plotting ROC curve using KNN



y_pred_knc_prob = knc.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, y_pred_knc_prob)

plt.title('ROC curve for test samples using KNN')

plt.show()
#Plotting ROC curve using Bagging Classifier



y_pred_bagclf_prob = bagclf.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, y_pred_bagclf_prob)

plt.title('ROC curve for test samples using Bagging Classifier')

plt.show()
#Plotting ROC curve using AdaBoost Classifier



y_pred_abc_prob = abc.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, y_pred_abc_prob)

plt.title('ROC curve for test samples using AdaBoost Classifier')

plt.show()
#Plotting ROC curve using GradientBoosting Classifier



y_pred_gbc_prob = gbc.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, y_pred_gbc_prob)

plt.title('ROC curve for test samples using GradientBoosting Classifier')

plt.show()
#Plotting ROC curve using XgBoost Classifier



y_pred_xbc_prob = xbc.predict_proba(X_test)

skplt.metrics.plot_roc(y_test, y_pred_xbc_prob)

plt.title('ROC curve for test samples using XgBoost Classifier')

plt.show()
#Setting Id as PassengerId and predicting survival 



df_test['Embarked_644']=0   # Adding one column which is missing in test dataset but available in training model

Id = df_test['PassengerId']

#predictions = gbc.predict(df_test.drop(columns=['PassengerId','Sex','Embarked','FamilyBucket','AgeGroup'], axis=1))

predictions = gbc.predict(df_test[['Fare', 'Sex_female', 'Sex_male', 'FamilyBucket_SmallFamily', 'Pclass', 'Age', 'Embarked_C', 'FamilyBucket_Single', 'AgeGroup_Infant', 'FamilyBucket_LargeFamily']])





#Converting output dataframe to csv file named "submission.csv"

output = pd.DataFrame({ 'PassengerId' : Id, 'Survived': predictions })

output.to_csv('submission.csv', index=False)