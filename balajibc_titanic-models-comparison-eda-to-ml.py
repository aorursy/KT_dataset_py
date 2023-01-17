# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re

sns.set()



pd.set_option('display.max_columns', None)

pd.set_option('display.max_colwidth', None)

pd.set_option('display.max_rows', None)
df_titanic_train=pd.read_csv('/kaggle/input/titanic/train.csv')

df_titanic_test_raw=pd.read_csv('/kaggle/input/titanic/test.csv')

df_titanic_train.head()
df_titanic_mod=df_titanic_train.copy()

df_titanic_mod.isnull().sum()
# Age - Replace missing values with Average Age

average_age=df_titanic_mod.Age.mean()

df_titanic_mod['Age'].fillna(average_age, inplace=True)
df_titanic_mod['Cabin'].value_counts().head()
# Cabin - Replace missing values with dummy code- 'NA' (Not Available)

df_titanic_mod['Cabin'].fillna('NA', inplace=True)



df_titanic_mod[ df_titanic_mod['Embarked'].isnull() ]
df_titanic_mod['Embarked'].value_counts()
df_titanic_mod['Embarked'].fillna('ZZ', inplace=True)

df_titanic_mod.isnull().sum()
df_titanic_mod.dtypes
df_titanic_mod['Sex'].value_counts()
df_titanic_mod['Sex']=df_titanic_mod['Sex'].astype('category').cat.codes

df_titanic_mod.head()
df_titanic_mod['Embarked_cd']=df_titanic_mod['Embarked'].astype('category').cat.codes

df_titanic_mod.drop('Embarked', axis=1, inplace=True)

df_titanic_mod.head()
def get_cabin_level(cabin_id):

    

    table=str.maketrans('', '', '0123456789') 

    cabins_list=cabin_id.translate(table)

    cabins_unq=''.join(set(cabins_list.split(' ')))

    

    return cabins_unq



df_titanic_mod['Cabin_level']=df_titanic_mod['Cabin'].apply(get_cabin_level)

df_titanic_mod.drop('Cabin', axis=1, inplace=True)

df_titanic_mod.head()
# Create buckets for Age attribute

bins=np.linspace(0,100,11)

age_divisions=list(range(1,11))



df_titanic_mod['Age_div']=pd.cut(df_titanic_mod['Age'], bins, labels=age_divisions, include_lowest=True)

df_titanic_mod['Age_div']=df_titanic_mod['Age_div'].astype('int')

df_titanic_mod.head()
# Extract the designations from 'Name' like Captain (Capt.), Major, Doctor (Dr.), Sir, Colonel (Col) etc..



df_titanic_mod['Title']=df_titanic_mod['Name'].str.extract('( Mr\. | Mrs\. | Miss\. | Ms\. | Mme\. | Mlle\. | Master\.| Rev\. | Dr\. | Major\. | Sir\. | Col\. | Capt\. )')

df_titanic_mod['Title']=df_titanic_mod['Title'].str.strip()
designation_info = df_titanic_mod.groupby(['Title', 'Survived']).agg(

        Row_cnt=pd.NamedAgg(column='PassengerId', aggfunc=np.size)

).reset_index(drop=False)



designation_info = designation_info.pivot(index='Title', columns='Survived', values='Row_cnt')

designation_info = designation_info.fillna(0)

designation_info.columns = ['Not survied', 'Survived']

designation_info.astype('int32')
# Convert categorical values to numeric

df_titanic_mod['Title'].replace(to_replace=['Mr.', 'Mrs.', 'Miss.', 'Ms.', 'Mme.', 'Mlle.', 'Master.', 'Rev.', 'Dr.', 'Major.', 'Sir.', 'Col.', 'Capt.'],

                                value=[1,2,3,4,5,6,7,8,9,10,11,12,13], inplace=True)

df_titanic_mod['Title'].fillna(0, inplace=True)

df_titanic_mod.head()
def seperate_name_with_braces(df_name):

    result=re.findall(r'\([\w*\d*].*\)', df_name)

    

    if len(result)==0:

        passengers=0

    else:

        passengers=1

    

    return passengers



df_titanic_mod['Co-Passenger']=df_titanic_mod['Name'].apply(seperate_name_with_braces)

df_titanic_mod.drop('Name', axis=1, inplace=True)

df_titanic_mod.head()
df_titanic_mod['Family_size'] = df_titanic_mod['SibSp'] + df_titanic_mod['Parch']

df_titanic_mod['Family_size'].value_counts()
def get_designation(df_name):

    split_results=re.split(r' ', df_name)

    

    if np.size(split_results) > 1:

        result=re.sub(r'[/.]', '', split_results[0])

    else:

        result='Ordinary'

    

    return result

    

df_titanic_mod['Ticket_level']=df_titanic_mod['Ticket'].apply(get_designation)

df_titanic_mod.drop('Ticket', axis=1, inplace=True)

df_titanic_mod.head()
ticket_info = df_titanic_mod.groupby(['Ticket_level', 'Survived']).agg(

        Row_cnt=pd.NamedAgg(column='PassengerId', aggfunc=np.size)

).reset_index(drop=False)



ticket_info = ticket_info.pivot(index='Ticket_level', columns='Survived', values='Row_cnt')

ticket_info = ticket_info.fillna(0)

ticket_info.columns = ['Not survied', 'Survived']

ticket_info = ticket_info.sort_values(by='Survived', ascending=False)

ticket_info.astype('int32').head()
df_tkt_lvl=pd.get_dummies(df_titanic_mod['Ticket_level'], prefix='Tkt_lvl')

df_tkt_lvl.head()
df_titanic_mod=pd.concat([df_titanic_mod, df_tkt_lvl], axis=1)

df_titanic_mod.drop('Ticket_level', axis=1, inplace=True)
df_cabin=pd.get_dummies(df_titanic_mod['Cabin_level'], prefix='Cabin')

df_cabin.head()
df_titanic_mod=pd.concat([df_titanic_mod, df_cabin], axis=1)

df_titanic_mod.drop('Cabin_level', axis=1, inplace=True)

df_titanic_mod.head()
titanic_attr=df_titanic_mod[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked_cd', 'Title', 'Co-Passenger', 'Family_size', 'Survived']]

plt.figure(figsize=(8,8))

sns.heatmap(titanic_attr.corr(), annot=True, fmt=".1f", cmap='plasma')

plt.show()
ax = sns.violinplot(x='Sex', y='Age', data=df_titanic_mod )

ax.set_xticklabels(['Male', 'Female'])

plt.show()
survived_fltr = df_titanic_mod['Survived'] == 1

not_survived_fltr = df_titanic_mod['Survived'] == 0

female_fltr = df_titanic_mod['Sex'] == 0

male_fltr = df_titanic_mod['Sex'] == 1 



fig, axes=plt.subplots(1,2, figsize=(16,6))



sns.distplot(df_titanic_mod[survived_fltr & male_fltr]['Age'], color='mediumblue', ax=axes[0], label='Survived', hist=False)

sns.distplot(df_titanic_mod[not_survived_fltr & male_fltr]['Age'], color='maroon', ax=axes[0], label='Not survived', hist=False)



sns.distplot(df_titanic_mod[survived_fltr & female_fltr]['Age'], color='green', ax=axes[1], label='Survived', hist=False)

sns.distplot(df_titanic_mod[not_survived_fltr & female_fltr]['Age'], color='crimson', ax=axes[1], label='Not survived', hist=False)



for ax in axes.ravel():

    ax.legend()

    ax.set_xlabel('Age')

    ax.set_xlim([0,100])



axes[0].set_title('Male')

axes[1].set_title('Female')

plt.show()
import numpy as np



Age_survival_inf = df_titanic_mod[['Age_div', 'Survived','PassengerId']].groupby(['Age_div','Survived']).agg(

                                        stats=pd.NamedAgg(column='PassengerId', aggfunc=np.size)

                                        )

Age_survival_inf.reset_index(drop=False, inplace=True)

Age_survival_pivot = Age_survival_inf.pivot(index='Age_div', columns='Survived', values='stats')



ax = Age_survival_pivot.plot(kind='bar', stacked=True, figsize=(10,6), color=['darkcyan','darkmagenta'])



lower_lmt = list(range(0,80,10))

upper_lmt = list(range(10,90,10))

age_buckets = [str(i)+'-'+str(j) for i,j in zip(lower_lmt, upper_lmt)]

ax.set_xticklabels(age_buckets)





ax.set_xlabel('')

ax.legend(['Not survived', 'Survived'])

ax.set_title('Age')

plt.show()
siblings_spouse_inf = df_titanic_mod[['SibSp', 'Survived','PassengerId']].groupby(['SibSp','Survived']).agg(

                                        stats=pd.NamedAgg(column='PassengerId', aggfunc=np.size)

                                        )

siblings_spouse_inf.reset_index(drop=False, inplace=True)

siblings_spouse_pivot = siblings_spouse_inf.pivot(index='SibSp', columns='Survived', values='stats')

siblings_spouse_pivot.rename(columns={0:'Not survived', 1:'Survived'}, inplace=True)

siblings_spouse_pivot.fillna(0, inplace=True)



ax = siblings_spouse_pivot.plot(kind='pie', figsize=(14,7), subplots=True, shadow = True, 

                           explode=(0.1,0,0,0,0,0,0), autopct="%1.1f%%", pctdistance=1.12, fontsize=10, labeldistance=None)



ax[0].set_title('Siblings spouse- Not Survived', fontsize=14)

ax[0].set_ylabel('')

ax[1].set_title('Siblings spouse- Survived', fontsize=14)

ax[1].set_ylabel('')

plt.show()
sns.catplot(x='Pclass',  y='Survived', col='Sex', data=df_titanic_train, kind="bar")

plt.show()
sns.pointplot(df_titanic_train['Pclass'], df_titanic_train['Survived'])

plt.show()
plt.figure(figsize=(7,7))

sns.boxplot(x='Survived', y='Fare', data=df_titanic_mod)

plt.show()
sns.jointplot(df_titanic_train['Age'], df_titanic_train['Fare'], color='teal', height=8, ratio=3)

plt.show()
plt.figure(figsize=(10,6))

sns.barplot(x='Parch', y='Survived', hue='Sex', data=df_titanic_train)

plt.legend(loc='upper right')

plt.show()
plt.figure(figsize=(10,6))

sns.catplot(x='Family_size',  y='Survived', col='Sex', data=df_titanic_mod, kind="bar")

plt.show()
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.tree import DecisionTreeClassifier



ind_features = df_titanic_mod.columns[ (df_titanic_mod.columns != 'Survived') & ((df_titanic_mod.columns != 'PassengerId')) ]
def get_top_k_features_by_mlxtend(data, features, target, top_k, direction=True, cv_cnt=0, show_results=True):

    

    X = data[features]

    y = data[target]

    

    model = DecisionTreeClassifier()

    

    sfs_model = SFS(model, 

                   k_features=top_k, 

                   forward=direction, 

                   floating=False, 

                   verbose=2,

                   scoring='f1',

                   cv=cv_cnt)

    

    sfs_model = sfs_model.fit(X, y)

    

    if show_results:

        print("Score : " , sfs_model.k_score_, "\n")

        print("Top" , top_k , " Feature Names : " , sfs_model.k_feature_names_, "\n")
get_top_k_features_by_mlxtend(df_titanic_mod, ind_features, 'Survived', top_k=10, direction=True, show_results=True)
from sklearn.feature_selection import RFE



fs_model=DecisionTreeClassifier()

rfe_features = RFE(fs_model, n_features_to_select=10)





X_idf = df_titanic_mod[ind_features]

y_rfe = df_titanic_mod['Survived']



X_rfe = rfe_features.fit_transform(X_idf, y_rfe)



fs_model.fit(X_rfe, y_rfe)



indx= 0 

feature_list = []



for col in X_idf.columns:

    

    if rfe_features.ranking_[indx] == 1:

        feature_list.append(col)

    indx = indx + 1



print ("RFE- Top 10 features:", feature_list)
features=['Pclass', 'Sex', 'Title', 'Age', 'Fare', 'Family_size', 'Cabin_NA', 'Tkt_lvl_Ordinary']





X=df_titanic_mod[features]

y=df_titanic_mod['Survived']



X.shape, y.shape
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0, shuffle=True)



print ("Train dataset size:{}, {} \nTest dataset size:{}, {}".format(X_train.shape, y_train.shape,

                                                                     X_test.shape, y_test.shape))
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score

from datetime import datetime



params={'max_depth':range(2,10)}



start_time = datetime.now()

DTree_grid=GridSearchCV(DecisionTreeClassifier(random_state=27), params, cv=10).fit(X_train, y_train)    

end_time = datetime.now()



print('Decision Tree traning elapsed time- {}\n'.format(end_time - start_time))

print('Grid search best params', DTree_grid.best_params_)



DT_train_score = DTree_grid.score(X_train, y_train)

DT_test_score = DTree_grid.score(X_test, y_test)

print ("\nTrain split score: {:.3f}  \nTest split score: {:.3f}".format(DT_train_score, DT_test_score))

X_4cv, _, y_4cv, _ = train_test_split(X, y, test_size=0.0001, random_state=0, shuffle=True)



DTree=DTree_grid.best_estimator_

CV_scores=cross_val_score(DTree, X_4cv, y_4cv, cv=10)

print ('Overall CV Score: ', np.round_(np.mean(CV_scores), 3))
yhat=DTree.predict(X_test)

print (classification_report(y_test, yhat, zero_division=1))
DTree_parameters=pd.DataFrame({'Feature':X.columns, 'Weights':DTree.feature_importances_}).round(2)

DTree_parameters[ DTree_parameters['Weights'] != 0.0].sort_values(by='Weights', ascending=False)
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve



def plot_model_eval_curves(clf_model, Xtest, ytest):



    fig, ax = plt.subplots(1,2, figsize=(12,6))



    model_probs = clf_model.predict_proba(Xtest)

    model_probs = model_probs[:, 1]



    uniq_, counts_ = np.unique(ytest, return_counts=True)

    majority_class = uniq_[counts_.argmax()]



    dclf_probs = [majority_class for _ in range(len(ytest))]



    prec, rec, _ = precision_recall_curve(ytest, model_probs)

    d_clf = len(y_test[ytest==1]) / len(ytest)



    fpr, tpr, _ = roc_curve(ytest, model_probs)

    d_fpr, d_tpr, _ = roc_curve(ytest, dclf_probs)



    ax[0].plot(prec, rec, marker='.', label='Actual Model')

    ax[0].plot([prec.min(), prec.max()], [d_clf, d_clf], linestyle='--', label='Dummy model')

    ax[0].set_xlabel('Precision')

    ax[0].set_ylabel('Recall')

    ax[0].set_title('Precision-Recall curve', fontsize=14)

    ax[0].legend(loc='lower left')



    ax[1].plot(fpr, tpr, marker='.', label='Actual Model')

    ax[1].plot(d_fpr, d_tpr, linestyle='--', label='Dummy model')

    ax[1].set_xlabel('TPR')

    ax[1].set_ylabel('FPR')

    ax[1].set_title('ROC curve', fontsize=14)

    ax[1].legend(loc='lower right')



    model_auc = roc_auc_score(ytest, model_probs) 

    print ("ROC AUC Score: {:.3f}".format(model_auc))

    

    return model_auc
DT_ROC = plot_model_eval_curves(DTree, X_test, y_test)
from sklearn.linear_model import LogisticRegression



logit_clf = LogisticRegression(max_iter=10000).fit(X_train, y_train)    





Log_train_score = logit_clf.score(X_train, y_train)

Log_test_score = logit_clf.score(X_test, y_test)

print ("Train split score: {:.3f}  \nTest split score: {:.3f}".format(Log_train_score, Log_test_score))
CV_scores=cross_val_score(logit_clf, X_4cv, y_4cv, cv=10)

print ('Overall Score: ', np.round_(np.mean(CV_scores), 3))
yhat=logit_clf.predict(X_test)

print (classification_report(y_test, yhat, zero_division=1))
Log_ROC = plot_model_eval_curves(logit_clf, X_test, y_test)
from sklearn.neighbors import KNeighborsClassifier



params={'n_neighbors':range(1,20), 'weights':['uniform','distance']}



start_time = datetime.now()  

KNN_grid=GridSearchCV(KNeighborsClassifier(), params, cv=10).fit(X_train, y_train)

end_time = datetime.now()



print('KNN traning elapsed time- {}\n'.format(end_time - start_time))

print('Grid search best params', KNN_grid.best_params_)



KNN_train_score = KNN_grid.score(X_train, y_train)

KNN_test_score = KNN_grid.score(X_test, y_test)

print ("\nTrain split score: {:.3f}  \nTest split score: {:.3f}".format(KNN_train_score, KNN_test_score))
KNN_clf=KNN_grid.best_estimator_



CV_scores=cross_val_score(KNN_clf, X_4cv, y_4cv, cv=10)

print ('Overall Score: ', np.round_(np.mean(CV_scores), 3))
yhat=KNN_clf.predict(X_test)

print (classification_report(y_test, yhat, zero_division=1))
KNN_ROC = plot_model_eval_curves(KNN_clf, X_test, y_test)
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler



X_std_train = StandardScaler().fit_transform(X_train)

X_std_test = StandardScaler().fit_transform(X_test)



params = {'C':[1, 5, 10],  'gamma':[0.001, 0.1, 1]}



start_time = datetime.now()  

SVM_grid=GridSearchCV(SVC(kernel='rbf', probability=True, random_state=27), params, cv=10).fit(X_std_train, y_train)

end_time = datetime.now()



print('SVC traning elapsed time- {}\n'.format(end_time - start_time))

print('Grid search best params', SVM_grid.best_params_)



SVC_train_score = SVM_grid.score(X_std_train, y_train)

SVC_test_score = SVM_grid.score(X_std_test, y_test)

print ("\nTrain split score: {:.3f}  \nTest split score: {:.3f}".format(SVC_train_score, SVC_test_score))
SVM_clf=SVM_grid.best_estimator_



X_std_4cv = StandardScaler().fit_transform(X_4cv)



CV_scores=cross_val_score(SVM_clf, X_std_4cv, y_4cv, cv=10)

print ('Overall Score: ', np.round_(np.mean(CV_scores), 3))
yhat=SVM_clf.predict(X_std_test)

print (classification_report(y_test, yhat, zero_division=1))
SVM_ROC = plot_model_eval_curves(SVM_clf, X_std_test, y_test)
from sklearn.ensemble import RandomForestClassifier



params={'n_estimators':range(50,200,50), 'max_depth':[3,4,5]}



start_time = datetime.now()  

Rforest_grid=GridSearchCV(RandomForestClassifier(random_state=27), params, cv=5).fit(X_train, y_train)

end_time = datetime.now()



print('Random Forest traning elapsed time- {}\n'.format(end_time - start_time))

print('Grid search best params', Rforest_grid.best_params_)



RF_train_score = Rforest_grid.score(X_train, y_train)

RF_test_score = Rforest_grid.score(X_test, y_test)

print ("\nTrain split score: {:.3f}  \nTest split score: {:.3f}".format(RF_train_score, RF_test_score))
Rforest_clf=Rforest_grid.best_estimator_



CV_scores=cross_val_score(Rforest_clf, X_4cv, y_4cv, cv=10)

print ('Overall Score: ', np.round_(np.mean(CV_scores), 3))
yhat=Rforest_clf.predict(X_test)

print (classification_report(y_test, yhat, zero_division=1))
RandomF_params=pd.DataFrame({'Feature':X.columns, 'Weights':Rforest_clf.feature_importances_}).round(2)

RandomF_params[ RandomF_params['Weights'] != 0.0].sort_values(by='Weights', ascending=False)
RF_ROC = plot_model_eval_curves(Rforest_clf, X_test, y_test)
from sklearn.ensemble import GradientBoostingClassifier



params={'n_estimators':[100, 150, 200], 'learning_rate':[0.01, 0.02, 0.04]}



start_time = datetime.now()  

GB_grid=GridSearchCV(GradientBoostingClassifier(random_state=27), params, cv=5).fit(X_train, y_train)

end_time = datetime.now()



print('Graident Boost traning elapsed time- {}\n'.format(end_time - start_time))

print('Grid search best params', GB_grid.best_params_)



GB_train_score = GB_grid.score(X_train, y_train)

GB_test_score = GB_grid.score(X_test, y_test)

print ("\nTrain split score: {:.3f}  \nTest split score: {:.3f}".format(GB_train_score, GB_test_score))

GB_clf=GB_grid.best_estimator_



CV_scores=cross_val_score(GB_clf, X_4cv, y_4cv, cv=10)

print ('Overall Score: ', np.round_(np.mean(CV_scores), 3))
yhat=GB_clf.predict(X_test)

print (classification_report(y_test, yhat, zero_division=1))
GradBoost_params=pd.DataFrame({'Feature':X.columns, 'Weights':GB_clf.feature_importances_}).round(2)

GradBoost_params[ GradBoost_params['Weights'] != 0.0].sort_values(by='Weights', ascending=False)
GB_ROC = plot_model_eval_curves(GB_clf, X_test, y_test)
from sklearn.ensemble import AdaBoostClassifier



params={'n_estimators':range(100,200,20), 'learning_rate':[0.01, 0.1, 0.2]}



start_time = datetime.now()  

AdaB_grid=GridSearchCV(AdaBoostClassifier(random_state=27), params, cv=5).fit(X_train, y_train)

end_time = datetime.now()



print('Ada Boost training Elapsed time- {}\n'.format(end_time - start_time))

print('Grid search best params', AdaB_grid.best_params_)



AdaB_train_score = AdaB_grid.score(X_train, y_train)

AdaB_test_score = AdaB_grid.score(X_test, y_test)

print ("\nTrain split score: {:.3f}  \nTest split score: {:.3f}".format(AdaB_train_score, AdaB_test_score))
AdaB_clf=AdaB_grid.best_estimator_



CV_scores=cross_val_score(AdaB_clf, X_4cv, y_4cv, cv=10)

print ('Overall Score: ', np.round_(np.mean(CV_scores), 3))
yhat=AdaB_clf.predict(X_test)

print (classification_report(y_test, yhat, zero_division=1))
AdaB_params=pd.DataFrame({'Feature':X.columns, 'Weights':AdaB_clf.feature_importances_}).round(2)

AdaB_params[ AdaB_params['Weights'] != 0.0].sort_values(by='Weights', ascending=False)
AdaB_ROC = plot_model_eval_curves(AdaB_clf, X_test, y_test)
import xgboost as xgb



params={'learning_rate':[0.005, 0.001, 0.01, 0.02]}



start_time = datetime.now()  

XG_grid=GridSearchCV(xgb.XGBClassifier(random_state=1), params, cv=10).fit(X_train, y_train)

end_time = datetime.now()



print('XG Boost traning elapsed time- {}\n'.format(end_time - start_time))

print('Grid search best params', XG_grid.best_params_)



XGB_train_score = XG_grid.score(X_train, y_train)

XGB_test_score = XG_grid.score(X_test, y_test)

print ("\nTrain split score: {:.3f}  \nTest split score: {:.3f}".format(XGB_train_score, XGB_test_score))

XGBoost_clf=XG_grid.best_estimator_



CV_scores=cross_val_score(XGBoost_clf, X_4cv, y_4cv, cv=10)

print ('Overall Score: ', np.round_(np.mean(CV_scores), 3))
yhat=XGBoost_clf.predict(X_test)

print (classification_report(y_test, yhat, zero_division=1))
XGBoost_params=pd.DataFrame({'Feature':X.columns, 'Weights':XGBoost_clf.feature_importances_}).round(2)

XGBoost_params[ XGBoost_params['Weights'] != 0.0].sort_values(by='Weights', ascending=False)
XGB_ROC = plot_model_eval_curves(XGBoost_clf, X_test, y_test)
from sklearn.naive_bayes import GaussianNB



NaiveG_clf = GaussianNB().fit(X_train, y_train)    



NaiveG_train_score = NaiveG_clf.score(X_train, y_train)

NaiveG_test_score = NaiveG_clf.score(X_test, y_test)

print ("Train split score: {:.3f}  \nTest split score: {:.3f}".format(NaiveG_train_score, NaiveG_test_score))
CV_scores=cross_val_score(NaiveG_clf, X_4cv, y_4cv, cv=10)

print ('Overall Score: ', np.round_(np.mean(CV_scores), 3))
yhat=NaiveG_clf.predict(X_test)

print (classification_report(y_test, yhat, zero_division=1))
NaiveG_ROC = plot_model_eval_curves(NaiveG_clf, X_test, y_test)
from sklearn.metrics import jaccard_score

from sklearn.metrics import f1_score, precision_score, recall_score



def get_model_eval_metrics(model, Xtrain, ytrain, Xtest, ytest):

        

    Train_score = model.score(Xtrain, ytrain)

    Test_score = model.score(Xtest, ytest)

    

    y_hat = model.predict(Xtest)

    

    F1 = f1_score(ytest, y_hat, average='weighted')

    Jaccard = jaccard_score(ytest, y_hat)

    Precision = precision_score(ytest, y_hat, average='weighted')

    Recall = recall_score(ytest, y_hat, average='weighted')

    

    return pd.Series([Train_score, Test_score, F1, Jaccard, Precision, Recall])
model_names = ['Decision Tree', 'Logit Reg', 'SVM Classifer', 'KNN', 'Random Forest', 'Gradient Boosting', 'AdaBoost', 'XGB', 'Gaussian Naive']

models = [DTree, logit_clf, SVM_clf, KNN_clf, Rforest_clf, GB_clf, AdaB_clf, XGBoost_clf, NaiveG_clf]

AUC_inf = [DT_ROC, Log_ROC, SVM_ROC, KNN_ROC, RF_ROC, GB_ROC, AdaB_ROC, XGB_ROC, NaiveG_ROC ]



df_models_comparison = pd.DataFrame()



for i in range(len(models)):

    

    if models[i] == SVM_clf:

        Xtrain_ = X_std_train

        Xtest_ = X_std_test

    else: 

        Xtrain_ = X_train

        Xtest_ = X_test

    

    model_scores = get_model_eval_metrics(models[i], Xtrain_, y_train, Xtest_, y_test)

    

    model_scores = model_scores.append(pd.Series(AUC_inf[i]), ignore_index=True)

    df_models_comparison = df_models_comparison.append(model_scores, ignore_index=True)

    

df_models_comparison.columns=['Train score', 'Test score', 'F1', 'Jaccard', 'Precision', 'Recall', 'ROC AUC score']

df_models_comparison.index = model_names
df_models_comparison.round(2)