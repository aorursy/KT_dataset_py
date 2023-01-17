import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('dark')

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.impute import SimpleImputer

from sklearn.model_selection import RandomizedSearchCV, cross_val_score

import re

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from scipy.stats import expon,norm

import os

%matplotlib inline
titanic_df=pd.read_csv('../input/titanic/train.csv')
test_df=pd.read_csv('../input/titanic/test.csv')
titanic_df.info()
test_df.info()
titanic_df.drop(columns=['PassengerId'], inplace=True)

PassengerIdTest=test_df['PassengerId']

test_df.drop(columns=['PassengerId'], inplace=True)
figure=plt.figure(figsize=(10,10))

g1=sns.heatmap(titanic_df.corr(), annot=True, fmt='.1f', cbar=False)
figure,ax=plt.subplots(1,2,figsize=(14,6))

g1=sns.countplot(x='Pclass', data=titanic_df, ax=ax[0])

ax[0].grid(True)

g2=sns.factorplot(x='Pclass', y='Survived', data=titanic_df, kind='bar', ax=ax[1])

ax[1].set_ylabel('Survival Probability');

ax[1].axhline(titanic_df['Survived'].describe()['mean'], ls='--', color='r')

ax[1].grid(True)

plt.close(2)

titanic_df=pd.get_dummies(titanic_df, columns=['Pclass'], drop_first=True)
titanic_df['Name'].unique()
def get_title_from_names(name):

    return name.split(',')[1].split('.')[0].strip()
titanic_df['Title']=titanic_df['Name'].apply(get_title_from_names)
titanic_df['Title'].value_counts()
titanic_df['Title'].replace(['Col','Major','Jonkheer','Ms','Mlle','Mme','Lady','Dona','the Countess','Sir','Capt','Don'],'RareTitle', inplace=True)
titanic_df.drop(columns=['Name'], inplace=True)

figure, ax=plt.subplots(1,2, figsize=(14,6))

g1=sns.countplot(x='Title', data=titanic_df, ax=ax[0])

g2=sns.factorplot(x='Title', y='Survived', kind='bar',data=titanic_df, ax=ax[1])

ax[1].set_ylabel('Survival Probability')

ax[1].axhline(titanic_df['Survived'].describe()['mean'], ls='--', color='r')

ax[0].grid(True)

ax[1].grid(True)

plt.close(2)
titanic_df=pd.get_dummies(titanic_df, columns=['Title'])

titanic_df.head()
titanic_df.describe()
figure, ax=plt.subplots(1,2, figsize=(14,6))

g1=sns.countplot(x='Sex', data=titanic_df, ax=ax[0])

g2=sns.factorplot(x='Sex', y='Survived', kind='bar',data=titanic_df, ax=ax[1])

ax[1].set_ylabel('Survival Probability')

ax[1].axhline(titanic_df['Survived'].describe()['mean'], ls='--', color='r')

ax[0].grid(True)

ax[1].grid(True)

plt.close(2)
titanic_df=pd.get_dummies(titanic_df, columns=['Sex'],drop_first=True)
titanic_df.head()
titanic_df['Age'].value_counts()
titanic_df['Age'].isnull().sum()
figure=plt.figure(figsize=(18,6))

g=sns.countplot(x='Age', data=titanic_df)

g.set_xticklabels(g.get_xticklabels(), rotation=90);

g.grid(True)

ages=titanic_df[titanic_df['Age'].notnull()]['Age'].values
def generate_new_age(null):

    return float(np.random.choice(ages, 1))

titanic_df['Age']=titanic_df['Age'].isnull().apply(generate_new_age)
def group_ages(age):

    return (age//5)
titanic_df['Age_Gr']=titanic_df['Age'].apply(group_ages)
titanic_df['Age_Gr'].value_counts()
figure, ax=plt.subplots(1,2, figsize=(14,6))

g1=sns.countplot(x='Age_Gr', data=titanic_df, ax=ax[0])

g2=sns.factorplot(x='Age_Gr', y='Survived', kind='bar',data=titanic_df, ax=ax[1])

ax[1].set_ylabel('Survival Probability');

ax[1].axhline(titanic_df['Survived'].describe()['mean'], ls='--', color='r')

ax[0].grid(True)

ax[1].grid(True)

plt.close(2)
titanic_df.drop(columns=['Age'], inplace=True)

titanic_df=pd.get_dummies(titanic_df, columns=['Age_Gr'],drop_first=True)

titanic_df.head()
fig, ax=plt.subplots(1,2, figsize=(14,6))

g1=sns.countplot(x='SibSp', data=titanic_df, ax=ax[0])

g2=sns.factorplot(x='SibSp', y='Survived', kind='bar', data=titanic_df, ax=ax[1])

ax[1].set_ylabel('Survival Probability');

ax[1].axhline(titanic_df.describe()['Survived']['mean'], ls='--', color='r')

ax[0].grid(True)

ax[1].grid(True)

plt.close(2)
fig, ax=plt.subplots(1,2, figsize=(14,6))

g1=sns.countplot(x='Parch', data=titanic_df, ax=ax[0])

g2=sns.factorplot(x='Parch', y='Survived', kind='bar', data=titanic_df, ax=ax[1])

ax[1].set_ylabel('Survival Probability');

ax[1].axhline(titanic_df.describe()['Survived']['mean'], ls='--', color='r')

ax[0].grid(True)

ax[1].grid(True)

plt.close(2)
def travels_alone(row):

    return int(row['SibSp']+row['Parch'])==0
titanic_df['Tr_Alone']=titanic_df.apply(travels_alone, axis=1)
fig, ax=plt.subplots(1,2, figsize=(14,6))

g1=sns.countplot(x='Tr_Alone', data=titanic_df, ax=ax[0])

g2=sns.factorplot(x='Tr_Alone', y='Survived', kind='bar', data=titanic_df, ax=ax[1])

ax[1].set_ylabel('Survival Probability');

ax[1].axhline(titanic_df.describe()['Survived']['mean'], ls='--', color='r')

ax[0].grid(True)

ax[1].grid(True)

plt.close(2)
print(titanic_df.corr()['Survived']['Tr_Alone'])
titanic_df.drop(columns=['SibSp', 'Parch'], inplace=True)

titanic_df=pd.get_dummies(titanic_df, columns=['Tr_Alone'], drop_first=True)
titanic_df['Ticket'].value_counts()
ticket_w_letter=titanic_df.loc[titanic_df['Ticket'].str.contains('[a-zA-Z]', regex=True)]

ticket_wout_letter=titanic_df.loc[~titanic_df['Ticket'].str.contains('[a-zA-Z]', regex=True)]
fig, ax=plt.subplots(1,2,figsize=(14,6))

g1=sns.countplot(x='Survived', data=ticket_wout_letter,ax=ax[0])

ax[0].set_title('Ticket with out letters')

g1=sns.countplot(x='Survived', data=ticket_w_letter, ax=ax[1])

ax[1].set_title('Ticket with letters')

plt.close(2)

figure=plt.figure(figsize=(18,6))

g=sns.countplot(x='Ticket', data=ticket_w_letter)

g.set_xticklabels(g.get_xticklabels(), rotation=90);

titanic_df.loc[titanic_df['Ticket'].str.contains('PC', regex=True),'Survived'].value_counts()
titanic_df.loc[titanic_df['Ticket'].str.contains('W./C.', regex=True),'Survived'].value_counts()
def pc_in_ticket(ticket):

    if re.search('PC', ticket):

        return True

    else:

        return False



def wc_in_ticket(ticket):

    if re.search('W./C.', ticket):

        return True

    else:

        return False
titanic_df['PC_inticket']=titanic_df['Ticket'].apply(pc_in_ticket)

titanic_df['WC_inticket']=titanic_df['Ticket'].apply(wc_in_ticket)
titanic_df=pd.get_dummies(titanic_df, columns=['PC_inticket', 'WC_inticket'], drop_first=True)
titanic_df.drop(columns=['Ticket'], inplace=True)
figure=plt.figure(figsize=(18,6))

g1=sns.countplot(titanic_df['Fare'].astype(np.int32))

most_frequent_fare=np.argmax(titanic_df['Fare'].value_counts())
most_frequent_fare
titanic_df['Fare_Gr']=pd.qcut(titanic_df['Fare'].astype(np.int32),7, precision=1)
titanic_df['Fare_Gr'].value_counts()
fig, ax=plt.subplots(1,2, figsize=(14,6))

g1=sns.countplot(x='Fare_Gr', data=titanic_df, ax=ax[0])

g2=sns.factorplot(x='Fare_Gr', y='Survived', kind='bar', data=titanic_df, ax=ax[1])

ax[1].set_ylabel('Survival Probability');

ax[1].axhline(titanic_df.describe()['Survived']['mean'], ls='--', color='r')

plt.close(2)
titanic_df.drop(columns=['Fare', 'Cabin'], inplace=True)

titanic_df=pd.get_dummies(titanic_df, columns=['Fare_Gr'],drop_first=True)
titanic_df['Embarked'].fillna(np.argmax(titanic_df['Embarked'].value_counts()), inplace=True)
fig, ax=plt.subplots(1,2, figsize=(14,6))

g1=sns.countplot(x='Embarked', data=titanic_df, ax=ax[0])

g2=sns.factorplot(x='Embarked', y='Survived', kind='bar', data=titanic_df, ax=ax[1])

ax[1].set_ylabel('Survival Probability');

ax[1].axhline(titanic_df.describe()['Survived']['mean'], ls='--', color='r')

plt.close(2)
titanic_df = pd.get_dummies(titanic_df, columns=['Embarked'],drop_first=True)

titanic_df.head()



figure=plt.figure(figsize=(16,16))

g1=sns.heatmap(titanic_df.corr(), annot=True, fmt='.1f', cbar=False)
X_train=titanic_df.drop(columns=['Survived','Title_Mr'])

y_train=titanic_df['Survived']

titanic_df.head()
param_dist_logistic={'solver':['newton-cg', 'liblinear','lbfgs', 'sag', 'saga'], 'penalty':['l2'], 'C':expon(scale=1)}

logistic_regressor=LogisticRegression()

rs_logistic=RandomizedSearchCV(logistic_regressor, param_distributions=param_dist_logistic, n_jobs=-1, cv=5, n_iter=250, scoring='accuracy')

rs_logistic.fit(X_train, y_train)
rs_logistic.best_score_
DTC=DecisionTreeClassifier(max_depth=1)

adaDTC=AdaBoostClassifier(DTC, random_state=0)

param_dist_ada = {'base_estimator__criterion':['gini', 'entropy'],

                  'base_estimator__splitter' :['best', 'random'],

                  'algorithm'                :['SAMME', 'SAMME.R'],

                  'n_estimators'             :[100,200,300,500],

                  'learning_rate'            :expon(scale=1)}

rs_ada=RandomizedSearchCV(adaDTC, param_distributions=param_dist_ada, n_jobs=-1, cv=5, n_iter=250, scoring='accuracy')

rs_ada.fit(X_train, y_train)
rs_ada.best_score_
param_dist_rf={'max_depth'        :  [4,5,6,7,8],

                 'max_features'     :[8,9,10,11],

                 'min_samples_split':[10],

                 'bootstrap'        :[False],

                 'n_estimators'     :[100,100,200,300],

                 'criterion'        :['gini', 'entropy']}

RFC=RandomForestClassifier()

rs_rf=RandomizedSearchCV(RFC, param_distributions=param_dist_rf, scoring='accuracy', n_jobs=-1, cv=5, n_iter=250 )

rs_rf.fit(X_train, y_train)
rs_rf.best_score_
param_dist_knc={'n_neighbors':[4,5,5,6,6,7], 'weights':['uniform', 'distance'], 'p':[1,2]}

KNC=KNeighborsClassifier()

rs_knc=RandomizedSearchCV(KNC, param_distributions=param_dist_knc, n_iter=250, n_jobs=-1, cv=5, scoring='accuracy')

rs_knc.fit(X_train, y_train)
rs_knc.best_score_
param_dist_nn={'activation':['logistic', 'relu', 'tanh'], 'solver':['lbfgs', 'sgd', 'adam'], 'alpha':expon(scale=0.2), 'learning_rate_init':expon(scale=0.2)}

nn=MLPClassifier()

rs_nn=RandomizedSearchCV(nn, param_distributions=param_dist_nn, n_jobs=-1, n_iter=250, cv=5, scoring='accuracy')

rs_nn.fit(X_train, y_train)
rs_nn.best_score_
param_dist_svc={'C':expon(scale=1), 'kernel':['rbf', 'poly', 'sigmoid'], 'degree':[2,3,4,5,6], 'coef0':expon(scale=0.01)}

svc=SVC()

rs_svc=RandomizedSearchCV(svc, param_distributions=param_dist_svc, n_jobs=-1, n_iter=250, scoring='accuracy', cv=5)

rs_svc.fit(X_train, y_train)
rs_svc.best_score_
param_dist_gaussian={'var_smoothing':expon(scale=0.001)}

gaussian=GaussianNB()

rs_gaussian=RandomizedSearchCV(gaussian, param_distributions=param_dist_gaussian, n_jobs=-1, scoring='accuracy', cv=5, n_iter=250)

rs_gaussian.fit(X_train, y_train)
rs_gaussian.best_score_
classifiers=[LogisticRegression(**rs_logistic.best_params_),

             AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',splitter='best'),n_estimators= 200,learning_rate=0.47560685124037505,algorithm= 'SAMME.R'),

             RandomForestClassifier(**rs_rf.best_params_),

             KNeighborsClassifier(**rs_knc.best_params_),

             MLPClassifier(**rs_nn.best_params_),

             SVC(**rs_svc.best_params_),

             GaussianNB(**rs_gaussian.best_params_)]

scores=[]

for classifier in classifiers:

    scores.append(cross_val_score(classifier, X_train, y_train, cv=5, n_jobs=-1, scoring='accuracy'))
accuracy_mean=[]

accuracy_std=[]

for score in scores:

    accuracy_mean.append(np.mean(score))

    accuracy_std.append(np.std(score))
print(f'La media de los valores de precisión de lo clasificadores es {np.array(accuracy_mean).mean():.4f}')

print(f'La media de los valores de variación estándar de lo clasificadores es {np.array(accuracy_std).mean():.4f}')



classifier_names=['LogisticRegression','AdaBoostClassifier','RandomForestClassifier','KNeighborsClassifier','MLPClassifier','SVC','GaussianNB']

figure, ax=plt.subplots(1,2,figsize=(14,6))

g1=sns.barplot(x=classifier_names, y=accuracy_mean, ax=ax[0])

ax[0].set_title('Accuracy Mean');

ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45);

g2=sns.barplot(x=classifier_names, y=accuracy_std, ax=ax[1])

ax[1].set_title('Accuracy Std')

ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45);

ax[0].axhline(np.array(accuracy_mean).mean(), color='red', ls='--')

ax[1].axhline(np.array(accuracy_std).mean(), color='red', ls='--')

ax[0].grid(True)

ax[1].grid(True)

plt.close(2)
votingC = VotingClassifier(estimators = [('lr',LogisticRegression(**rs_logistic.best_params_)),

                                         ('ada', AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',splitter='best'),n_estimators= 200,learning_rate=0.47560685124037505,algorithm= 'SAMME.R')),

                                         ('rf',RandomForestClassifier(**rs_rf.best_params_)),

                                         ('knc',KNeighborsClassifier(**rs_knc.best_params_)),

                                         ('nn',MLPClassifier(**rs_nn.best_params_)),

                                         ('svc',SVC(**rs_svc.best_params_)),

                                         ('gaussian',GaussianNB(**rs_gaussian.best_params_))                                         

                                        ],n_jobs=-1)

print(votingC)

votingC = votingC.fit(X_train, y_train)
voting_cross_val_score=cross_val_score(votingC, X_train, y_train, cv=5, n_jobs=-1, scoring='accuracy')
np.mean(voting_cross_val_score), np.array(voting_cross_val_score).std()
np.array(accuracy_mean).max()
classifier_names=['LogisticRegression','AdaBoostClassifier','RandomForestClassifier','KNeighborsClassifier','MLPClassifier','SVC','GaussianNB', 'VotingC']

figure, ax=plt.subplots(1,2,figsize=(14,6))

g1=sns.barplot(x=classifier_names, y=list(accuracy_mean)+[np.mean(voting_cross_val_score)], ax=ax[0])

ax[0].set_title('Accuracy Mean');

ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45);

ax[0].axhline(np.mean(accuracy_mean), ls='--', color='black');

g2=sns.barplot(x=classifier_names, y=list(accuracy_std)+[np.std(voting_cross_val_score)], ax=ax[1])

ax[1].set_title('Accuracy Std')

ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45);

ax[1].axhline(np.mean(accuracy_std), ls='--', color='black');

plt.close(2)
test_df['Title']=test_df['Name'].apply(get_title_from_names)
test_df['Title'].replace(['Col','Major','Jonkheer','Ms','Mlle','Mme','Lady','Dona','the Countess','Sir','Capt','Don'],'RareTitle', inplace=True)
test_df.drop(columns=['Name'], inplace=True)
test_df['Age']=test_df['Age'].isnull().apply(generate_new_age)
test_df['Age_Gr']=test_df['Age'].apply(group_ages)

test_df.drop(columns=['Age'], inplace=True)
test_df['Tr_Alone']=test_df.apply(travels_alone, axis=1)
test_df.drop(columns=['SibSp', 'Parch'], inplace=True)
test_df['PC_inticket']=test_df['Ticket'].apply(pc_in_ticket)

test_df['WC_inticket']=test_df['Ticket'].apply(wc_in_ticket)
test_df.drop(columns=['Ticket'], inplace=True)
def identify_interval(row):

    if (int(row['Fare'])>=0) and (int(row['Fare'])<7):

        return '(-0.1, 7.0]'

    if (int(row['Fare'])>=12) and (int(row['Fare'])<19):

        return '(12.0, 19.0]'

    if (int(row['Fare'])>=56) and (int(row['Fare'])<512):

        return '(56.0, 512.0]' 

    if (int(row['Fare'])>=27) and (int(row['Fare'])<56):

        return '(27.0, 56.0]'  

    if (int(row['Fare'])>=19) and (int(row['Fare'])<27):

        return '(19.0, 27.0]'  

    if (int(row['Fare'])>=8) and (int(row['Fare'])<12):

        return '(8.0, 12.0]' 

    if (int(row['Fare'])>=7) and (int(row['Fare'])<8):

        return '(7.0, 8.0]' 
test_df.loc[test_df['Fare'].isnull(), 'Fare']=most_frequent_fare
test_df['Fare_Gr']=test_df.apply(identify_interval, axis=1)
test_df.loc[test_df['Embarked'].isnull(), 'Embarked']='S'
test_df.drop(columns=['Cabin'], inplace=True)
test_df = pd.get_dummies(test_df, columns=['Pclass','Embarked','Title','Fare_Gr','Sex','Age_Gr', 'Tr_Alone','WC_inticket','PC_inticket'])

for col in X_train.columns:

    if col not in test_df.columns:

        test_df[col]=np.zeros(418, dtype=np.int8)
test_df=test_df[X_train.columns]

test_predictions=rs_svc.best_estimator_.predict(test_df)

dataframe_results=pd.DataFrame(np.c_[PassengerIdTest.values, test_predictions], columns=['PassengerId', 'Survived'])

dataframe_results.to_csv('predictions.csv',index=False)