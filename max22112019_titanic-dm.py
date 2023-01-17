# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train_len = len(train)
#percentage of nan values present in each feature
features_nan=[feature for feature in train.columns if train[feature].isnull().sum()>0]

for feature in features_nan:
    print(feature, np.round(train[feature].isnull().mean()*100, 2),  ' % missing values')
num = train[['Age','SibSp','Parch','Fare']]
cat = train[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]
#num, #cat
#distributions for all numeric variables 
for i in num.columns:
    plt.hist(num[i])
    plt.title(i)
    plt.show()
print(num.corr())
#compare survival rate across Age, SibSp, Parch, and Fare 
#pd.pivot_table(train, index = 'Survived', values = ['Age','SibSp','Parch','Fare'])
for i in cat.columns:
    sns.barplot(cat[i].value_counts().index,cat[i].value_counts()).set_title(i)
    plt.show()
print(pd.pivot_table(train, index = 'Survived', columns = 'Pclass', values = 'Ticket' ,aggfunc ='count'))
print()
print(pd.pivot_table(train, index = 'Survived', columns = 'Sex', values = 'Ticket' ,aggfunc ='count'))
print()
print(pd.pivot_table(train, index = 'Survived', columns = 'Embarked', values = 'Ticket' ,aggfunc ='count'))
# Creating Deck column from the first letter of the Cabin column (M stands for Missing)
train['Deck'] = train['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
train.Deck.unique()
train_deck = train.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 
                                                                        'Fare', 'Embarked', 'Cabin', 
                                                                        'PassengerId', 'Ticket']).rename(columns={'Name': 'Count'}).transpose()
train_deck
# Passenger in the T deck is changed to A
idx = train[train['Deck'] == 'T'].index
train.loc[idx, 'Deck'] = 'A'
train_survived = train.groupby(['Deck', 'Survived']).count().drop(columns=['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                                                                                   'Embarked', 'Pclass',
                                                                                   'Cabin', 'PassengerId', 
                                                                                   'Ticket']).rename(columns={'Name':'Count'}).transpose()
train_survived
pd.pivot_table(train,index='Survived',columns='Deck', values = 'Name', aggfunc='count')
deck_mapping = {"A": 1, "B": 2, "C": 3, "D": 4,"E": 5,"F": 6,"G": 7,"M": 8 }
train['Deck'] = train['Deck'].map(deck_mapping)
train.head()
#percentage of nan values present in each feature
features_nan=[feature for feature in test.columns if train[feature].isnull().sum()>0]

for feature in features_nan:
    print(feature, np.round(test[feature].isnull().mean()*100, 2),  ' % missing values')
test['Deck'] = test['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
test.Deck.unique()
test_deck = test.groupby(['Deck', 'Pclass']).count().drop(columns=['Sex', 'Age', 'SibSp', 'Parch', 
                                                                        'Fare', 'Embarked', 'Cabin', 
                                                                        'PassengerId', 'Ticket']).rename(columns={'Name': 'Count'}).transpose()
test_deck
deck_mapping = {"A": 1, "B": 2, "C": 3, "D": 4,"E": 5,"F": 6,"G": 7,"M": 8 }
test['Deck'] = test['Deck'].map(deck_mapping)
test.head()
dataset = pd.concat([train, test], sort=True).reset_index(drop=True)
dataset.head()
sns.distplot(dataset['Age'].dropna())
figure=dataset.Age.hist(bins=50)
figure.set_title('Age')
figure.set_xlabel('Age')
figure.set_ylabel('No of passenger')
figure=dataset.boxplot(column="Age")
dataset['Age'].describe()
uppper_boundary=dataset['Age'].mean() + 3* dataset['Age'].std()
lower_boundary=dataset['Age'].mean() - 3* dataset['Age'].std()
print(lower_boundary), print(uppper_boundary),print(dataset['Age'].mean())
dataset.loc[dataset['Age']>=73,'Age']=73
figure=dataset.Fare.hist(bins=50)
figure.set_title('Fare')
figure.set_xlabel('Fare')
figure.set_ylabel('No of passenger')
dataset.boxplot(column="Fare")
dataset['Fare'].describe()
#### Lets compute the Interquantile range to calculate the boundaries
IQR=dataset.Fare.quantile(0.75)-dataset.Fare.quantile(0.25)
lower_bridge=dataset['Fare'].quantile(0.25)-(IQR*3)
upper_bridge=dataset['Fare'].quantile(0.75)+(IQR*3)
print(lower_bridge), print(upper_bridge)
dataset.loc[dataset['Fare']>=100,'Fare']=101
figure=dataset.Age.hist(bins=50)
figure.set_title('Fare')
figure.set_xlabel('Fare')
figure.set_ylabel('No of passenger')
figure=dataset.Fare.hist(bins=50)
figure.set_title('Fare')
figure.set_xlabel('Fare')
figure.set_ylabel('No of passenger')
dataset['num_Ticket'] = dataset.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
dataset['num_Ticket'].value_counts()
dataset['Ticket_letters'] = dataset.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').
                                             replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
pd.set_option("max_rows", None)
dataset['Ticket_letters'].value_counts()
dataset.head()
#extract Title from Name 
dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(dataset['Title'], dataset['Sex'])
title_mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs' }
dataset.replace({'Title': title_mapping}, inplace=True)
#titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
#dataset['Title'] = dataset['Title'].map(title_mapping)
#encode title variables
title_num_mapping = {"Dr": 1, "Master": 2, "Miss": 3, "Mr": 4,"Mrs": 5,"Rev": 6}
dataset['Title'] = dataset['Title'].map(title_num_mapping)
dataset.Title.unique()
#dataset['Title'] = dataset['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'Countess', 'Dona'], 'Miss/Mrs/Ms')
#dataset['Title'] = dataset['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')
# Extracting surnames from Name
dataset['Last_Name'] = dataset['Name'].apply(lambda x: str.split(x, ",")[0])
dataset.Last_Name.unique()
#encode Embarked 
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
#encode Sex 
sex_mapping = {"female": 1, "male": 2}
dataset['Sex'] = dataset['Sex'].map(sex_mapping)
dataset.head()
dataset[dataset['Age'].isnull()]
#null_data = df[df.isnull().any(axis=1)]
#df_all[df_all['Embarked'].isnull()]
dataset['Age'].isnull().sum()
age_corr=dataset.corr().abs().unstack().sort_values(ascending=False).reset_index()
#dataframe index is : [1,5,6,10,11] and I would like to reset it to [0,1,2,3,4] using reset_index()
#Reshaping the data using stack(), the column is stacked row wise
#default sorting algorithm is 'quicksort'
age_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
age_corr[age_corr['Feature 1'] == 'Age']
age_by_pclass_sex = dataset.groupby(['Sex', 'Pclass','Title','Deck','SibSp','Parch']).median()['Age']
age_by_pclass_sex
#taking care of missing values in Age
dataset['Age'] = dataset.groupby(['Pclass','SibSp'])['Age'].apply(lambda x: x.fillna(x.median()))
dataset.Age.isnull().sum()
#create an Ageband 
dataset['Ageband'] = pd.qcut(dataset['Age'], 10)
pd.pivot_table(dataset,index='Survived',columns='Ageband', values = 'Name', aggfunc='count')
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
dataset['Ageband'] = encoder.fit_transform(dataset['Ageband'])
#dataset['AgeBin'] = pd.qcut(dataset['Age'], 4).astype(int)

#label = LabelEncoder()
#dataset['AgeBand'] = encoder.fit_transform(dataset['AgeBin'])

#dataset.drop(['Age'], 1, inplace=True)
#taking care of missing values in Embarked
dataset['Embarked'] = dataset['Embarked'].fillna('1')#S:1
fare_corr=dataset.corr().abs().unstack().sort_values(ascending=False).reset_index()
#dataframe index is : [1,5,6,10,11] and I would like to reset it to [0,1,2,3,4] using reset_index()
#Reshaping the data using stack(), the column is stacked row wise
#default sorting algorithm is 'quicksort'
fare_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
fare_corr[fare_corr['Feature 1'] == 'Fare']
dataset[dataset['Fare'].isnull()]
med_fare=dataset.groupby(['SibSp', 'Parch', 'Pclass','Sex'])['Fare'].median()[0][0][3][2]
med_fare
dataset['Fare']=dataset['Fare'].fillna(med_fare)
med_fare
dataset.isnull().sum()
#create Family_Size and IsAlone
dataset['Family_Size'] = dataset['SibSp'] + dataset['Parch'] + 1
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
dataset['Family_Size_Band'] = dataset['Family_Size'].map(family_map)
dataset['IsAlone'] = dataset['Family_Size'].map(lambda s: 1
                                                   if s == 1 else 0)
DEFAULT_SURVIVAL_VALUE = 0.5
dataset['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in dataset[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                dataset.loc[dataset['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                dataset.loc[dataset['PassengerId'] == passID, 'Family_Survival'] = 0
                
for _, grp_df in dataset.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    dataset.loc[dataset['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    dataset.loc[dataset['PassengerId'] == passID, 'Family_Survival'] = 0
                        

sns.catplot(x="Family_Size",y="Survived",data = dataset.iloc[:train_len],kind="bar")
#create a Fareband
dataset['Fareband'] = pd.qcut(dataset['Fare'], 13)
pd.pivot_table(dataset,index='Survived',columns='Fareband', values = 'Name', aggfunc='count')
dataset.head()
#title_mapping = {"Mr": 0, "Miss/Mrs/Ms": 1, "Master": 2, "Dr/Military/Noble/Clergy": 3}
#dataset['Title'] = dataset['Title'].map(title_mapping)
#sex_mapping = {"female": 0, "male": 1}
#dataset['Sex'] = dataset['Sex'].map(sex_mapping)
#embarked_mapping = {"S": 0, "C": 1, "Q": 2}
#dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
dataset.Fareband=encoder.fit_transform(dataset.Fareband)
dataset.Family_Size_Band=encoder.fit_transform(dataset.Family_Size_Band)
#remove unwanted columns part 1
dataset.drop(labels=['Cabin', 'Fare', 'Age', 'Name', 'PassengerId', 'Ticket', ], axis=1, inplace=True)
dataset.drop(labels=['Parch','SibSp','Survived','Last_Name'], axis=1, inplace=True)
dataset.drop(labels=['Ticket_letters'], axis=1, inplace=True)
dataset.head()
dataset.shape
#created dummy variables from categories (also can use OneHotEncoder)
#dummies = pd.get_dummies(dataset[['Deck','Pclass','Sex','Fareband',
                                  #'Embarked','Ageband', 'Family_Size','Family_Size_Band','num_Ticket','Title',
                                 #'IsAlone','Family_Survival']])
#dummies.shape
#dummies.head()
train_df = dataset.loc[:890]
test_df = dataset.loc[891:]
train_df.columns
test_df.shape
# Scale data 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(train_df)
X_test=scaler.fit_transform(test_df)
y_train=train.Survived
y_train.head()
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
gnb = GaussianNB()
cv = cross_val_score(gnb,X_train,y_train,cv=5)
print(cv)
print(cv.mean()*100)
lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,X_train,y_train,cv=5)
print(cv)
print(cv.mean()*100)
dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt,X_train,y_train,cv=5)
print(cv)
print(cv.mean()*100)
knn = KNeighborsClassifier()
cv = cross_val_score(knn,X_train,y_train,cv=5)
print(cv)
print(cv.mean()*100)
rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf,X_train,y_train,cv=5)
print(cv)
print(cv.mean()*100)
svc = SVC(probability = True)
cv = cross_val_score(svc,X_train,y_train,cv=5)
print(cv)
print(cv.mean()*100)
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state =1)
cv = cross_val_score(xgb,X_train,y_train,cv=5)
print(cv)
print(cv.mean()*100)
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators = [('lr',lr),('knn',knn),
                                            ('rf',rf),('gnb',gnb),('svc',svc),
                                            ('xgb',xgb),('dt',dt)], voting = 'soft') 
cv = cross_val_score(voting_clf,X_train,y_train,cv=5)
print(cv)
print(cv.mean())
voting_clf.fit(X_train,y_train)
y_hat_base_vc = voting_clf.predict(X_test).astype(int)
testing=pd.read_csv('../input/titanic/test.csv')
basic_submission_vc = {'PassengerId': testing.PassengerId, 'Survived': y_hat_base_vc}
base_submission_vc = pd.DataFrame(data=basic_submission_vc)
base_submission_vc.to_csv('base_submission_votingclf.csv', index=False)
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV 
#simple performance reporting function
def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: ' + str(classifier.best_score_))
    print('Best Parameters: ' + str(classifier.best_params_))
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve, cross_val_score
kfold = StratifiedKFold(n_splits=8)
lr = LogisticRegression()
param_grid = {'max_iter' : [2000],
              'penalty' : ['l1', 'l2'],
              'C' : np.logspace(-4, 4, 20),
              'solver' : ['liblinear']}

clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = kfold, verbose = True, n_jobs = -1)
best_clf_lr = clf_lr.fit(X_train,y_train)
clf_performance(best_clf_lr,'Logistic Regression')
'''Logistic Regression
Best Score: 0.8328004343629344
Best Parameters: {'C': 0.23357214690901212, 'max_iter': 2000, 'penalty': 'l1', 'solver': 'liblinear'}'''
knn = KNeighborsClassifier()
param_grid = {'n_neighbors' : [3,5,7,9],
              'weights' : ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree','kd_tree'],
              'p' : [1,2]}
clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_knn = clf_knn.fit(X_train,y_train)
clf_performance(best_clf_knn,'KNN')
'''KNN
Best Score: 0.8350072186303434
Best Parameters: {'algorithm': 'ball_tree', 'n_neighbors': 9, 'p': 1, 'weights': 'uniform'}'''
#here, we got better score for cv=5 than cv=kfold!!!
y_hat_knn = clf_knn.best_estimator_.predict(X_test).astype(int)
testing=pd.read_csv('../input/titanic/test.csv')
basic_submission_knn = {'PassengerId': testing.PassengerId, 'Survived': y_hat_knn}
submission_knn = pd.DataFrame(data=basic_submission_knn)
submission_knn.to_csv('submission_clf_knn.csv', index=False)
'''svc = SVC(probability = True)
param_grid = tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10],
                                  'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['linear'], 'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['poly'], 'degree' : [2,3,4,5], 'C': [.1, 1, 10, 100, 1000]}]
clf_svc = GridSearchCV(svc, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_svc = clf_svc.fit(X_train,y_train)
clf_performance(best_clf_svc,'SVC')

y_hat_svc = clf_svc.best_estimator_.predict(X_test).astype(int)
testing=pd.read_csv('../input/titanic/test.csv')
svc_submission = {'PassengerId': testing.PassengerId, 'Survived': y_hat_svc}
submission_svc = pd.DataFrame(data=svc_submission)
submission_svc.to_csv('submission_clf_svc.csv', index=False)'''

'''SVC
Best Score: 0.8428410018203504
Best Parameters: {'C': 10, 'degree': 2, 'kernel': 'poly'}'''
svc = SVC(probability = True)
param_grid = tuned_parameters = [{'kernel': ['poly'], 'degree' : [2], 'C': [10]}]
clf_svc = GridSearchCV(svc, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_svc = clf_svc.fit(X_train,y_train)
clf_performance(best_clf_svc,'SVC')
y_hat_svc = clf_svc.best_estimator_.predict(X_test).astype(int)
testing=pd.read_csv('../input/titanic/test.csv')
svc_submission = {'PassengerId': testing.PassengerId, 'Survived': y_hat_svc}
submission_svc = pd.DataFrame(data=svc_submission)
submission_svc.to_csv('submission_clf_svc.csv', index=False)
'''rf = RandomForestClassifier(random_state = 1)
param_grid =  {'n_estimators': [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800], 
                                  'bootstrap': [True,False],
                                  'max_depth': [3,4,5,6,7,8,9,10,15,20,50,None],
                                  'max_features': [3,'auto','sqrt','log2'],
                                  'bootstrap': [False, True],
                                  'criterion': ['gini', 'entropy'],
                                  'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10],
                                  'min_samples_split': [2 ,3,4,5,6,7,8,9,10]}
                                  
clf_rf_rnd = RandomizedSearchCV(rf, param_distributions = param_grid, n_iter = 200, 
cv = 5, verbose = True, n_jobs = -1)
best_clf_rf_rnd = clf_rf_rnd.fit(X_train,y_train)
clf_performance(best_clf_rf_rnd,'Random Forest')'''

'''y_hat_rf_rnd = clf_rf_rnd.best_estimator_.predict(X_test).astype(int)
testing=pd.read_csv('../input/titanic/test.csv')
rf_submission = {'PassengerId': testing.PassengerId, 'Survived': y_hat_rf_rnd}
submission_rf = pd.DataFrame(data=rf_submission)
submission_rf.to_csv('submission_clf_rf_rnd.csv', index=False)'''

'''Random Forest
Best Score: 0.8495762977841943
Best Parameters: {'n_estimators': 100, 'min_samples_split': 3, 
'min_samples_leaf': 9, 'max_features': 'sqrt', 'max_depth': 4, 'criterion': 'gini', 'bootstrap': True}'''
'''#with stratified kfold
rf = RandomForestClassifier(random_state = 1)
param_grid =  {'n_estimators': [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800], 
                                  'bootstrap': [True,False],
                                  'max_depth': [3,4,5,6,7,8,9,10,15,20,50,None],
                                  'max_features': [3,'auto','sqrt','log2'],
                                  'bootstrap': [False, True],
                                  'criterion': ['gini', 'entropy'],
                                  'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10],
                                  'min_samples_split': [2 ,3,4,5,6,7,8,9,10]}
                                  
clf_rf_rnd_2 = RandomizedSearchCV(rf, param_distributions = param_grid, n_iter = 200, cv = kfold,
verbose = True, n_jobs = -1)
best_clf_rf_rnd_2 = clf_rf_rnd_2.fit(X_train,y_train)
clf_performance(best_clf_rf_rnd_2,'Random Forest')'''

'''y_hat_rf_rnd_2 = clf_rf_rnd_2.best_estimator_.predict(X_test).astype(int)
testing=pd.read_csv('../input/titanic/test.csv')
rf_submission = {'PassengerId': testing.PassengerId, 'Survived': y_hat_rf_rnd_2}
submission_rf = pd.DataFrame(data=rf_submission)
submission_rf.to_csv('submission_clf_rf_rnd_2.csv', index=False)'''

'''Random Forest
Best Score: 0.8529701576576576
Best Parameters: {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 6,
'max_features': 3, 'max_depth': 4, 'criterion': 'gini', 'bootstrap': True}'''
#with stratified kfold
rf = RandomForestClassifier(random_state = 1)
param_grid =  {'n_estimators': [400], 
                                  'bootstrap': [True],
                                  'max_depth': [4],
                                  'max_features': [3],
                                  'bootstrap': [True],
                                  'criterion': ['gini'],
                                  'min_samples_leaf': [6],
                                  'min_samples_split': [2]}
                                  
clf_rf_rnd_2 = RandomizedSearchCV(rf, param_distributions = param_grid, n_iter = 200, cv = kfold, verbose = True, n_jobs = -1)
best_clf_rf_rnd_2 = clf_rf_rnd_2.fit(X_train,y_train)
clf_performance(best_clf_rf_rnd_2,'Random Forest')
'''Random Forest
Best Score: 0.8529701576576576
Best Parameters: {'n_estimators': 400, 'min_samples_split': 2,
'min_samples_leaf': 6, 'max_features': 3, 'max_depth': 4, 'criterion': 'gini', 'bootstrap': True}'''
y_hat_rf_rnd_2 = clf_rf_rnd_2.best_estimator_.predict(X_test).astype(int)
testing=pd.read_csv('../input/titanic/test.csv')
submission_rf_rnd = {'PassengerId': testing.PassengerId, 'Survived': y_hat_rf_rnd_2}
submission_rf_rnd_2 = pd.DataFrame(data=submission_rf_rnd)
submission_rf_rnd_2.to_csv('submission_clf_rf_rnd_2.csv', index=False)
'''rf = RandomForestClassifier(random_state = 1)
param_grid =  {'n_estimators': [100,150,200,300,400,500],
               'criterion':['gini','entropy'],
                                  'bootstrap': [True],
                                  'max_depth': [3,4,5,6],
                                  'max_features': ['sqrt'],
                                  'min_samples_leaf': [1,5,9],
                                  'min_samples_split': [2,3,4,5]}
                                  
clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_rf = clf_rf.fit(X_train,y_train)
clf_performance(best_clf_rf,'Random Forest')'''

'''Random Forest
Best Score: 0.8529470843010483
Best Parameters: {'bootstrap': True, 'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt', 
'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 100}'''
'''rf = RandomForestClassifier(random_state = 1)
param_grid =  {'n_estimators': [100,200,300,400,500],
               'criterion':['gini'],
                                  'bootstrap': [False, True],
                                  'max_depth': [3,4,5,None],
                                  'max_features': ['sqrt'],
                                  'min_samples_leaf': [4,5,6,7],
                                  'min_samples_split': [1,2,3,4]}
                                  
clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = kfold, verbose = True, n_jobs = -1)
best_clf_rf = clf_rf.fit(X_train,y_train)
clf_performance(best_clf_rf,'Random Forest')'''

'''Random Forest
Best Score: 0.8541063384813385
Best Parameters: {'bootstrap': True, 'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt',
'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 100}'''
rf = RandomForestClassifier(random_state = 1)
param_grid =  {'n_estimators': [100],
               'criterion':['gini'],
                                  'bootstrap': [True],
                                  'max_depth': [4],
                                  'max_features': ['sqrt'],
                                  'min_samples_leaf': [4],
                                  'min_samples_split': [2]}
                                  
clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = kfold, verbose = True, n_jobs = -1)
best_clf_rf = clf_rf.fit(X_train,y_train)
clf_performance(best_clf_rf,'Random Forest')
'''Random Forest
Best Score: 0.8541063384813385
Best Parameters: {'bootstrap': True, 'criterion': 'gini',
'max_depth': 4, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 100}'''
y_hat_rf_grid = best_clf_rf.best_estimator_.predict(X_test).astype(int)
testing=pd.read_csv('../input/titanic/test.csv')
rf_submission_grid = {'PassengerId': testing.PassengerId, 'Survived': y_hat_rf_grid}
submission_rf_grid = pd.DataFrame(data=rf_submission_grid)
submission_rf_grid.to_csv('submission_clf_rf_grid.csv', index=False)
best_rf = best_clf_rf.best_estimator_.fit(X_train,y_train)
feat_importances = pd.Series(best_rf.feature_importances_, index=train_df.columns)
feat_importances.nlargest(20).plot(kind='barh')
clf_rf_3=RandomForestClassifier(criterion='gini', n_estimators=1100,
                                           max_depth=5,
                                           min_samples_split=4,
                                           min_samples_leaf=5,
                                           max_features='auto',
                                           oob_score=True,
                                           random_state=1,
                                           n_jobs=-1,
                                           verbose=1)
best_clf_rf_3 = clf_rf_3.fit(X_train,y_train)
y_hat_rf_3 = best_clf_rf_3.predict(X_test).astype(int)
testing=pd.read_csv('../input/titanic/test.csv')
rf_submission_3 = {'PassengerId': testing.PassengerId, 'Survived': y_hat_rf_3}
submission_rf_3 = pd.DataFrame(data=rf_submission_3)
submission_rf_3.to_csv('submission_rf_3.csv', index=False)
clf_rf_4=RandomForestClassifier(criterion='gini',
                                           n_estimators=1750,
                                           max_depth=7,
                                           min_samples_split=6,
                                           min_samples_leaf=6,
                                           max_features='auto',
                                           oob_score=True,
                                           random_state=1,
                                           n_jobs=-1,
                                           verbose=1)
best_clf_rf_4 = clf_rf_4.fit(X_train,y_train)
y_hat_rf_4 = best_clf_rf_4.predict(X_test).astype(int)
testing=pd.read_csv('../input/titanic/test.csv')
rf_submission_4 = {'PassengerId': testing.PassengerId, 'Survived': y_hat_rf_4}
submission_rf_4 = pd.DataFrame(data=rf_submission_4)
submission_rf_4.to_csv('submission_rf_4.csv', index=False)
rf_param_grid_best = {"max_depth": [None],
              "max_features": [3],
              "min_samples_split": [4],
              "min_samples_leaf": [5],
              "bootstrap": [False],
              "n_estimators" :[200],
              "criterion": ["gini"]}

gs_rf = GridSearchCV(rf, param_grid = rf_param_grid_best, cv=kfold, n_jobs= -1, verbose = 1)

gs_rf.fit(X_train, y_train)
rf_best = gs_rf.best_estimator_
print(f'RandomForest GridSearch best params: {gs_rf.best_params_}')
print(f'RandomForest GridSearch best score: {gs_rf.best_score_}')
y_hat_gs_rf = gs_rf.predict(X_test).astype(int)
testing=pd.read_csv('../input/titanic/test.csv')
rf_submission_gs = {'PassengerId': testing.PassengerId, 'Survived': y_hat_gs_rf}
submission_rf_gs = pd.DataFrame(data=rf_submission_gs)
submission_rf_gs.to_csv('submission_gs_rf.csv', index=False)
'''xgb = XGBClassifier(random_state = 1)

param_grid = {
    'n_estimators': [20, 50, 100, 250,300,400, 500,600,700,1000],
    'colsample_bytree': [0.2, 0.5, 0.7,0.75, 0.8,0.85, 0.9, 1],
    'max_depth': [2, 5, 10, 15, 20, 25, None],
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [1, 1.5, 2,2.5,3,4],
    'subsample': [0.5,0.55,0.6,0.65,0.7, 0.8, 0.9],
    'learning_rate':[.01,0.05,0.1,0.2,0.3,0.5,0.6,0.7,0.9],
    'gamma':[0,.01,.1,.5,1,10,100],
    'min_child_weight':[0,.01,0.05,0.1,1,10,100],
    'sampling_method': ['uniform', 'gradient_based']
}

clf_xgb_rnd = RandomizedSearchCV(xgb, param_distributions = param_grid, n_iter = 2000, 
cv = kfold, verbose = True, n_jobs = -1)
best_clf_xgb_rnd = clf_xgb_rnd.fit(X_train,y_train)
clf_performance(best_clf_xgb_rnd,'XGB')'''

'''XGB
Best Score: 0.8575048262548263
Best Parameters: {'subsample': 0.7, 'sampling_method': 'uniform', 
'reg_lambda': 2, 'reg_alpha': 0.5, 'n_estimators': 700, 'min_child_weight': 0, 'max_depth': 10,
'learning_rate': 0.01, 'gamma': 0.01, 'colsample_bytree': 0.75}'''
'''xgb = XGBClassifier(random_state = 1)

param_grid = {
    'n_estimators': [550,600,650],
    'colsample_bytree': [0.9,0.95,1],
    'max_depth': [12,15,20],
    'reg_alpha': [0.5,1],
    'reg_lambda': [0.5,1],
    'subsample': [0.85,0.9,0.95],
    'learning_rate':[0.85,0.9,0.95],
    'gamma':[8,10,12],
    'min_child_weight':[0],
    'sampling_method': ['uniform']
}

clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = kfold, verbose = True, n_jobs = -1)
best_clf_xgb = clf_xgb.fit(X_train,y_train)
clf_performance(best_clf_xgb,'XGB')'''
xgb = XGBClassifier(random_state = 1)

param_grid = {
    'n_estimators': [700],
    'colsample_bytree': [0.75],
    'max_depth': [10],
    'reg_alpha': [0.5],
    'reg_lambda': [2],
    'subsample': [0.7],
    'learning_rate':[0.01],
    'gamma':[.01],
    'min_child_weight':[0],
    'sampling_method': ['uniform']
}

clf_xgb_rnd = RandomizedSearchCV(xgb, param_distributions = param_grid, n_iter = 2000, cv = kfold, verbose = True, n_jobs = -1)
best_clf_xgb_rnd = clf_xgb_rnd.fit(X_train,y_train)
clf_performance(best_clf_xgb_rnd,'XGB')
'''XGB
Best Score: 0.8575048262548263
Best Parameters: {'subsample': 0.7, 'sampling_method': 'uniform', 'reg_lambda': 2, 'reg_alpha': 0.5, 'n_estimators': 700, 'min_child_weight': 0, 'max_depth': 10,
'learning_rate': 0.01, 'gamma': 0.01, 'colsample_bytree': 0.75}'''
y_hat_xgb_rnd = best_clf_xgb_rnd.best_estimator_.predict(X_test).astype(int)
testing=pd.read_csv('../input/titanic/test.csv')
xgb_submission_rnd = {'PassengerId': testing.PassengerId, 'Survived': y_hat_xgb_rnd}
submission_xgb = pd.DataFrame(data=xgb_submission_rnd)
submission_xgb.to_csv('submission_xgb_rnd.csv', index=False)
xgb_param_grid_best = {'learning_rate':[0.1], 
                  'reg_lambda':[0.3],
                  'gamma': [1],
                  'subsample': [0.8],
                  'max_depth': [2],
                  'n_estimators': [300]
              }

gs_xgb = GridSearchCV(xgb, param_grid = xgb_param_grid_best, cv=kfold, n_jobs= -1, verbose = 1)

gs_xgb.fit(X_train,y_train)

xgb_best = gs_xgb.best_estimator_
print(f'XGB GridSearch best params: {gs_xgb.best_params_}')
print(f'XGB GridSearch best score: {gs_xgb.best_score_}')
y_hat_gs_xgb = gs_xgb.predict(X_test).astype(int)
testing=pd.read_csv('../input/titanic/test.csv')
gs_xgb_submission = {'PassengerId': testing.PassengerId, 'Survived': y_hat_gs_xgb}
submission_gs_xgb = pd.DataFrame(data=gs_xgb_submission)
submission_gs_xgb.to_csv('submission_gs_xgb_new.csv', index=False)
best_lr = best_clf_lr.best_estimator_
best_knn = best_clf_knn.best_estimator_
best_svc = best_clf_svc.best_estimator_
best_rf = best_clf_rf
best_xgb = best_clf_xgb_rnd.best_estimator_

voting_clf_hard = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc)], voting = 'hard') 
voting_clf_soft = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc)], voting = 'soft') 
voting_clf_all = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc),
                                                ('lr', best_lr)], voting = 'soft') 
voting_clf_xgb = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), 
                                                ('xgb', best_xgb),('lr', best_lr)], voting = 'soft')

print('voting_clf_hard :',cross_val_score(voting_clf_hard,X_train,y_train,cv=kfold))
print('voting_clf_hard mean :',cross_val_score(voting_clf_hard,X_train,y_train,cv=kfold).mean())

print('voting_clf_soft :',cross_val_score(voting_clf_soft,X_train,y_train,cv=kfold))
print('voting_clf_soft mean :',cross_val_score(voting_clf_soft,X_train,y_train,cv=kfold).mean())

print('voting_clf_all :',cross_val_score(voting_clf_all,X_train,y_train,cv=kfold))
print('voting_clf_all mean :',cross_val_score(voting_clf_all,X_train,y_train,cv=kfold).mean())

print('voting_clf_xgb :',cross_val_score(voting_clf_xgb,X_train,y_train,cv=kfold))
print('voting_clf_xgb mean :',cross_val_score(voting_clf_xgb,X_train,y_train,cv=kfold).mean())
#in a soft voting classifier you can weight some models more than others. I used a grid search to explore different weightings
#no new results here
params = {'weights' : [[1,1,1],[1,2,1],[1,1,2],[2,1,1],[2,2,1],[1,2,2],[2,1,2]]}

vote_weight = GridSearchCV(voting_clf_soft, param_grid = params, cv = kfold, verbose = True, n_jobs = -1)
best_clf_weight = vote_weight.fit(X_train,y_train)
clf_performance(best_clf_weight,'VC Weights')
voting_clf_sub = best_clf_weight.best_estimator_.predict(X_test)
#Make Predictions 
voting_clf_hard.fit(X_train, y_train)
voting_clf_soft.fit(X_train, y_train)
voting_clf_all.fit(X_train, y_train)
voting_clf_xgb.fit(X_train, y_train)

best_rf.fit(X_train, y_train)
y_hat_vc_hard = voting_clf_hard.predict(X_test).astype(int)
y_hat_rf = best_rf.predict(X_test).astype(int)
y_hat_vc_soft =  voting_clf_soft.predict(X_test).astype(int)
y_hat_vc_all = voting_clf_all.predict(X_test).astype(int)
y_hat_vc_xgb = voting_clf_xgb.predict(X_test).astype(int)
#convert output to dataframe 
final_data = {'PassengerId': testing.PassengerId, 'Survived': y_hat_rf}
submission = pd.DataFrame(data=final_data)

final_data_2 = {'PassengerId': testing.PassengerId, 'Survived': y_hat_vc_hard}
submission_2 = pd.DataFrame(data=final_data_2)

final_data_3 = {'PassengerId': testing.PassengerId, 'Survived': y_hat_vc_soft}
submission_3 = pd.DataFrame(data=final_data_3)

final_data_4 = {'PassengerId': testing.PassengerId, 'Survived': y_hat_vc_all}
submission_4 = pd.DataFrame(data=final_data_4)

final_data_5 = {'PassengerId': testing.PassengerId, 'Survived': y_hat_vc_xgb}
submission_5 = pd.DataFrame(data=final_data_5)

final_data_comp = {'PassengerId': testing.PassengerId, 'Survived_vc_hard': y_hat_vc_hard, 'Survived_rf': y_hat_rf, 'Survived_vc_soft' : y_hat_vc_soft, 'Survived_vc_all' : y_hat_vc_all,  'Survived_vc_xgb' : y_hat_vc_xgb}
comparison = pd.DataFrame(data=final_data_comp)
#track differences between outputs 
comparison['difference_rf_vc_hard'] = comparison.apply(lambda x: 1 if x.Survived_vc_hard != x.Survived_rf else 0, axis =1)
comparison['difference_soft_hard'] = comparison.apply(lambda x: 1 if x.Survived_vc_hard != x.Survived_vc_soft else 0, axis =1)
comparison['difference_hard_all'] = comparison.apply(lambda x: 1 if x.Survived_vc_all != x.Survived_vc_hard else 0, axis =1)
#prepare submission files 
submission.to_csv('submission_rf_aug.csv', index =False)
submission_2.to_csv('submission_vc_hard_aug.csv',index=False)
submission_3.to_csv('submission_vc_soft_aug.csv', index=False)
submission_4.to_csv('submission_vc_all_aug.csv', index=False)
submission_5.to_csv('submission_vc_xgb_aug.csv', index=False)