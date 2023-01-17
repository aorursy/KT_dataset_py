# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are availablein the read-only "../input/" directory
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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score,make_scorer
import string
train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')

dataset=pd.concat([train,test])


EDA=pd.DataFrame()


#train.describe(include='all')
#test.describe(include='all')
dataset.describe(include='all')
print(train.Pclass.nunique())

fig1=sns.distplot(train.Pclass[train.Survived==1],label='Survived',bins=6)
fig1=sns.distplot(train.Pclass[train.Survived==0],label='Not Survived',bins=6)
fig1.set_title('Pclass')
fig1.set_xlabel('Pclass');fig1.set_ylabel('Freqency')
fig1.legend()
EDA=pd.concat([EDA,pd.get_dummies(dataset["Pclass"],columns=['Pclass'] , prefix='P')],axis=1)

EDA
dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')
train_x=dataset['Title'][:891]
train_y=dataset['Survived'][:891]
pd.crosstab(train_x,train_y)
dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don','Dona', 'Dr', 'Jonkheer',
                                               'Lady','Major', 'Rev', 'Sir'], 'Other')
dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'].astype(str)

train_x=dataset['Title'][:891]

hist=pd.crosstab(train_x,train_y)
hist['ratio']=hist[1]/hist.sum(axis=1)

fig2=plt.figure("Title")
plt.title("Title")
plt.xlabel("Title");plt.ylabel("Counts")
plt.hist(train_x[train_y==1],label="Survived",alpha=0.4)
plt.hist(train_x[train_y==0],label="Not Survived",alpha=0.4)
plt.legend()
plt.show()


hist
EDA
# Mrs : 1 , Miss : 2, Master : 3, Other: 4, Mr: 5
EDA=pd.concat([EDA,pd.get_dummies(dataset['Title'],columns=['Title'] , prefix='T')],axis=1)

EDA
dataset["Name_len"]=dataset.Name.apply(lambda x: len(x))

train_x=dataset.Name_len[:891]

fig11=sns.distplot(train_x[train_y==1],label='Survived')
fig11=sns.distplot(train_x[train_y==0],label='Not Survived' )
fig11.set_xlabel('Name_len');fig11.set_ylabel('Freqency')
fig11.set_title("Name_len")
fig11.legend()
plt.show()
dataset["Nam_ran"]=dataset.Name_len.map(lambda x: 1 if x <=20 else (2 if 20< x<=40 else 3 ))

train_x=dataset.Nam_ran[:891]

fig12=plt.figure("Nam_ran")
plt.title("Nam_ran")
plt.xlabel("Nam_ran");plt.ylabel("Counts")
plt.hist(train_x[train_y==1],label="Survived",alpha=0.4)
plt.hist(train_x[train_y==0],label="Not Survived",alpha=0.4)
plt.legend()
plt.show()

hist=pd.crosstab(train_x,train_y)
hist['ratio']=hist[1]/hist.sum(axis=1)

hist

EDA["Nam_len"]=dataset["Nam_ran"]

EDA
fig3=plt.figure("Sex")
plt.title("Sex")
plt.xlabel("Sex");plt.ylabel("Counts")

train_x=dataset['Sex'][:891]

plt.hist(train_x[train_y==1],label="Survived",alpha=0.4,bins=3)
plt.hist(train_x[train_y==0],label="Not Survived",alpha=0.4,bins=3)
plt.legend()
plt.show()
#male: 1 female: 2
EDA=pd.concat([EDA,pd.get_dummies(dataset['Sex'],columns=['Sex'] , prefix='S')],axis=1)

EDA
print("Num of NaN:",dataset.Age.isnull().sum())
print("Rate of NaN: ", dataset.Age.isnull().sum()/len(dataset.Age)*100,"%")

# Master : 0 Miss : 1 , Mr,Other : 2, Mrs: 3
dataset['Title']=dataset.Title.replace(['Master'], 0)
dataset['Title']=dataset.Title.replace(['Miss'], 1)
dataset['Title']=dataset.Title.replace(['Mr','Other'],2 )
dataset['Title']=dataset.Title.replace(['Mrs'], 3)
dataset
#I copied it from: https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial#1.-Exploratory-Data-Analysis
data_corr=dataset.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
data_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
data_corr[data_corr['Feature 1'] == 'Age']
Age_by_title_pclass=dataset.dropna(subset=['Age']).groupby(['Title','Pclass']).median()['Age']
Age_by_title_pclass
# Master : 0 Miss : 1 , Mr,Other : 2, Mrs: 3
dataset.head(6)
# I cannot think of more good way to deal with this. I tried pandas.map , panda.transform, but I should change Age while handling Title.
nul_age=[]
for tit,pcl in zip(dataset[dataset.Age.isnull()].Title,dataset[dataset.Age.isnull()].Pclass):
    nul_age.append(Age_by_title_pclass[tit,pcl])

dataset.Age[dataset.Age.isnull()]=nul_age

dataset.head(6)
train_x=dataset['Age'][:891]

fig4=sns.distplot(train_x[train_y==1],label='Survived')
fig4=sns.distplot(train_x[train_y==0],label='Not Survived' )
fig4.set_xlabel('Age');fig1.set_ylabel('Freqency')
fig4.set_title("Age")
fig4.legend()
plt.xlim(20)
plt.show()
#"child": (0,5] "teen": (5,18], "adult": (18,34] "mid": (34,50] "elder": (50,inf)

dataset['A_ran']=dataset.Age.map(lambda x: "child" if x<=5 else( "teen" if 10< x <=18 else ("adult" if 18<x<=34 else("mid" if 34<x<=50 else "elder"))))
train_x=dataset['A_ran'][:891]

fig5=plt.figure("Age_range")
plt.title("A_ran")
plt.xlabel("Age_ran");plt.ylabel("Counts")
plt.hist(train_x[train_y==1],label="Survived",alpha=0.4)
plt.hist(train_x[train_y==0],label="Not Survived",alpha=0.4)
plt.legend()
plt.show()

hist=pd.crosstab(train_x,train_y)
hist['ratio']=hist[1]/hist.sum(axis=1)

hist
EDA['A_rank']=dataset.A_ran.replace(['child','mid','teen','elder','adult'],[1,2,3,4,5])

EDA

dataset['Family_size']=dataset['SibSp']+dataset['Parch']+1

train_x=dataset['Family_size'][:891]
fig6=sns.distplot(train_x[train_y==1],label='Survived')
fig6=sns.distplot(train_x[train_y==0],label='Not Survived' )
fig6.set_xlabel('Family_size');fig1.set_ylabel('Freqency')
fig6.legend()
#plt.xlim(2,8)
plt.show()

dataset['F_ran']=dataset.Family_size.map(lambda x: "alone" if x==1 else( "small" if 2<= x <5 else "large"))

train_x=dataset['F_ran'][:891]
fig6=plt.figure("Family Size")
plt.title("Family size Range")
plt.xlabel("Family size Range");plt.ylabel("Counts")
plt.hist(train_x[train_y==1],label="Survived",alpha=0.4)
plt.hist(train_x[train_y==0],label="Not Survived",alpha=0.4)
plt.legend()
plt.show()

hist=pd.crosstab(train_x,train_y)
hist['ratio']=hist[1]/hist.sum(axis=1)

hist
EDA=pd.concat([EDA,pd.get_dummies(dataset['F_ran'],columns=['F_ran'] , prefix='F')],axis=1)

EDA
Ticket=[]
for tic in dataset.Ticket:
    if not tic.isdigit():
        Ticket.append(tic.replace(".","").replace("/","").strip().split(" ")[0])
    else:
        Ticket.append(None)
dataset['Tic_pre']=Ticket

dataset.Tic_pre
dataset['Ticket_Frequency'] = dataset.groupby('Ticket')['Ticket'].transform('count')

train_x=dataset['Ticket_Frequency'][:891]
fig6=sns.distplot(train_x[train_y==1],label='Survived')
fig6=sns.distplot(train_x[train_y==0],label='Not Survived' )
fig6.set_xlabel('Ticket_Frequency');fig1.set_ylabel('Freqency')
fig6.legend()
plt.ylim(0,0.7)
plt.show()

hist=pd.crosstab(train_x,train_y)
hist['ratio']=hist[1]/hist.sum(axis=1)


hist
EDA['Ticket_freq']=dataset['Ticket_Frequency']

EDA
print(dataset[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).median())
print(dataset[dataset["Fare"].isnull()])
dataset['Fare']=dataset['Fare'].fillna(8.0500)
print(dataset.iloc[891+152])
train_x=dataset['Fare'][:891]
fig7=sns.distplot(train_x[train_y==1],label='Survived')
fig7=sns.distplot(train_x[train_y==0],label='Not Survived' )
fig7.set_xlabel('Fare');fig1.set_ylabel('Freqency')
fig7.legend()
#plt.xlim(30)
plt.show()

print('Difference: ',train.Fare[train.Survived==1].mean()/train.Fare[train.Survived==0].mean()*100%100," %")
dataset["Fare_log"]=dataset.Fare.map(lambda x: np.log(x) if x != 0 else 0)

train_x=dataset["Fare_log"][:891]
fig8=sns.distplot(train_x[train_y==1],label='Survived')
fig8=sns.distplot(train_x[train_y==0],label='Not Survived' )
fig8.set_xlabel('Fare $[\log]$');fig1.set_ylabel('Freqency')
fig8.legend()
#plt.xlim(2,)
plt.show()
dataset['Fa_ran']=dataset.Fare_log.map(lambda x: "poor" if x<=1 else( "mid" if 1< x <=2.5 else "rich"))

train_x=dataset["Fa_ran"][:891]

fig8=plt.figure("Fare")
plt.title("Fare Range")
plt.xlabel("Fare Range");plt.ylabel("Counts")
plt.hist(train_x[train_y==1],label="Survived",alpha=0.4)
plt.hist(train_x[train_y==0],label="Not Survived",alpha=0.4)
plt.legend()
plt.show()

hist=pd.crosstab(train_x,train_y)
hist['ratio']=hist[1]/hist.sum(axis=1)

hist

EDA['Fa_rank']=dataset.Fa_ran.replace(['poor','mid','rich'],[1,2,3])

EDA
dataset['floor'] = dataset.Cabin.str[0]
dataset['floor'] = dataset.floor.fillna(0)
dataset['floor'] = dataset.floor.replace(['A', 'B', 'C', 'D','E', 'F','G','T'], [1,2,3,4,5,6,7,8])

dataset.head()
train_x=dataset["floor"][:891]

fig9=plt.figure("Floor")
plt.title("Floors")
plt.xlabel("Floors");plt.ylabel("Counts")
plt.hist(train_x[train_y==1],label="Survived",alpha=0.4)
plt.hist(train_x[train_y==0],label="Not Survived",alpha=0.4)
plt.ylim(0,40)
plt.legend()
plt.show()

hist=pd.crosstab(train_x,train_y)
hist['ratio']=hist[1]/hist.sum(axis=1)

hist
dataset['floor'] = dataset.floor.replace([1,2,3,8],'ABC')
dataset['floor'] = dataset.floor.replace([4,5],'DE')
dataset['floor'] = dataset.floor.replace([6,7],'FG')
dataset['floor'] = dataset.floor.replace([0],'X')

train_x=dataset["floor"][:891]

fig10=plt.figure("Floor")
plt.title("Floors")
plt.xlabel("Floors");plt.ylabel("Counts")
plt.hist(train_x[train_y==1],label="Survived",alpha=0.4)
plt.hist(train_x[train_y==0],label="Not Survived",alpha=0.4)
#plt.ylim(0,40)
plt.legend()
plt.show()

hist=pd.crosstab(train_x,train_y)
hist['ratio']=hist[1]/hist.sum(axis=1)

hist
EDA=pd.concat([EDA,pd.get_dummies(dataset['floor'],columns=['floor'] , prefix='Fl')],axis=1)

EDA
dataset['Embarked']=dataset.Embarked.fillna('S').astype(str)

train_x=dataset['Embarked'][:891]

fig10=plt.figure("Embarked")
plt.title("Embarked")
plt.xlabel("Embarked");plt.ylabel("Counts")
plt.hist(train_x[train_y==1],label="Survived",alpha=0.4)
plt.hist(train_x[train_y==0],label="Not Survived",alpha=0.4)
plt.legend()
plt.show()
hist=pd.crosstab(train_x,train_y)
hist['ratio']=hist[1]/hist.sum(axis=1)

hist

dataset['Embarked']=dataset.Embarked.fillna('S').astype(str)
EDA=pd.concat([EDA,pd.get_dummies(dataset['Embarked'],columns=['Embarked'] , prefix='E')],axis=1)

EDA
def extract_surname(data):    
    
    families = []
    
    for i in range(len(data)):        
        name = data.iloc[i]

        if '(' in name:
            name_no_bracket = name.split('(')[0] 
        else:
            name_no_bracket = name
            
        family = name_no_bracket.split(',')[0]
        title = name_no_bracket.split(',')[1].strip().split(' ')[0]
        
        for c in string.punctuation:
            family = family.replace(c, '').strip()
            
        families.append(family)
            
    return families
dataset['Family'] = extract_surname(dataset['Name'])
train_x = dataset.iloc[:891]
test_x = dataset.iloc[891:]
datas = [train_x, test_x]

dataset.Family
non_unique_families = [x for x in train_x['Family'].unique() if x in test_x['Family'].unique()]
non_unique_tickets = [x for x in train_x['Ticket'].unique() if x in test_x['Ticket'].unique()]

df_family_survival_rate = train_x.groupby('Family')['Survived', 'Family','Family_size'].median()
df_ticket_survival_rate = train_x.groupby('Ticket')['Survived', 'Ticket','Ticket_Frequency'].median()

family_rates = {}
ticket_rates = {}

for i in range(len(df_family_survival_rate)):
    # Checking a family exists in both training and test set, and has members more than 1
    if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:
        family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]

for i in range(len(df_ticket_survival_rate)):
    # Checking a ticket exists in both training and test set, and has members more than 1
    if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:
        ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]

mean_survival_rate = np.mean(train_x['Survived'])

train_family_survival_rate = []
train_family_survival_rate_NA = []
test_family_survival_rate = []
test_family_survival_rate_NA = []

for i in range(len(train_x)):
    if train_x['Family'][i] in family_rates:
        train_family_survival_rate.append(family_rates[train_x['Family'][i]])
        train_family_survival_rate_NA.append(1)
    else:
        train_family_survival_rate.append(mean_survival_rate)
        train_family_survival_rate_NA.append(0)
        
for i in range(len(test_x)):
    if test_x['Family'].iloc[i] in family_rates:
        test_family_survival_rate.append(family_rates[test_x['Family'].iloc[i]])
        test_family_survival_rate_NA.append(1)
    else:
        test_family_survival_rate.append(mean_survival_rate)
        test_family_survival_rate_NA.append(0)
        
train_x['Family_Survival_Rate'] = train_family_survival_rate
train_x['Family_Survival_Rate_NA'] = train_family_survival_rate_NA
test_x['Family_Survival_Rate'] = test_family_survival_rate
test_x['Family_Survival_Rate_NA'] = test_family_survival_rate_NA

train_ticket_survival_rate = []
train_ticket_survival_rate_NA = []
test_ticket_survival_rate = []
test_ticket_survival_rate_NA = []

for i in range(len(train_x)):
    if train_x['Ticket'][i] in ticket_rates:
        train_ticket_survival_rate.append(ticket_rates[train_x['Ticket'][i]])
        train_ticket_survival_rate_NA.append(1)
    else:
        train_ticket_survival_rate.append(mean_survival_rate)
        train_ticket_survival_rate_NA.append(0)
        
for i in range(len(test_x)):
    if test_x['Ticket'].iloc[i] in ticket_rates:
        test_ticket_survival_rate.append(ticket_rates[test_x['Ticket'].iloc[i]])
        test_ticket_survival_rate_NA.append(1)
    else:
        test_ticket_survival_rate.append(mean_survival_rate)
        test_ticket_survival_rate_NA.append(0)
        
train_x['Ticket_Survival_Rate'] = train_ticket_survival_rate
train_x['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA
test_x['Ticket_Survival_Rate'] = test_ticket_survival_rate
test_x['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA

for df in [train_x, test_x]:
    df['Survival_Rate'] = (df['Ticket_Survival_Rate'] + df['Family_Survival_Rate']) / 2
    df['Survival_Rate_NA'] = (df['Ticket_Survival_Rate_NA'] + df['Family_Survival_Rate_NA']) / 2    

    
EDA['Survival_Rate']=pd.concat([train_x['Survival_Rate'],test_x['Survival_Rate']])
EDA['Survival_Rate_NA']=pd.concat([train_x['Survival_Rate_NA'],test_x['Survival_Rate_NA']])

EDA
np.sum(EDA.isnull().any()) #null data check
EDA_train=EDA.iloc[:891,:]
EDA_test=EDA.iloc[891:,:]
PassengerIds=test['PassengerId']
print(np.shape(EDA_train),np.shape(EDA_test))
EDA_test
X_train, X_test, y_train, y_test = train_test_split(EDA_train,train_y, test_size=0.2,random_state=0)
rf=RandomForestClassifier(random_state=0)

rf_param_list = {"n_estimators": [500,1000,1500],
                 'criterion': ['gini', 'entropy'],
              "max_depth": [4, 6, 8, 10, None],
              "min_samples_leaf": [3,5,7,10],
                  'oob_score': [True],
                 'random_state':[0]}

rf_param_list2 = {"n_estimators": [1500],
                 'criterion': ['gini', 'entropy'],
              "max_depth": np.arange(0,10,2),
              "min_samples_leaf": np.arange(3,9,1),
                  'oob_score': [True],
                 'random_state':[0]}

rf_grid= GridSearchCV(estimator=rf, param_grid = rf_param_list,cv = 5, n_jobs=-1)
#rf_grid= RandomizedSearchCV(estimator=rf, param_distributions = rf_param_list2,n_iter=60,cv = 5, n_jobs=-1)
rf_grid.fit(X_train, y_train)

print("best parameters:",rf_grid.best_params_)
print('best score:', rf_grid.best_score_)

rf = rf_grid.best_estimator_
rf.fit(X_train,y_train)
result = rf.predict(EDA_test).astype(int)
output=pd.DataFrame({ 'PassengerId' : PassengerIds, 'Survived': result })
output.to_csv('rf_submission.csv', index=False)

tree=DecisionTreeClassifier(random_state=0)

tree_param_list = {'criterion':['gini','entropy'],
                    'max_depth':[2, 4, 6, 8, 10, None], 
                    'max_leaf_nodes':[None,2,3,4,5,6,7],
                    'min_samples_split':[1,2,3,4],
                    'min_samples_leaf':[3,5,7,10]}

tree_grid= GridSearchCV(estimator=tree, param_grid = tree_param_list,cv = 5, n_jobs=-1)
tree_grid.fit(X_train, y_train)

print("best parameters:",tree_grid.best_params_)
print('best score:', tree_grid.best_score_)

tree = tree_grid.best_estimator_
tree.fit(X_train,y_train)
result = tree.predict(EDA_test).astype(int)
output=pd.DataFrame({ 'PassengerId' : PassengerIds, 'Survived': result })
output.to_csv('tree_submission.csv', index=False)
ada=AdaBoostClassifier(random_state=13)

ada_param_list = {'n_estimators': [500,1000,1500],
            'learning_rate': [.01, .03, .05, .1, .25],
            'random_state':[0]}

ada_grid= GridSearchCV(estimator=ada, param_grid = ada_param_list,cv = 5, n_jobs=-1)
ada_grid.fit(X_train, y_train)

print("best parameters:",ada_grid.best_params_)
print('best score:', ada_grid.best_score_)

ada = ada_grid.best_estimator_
ada.fit(X_train,y_train)
result = ada.predict(EDA_test).astype(int)
output=pd.DataFrame({ 'PassengerId' : PassengerIds, 'Survived': result })
output.to_csv('ada_submission.csv', index=False)
svc=SVC()

svc_param_list = {'C': [0.5,0.75,1,2],
            'gamma':[ .5, .75, 1.0,1.25,1.5],
            'decision_function_shape': ['ovo', 'ovr'],
            'probability': [True],
            'random_state': [0]}

svc_grid= GridSearchCV(estimator=svc, param_grid = svc_param_list,cv = 5, n_jobs=-1)
svc_grid.fit(X_train, y_train)

print("best parameters:",svc_grid.best_params_)
print('best score:', svc_grid.best_score_)

svc = svc_grid.best_estimator_
svc.fit(X_train,y_train)
result = svc.predict(EDA_test).astype(int)
output=pd.DataFrame({ 'PassengerId' : PassengerIds, 'Survived': result })
output.to_csv('svc_submission.csv', index=False)
xgb=XGBClassifier()


xgb_param_list={ 'max_depth':[1,2,4,6,8,10],
                 'min_child_weight':[1,3,5],
                 'gamma':[0,1,2,3],
                 'learning_rate': [.01, .03, .05, .1, .25],
                 'seed':[0]}


xgb_grid= GridSearchCV(estimator=xgb, param_grid = xgb_param_list,cv = 5, n_jobs=-1)
xgb_grid.fit(X_train, y_train)

print("best parameters:",xgb_grid.best_params_)
print('best score:', xgb_grid.best_score_)

xgb = xgb_grid.best_estimator_
xgb.fit(X_train,y_train)
result = xgb.predict(EDA_test).astype(int)
output=pd.DataFrame({ 'PassengerId' : PassengerIds, 'Survived': result })
output.to_csv('xgb_submission.csv', index=False)
vote_clf=[('rf',rf),('tree',tree),('ada',ada),('svc',svc),('xgb',xgb)]
vote_soft = VotingClassifier(estimators = vote_clf , voting = 'soft',n_jobs=-1)
vote_soft_cv = cross_validate(vote_soft, X_train, y_train, cv  = 5)
vote_soft.fit(X_train,y_train)
y_preds=vote_soft.predict(X_test)

print(sum(y_preds==y_test)/len(y_preds)*100)
result = vote_soft.predict(EDA_test).astype(int)
output=pd.DataFrame({ 'PassengerId' : PassengerIds, 'Survived': result })
output.to_csv('voting_submission.csv', index=False)
ftr_importances_values = rf.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_top30 = ftr_importances.sort_values(ascending=False)[:30]

plt.figure(figsize=(8,6))
plt.title('Top 30 Feature Importances')
sns.barplot(x=ftr_top30, y=ftr_top30.index)
plt.show()
test_Survived_rb = pd.Series(rf.predict(EDA_test), name="rf")
test_Survived_tree = pd.Series(tree.predict(EDA_test), name="tree")
test_Survived_ada = pd.Series(ada.predict(EDA_test), name="ada")
test_Survived_svc = pd.Series(svc.predict(EDA_test), name="svc")
test_Survived_xgb = pd.Series(xgb.predict(EDA_test), name="xgb")
test_Survived_vote = pd.Series(vote_soft.predict(EDA_test), name="voting")

ensemble_results = pd.concat([test_Survived_rb,test_Survived_tree,test_Survived_ada,test_Survived_svc, test_Survived_xgb, test_Survived_vote],axis=1)
fig10= sns.heatmap(ensemble_results.corr(),annot=True)
