import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline

dataset_exp = pd.read_csv('../input/train.csv')
dataset_exp.head()
dataset_exp.drop('PassengerId',axis = 1, inplace = True)
dataset_exp.info()
#Embarked missing value : only 2 missing, use mode here, but in the later prerprocessing, more detail will be considered.

Embark_mode = dataset_exp['Embarked'].mode()[0]

dataset_exp['Embarked'].fillna(value = Embark_mode, inplace = True)
# transfer categorical variables to numericial

dataset_exp['Sex'][dataset_exp['Sex']=='male']=1

dataset_exp['Sex'][dataset_exp['Sex']=='female']=0

dataset_exp['Sex'] = dataset_exp['Sex'].astype(np.int64)



# for age, use Random Forest to predict missing value first

age_set = dataset_exp[['Age','Survived','Pclass','Sex','SibSp','Parch','Fare']]

X_train = age_set[age_set['Age'].notnull()].iloc[:,1:].values

X_test = age_set[age_set['Age'].isnull()].iloc[:,1:].values

Y_train =age_set[age_set['Age'].notnull()].iloc[:,0].values



from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators= 1000,random_state = 0)

regressor.fit(X_train,Y_train)



Y_pred = regressor.predict(X_test)

Y_pred = np.round(Y_pred,0)

dataset_exp.loc[dataset_exp['Age'].isnull(),'Age'] = Y_pred
# we can see lots of missing value here, but how relevant is it? Should we discard this variable?

dataset_exp['Cabin'][dataset_exp['Cabin'].isnull()]='U'
dataset_exp.info()
# Among all the people, the proportion of sex:

#Total population Comparison

total = dataset_exp['Sex']

total = total.value_counts()



# Among Survivors

sex_sur = dataset_exp['Sex'][dataset_exp['Survived']==1]

sex_sur[sex_sur==0] ='female'

sex_sur[sex_sur==1] ='male'

sex_sur = sex_sur.value_counts()



fig,ax = plt.subplots(1,2,figsize = (8,5))

label0 = ['male','female']

label1 = ['female','male']

fig.suptitle('Whole Population')

ax[0].pie(total,autopct='%1.1f%%',labels = label0,colors = ['blue','orange'],startangle = 90,shadow=True)

ax[0].set_title('All people')

ax[1].pie(sex_sur,autopct='%1.1f%%',labels = label1,colors = ['orange','blue'],startangle = -90,shadow=True)

ax[1].set_title('Survivors')

plt.show()

pc = dataset_exp[['Pclass','Survived']].groupby('Pclass').mean()

pc.reset_index(inplace = True)

bar_width = 0.15

ind = np.arange(0.2,1.19,1/3)

plt.figure(figsize = (6,6))

plt.bar(ind,pc['Survived'],width = bar_width) 

plt.xticks(ind,label = pc['Pclass'])

for i in range(0,len(pc['Pclass'])):

    plt.text(ind[i],pc['Survived'][i]+0.008,str(round(100*pc['Survived'][i],2))+'%',ha = 'center',size = 9,)

plt.title('Survival Rate of different Pclass')

plt.xlabel('Pclass')

plt.ylabel('Survival Rate')

plt.show()
facet = sns.FacetGrid(data = dataset_exp,hue = 'Survived',size = 6)

facet.map(sns.kdeplot,'Age',shade = True)

facet.set(xlim=(0,max(dataset_exp['Age'][dataset_exp['Age'].notnull()])))

facet.add_legend()

plt.show()



# average survival rate by age

dataset_age = dataset_exp.copy()

dataset_age['Age'] = dataset_age['Age'].apply(lambda x:round(x,2))



dataset_age['Age'] = dataset_age['Age'].apply(str)

age_ave = dataset_age[['Age','Survived']].groupby('Age').mean()

age_ave.reset_index(inplace = True)

age_ave['Age'] = age_ave['Age'].apply(float)

age_ave.sort_values('Age',inplace = True)

age_ave['Age'] = age_ave['Age'].apply(int)

dataset_age['Age'] = dataset_age['Age'].apply(str)



plt.figure(figsize = (20,6))

sns.barplot('Age','Survived',data = age_ave,ci=1)

plt.title('average survival rate by age')

plt.show()
fig,ax = plt.subplots(1,2,figsize = (12,6))

fig.suptitle('Survival distribution along Age')



sns.violinplot('Pclass','Age',hue = 'Survived',data = dataset_exp, split = True,ax = ax[0])

ax[0].set_title('Age ~ Pclass')

ax[0].set_yticks(range(0,100,20))

 

sns.violinplot('Sex','Age',hue = 'Survived',data = dataset_exp, split = True,ax = ax[1])

ax[1].set_title('Age ~ Sex')

ax[1].set_yticks(range(0,100,20))

plt.show()



#Then compare the shape with population Age distribution

fig,ax = plt.subplots(1,2,figsize = (12,6))

fig.suptitle('Population Age Distribution')

sns.distplot(dataset_exp['Age'],bins = 30, hist = True, kde = True,ax = ax[0],color = 'blue')



sns.boxplot(y='Age',data = dataset_exp,ax=ax[1], width = 0.4)

plt.show()

# Same distribution 
dataset_exp['Child'] = 'None'

dataset_exp['Age'] = dataset_exp['Age'].apply(np.float64)

dataset_exp['Child'][dataset_exp['Age']>15]='adult'

dataset_exp['Child'][dataset_exp['Age']<=15]='child'

sns.set({'figure.figsize':(6,6)})

sns.barplot('Child','Survived',data  = dataset_exp)

plt.show()
# Sib/sp and Par/ch

sibsp = dataset_exp[['SibSp','Survived']]

sibsp['SibSp'] = sibsp['SibSp'].apply(str)

sibsp_sur = sibsp.groupby('SibSp').mean()

sibsp_sur.reset_index(inplace = True)



parch= dataset_exp[['Parch','Survived']]

parch['Parch'] = parch['Parch'].apply(str)

parch_sur = parch.groupby('Parch').mean()

parch_sur.reset_index(inplace = True)



# create new variable: family size = sum of these two

dataset_exp['FamilySize'] = dataset_exp['SibSp']+dataset_exp['Parch']+1

fam= dataset_exp[['FamilySize','Survived']]

fam['FamilySize'] = fam['FamilySize'].apply(str)

fam_sur = fam.groupby('FamilySize').mean()

fam_sur.reset_index(inplace = True)

fam_sur['FamilySize'] = fam_sur['FamilySize'].apply(int)

fam_sur.sort_values('FamilySize',inplace = True)

fam_sur['FamilySize'] = fam_sur['FamilySize'].apply(str)

fam_sur.reset_index(inplace = True)

fam_sur.drop('index',axis=1,inplace = True)



# plot

fig,ax = plt.subplots(1,3,figsize = (9,5))

fig.suptitle('Survival Rate ~ Family members')

sns.barplot('SibSp','Survived',data = sibsp_sur,ax=ax[0],ci=0,color = 'blue')

ax[0].set_xlabel('Number of siblings/spouse')

ax[0].set_ylabel('Survive Rate')



sns.barplot('Parch','Survived',data = parch_sur,ax=ax[1],ci=0,color = 'blue')

ax[1].set_xlabel('Number of Parent/children')

ax[1].set_ylabel(' ')



sns.barplot('FamilySize','Survived',data = fam_sur,ax=ax[2],ci=0,color = 'blue',order =fam_sur['FamilySize'])

ax[2].set_xlabel('FamilySize')

ax[2].set_ylabel(' ')

plt.show()

dataset_exp['FamilySize'][dataset_exp['FamilySize']>7]=0

dataset_exp['FamilySize'][(dataset_exp['FamilySize']==1)|((dataset_exp['FamilySize']>=5)&(dataset_exp['FamilySize']<=7))]=1

dataset_exp['FamilySize'][(dataset_exp['FamilySize']>=2)&(dataset_exp['FamilySize']<=4)]=2

dataset_exp['FamilySize'] = dataset_exp['FamilySize'].apply(str)

sns.barplot('FamilySize','Survived',data = dataset_exp)

plt.show()
# General Distribution

fare = dataset_exp['Fare']

sns.distplot(fare,bins=70,color = 'blue')

plt.show()



# Compare the average fare between differnet pclass

fare_pc = dataset_exp[['Fare','Pclass']]

plt.figure(figsize = (5,10))

sns.boxplot('Pclass','Fare',data = fare_pc)

plt.ylim([-50,300])

plt.yticks(range(0,300,100))

plt.show()
# compare the mean fare of survivors and victims

fare_surv = dataset_exp[dataset_exp['Survived']==1]['Fare'].mean()

fare_vict = dataset_exp[dataset_exp['Survived']==0]['Fare'].mean()

ind = [0.5,0.7]

fare_y = [fare_surv,fare_vict]

plt.bar(ind,fare_y,width = 0.1)

plt.xticks(ind,['Survivors','Victims'])

for i in range(0,2):

    plt.text(ind[i],fare_y[i]+1,round(fare_y[i],2),size =10,ha = 'center')

plt.ylabel('Average Fare')

plt.ylim([0,60])

plt.title('Average fare of survivors and victims')

plt.show()



facet_fare = sns.FacetGrid(data = dataset_exp,hue = 'Survived',aspect = 2)

facet_fare.map(sns.kdeplot,'Fare',shade = True)

facet_fare.set(xlim = (0,max(dataset_exp['Fare'])))

plt.legend(labels=['die','survive'])

plt.title('Survival~Fare')

plt.show()
dataset_exp['Fare'][dataset_exp['Fare']<50]=0

dataset_exp['Fare'][(dataset_exp['Fare']>=50)&(dataset_exp['Fare']<74)]=1

dataset_exp['Fare'][(dataset_exp['Fare']>=74)&(dataset_exp['Fare']<300)]=2

dataset_exp['Fare'][dataset_exp['Fare']>=300]=3



sns.set({'figure.figsize':(8,6)})

sns.barplot('Fare','Survived',data = dataset_exp)

plt.show()
#Embarked: surv/vict for each port

sns.countplot('Embarked',hue = 'Survived', data = dataset_exp)

plt.title('Survivals for each port')

plt.xlabel('Port Embarked')

plt.show()



sns.pointplot(x = 'Embarked',y = 'Survived',data = dataset_exp)

plt.title('Survivals Rate of each port')

plt.xlabel('Port Embarked')

plt.ylabel('Survival Rate')

plt.show()

# port C has the highest survival rate

dataset_exp['Cabin'].fillna('U',inplace = True)
dataset_exp['Has_Cabin'] = dataset_exp['Cabin'].apply(lambda x : 0 if x== 'U' else 1)

dataset_exp['Has_Cabin'] = dataset_exp['Has_Cabin'].apply(str)

sns.barplot(x ='Has_Cabin',y = 'Survived',data = dataset_exp)

plt.show()
dataset_exp['Cabin_code'] = dataset_exp['Cabin'].str.get(0)

sns.barplot(x ='Cabin_code',y='Survived',data = dataset_exp,estimator=np.mean)

plt.show()
pattern = re.compile('\w+, (.+?)[.]')

dataset_exp['Title'] = 'None'

for i in range(0,len(dataset_exp['Name'])):

    try:    

        dataset_exp['Title'][i] = re.search(pattern,list(dataset_exp['Name'].values)[i]).group(1) 

    except:

        continue       

Title_dict = dict()

Title_dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'],'Officer'))

Title_dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'],'Royalty'))

Title_dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'],'Mrs'))

Title_dict.update(dict.fromkeys(['Mlle', 'Miss'],'Miss'))

Title_dict.update(dict.fromkeys(['Mr'],'Mr'))

Title_dict.update(dict.fromkeys(['Master','Jonkheer'],'Master'))

dataset_exp['Title'] = dataset_exp['Title'].map(Title_dict)
sns.barplot('Title','Survived',data = dataset_exp)

plt.show()
ticket_counts = dataset_exp['Ticket'].value_counts()

ticket_unique =list(dataset_exp['Ticket'].unique())

ticket_dict = dict()

for i in range(0,len(ticket_unique)):

    ticket_dict.update(dict.fromkeys([ticket_unique[i]],ticket_counts.loc[ticket_unique[i],]))

dataset_exp['Ticket'] = dataset_exp['Ticket'].map(ticket_dict).apply(str) 

sns.barplot(x ='Ticket',y='Survived',data = dataset_exp)

plt.show()
# divide Tickets into two classes

dataset_exp['Ticket_nums'] = dataset_exp['Ticket'].apply(int)   

dataset_exp['Ticket_nums'][(dataset_exp['Ticket_nums']==1)|(dataset_exp['Ticket_nums']>=5)]=0 #1,5,6,7

dataset_exp['Ticket_nums'][(dataset_exp['Ticket_nums']>=2)&(dataset_exp['Ticket_nums']<=4)]=1 #2,3,4

dataset_exp.info()
# concatenate train and test and preprocess them together- to have uniform format

train_set = pd.read_csv('../input/train.csv')

train_set_y =train_set.iloc[:,1] 

train_set = train_set.drop('Survived',axis=1)

test_set = pd.read_csv('../input/test.csv')

dataset = train_set.append(test_set)



#reset index

dataset.reset_index(inplace = True)

dataset.drop('index',axis=1,inplace = True)

dataset.drop('PassengerId',axis = 1, inplace = True)
#Embarked missing value : looking into observations with missing Embarked port

emb_missing = dataset[dataset['Embarked'].isnull()]

# these two observations share variables: Pclass and Ticket, so use these variables to speculate Embarked port:

# Tickets starts with 113:

emb_missing_pattern = '^113.+'

emb_missing_ticket = dataset['Ticket'].apply(lambda x:bool(re.search(emb_missing_pattern,x)))

ticket_113 = dataset[emb_missing_ticket]  #all observations with tickets starting with 113

ticket_113['Ticket'] = ticket_113['Ticket'].apply(int)

ticket_113.sort_values('Ticket',inplace = True)

ticket_113 = ticket_113[['Ticket','Embarked']] 

ticket_113['Embarked'].value_counts()

# 52 S vs 10 C, and around 1135XX, most of them are S, so here fill missing values with S

dataset['Embarked'].fillna('C',inplace = True)

    

# Name : extract the title from name using regex

pattern = re.compile('\w+, (.+?)[.]')

dataset['Title'] = 'None'

for i in range(0,len(dataset['Name'])):

    try:    

        dataset['Title'][i] = re.search(pattern,list(dataset['Name'].values)[i]).group(1) 

    except:

        continue

    

dataset.drop('Name',axis=1,inplace = True)



# Title: Unify to English format and implement dummy encoding

title_dict = dict()

title_dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))

title_dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))

title_dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))

title_dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))

title_dict.update(dict.fromkeys(['Mr'], 'Mr'))

title_dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))

dataset['Title'] = dataset['Title'].map(title_dict)



dataset['Title'][dataset['Title'].isin(('Mr','Officer'))] = 0

dataset['Title'][dataset['Title'] =='Master'] = 1

dataset['Title'][dataset['Title'].isin(('Mrs','Miss','Royalty'))]=2

dataset['Title'] = dataset['Title'].apply(int)





# FamilySize = Sibsp + Parch + 1, further classified as 0(familysize>7),1(familysize=1 or 5,6,7) ,2(2<=familysize<=4)

dataset['FamilySize'] = dataset['SibSp']+dataset['Parch']+1

dataset['FamilySize'][dataset['FamilySize']>7]=0

dataset['FamilySize'][(dataset['FamilySize']==1)|((dataset['FamilySize']>=5)&(dataset['FamilySize']<=7))]=1

dataset['FamilySize'][(dataset['FamilySize']>=2)&(dataset['FamilySize']<=4)]=2

dataset.drop(['SibSp','Parch'],axis=1,inplace = True)



# Fare has 1 missing value - look into the dataset:

missing_fare_ob = dataset[dataset['Fare'].isnull()] 

#this person's ticket is 3701, then check the similar ticket number's fare pattern:

ticket_37_pattern = r'^3\d\d\d$'

ticket_37 = dataset[dataset['Ticket'].apply(lambda x:bool(re.search(ticket_37_pattern,x)))]

# all the fare are similar, it's reasonable to use mean to fill

mean_fare = np.mean(ticket_37[ticket_37['Fare'].notnull()]['Fare'])

dataset['Fare'].fillna(mean_fare,inplace = True)



# Cabin

#two groups: 0(Unknown),1(known)

dataset['Cabin'][dataset['Cabin'].isnull()] ='U'

dataset['Cabin']=dataset['Cabin'].str.get(0)

dataset['Cabin'] = dataset['Cabin'].apply(lambda x: 0 if x=='U' else 2 if x in ('E','D','B') else 1)





#Ticket: same pattern as family size, classify them into groups

ticket_counts = dataset['Ticket'].value_counts()

ticket_unique =list(dataset['Ticket'].unique())

ticket_dict = dict()

for k in range(0,len(ticket_unique)):

    ticket_dict.update(dict.fromkeys([ticket_unique[k]],ticket_counts.loc[ticket_unique[k]]))

dataset['Ticket'] = dataset['Ticket'].map(ticket_dict)

dataset['Ticket_nums'] = dataset['Ticket']

# divide Tickets into three classes 

dataset['Ticket_nums'][(dataset['Ticket_nums']==5)|(dataset['Ticket_nums']==6)|(dataset['Ticket_nums']>=8)]=0 # 5,6,more than 8 -- group 0

dataset['Ticket_nums'][(dataset['Ticket_nums']==1)|(dataset['Ticket_nums']==7)]=1 #1,7 --group 1

dataset['Ticket_nums'][(dataset['Ticket_nums']>=2)&(dataset['Ticket_nums']<=5)]=2 #2,3,4 --group 2

dataset.drop('Ticket',axis=1,inplace = True)
#use Sex,Title,Pclass to predict Age

age_set = dataset[['Age','Pclass','Title','Sex']]



# preprocessing: Pclass Sex and Title -> dummy variables

age_set = pd.get_dummies(age_set)



age_train_x = age_set[age_set['Age'].notnull()].iloc[:,1:]

age_train_y = age_set[age_set['Age'].notnull()].iloc[:,0]

age_test_x = age_set[age_set['Age'].isnull()].iloc[:,1:]



 #Random Forest

 #Grid Search to find optimal parameters

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

#use grid search to find optimal parameters

'''

regressor_age_rf = RandomForestRegressor()

age_rf_params = [{'n_estimators':[100,500,1000,2000],'max_depth':[4,5,6,7],'min_samples_split':[6]}]

age_rf_grid_search = GridSearchCV(estimator=regressor_age_rf,param_grid = age_rf_params,scoring = 'neg_mean_squared_error',cv=10,n_jobs=-1)

age_rf_gs = age_rf_grid_search.fit(age_train_x,age_train_y)

age_rf_gs.best_params_

age_rf_gs.best_score_

# best params:{'max_depth': 6, 'min_samples_split': 7, 'n_estimators': 100}

# best score:-151.83

'''

#these parameters end up to be best

regressor_age_rf = RandomForestRegressor(n_estimators = 100,max_depth = 6, min_samples_split = 5, random_state=0)

regressor_age_rf.fit(age_train_x,age_train_y)

'''

features = list(age_train_x.columns)

importance = list(regressor_age_rf.feature_importances_)

fea_imp = {'feature':features,'importance':importance}

imp = pd.DataFrame(data =fea_imp )

imp.sort_values('importance',inplace = True,ascending = False)

sns.barplot(x = 'feature',y='importance',data = imp)

plt.show()

'''

age_pred_y = regressor_age_rf.predict(age_test_x)

dataset['Age_RF']=dataset['Age']

dataset['Age_RF'][dataset['Age'].isnull()] = age_pred_y

 



#Gradient Boosting

#Grid Search to find optimal parameters



from sklearn.ensemble import GradientBoostingRegressor

'''

regressor_gb = GradientBoostingRegressor()

gb_params = [{'n_estimators':[100,1000,2000],'max_depth':[3],'min_samples_split':[2,3,4]}]

gb_grid_search = GridSearchCV(estimator=regressor_gb,param_grid = gb_params,scoring = 'neg_mean_squared_error',cv=10,n_jobs=-1)

gb_gs = gb_grid_search.fit(age_train_x,age_train_y)

gb_gs.best_params_

gb_gs.best_score_'''

# best params: {'max_depth': 3, 'min_samples_split': 3, 'n_estimators': 100}

# best score: -141.40474456016958

regressor_gb = GradientBoostingRegressor(n_estimators=100,max_depth=3, min_samples_split=3,random_state=0)

regressor_gb.fit(age_train_x,age_train_y)

'''

#Feature Selection

regressor_gb.feature_importances_

features = list(age_train_x.columns)

importance = list(regressor_gb.feature_importances_)

fea_imp = {'feature':features,'importance':importance}

imp = pd.DataFrame(data =fea_imp )

imp.sort_values('importance',inplace = True,ascending = False)

sns.barplot(x = 'feature',y='importance',data = imp)

plt.show()

'''

age_pred_y_gb = regressor_gb.predict(age_test_x)

dataset['Age_GB']=dataset['Age']

dataset['Age_GB'][dataset['Age'].isnull()] = age_pred_y_gb



# Merging two models ---mean

dataset['Age'] =(dataset['Age_GB']+dataset['Age_RF'])/2

dataset.drop(['Age_GB','Age_RF'],axis=1,inplace = True)



# divide people into child and adult(>15) groups

dataset['Child']='None'

dataset['Child'][dataset['Age']<=15]='Child'

dataset['Child'][dataset['Age']>15]='Adult'

dataset.drop('Age',axis=1,inplace = True)
dataset.head()
# dummy encoding

dataset = pd.get_dummies(dataset)

dataset.head()
# test_train split

train_set_x = dataset.iloc[:891,:]

test_set_x = dataset.iloc[891:,:]

#train_set_y has been defined at the beginning



X_train = train_set_x.values

y_train = train_set_y.values

X_test = test_set_x.values
from sklearn.pipeline import Pipeline,make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.feature_selection import f_classif

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold
'''

# K-best and Grid Search for RF

pipe=Pipeline([('kb',SelectKBest(f_classif,k='all')), 

               ('rf', RandomForestClassifier(random_state = 10, max_features = 'sqrt'))])



param_test = {'kb__k':[9],

              'kb__score_func':[chi2],

              'rf__n_estimators':[554], 

              'rf__max_depth':[7],

              'rf__min_samples_split':[2],

              'rf__min_samples_leaf':[2],

              'rf__criterion':['entropy']

              }





gs_rf = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='roc_auc', cv=10,n_jobs=-1)

gs_rf.fit(X_train,y_train)

print(gs_rf.best_params_, gs_rf.best_score_)

'''
from sklearn.model_selection import learning_curve

kb_rf = SelectKBest(chi2,k = 9)

clf_rf = RandomForestClassifier(random_state = 10, 

                                  n_estimators = 554,

                                  max_depth = 7, 

                                  criterion= 'entropy',

                                  min_samples_leaf = 2,

                                  min_samples_split = 2,

                                  max_features = 'sqrt')

pipeline_rf = make_pipeline(kb_rf, clf_rf)

train_sizes,train_scores,test_scores = learning_curve(pipeline_rf,X_train,y_train,cv=10,train_sizes = np.linspace(0.1,1.0,5),n_jobs=-1)

train_scores_mean = np.mean(train_scores,axis=1)

test_scores_mean = np.mean(test_scores,axis=1)

plt.plot(train_sizes,train_scores_mean,color = 'red',marker='.',markersize = 20,linewidth = 3,label='training curve')

plt.plot(train_sizes,test_scores_mean,color = 'green',marker='.',markersize = 20,linewidth = 3,label = 'valication curve')

plt.ylim([0.65,1])

plt.legend()

plt.title('Random Forest Learning Curve')

plt.show()
# RF predict 

kb_rf = SelectKBest(chi2,k = 9)

clf_rf = RandomForestClassifier(random_state = 10, 

                                  n_estimators = 554,

                                  max_depth = 7, 

                                  criterion= 'entropy',

                                  min_samples_leaf = 2,

                                  min_samples_split = 2,

                                  max_features = 'sqrt')

rf_pipeline = make_pipeline(kb_rf, clf_rf)
from sklearn.model_selection import learning_curve

kb_et = SelectKBest(chi2,k = 10)

clf_et = ExtraTreesClassifier(random_state = 10, 

                                  n_estimators = 941,

                                  max_depth = 7, 

                                  criterion= 'entropy',

                                  min_samples_leaf = 2,

                                  min_samples_split = 2,

                                  max_features = 'sqrt')

               

pipeline_et = make_pipeline(kb_et, clf_et)

train_sizes,train_scores,test_scores = learning_curve(pipeline_et,X_train,y_train,cv=10,train_sizes = np.linspace(0.1,1.0,5),n_jobs=-1)

train_scores_mean = np.mean(train_scores,axis=1)

test_scores_mean = np.mean(test_scores,axis=1)

plt.plot(train_sizes,train_scores_mean,color = 'red',marker='.',markersize = 20,linewidth = 3,label='training curve')

plt.plot(train_sizes,test_scores_mean,color = 'green',marker='.',markersize = 20,linewidth = 3,label = 'valication curve')

plt.ylim([0.65,1])

plt.legend()

plt.title('Extra Tree Learning Curve')

plt.show()
# ET predict 

kb_et = SelectKBest(chi2,k = 10)

clf_et = ExtraTreesClassifier(random_state = 10, 

                                  n_estimators = 941,

                                  max_depth = 7, 

                                  criterion= 'entropy',

                                  min_samples_leaf = 2,

                                  min_samples_split = 2,

                                  max_features = 'sqrt')

et_pipeline = make_pipeline(kb_et, clf_et)
# Feature scaling:Fare,Age,FamilySize

dataset_scaled = dataset.copy()

s_fare = StandardScaler()

fare_scaled = s_fare.fit_transform(dataset_scaled['Fare'].values.reshape(-1,1))

dataset_scaled['Fare'] = fare_scaled



s_pc = StandardScaler()

pc_scaled = s_pc.fit_transform(dataset_scaled['Pclass'].values.reshape(-1,1))

dataset_scaled['Pclass'] = pc_scaled



s_cb = StandardScaler()

cb_scaled = s_cb.fit_transform(dataset_scaled['Cabin'].values.reshape(-1,1))

dataset_scaled['Cabin'] = cb_scaled



s_tt = StandardScaler()

tt_scaled = s_tt.fit_transform(dataset_scaled['Title'].values.reshape(-1,1))

dataset_scaled['Title'] = tt_scaled



s_fs = StandardScaler()

fs_scaled = s_fs.fit_transform(dataset_scaled['FamilySize'].values.reshape(-1,1))

dataset_scaled['FamilySize'] = fs_scaled





s_tn = StandardScaler()

tn_scaled = s_tn.fit_transform(dataset_scaled['Ticket_nums'].values.reshape(-1,1))

dataset_scaled['Ticket_nums'] = tn_scaled



# scaled datasets

train_set_x_s = dataset_scaled.iloc[:891,:]

test_set_x_s = dataset_scaled.iloc[891:,:]

 

# train/test set after scaling

X_train_s = train_set_x_s.values

y_train_s = train_set_y.values

X_test_s = test_set_x_s.values
from sklearn.model_selection import learning_curve

kb_svm = SelectKBest(f_classif,k=12)

clf_svm = SVC(C = 1.41, gamma = 0.1, kernel = 'rbf')

 

pipeline_svm = make_pipeline(kb_svm, clf_svm)

train_sizes,train_scores,test_scores = learning_curve(pipeline_svm,X_train_s,y_train_s,cv=10,train_sizes = np.linspace(0.1,1.0,5),n_jobs=-1)

train_scores_mean = np.mean(train_scores,axis=1)

test_scores_mean = np.mean(test_scores,axis=1)

plt.plot(train_sizes,train_scores_mean,color = 'red',marker='.',markersize = 20,linewidth = 3,label='training curve')

plt.plot(train_sizes,test_scores_mean,color = 'green',marker='.',markersize = 20,linewidth = 3,label = 'valication curve')

plt.ylim([0.65,1])

plt.legend()

plt.title('SVM Learning Curve')

plt.show()
# SVM predict 

kb_svm = SelectKBest(f_classif,k=12)

clf_svm = SVC(C = 1.41, gamma = 0.1, kernel = 'rbf')

svm_pipeline = make_pipeline(kb_svm, clf_svm)
#Stacking: 

 #LEVEL 1:  bagging RF+ET+SVM

  # k-fold on un-scaled training set

def get_out_of_kfold(model,fold,train_x,train_y,test_x):

   

    kfold_uns = KFold(n_splits = fold,shuffle = False)

    train_pred_results =np.array([])

    test_pred_results =np.zeros((X_test.shape[0],))

    for kf_data in kfold_uns.split(train_x):

        train_index = kf_data[0]

        test_index = kf_data[1]

        # training set(folds-1),test set(1)

        kf_train_x = train_x[train_index]

        kf_train_y = train_y[train_index]

        kf_test = train_x[test_index]

       

        model.fit(kf_train_x,kf_train_y)

        train_pred_results = np.hstack((train_pred_results,model.predict(kf_test)))

        test_pred_results =test_pred_results + model.predict(test_x) 

    test_pred_results = np.round(test_pred_results/fold) 

    result_list = list()

    result_list.append(train_pred_results)

    result_list.append(test_pred_results)

    return result_list
NFold = 9  # 9 fold

rf_list = get_out_of_kfold(rf_pipeline,NFold,X_train,y_train,X_test)

et_list = get_out_of_kfold(et_pipeline,NFold,X_train,y_train,X_test)

#for svm, use scaled dataset

svm_list = get_out_of_kfold(svm_pipeline,NFold,X_train_s,y_train_s,X_test_s)



# Create train/test set for level2

bagging_dict_train = {'RF':rf_list[0],'ET':et_list[0],'SVM':svm_list[0]}

X_train_lv1 =pd.DataFrame(bagging_dict_train)

y_train_lv1 = y_train



bagging_dict_test = {'RF':rf_list[1],'ET':et_list[1],'SVM':svm_list[1]}

X_test_lv2 = pd.DataFrame(bagging_dict_test)

# meta classifier 

# K-Nearest Neighbors

# Learning curve on selected model



from sklearn.model_selection import learning_curve

kb_knn = SelectKBest(f_classif,k='all' )

clf_knn = KNeighborsClassifier(algorithm= 'auto', n_neighbors=28, weights='uniform')

 

pipeline_knn = make_pipeline(kb_knn, clf_knn)

train_sizes,train_scores,test_scores = learning_curve(pipeline_knn,X_train_lv1,y_train_lv1,cv=10,train_sizes = np.linspace(0.1,1.0,5),n_jobs=-1)

train_scores_mean = np.mean(train_scores,axis=1)

test_scores_mean = np.mean(test_scores,axis=1)

plt.plot(train_sizes,train_scores_mean,color = 'red',marker='.',markersize = 20,linewidth = 3,label='training curve')

plt.plot(train_sizes,test_scores_mean,color = 'green',marker='.',markersize = 20,linewidth = 3,label = 'valication curve')

plt.ylim([0.7,1])

plt.legend()

plt.title('K-NN Learning Curve')

plt.show()

# K-NN final prediction

kb_knn = SelectKBest(f_classif,k='all' )

clf_knn = KNeighborsClassifier(algorithm= 'auto', n_neighbors=28, weights='uniform')

pipeline_knn = make_pipeline(kb_knn, clf_knn)

pipeline_knn.fit(X_train_lv1,y_train_lv1)

knn_pred = pipeline_knn.predict(X_test_lv2)
test_set_x.reset_index(inplace = True)

test_set_x.drop('index',axis = 1, inplace = True)

test_set_x['KNN_pred']=knn_pred

# final result output

final_result = pd.DataFrame()

final_result['PassengerId'] = test_set['PassengerId']

final_result['Survived'] = test_set_x['KNN_pred']

#final_result.to_csv('Thoffy-submission-knn.csv')
