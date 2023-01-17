import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



#사소한 Error메시지 제거

import warnings

warnings.filterwarnings(action='ignore')
#훈련데이터와 테스트 데이터 셋을 불러오기

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#훈련데이터셋의 형태

train.head()
#훈련데이터셋의 컬럼의 종류와 속성 및 각 컬럼의 개수확인

train.info()
def check_null(dataset):

    df_null = pd.DataFrame(dataset.isnull().sum().sort_values(ascending=False)).reset_index()

    df_null.columns = ['var_name','count']

    return df_null[df_null['count']>0]
#훈련데이터 셋에 null값의 개수를 파악

check_null(train)
#테스트데이터 셋에 null값의 개수를 파악

check_null(test)
#PassengerId를 제외한 모든 컬럼과 Survived[Target value]와의 상관계를 heatmap화

fig = plt.figure()

fig,ax = plt.subplots(figsize=[12,12])

train_corr = train.drop('PassengerId',axis=1).dropna().corr()

sns.heatmap(train_corr,annot=True,square=True,vmax=0.6,cmap=plt.cm.summer)

plt.suptitle('Correlation Heatmap of Numeric Features',fontsize=18)
n=len(train_corr.columns)

train_corr.nlargest(n,columns='Survived')['Survived']
fig = plt.figure(figsize=[20,20])

ax = sns.pairplot(train.drop('PassengerId',axis=1).dropna(),palette={'red','blue'},hue='Survived')
fig = plt.figure()

fig,ax = plt.subplots(3,3,figsize=[15,15])



surv = train[train.Survived == 1]

nosurv = train[train.Survived == 0]



ax1 = plt.subplot(3,3,1)

ax1 = sns.distplot(surv.dropna().Age,bins=range(0,surv.dropna().Age.max().astype('int')+1,1),kde=False,color='blue',label='Survived')

ax1 = sns.distplot(nosurv.dropna().Age,bins=range(0,surv.dropna().Age.max().astype('int')+1,1),kde=False,color='red',label='Died')

ax1.set_ylabel('# of Passengers')

ax1.legend()



ax2 = plt.subplot(3,3,2)

ax2 = sns.barplot(x='Pclass',y='Survived',data=train)



ax3 = plt.subplot(3,3,3)

ax3 = sns.barplot(x='SibSp',y='Survived',data=train)



ax4 = plt.subplot(3,3,4)

ax4 = sns.barplot(x='Parch',y='Survived',data=train)



ax5 = plt.subplot(3,3,5)

ax5 = sns.distplot(surv.Fare,kde=False,color='blue',label='Survived')

ax5 = sns.distplot(nosurv.Fare,kde=False,color='red',label='Died')

ax5.set_ylabel('# of Passengers')

ax5.legend()



ax6 = plt.subplot(3,3,6)

ax6 = sns.barplot(x='Embarked',y='Survived',data=train)



ax7 = plt.subplot(3,3,7)

ax7 = sns.barplot(x='Sex',y='Survived',data=train)



plt.suptitle("Detail relationship btw Survived and various features",fontsize=18)
#2. 탑승칸의 등급과 Survived

fig = plt.figure()

fig,ax = plt.subplots(1,2,figsize=[15,5])

ax1 = plt.subplot(1,3,1)

ax1 = sns.barplot(x='Pclass',y='Survived',data=train)

ax2 = plt.subplot(1,3,2)

ax2 = sns.countplot(x='Pclass',data=train)

ax3 = plt.subplot(1,3,3)

ax3 = sns.countplot(x='Pclass',hue='Survived',data=train)



plt.suptitle('Detail about Pclass with Survived',fontsize=18)
#3,4 SibSp, Parch의 인원수와 Survived

tab = pd.crosstab(train['SibSp'],train['Survived'])

ax1 = tab.div(tab.sum(1),axis=0).plot(kind='bar',stacked=True)

title = ax1.title

title.set_position([.5,1.2])

ax1.set_title('Survivor Ratio by # of SibSp')

plt.suptitle("Survivor Ratio by # of SibSp and Parch")



tab = pd.crosstab(train['Parch'],train['Survived'])

ax2 = tab.div(tab.sum(1),axis=0).plot(kind='bar',stacked=True)

title = ax2.title

title.set_position([.5,1.1])

ax2.set_title('Survivor Ratio by # of Parch')
#돈을 많이낸 승객과 Survived지표

fig = plt.figure()

fig,ax = plt.subplots(1,3,figsize=[18,6])



ax1 = plt.subplot(1,3,1)

ax1 = sns.distplot(surv[surv.Pclass==1].Fare,color='blue',label='Survived',kde=False)

ax1 = sns.distplot(nosurv[nosurv.Pclass==1].Fare,color='red',label='Died',kde=False)

ax1.set_ylabel('# of Passengers')

ax1.set_title('Pclass 1')

ax1.legend()



ax2 = plt.subplot(1,3,2)

ax2 = sns.distplot(surv[surv.Pclass==2].Fare,color='blue',label='Survived',kde=False)

ax2 = sns.distplot(nosurv[nosurv.Pclass==2].Fare,color='red',label='Died',kde=False)

ax2.set_ylabel('# of Passengers')

ax2.set_title('Pclass 2')

ax2.legend()



ax3 = plt.subplot(1,3,3)

ax3 = sns.distplot(surv[surv.Pclass==3].Fare,color='blue',label='Survived',kde=False)

ax3 = sns.distplot(nosurv[nosurv.Pclass==3].Fare,color='red',label='Died',kde=False)

ax3.set_ylabel('# of Passengers')

ax3.set_title('Pclass 3')

ax3.legend()



plt.suptitle('Passenger Insight by paid Fare',fontsize=18)
fig = plt.figure()

fig,ax = plt.subplots(1,2,figsize=[10,5])



male = train[train.Sex=='male']

female = train[train.Sex=='female']



male_surv = male[male.Survived==1]

male_nosurv = male[male.Survived==0]

fem_surv = female[female.Survived==1]

fem_nosurv = female[female.Survived==0]



ax1 = plt.subplot(1,2,1)

ax1 = sns.distplot(male_surv.Age.dropna(),bins=range(0,surv.Age.dropna().max().astype('int')+1,1),kde=False,color='blue',label='Survived')

ax1 = sns.distplot(male_nosurv.Age.dropna(),bins=range(0,surv.Age.dropna().max().astype('int')+1,1),kde=False,color='red',label='Died')



ax1.set_ylabel('# of Passengers')

ax1.set_title('Age Distribution of Males by Survived')

ax1.legend()



ax2 = plt.subplot(1,2,2)

ax2 = sns.distplot(fem_surv.Age.dropna(),bins=range(0,nosurv.Age.dropna().max().astype('int')+1,1),kde=False,color='blue',label='Survived')

ax2 = sns.distplot(fem_nosurv.Age.dropna(),bins=range(0,nosurv.Age.dropna().max().astype('int')+1,1),kde=False,color='red',label='Died')



ax2.set_ylabel('# of Passengers')

ax2.set_title('Age Distribution of Females by Survived')

ax2.legend()



plt.suptitle('Age Distribution by Survived value',fontsize=18)
fig = plt.figure()

fig, ax = plt.subplots(1,2,figsize=[12,6])



#pclass와 탑승자의 연령과 Survived

ax1 = plt.subplot(1,2,1)

ax1 = sns.violinplot(x='Pclass',y='Age',hue='Survived',data=train,split=True)



#pclass와 Embarked와 Survived

ax2 = sns.factorplot(x='Pclass',y='Survived',hue='Sex',col='Embarked',data=train,split=True)
#Pclass별 생존자 현황

tab = pd.crosstab(train['Pclass'],train['Survived'])

tab.div(tab.sum(1),axis=0).plot(kind='bar',stacked=True)
#Pclass,Sex 그리고 Survived

sns.barplot(x='Pclass',y='Survived',hue='Sex',data=train)
tab = pd.crosstab(train['Embarked'],train['Sex'])

tab.div(tab.sum(1),axis=0).plot(kind='bar',stacked=True)
tab = pd.crosstab(train['Embarked'],train['Pclass'])

tab.div(tab.sum(1),axis=0).plot(kind='bar',stacked=True)
fig = plt.figure()

fig, ax = plt.subplots(1,2,figsize=[12,6])



ax1 = plt.subplot(1,2,1)

ax1 = sns.barplot(x='Embarked',y='Survived',hue='Pclass',data=train)



ax2 = plt.subplot(1,2,2)

ax2 = sns.violinplot(x='Embarked',y='Age',hue='Survived',data=train,split=True) 
check_null(train)
check_null(test)
train.Embarked.value_counts(ascending=False).idxmax() # S 664
train.loc[train.Embarked.isnull()].index
train.loc[train.Embarked.isnull(),'Embarked'] = 'S'
test.loc[test.Fare.isnull(),'Fare'] = test.Fare.median()
# combine = pd.concat([train,test],axis=0)



# combine_null = combine[combine.Age.isnull()]

# combine_notnull = combine[~combine.Age.isnull()]



# remove_cols = ['PassengerId','Survived','Name','Ticket','Cabin']

# new_cols = []

# for col in list(combine.columns):

#     if col in remove_cols:

#         continue

#     else:

#         new_cols.append(col)

# combine_null = combine_null.loc[:,new_cols]

# combine_notnull = combine_notnull.loc[:,new_cols]

# combine_null = combine_null.drop('Age',axis=1)

# X_combine_notnull = combine_notnull.drop('Age',axis=1)

# y_combine_notnull = combine_notnull['Age']
# for col in ['Sex','Embarked','Pclass']:

#     combine_null[col] = combine_null[col].astype('category')

#     X_combine_notnull[col] = X_combine_notnull[col].astype('category')



# for col in ['Fare','SibSp','Parch']:

#     if col == 'Fare':

#         combine_null[col] = pd.to_numeric(combine_null[col])

#         X_combine_notnull[col] = pd.to_numeric(X_combine_notnull[col])

#     else:

#         combine_null[col] = pd.to_numeric(combine_null[col],downcast='integer')

#         X_combine_notnull[col] = pd.to_numeric(X_combine_notnull[col],downcast='integer')
# w_o_Pclass = pd.concat([combine_null,pd.get_dummies(combine_null.Pclass)],axis=1).drop('Pclass',axis=1)

# w_o_Sex = pd.concat([w_o_Pclass,pd.get_dummies(combine_null.Sex)],axis=1).drop('Sex',axis=1)

# w_o_Emb = pd.concat([w_o_Sex,pd.get_dummies(combine_null.Embarked)],axis=1).drop('Embarked',axis=1)

# combine_null = w_o_Emb



# w_o_Pclass = pd.concat([X_combine_notnull,pd.get_dummies(combine_notnull.Pclass)],axis=1).drop('Pclass',axis=1)

# w_o_Sex = pd.concat([w_o_Pclass,pd.get_dummies(X_combine_notnull.Sex)],axis=1).drop('Sex',axis=1)

# w_o_Emb = pd.concat([w_o_Sex,pd.get_dummies(X_combine_notnull.Embarked)],axis=1).drop('Embarked',axis=1)

# X_combine_notnull = w_o_Emb
# from sklearn.ensemble import RandomForestRegressor

# rf = RandomForestRegressor(random_state=200)

# rf.fit(X_combine_notnull,y_combine_notnull)

# new_Age = []

# for element in rf.predict(combine_null):

#     new_Age.append(round(element,2))

# new_Age = np.array(new_Age)
# combine_null = combine[combine.Age.isnull()]

# combine_notnull = combine[~combine.Age.isnull()]

# combine_null['Age'] = new_Age

# combine = pd.concat([combine_notnull,combine_null],axis=0)

# combine.drop('Cabin',axis=1,inplace=True)
combine = pd.concat([train,test],axis=0)
combine.head()
ticket = (combine.Ticket.value_counts() > 1).reset_index()

sharedTicketList = ticket[ticket.Ticket == True]['index'].values



def SharedTicket(ticket):

    if ticket in sharedTicketList:

        return 1

    else: 

        return 0
combine['Alone'] = (combine.Parch + combine.SibSp == 0)

combine['Family'] = (combine.Parch + combine.SibSp)

combine['Large_family'] = (combine.Parch + combine.SibSp > 5)

combine['Title'] = combine.Name.apply(lambda x:x.split(',')[1].split('.')[0].strip())

combine['Child'] = combine.Age < 10

combine['Young'] = np.logical_or(combine.Age <= 30, combine.Title.isin(['Miss','Master','Mlle']))

combine['Fare_cat'] = pd.DataFrame(np.floor(np.log10(combine.Fare+1))).astype('int')

combine['SharedTicket'] = combine.Ticket.apply(SharedTicket)

combine['Age_known'] = ~combine.Age.isnull()

combine['Cabin_known'] = ~combine.Cabin.isnull()

combine['Deck']= combine.Cabin.str[0]

combine['Deck']= combine['Deck'].fillna(value='U')
combine.head()
neededToChange = ['Embarked','Sex','Deck']



for col in neededToChange:

    combine[col] = combine[col].astype('category')

    print(combine[col].cat.categories)



combine['Embarked'].cat.categories = range(0,3) # C:0,Q:1,S:2

combine['Embarked'] = combine['Embarked'].astype('int')

combine['Sex'].cat.categories = range(0,2) #female:

combine['Sex'] = combine['Sex'].astype('int')

combine['Deck'].cat.categories = range(0,9) #A:0,B:1,C:2,D:3,E:4,F:5,G:6,T:7,U:8

combine['Deck'] = combine['Deck'].astype('int')
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(np.float64(combine['Fare']).reshape((len(combine['Fare']),1)))

std_Scaled_Fare = scaler.transform(np.float64(combine['Fare']).reshape(len(combine['Fare']),1))
print(std_Scaled_Fare.min(),std_Scaled_Fare.max())
combine['std_Scaled_Fare'] = std_Scaled_Fare
combine.columns
combine = combine[['PassengerId','Age', 'Cabin', 'Embarked', 'Fare', 'Name', 'Parch', 

       'Pclass', 'Sex', 'SibSp', 'Ticket', 'Alone', 'Family',

       'Large_family', 'Title', 'Child', 'Young', 'Fare_cat', 'SharedTicket',

       'Age_known', 'Cabin_known', 'Deck', 'std_Scaled_Fare','Survived']]
train = combine[~combine.Survived.isnull()]

test = combine[combine.Survived.isnull()]
train.info()
test.info()
fig = plt.figure(figsize=[20,20])

train_corr = train.drop('PassengerId',axis=1).corr()

sns.heatmap(train_corr,square=True,annot=True,cmap=plt.cm.summer)
n=len(train_corr.columns)

df_train_corr = train_corr.nlargest(n,columns='Survived')['Survived'].reset_index()
df_train_corr
features = df_train_corr[abs(df_train_corr.Survived) >0.08]['index'].values

features = [feature for feature in features if feature!='Fare']

features
plt.figure(figsize=[15,15])

sns.heatmap(train.loc[:,features].corr(),annot=True,square=True)
features.append('PassengerId')



combine = combine.loc[:,features]

combine.head()
combine.columns
combine = combine[['PassengerId', 'Cabin_known', 'Fare_cat', 'std_Scaled_Fare',

       'SharedTicket', 'Young', 'Child', 'Age_known', 'Parch', 'Embarked',

       'Alone', 'Deck', 'Pclass', 'Sex', 'Survived']]
# from sklearn.preprocessing import LabelBinarizer



# binarizer_output={}



# neededToChange = ['Fare_cat','Parch','Pclass','Embarked','Deck']

# for col in neededToChange:

#     combine[col] = combine[col].astype('category')

#     binarizer = LabelBinarizer()

#     binarizer.fit(combine[col])

#     print(binarizer.classes_)

#     binarizer_output[col] = pd.DataFrame(binarizer.transform(combine[col]))
neededToChange = ['Fare_cat','Parch','Pclass','Embarked','Deck']



for col in neededToChange:

    combine[col] = combine[col].astype('category')

    binerized = pd.get_dummies(combine[col],prefix=col)

    

    combine = pd.concat([combine,binerized],axis=1)

    combine = combine.drop(col,axis=1)
combine.info()
# final_features = []



# for col in combine.columns:

#     if col in features:

#         final_features.append(col)
# features = features[-1::-1]

# combine = combine.loc[:,features]
# combine = combine.reset_index()

# combine.drop('index',axis=1,inplace=True)
# for col in neededToChange:

#     combine.drop(col,axis=1,inplace=True)

#     combine = pd.concat([combine,binarizer_output[col]],axis=1)
# combine['Embarked'] = combine['Embarked'].astype('category')

# combine.Embarked.cat.categories = [0,1,2] # C,Q,S

# combine['Embarked'] = pd.to_numeric(combine['Embarked'],downcast='integer')

# combine['Sex'] = combine['Sex'].astype('category')

# combine.Sex.cat.categories = [0,1] #female,male

# combine['Sex'] = pd.to_numeric(combine['Sex'],downcast='integer')
# combine = combine[['PassengerId', 'Age', 'Embarked', 'Fare', 'Name', 'Parch', 

#        'Pclass', 'Sex', 'SibSp', 'Ticket', 'Alone', 'Family', 'Large_family',

#        'Title', 'Child', 'Young', 'Fare_cat', 'SharedTicket','Survived']]
# train = combine.loc[~combine.Survived.isnull()]

# test = combine.loc[combine.Survived.isnull()]

# test.drop('Survived',axis=1,inplace=True)
# fig = plt.figure(figsize=[20,20])

# train_corr = train.drop('PassengerId',axis=1).corr()

# sns.heatmap(train_corr,square=True,annot=True,vmax=0.6,cmap=plt.cm.summer)
# df = train_corr.nlargest(n=15,columns='Survived')['Survived'].reset_index()

# df = df.drop(df.loc[(df['index'] == 'Survived')].index,axis=0)

# index = df['index']

# ratio = df['Survived']

# fig = plt.figure(figsize=[10,6])

# ax = sns.barplot(x=index,y=ratio)

# ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

# ax.set_title('Correlationship with Survived feature',fontsize=18)

# ax.set_xlabel('Features',fontsize=14)
# df_usingcol = (abs(train_corr.nlargest(n=15,columns='Survived')['Survived']) > 0.1).reset_index()

# usingcols = list(df_usingcol[df_usingcol.Survived==True]['index'].values)

# # usingcols.remove('SharedTicket')

# finalcols = []

# finalcols.append('PassengerId')

# for col in usingcols:

#     finalcols.append(col)
# combine = pd.concat([train,test],axis=0)

# combine = combine.loc[:,finalcols]
# # cat_cols = ['Fare_cat','Child','Young','Alone','Pclass','Sex','Embarked']

# cat_cols = ['Fare_cat','Pclass','Embarked']

# for col in cat_cols:

#     combine[col] = combine[col].astype('category')

#     combine = pd.concat([combine,pd.get_dummies(combine[col])],axis=1)

#     combine.drop(col,axis=1,inplace=True)
train = combine.loc[~combine.Survived.isnull()]

test = combine.loc[combine.Survived.isnull()]

test.drop('Survived',axis=1,inplace=True)
train['Survived'] = train['Survived'].astype('int')
train = train.sort_values(by='PassengerId')

test = test.sort_values(by='PassengerId')
from sklearn.model_selection import train_test_split

X = train.drop(['Survived','PassengerId'],axis=1)

y = train['Survived']

test = test.drop('PassengerId',axis=1)



X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold



SKFold = StratifiedKFold(n_splits=5)



knn = KNeighborsClassifier()

knn_params={'n_neighbors':range(1,11,1),'weights':['distance']}

grid_knn = GridSearchCV(knn,knn_params,scoring='accuracy',cv=SKFold)

grid_knn.fit(X_train,y_train)

grid_knn.score(X_test,y_test)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr_params = {'C':[0.001,0.01,0.1,1,10,100,1000]}

grid_lr = GridSearchCV(lr,lr_params,scoring='accuracy',cv=SKFold)

grid_lr.fit(X_train,y_train)

grid_lr.score(X_test,y_test)
from sklearn.svm import LinearSVC



lsvc = LinearSVC()

lsvc_params = {'C':[0.001,0.01,0.1,1,10,100,1000]}

grid_lsvc = GridSearchCV(lsvc,lsvc_params,scoring='accuracy',cv=SKFold)

grid_lsvc.fit(X_train,y_train)

grid_lsvc.score(X_test,y_test)
from sklearn.naive_bayes import BernoulliNB



nb = BernoulliNB()

nb_params = {'alpha':[0.001,0.01,0.1,1,10,100,1000]}

grid_nb = GridSearchCV(nb,nb_params,scoring='accuracy',cv=SKFold)

grid_nb.fit(X_train,y_train)

grid_nb.score(X_test,y_test)
from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier()

tree_params = {'max_depth':range(1,10,1)}

grid_tree = GridSearchCV(tree,tree_params,scoring='accuracy',cv=SKFold)

grid_tree.fit(X_train,y_train)

grid_tree.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(random_state=42)

rf.fit(X_train,y_train)

rf.score(X_test,y_test)
from sklearn.ensemble import GradientBoostingClassifier



gb = GradientBoostingClassifier(

 learning_rate= 0.1,

 max_depth= 5,

 max_features= 'sqrt',

 min_samples_leaf= 1,

 min_samples_split= 2,

 min_weight_fraction_leaf= 0.0,

 random_state= 12,

 subsample= 1.0

)

gb_params = {'n_estimators':range(20,100,20)}

grid_gb1 = GridSearchCV(gb,gb_params,scoring='accuracy',cv=5)

grid_gb1.fit(X_train,y_train)
grid_gb1.cv_results_,grid_gb1.best_params_,grid_gb1.best_score_
from sklearn.ensemble import GradientBoostingClassifier



gb = GradientBoostingClassifier(

 learning_rate= 0.1,

#  max_depth= 5,

 max_features= 'sqrt',

 min_samples_leaf= 1,

#  min_samples_split= 2,

 min_weight_fraction_leaf= 0.0,

 n_estimators= 60,

 random_state= 12,

 subsample= 1.0

)

gb_params = {'max_depth':range(3,15,3),'min_samples_split':range(10,101,20)}

grid_gb2 = GridSearchCV(gb,gb_params,scoring='accuracy',cv=5)

grid_gb2.fit(X_train,y_train)
grid_gb2.cv_results_,grid_gb2.best_params_,grid_gb2.best_score_
gb = GradientBoostingClassifier(

 learning_rate= 0.1,

 max_depth= 12,

 max_features= 'sqrt',

#  min_samples_leaf= 1,

 min_samples_split= 70,

 min_weight_fraction_leaf= 0.0,

 n_estimators= 60,

 random_state= 12,

 subsample= 1.0

)

gb_params = {'min_samples_leaf':range(1,20,3)}

grid_gb3 = GridSearchCV(gb,gb_params,scoring='accuracy',cv=5)

grid_gb3.fit(X_train,y_train)
grid_gb3.cv_results_,grid_gb3.best_params_,grid_gb3.best_score_
gb = GradientBoostingClassifier(

 learning_rate= 0.1,

 max_depth= 12,

#  max_features= 'sqrt',

 min_samples_leaf= 10,

 min_samples_split= 70,

 min_weight_fraction_leaf= 0.0,

 n_estimators= 60,

 random_state= 12,

 subsample= 1.0

)

gb_params = {'max_features':range(6,34,4)}

grid_gb4 = GridSearchCV(gb,gb_params,scoring='accuracy',cv=5)

grid_gb4.fit(X_train,y_train)
grid_gb4.cv_results_,grid_gb4.best_params_,grid_gb4.best_score_
gb = GradientBoostingClassifier(

 learning_rate= 0.1,

 max_depth= 12,

 max_features= 22,

 min_samples_leaf= 10,

 min_samples_split= 70,

 min_weight_fraction_leaf= 0.0,

 n_estimators= 60,

 random_state= 12,

#  subsample= 1.0

)

gb_params = {'subsample':[0.5,0.6,0.7,0.8,0.9,1.0]}

grid_gb5 = GridSearchCV(gb,gb_params,scoring='accuracy',cv=5)

grid_gb5.fit(X_train,y_train)
gb = GradientBoostingClassifier(

 learning_rate= 0.1,

 max_depth= 12,

 max_features= 22,

 min_samples_leaf= 10,

 min_samples_split= 70,

 min_weight_fraction_leaf= 0.0,

 n_estimators= 60,

 random_state= 12,

 subsample= 1.0

)

gb_params = {'subsample':[0.5,0.6,0.7,0.8,0.9,1.0]}

grid_gb5 = GridSearchCV(gb,gb_params,scoring='accuracy',cv=5)

grid_gb5.fit(X_train,y_train)
grid_gb5.cv_results_,grid_gb5.best_params_,grid_gb5.best_score_
from sklearn.metrics import accuracy_score



gb = GradientBoostingClassifier(

 learning_rate= 0.1,

 max_depth= 12,

 max_features= 22,

 min_samples_leaf= 10,

 min_samples_split= 70,

 min_weight_fraction_leaf= 0.0,

 n_estimators= 60,

 random_state= 12,

 subsample= 1.0

)

# gb_params = {'subsample':[0.5,0.6,0.7,0.8,0.9,1.0]}

# grid_gb5 = GridSearchCV(gb,gb_params,scoring='accuracy',cv=5)

gb.fit(X_train,y_train)

print(accuracy_score(y_train,gb.predict(X_train)),accuracy_score(y_test,gb.predict(X_test)))
from sklearn.metrics import accuracy_score



gb = GradientBoostingClassifier(

 learning_rate= 0.05,

 max_depth= 12,

 max_features= 22,

 min_samples_leaf= 10,

 min_samples_split= 70,

 min_weight_fraction_leaf= 0.0,

 n_estimators= 120,

 random_state= 12,

 subsample= 1.0

)

# gb_params = {'subsample':[0.5,0.6,0.7,0.8,0.9,1.0]}

# grid_gb5 = GridSearchCV(gb,gb_params,scoring='accuracy',cv=5)

gb.fit(X_train,y_train)

accuracy_score(y_train,gb.predict(X_train))

print(accuracy_score(y_train,gb.predict(X_train)),accuracy_score(y_test,gb.predict(X_test)))
from sklearn.metrics import accuracy_score



gb = GradientBoostingClassifier(

 learning_rate= 0.01,

 max_depth= 12,

 max_features= 22,

 min_samples_leaf= 10,

 min_samples_split= 70,

 min_weight_fraction_leaf= 0.0,

 n_estimators= 600,

 random_state= 12,

 subsample= 1.0

)

# gb_params = {'subsample':[0.5,0.6,0.7,0.8,0.9,1.0]}

# grid_gb5 = GridSearchCV(gb,gb_params,scoring='accuracy',cv=5)

gb.fit(X_train,y_train)

accuracy_score(y_train,gb.predict(X_train))

print(accuracy_score(y_train,gb.predict(X_train)),accuracy_score(y_test,gb.predict(X_test)))
from sklearn.metrics import accuracy_score



gb = GradientBoostingClassifier(

 learning_rate= 0.005,

 max_depth= 12,

 max_features= 22,

 min_samples_leaf= 10,

 min_samples_split= 70,

 min_weight_fraction_leaf= 0.0,

 n_estimators= 1200,

 random_state= 12,

 subsample= 1.0

)

# gb_params = {'subsample':[0.5,0.6,0.7,0.8,0.9,1.0]}

# grid_gb5 = GridSearchCV(gb,gb_params,scoring='accuracy',cv=5)

gb.fit(X_train,y_train)

accuracy_score(y_train,gb.predict(X_train))

print(accuracy_score(y_train,gb.predict(X_train)),accuracy_score(y_test,gb.predict(X_test)))
# from xgboost import XGBClassifier

# import xgboost





# # xgb.get_params()

# params = {'base_score': 0.5,

#  'booster': 'gbtree',

#  'colsample_bylevel': 0.8,

#  'colsample_bytree': 1,

#  'gamma': 0,

#  'learning_rate': 0.1,

#  'max_delta_step': 0,

#  'max_depth': 5,

#  'min_child_weight': 0.8,

#  'missing': None,

#  'n_estimators': 100,

#  'n_jobs': 1,

#  'nthread': None,

#  'objective': 'binary:logistic',

#  'random_state': 0,

#  'reg_alpha': 0,

#  'reg_lambda': 1,

#  'scale_pos_weight': 1,

#  'seed': None,

#  'silent': True,

#  'subsample': 0.8}



# # X_xgb_train = xgboost.DMatrix(X_train)

# XGBClassifier.fit(params,X_train,y_train,eval_metric='accuracy')
score = []

models = [grid_knn,grid_lr,grid_lsvc,grid_nb,grid_tree,rf,gb] 

for model in models:

    score.append(model.score(X_test,y_test))

cols = ['knn','LogisticRegression','Linear Support Vector Machine','Naive Bayes','DecisionTree','RandomForest','GradientBoost']
fig = plt.figure(figsize=[10,10])

ax = sns.barplot(x=cols,y=score)

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)



height = [p.get_height() for p in ax.patches]

# Looping through bars

for i, p in enumerate(ax.patches):    

    # Adding percentages

    ax.text(p.get_x()+p.get_width()/2, height[i]*1.01,

            '{:1.1%}'.format(height[i]), ha="center", size=14) 
X = X.values
# X_predict = np.zeros(len(X))

test_predict = np.array(len(test))
X_test.head()
First = True



for i,(train_level,valid_level) in enumerate(SKFold.split(X,y)):

    X_train, X_valid = X[train_level,:], X[valid_level,:]

    y_train, y_valid = y[train_level], y[valid_level]

    

    gb.fit(X_train,y_train)

    train_set = gb.predict(X_train)

    valid_set = gb.predict(X_valid)

    test_set = gb.predict(test)

    

    if First:

        print("#"*30,i+1)

        test_predict = pd.Series(test_set)

        First=False

    else:

        print("#"*30,i+1)

        test_predict += pd.Series(test_set)
def decision(row):

    if row < 3:

        return 0

    else:

        return 1
test_predict = test_predict.apply(decision)
# from sklearn.ensemble import VotingClassifier



# estimators = [('lsvc',grid_lsvc),('d_tree',grid_tree),('gb',grid_gb)]

# voting = VotingClassifier(estimators=estimators,voting='hard')

# voting.fit(X_train,y_train)

# voting.score(X_test,y_test)
# test.drop('PassengerId',axis=1,inplace=True)

# predicts = grid_gb.predict(test).astype('int')
submission = pd.read_csv('../input/sample_submission.csv')

submission['Survived'] = test_predict

submission.to_csv('../working/submit.csv',index=False)