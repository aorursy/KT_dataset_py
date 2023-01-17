import math, time, random, datetime
# 기본 데이터 정리 및 처리

import pandas as pd

import numpy as np



# 시각화

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('seaborn-whitegrid')

import missingno



# 전처리 및 머신 러닝 알고리즘

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import VotingClassifier



# 모델 튜닝 및 평가

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_predict

from sklearn import model_selection



# 경고 제거

import sys

import warnings



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
# 병합 준비

ntrain = train.shape[0]

ntest = test.shape[0]



# 아래는 따로 잘 모셔 둡니다.

y_train = train['Survived'].values

passId = test['PassengerId']



# 병함 파일 만들기

data = pd.concat((train, test))



# 데이터 행과 열의 크기는

print("data size is: {}".format(data.shape))
data.head()
train.head()
data.describe()
data.columns
data.columns[3]
missingno.matrix(data, figsize = (15,8))
data.isnull().sum() #비어 있는 값들을 체크해 본다.
data.Age.isnull().any()
data.dtypes
train.head()
# Co-relation 매트릭스

corr = data.corr()

# 마스크 셋업

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# 그래프 셋업

plt.figure(figsize=(14, 8))

# 그래프 타이틀

plt.title('Overall Correlation of Titanic Features', fontsize=18)

#  Co-relation 매트릭스 런칭

sns.heatmap(corr, mask=mask, annot=False,cmap='RdYlGn', linewidths=0.2, annot_kws={'size':20})

plt.show()
fig = plt.figure(figsize=(10,2))

sns.countplot(y='Survived', data=train)

print(train.Survived.value_counts())
f,ax=plt.subplots(1,2,figsize=(15,6))

train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=train,ax=ax[1])

ax[1].set_title('Survived')

plt.show()
def piecount(a):

    f,ax=plt.subplots(1,2,figsize=(15,6))

    train[a].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

    ax[0].set_title(a)

    ax[0].set_ylabel('')

    sns.countplot(a,data=train,ax=ax[1])

    ax[1].set_title(a)

    plt.show()



piecount('Survived')
def piecount3(a):

      f,ax=plt.subplots(1,2,figsize=(15,6))

      train[a].value_counts().plot.pie(explode=[0,0.0,0],autopct='%0.4f%%',ax=ax[1],shadow=True)

      ax[1].set_title(a)

      ax[1].set_ylabel('')

      sns.countplot(a,data=train,ax=ax[0])

      ax[0].set_title(a)

      plt.show()



piecount3("Pclass")
train.groupby(['Pclass','Survived'])['Survived'].count()
pd.crosstab(train.Pclass,train.Survived,margins=True).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(12,6))

train[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived per Pcalss')

sns.countplot('Pclass',hue='Survived',data=train,ax=ax[1])

ax[1].set_title('Pcalss Survived vs Not Survived')

plt.show()
piecount3("Pclass")
train.groupby('Pclass').Survived.mean()
data.Name.value_counts()
temp = data.copy()

temp['Initial']=0

for i in train:

    temp['Initial']=data.Name.str.extract('([A-Za-z]+)\.')
pd.crosstab(temp.Initial,temp.Sex).T.style.background_gradient(cmap='summer_r')
def survpct(a):

  return temp.groupby(a).Survived.mean()



survpct('Initial')
pd.crosstab(temp.Initial,temp.Survived).T.style.background_gradient(cmap='summer_r')
def bag(a,b,c,d):

  f,ax=plt.subplots(1,2,figsize=(20,8))

  train[[a,b]].groupby([a]).mean().plot.bar(ax=ax[0])

  ax[0].set_title(c)

  sns.countplot(a,hue=b,data=train,ax=ax[1])

  ax[1].set_title(d)

  plt.show()



bag('Sex','Survived','Survived per Sex','Sex Survived vs Not Survived')  
pd.crosstab([train.Sex,train.Survived],train.Pclass,margins=True).style.background_gradient(cmap='summer_r')
print('Oldest Passenger was ',data['Age'].max(),'Years')

print('Youngest Passenger was ',data['Age'].min(),'Years')

print('Average Age on the ship was ',int(data['Age'].mean()),'Years')
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Pclass","Age", hue="Survived", data=train,split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=train,split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
data.head()
temp.groupby('Initial')['Age'].mean() #이니셜 별 평균 연령 체크
temp['Newage']=temp['Age']



def newage(k,n):

  temp.loc[(temp.Age.isnull())&(temp.Initial==k),'Newage']= n

      

newage('Capt',int(70.000000))

newage('Col',int(54.000000))

newage('Countess',int(33.000000))

newage('Don',int(40.000000))

newage('Dona',int(39.000000))

newage('Dr',int(43.571429))

newage('Jonkheer',int(38.000000))

newage('Lady',int(48.000000))

newage('Major',int(48.500000))

newage('Master',int(5.482642))

newage('Miss',int(21.774238))

newage('Mlle',int(24.000000))

newage('Mme',int(24.000000))

newage('Mr',int(32.252151))

newage('Mrs',int(36.994118))

newage('Ms',int(28.000000))

newage('Rev',int(41.250000))

newage('Sir',int(49.000000))



temp['Age'][70:80]
temp['Newage'][70:80]
survpct('Newage')

pd.crosstab(temp.Newage,temp.Survived,margins=True).style.background_gradient(cmap='summer_r')
temp['Age_Range']=pd.qcut(temp['Newage'],10)

def groupmean(a,b):

  return temp.groupby([a])[b].mean().to_frame().style.background_gradient(cmap='summer_r')



groupmean('Age_Range', 'Newage')
groupmean('Age_Range', 'Survived')
temp.head(10)
temp['Gender']= temp['Sex']



for n in range(1,4):

  temp.loc[(temp['Sex'] == 'male') & (temp['Pclass'] == n),'Gender']= 'm'+str(n)

  temp.loc[(temp['Sex'] == 'female') & (temp['Pclass'] == n),'Gender']= 'w'+str(n)



temp.loc[(temp['Gender'] == 'm3'),'Gender']= 'm2'

temp.loc[(temp['Gender'] == 'w3'),'Gender']= 'w2'

temp.loc[(temp['Age'] <= 1.0),'Gender']= 'baby'

temp.loc[(temp['Age'] > 75.0),'Gender']= 'old'



groupmean('Gender', 'Survived')
temp['Agroup']=0



temp.loc[temp['Newage']<1.0,'Agroup']= 1

temp.loc[(temp['Newage']>=1.0)&(temp['Newage']<=3.0),'Agroup']= 2

temp.loc[(temp['Newage']>3.0)&(temp['Newage']<11.0),'Agroup']= 7

temp.loc[(temp['Newage']>=11.0)&(temp['Newage']<15.0),'Agroup']= 13

temp.loc[(temp['Newage']>=15.0)&(temp['Newage']<18.0),'Agroup']= 16

temp.loc[(temp['Newage']>=18.0)&(temp['Newage']<= 20.0),'Agroup']= 18

temp.loc[(temp['Newage']> 20.0)&(temp['Newage']<=22.0),'Agroup']= 21

temp.loc[(temp['Newage']>22.0)&(temp['Newage']<=26.0),'Agroup']= 24

temp.loc[(temp['Newage']>26.0)&(temp['Newage']<=30.0),'Agroup']= 28

temp.loc[(temp['Newage']>30.0)&(temp['Newage']<=32.0),'Agroup']= 31

temp.loc[(temp['Newage']>32.0)&(temp['Newage']<=34.0),'Agroup']= 33

temp.loc[(temp['Newage']>34.0)&(temp['Newage']<=38.0),'Agroup']= 36

temp.loc[(temp['Newage']>38.0)&(temp['Newage']<=52.0),'Agroup']= 45

temp.loc[(temp['Newage']>52.0)&(temp['Newage']<=75.0),'Agroup']= 60

temp.loc[temp['Newage']>75.0,'Agroup']= 78
groupmean('Agroup', 'Survived')
groupmean('Agroup', 'Age')
temp.head()
temp['Alone']=0



temp.loc[(temp['SibSp']==0)& (temp['Parch']==0),'Alone']= 1
temp.head(n=10)
temp['Family']=0



for i in temp:

  temp['Family'] = temp['Parch'] + temp['SibSp'] +1
temp.head(20)
survpct('Family')
bag('Parch','Survived','Survived per Parch','Parch Survived vs Not Survived') 
pd.crosstab([temp.Family,temp.Survived],temp.Pclass,margins=True).style.background_gradient(cmap='summer_r')
temp.Ticket.head(n=20)
temp.Ticket.isnull().any()
temp['Initick'] = 0

for s in temp:

    temp['Initick']=temp.Ticket.str.extract('^([A-Za-z]+)')

for s in temp:

    temp.loc[(temp.Initick.isnull()),'Initick']='X'

temp.head()
temp.groupby(['Initick'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
def groupmean(a,b):

  return temp.groupby([a])[b].mean().to_frame().style.background_gradient(cmap='summer_r')



groupmean('Initick', 'Survived')
pd.crosstab([temp.Pclass,temp.Survived],temp.Initick == 'X',margins=True).style.background_gradient(cmap='summer_r')
train['Tgroup'] = 0



temp['Tgroup'] = 0



temp.loc[(temp['Initick']=='X')& (temp['Pclass']==1),'Tgroup']= 1

temp.loc[(temp['Initick']=='X')& (temp['Pclass']==2),'Tgroup']= 2

temp.loc[(temp['Initick']=='X')& (temp['Pclass']==3),'Tgroup']= 3

temp.loc[(temp['Initick']=='Fa'),'Tgroup']= 3

temp.loc[(temp['Initick']=='SCO'),'Tgroup']= 4

temp.loc[(temp['Initick']=='A'),'Tgroup']= 5

temp.loc[(temp['Initick']=='CA'),'Tgroup']= 6

temp.loc[(temp['Initick']=='W'),'Tgroup']= 7

temp.loc[(temp['Initick']=='S'),'Tgroup']= 8

temp.loc[(temp['Initick']=='SOTON'),'Tgroup']= 9

temp.loc[(temp['Initick']=='LINE'),'Tgroup']= 10

temp.loc[(temp['Initick']=='STON'),'Tgroup']= 11

temp.loc[(temp['Initick']=='C'),'Tgroup']= 12

temp.loc[(temp['Initick']=='P'),'Tgroup']= 13

temp.loc[(temp['Initick']=='WE'),'Tgroup']= 14

temp.loc[(temp['Initick']=='SC'),'Tgroup']= 15

temp.loc[(temp['Initick']=='F'),'Tgroup']= 16

temp.loc[(temp['Initick']=='PP'),'Tgroup']= 17

temp.loc[(temp['Initick']=='PC'),'Tgroup']= 17

temp.loc[(temp['Initick']=='SO'),'Tgroup']= 18

temp.loc[(temp['Initick']=='SW'),'Tgroup']= 19

groupmean('Tgroup', 'Survived')
print('Highest Fare was:',temp['Fare'].max())

print('Lowest Fare was:',temp['Fare'].min())

print('Average Fare was:',temp['Fare'].mean())
f,ax=plt.subplots(1,3,figsize=(20,8))

sns.distplot(train[train['Pclass']==1].Fare,ax=ax[0])

ax[0].set_title('Fares in Pclass 1')

sns.distplot(train[train['Pclass']==2].Fare,ax=ax[1])

ax[1].set_title('Fares in Pclass 2')

sns.distplot(train[train['Pclass']==3].Fare,ax=ax[2])

ax[2].set_title('Fares in Pclass 3')

plt.show()
temp['Fare_Range']=pd.qcut(train['Fare'],10)

groupmean('Fare_Range', 'Fare')
temp['Fgroup']=0



temp.loc[temp['Fare']<= 7.125,'Fgroup']=5.0

temp.loc[(temp['Fare']>7.125)&(temp['Fare']<=7.9),'Fgroup']= 7.5

temp.loc[(temp['Fare']>7.9)&(temp['Fare']<=8.03),'Fgroup']= 8.0

temp.loc[(temp['Fare']>8.03)&(temp['Fare']<10.5),'Fgroup']= 9.5

temp.loc[(temp['Fare']>=10.5)&(temp['Fare']<23.0),'Fgroup']= 16.0

temp.loc[(temp['Fare']>=23.0)&(temp['Fare']<=27.8),'Fgroup']= 25.5

temp.loc[(temp['Fare']>27.8)&(temp['Fare']<=51.0),'Fgroup']= 38.0

temp.loc[(temp['Fare']>51.0)&(temp['Fare']<=73.5),'Fgroup']= 62.0

temp.loc[temp['Fare']>73.5,'Fgroup']= 100.0
temp.head()
temp.Cabin.value_counts()
temp.Cabin.isnull().sum()
temp['Inicab'] = 0

for i in temp:

    temp['Inicab']=temp.Cabin.str.extract('^([A-Za-z]+)')

    temp.loc[((temp.Cabin.isnull()) & (temp.Pclass.values == 1 )),'Inicab']='X'

    temp.loc[((temp.Cabin.isnull()) & (temp.Pclass.values == 2 )),'Inicab']='Y'

    temp.loc[((temp.Cabin.isnull()) & (temp.Pclass.values == 3 )),'Inicab']='Z'

temp.head(n=20)
temp.Inicab.value_counts()
temp['Inicab'].replace(['A','B', 'C', 'D', 'E', 'F', 'G','T', 'X', 'Y', 'Z'],[1,2,3,4,5,6,7,8,9,10,11],inplace=True)
temp.head()
pd.crosstab([temp.Embarked,temp.Pclass],[temp.Sex,temp.Survived],margins=True).style.background_gradient(cmap='summer_r')

sns.factorplot('Embarked','Survived',data=temp)

fig=plt.gcf()

fig.set_size_inches(5,3)

plt.show()
f,ax=plt.subplots(2,2,figsize=(20,15))

sns.countplot('Embarked',data=temp,ax=ax[0,0])

ax[0,0].set_title('No. Of Passengers Boarded')

sns.countplot('Embarked',hue='Sex',data=temp,ax=ax[0,1])

ax[0,1].set_title('Male-Female Split for Embarked')

sns.countplot('Embarked',hue='Survived',data=temp,ax=ax[1,0])

ax[1,0].set_title('Embarked vs Survived')

sns.countplot('Embarked',hue='Pclass',data=temp,ax=ax[1,1])

ax[1,1].set_title('Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2,hspace=0.5)

plt.show()
temp.loc[(temp.Embarked.isnull())]
temp.loc[(temp.Ticket == '113572')]
temp.sort_values(['Ticket'], ascending = True)[35:45]
temp.loc[(train.Embarked.isnull()),'Embarked']='S'
temp.sort_values(['Ticket'], ascending = True)[35:45]
temp.head()
temp.groupby('Initial').Survived.mean()
temp['Initial'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dona' , 'Dr', 'Jonkheer', 'Lady', 'Major', 'Master',  'Miss'  ,'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev', 'Sir'],[1, 2, 3, 4, 5, 6, 4, 3, 2, 8, 9, 3, 3, 4, 5, 3, 1, 3 ],inplace=True)
temp.groupby('Initial').Survived.mean()
temp.groupby('Embarked').Survived.mean()
temp["Embarked"].replace(['C','Q', 'S'], [1,2,3], inplace =True )
temp["Gender"].replace(['baby','m1', 'm2', 'old', 'w1', 'w2'], [1,2,3,4,5,6], inplace =True )
temp.head()
missingno.matrix(temp, figsize = (10,5))
df = pd.DataFrame()
def sub(a,b):

  df[a]=temp[b]



sub('Pclass', 'Pclass')

sub('Name', 'Initial')

sub('Sex', 'Gender')

sub('Age', 'Agroup')

sub('Alone', 'Alone')

sub('Family', 'Family')

sub('Ticket', 'Tgroup')

sub('Fare', 'Fgroup')

sub('Cabin', 'Inicab')

sub('Embarked', 'Embarked')





df.head()
df.isnull().sum()
len(df)
df.head()
score = df.copy()
len(score)
score['Survived'] = temp['Survived']
score['Score'] = 0
def see(a):

  return score.groupby(a).Survived.mean()



see('Pclass')
see('Name')
see('Sex')
see('Age')
see('Family')
see('Ticket')
see('Fare')
see('Cabin')
see('Embarked')
score['Class'] = 0

score['CE'] = 0

score['CN'] = 0

score['CP'] = 0



for i in score:

   score.loc[((score.Embarked.values == 1 )),'CE']=1

   score.loc[((score.Name.values == 2 )),'CN']=1

   score.loc[((score.Name.values == 3 )),'CN']=5

   score.loc[((score.Pclass.values == 1 )),'Class']=1

   score.loc[((score.Pclass.values == 3 )),'Class']=-1





score['Class'] = score['CE'] + score['CN'] + score['CP']



score.head(3)
score['Wealth'] = 0

score['WC'] = 0

score['WF'] = 0

score['WT'] = 0



for i in score:

   score.loc[((score.Cabin.values == 8 )),'WC']=-5

   score.loc[((score.Cabin.values == 11 )),'WC']=-1

   score.loc[((score.Cabin.values == 3 )),'WC']=1

   score.loc[((score.Cabin.values == 6 )),'WC']=1

   score.loc[((score.Cabin.values == 7 )),'WC']=1

   score.loc[((score.Cabin.values == 2 )),'WC']=3

   score.loc[((score.Cabin.values == 4 )),'WC']=3

   score.loc[((score.Cabin.values == 5 )),'WC']=3

   score.loc[((score.Fare.values <= 5 )),'WF']=-5

   score.loc[((score.Fare.values == 9.5 )),'WF']=-3

   score.loc[((score.Fare.values == 7.5 )),'WF']=-1

   score.loc[((score.Fare.values == 62 )),'WF']=1

   score.loc[((score.Fare.values >= 100 )),'WF']=3

   score.loc[((score.Ticket.values >= 4 ) & (score.Ticket.values <= 7 )),'WT']=-5

   score.loc[((score.Ticket.values >= 8 ) & (score.Ticket.values <= 9 )),'WT']=-3

   score.loc[((score.Ticket.values == 10 )),'WT']=-1

   score.loc[((score.Ticket.values == 1 )),'WT']=1

   score.loc[((score.Ticket.values >= 13 ) & (score.Ticket.values <= 17 )),'WT']=1

   score.loc[((score.Ticket.values >= 18)),'WT']=5





score['Wealth'] = score['WC'] + score['WF'] + score['WT'] 



score.head(20)
score['Priority'] = 0

score['PA'] = 0

score['PN'] = 0

score['PS'] = 0



for i in score:

   score.loc[((score.Age.values == 1 )),'PA']=5

   score.loc[((score.Age.values == 13 )),'PA']=1

   score.loc[((score.Age.values == 2 )),'PA']=1

   score.loc[((score.Age.values == 31 )),'PA']=-1

   score.loc[((score.Age.values == 7 )),'PA']=1

   score.loc[((score.Age.values == 78 )),'PA']=5

   score.loc[((score.Name.values == 4 )),'PN']=-1

   score.loc[((score.Name.values == 5 )),'PN']=3

   score.loc[((score.Name.values == 8)),'PN']=1

   score.loc[((score.Name.values == 9 )),'PN']=1

   score.loc[((score.Sex.values == 1 )),'PS']=3

   score.loc[((score.Sex.values == 3 )),'PS']=-3

   score.loc[((score.Sex.values == 4 )),'PS']=5

   score.loc[((score.Sex.values == 5 )),'PS']=5

   score.loc[((score.Sex.values >= 6)),'PS']=1





score['Priority'] = score['PA'] + score['PN'] + score['PS'] 



score.head(3)
score['Situation'] = 0

score['SA'] = 0

score['SF'] = 0





for i in score:

   score.loc[((score.Age.values == 36 )),'SA']=1

   score.loc[((score.Family.values == 2 )),'SF']=1

   score.loc[((score.Family.values == 3 )),'SF']=1

   score.loc[((score.Family.values == 4)),'SF']=3





score['Situation'] = score['SA'] + score['SF'] 



score.head(20)
score['Sacrificed'] = 0

score['SN'] = 0

score['FS'] = 0





for i in score:

   score.loc[((score.Name.values == 1 )),'SN']=-5

   score.loc[((score.Family.values == 5 )),'FS']=-1

   score.loc[((score.Family.values == 6 )),'FS']=-3

   score.loc[((score.Family.values == 8)),'FS']=-5

   score.loc[((score.Family.values >= 9)),'FS']=-5



score['Sacrificed'] = score['SN'] + score['FS'] 



score.head(3)
score['Score'] = score['Class'] + score['Wealth'] + score['Priority'] + score['Situation']  + score['Sacrificed'] 
df_new = pd.DataFrame()
def ch(a):

  df_new[a] = score[a]



ch('Pclass')

ch('Name')

ch('Sex')

ch('Age')

ch('Embarked')

ch('Cabin')

ch('Score')

ch('Class')

ch('Wealth')

ch('Priority')

ch('Situation')

ch('Sacrificed')



df_new.head()
len(df_new)
df_enc = df_new.apply(LabelEncoder().fit_transform)

                          

df_enc.head()
train = df_enc[:ntrain]

test = df_enc[ntrain:]
X_test = test

X_train = train
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train
ran = RandomForestClassifier(random_state=1)

knn = KNeighborsClassifier()

log = LogisticRegression()

xgb = XGBClassifier()

gbc = GradientBoostingClassifier()

svc = SVC(probability=True)

ext = ExtraTreesClassifier()

ada = AdaBoostClassifier()

gnb = GaussianNB()

gpc = GaussianProcessClassifier()

bag = BaggingClassifier()



# Prepare lists

models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         

scores = []



# Sequentially fit and cross validate all models

for mod in models:

    mod.fit(X_train, y_train)

    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 10)

    scores.append(acc.mean())
# 결과 테이블을 만듭니다.

results = pd.DataFrame({

    'Model': ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression', 'XGBoost', 'Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier'],

    'Score': scores})



result_df = results.sort_values(by='Score', ascending=False).reset_index(drop=True)

result_df.head(11)
# Plot results

sns.barplot(x='Score', y = 'Model', data = result_df, color = 'c')

plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')

plt.xlim(0.80, 0.84)
# 중요도를 보는 함수를 만듭니다.

def importance_plotting(data, x, y, palette, title):

    sns.set(style="whitegrid")

    ft = sns.PairGrid(data, y_vars=y, x_vars=x, size=5, aspect=1.5)

    ft.map(sns.stripplot, orient='h', palette=palette, edgecolor="black", size=15)

    

    for ax, title in zip(ft.axes.flat, titles):

    # 각 그래프마다 새로운 타이틀을 줍니다.

        ax.set(title=title)

    # 그래프를 바로 세워 봅니다.

        ax.xaxis.grid(False)

        ax.yaxis.grid(True)

    plt.show()
# 데이터 프레임에 항목 중요도를 넣습니다.

fi = {'Features':train.columns.tolist(), 'Importance':xgb.feature_importances_}

importance = pd.DataFrame(fi, index=None).sort_values('Importance', ascending=False)
# 그래프 제목

titles = ['The most important features in predicting survival on the Titanic: XGB']



# 그래프 그리기

importance_plotting(importance, 'Importance', 'Features', 'Reds_r', titles)
# 중요도를 데이터프레임에 넣습니다. Logistic regression에서는 중요도보다 coefficients를 사용합니다. 

fi = {'Features':train.columns.tolist(), 'Importance':np.transpose(log.coef_[0])}

importance = pd.DataFrame(fi, index=None).sort_values('Importance', ascending=False)
importance.head()
# 그래프 타이틀

titles = ['The most important features in predicting survival on the Titanic: Logistic Regression']



# 그래프 그리기

importance_plotting(importance, 'Importance', 'Features', 'Reds_r', titles)
# 5가지 모델에 대한 항목 중요도 얻기

gbc_imp = pd.DataFrame({'Feature':train.columns, 'gbc importance':gbc.feature_importances_})

xgb_imp = pd.DataFrame({'Feature':train.columns, 'xgb importance':xgb.feature_importances_})

ran_imp = pd.DataFrame({'Feature':train.columns, 'ran importance':ran.feature_importances_})

ext_imp = pd.DataFrame({'Feature':train.columns, 'ext importance':ext.feature_importances_})

ada_imp = pd.DataFrame({'Feature':train.columns, 'ada importance':ada.feature_importances_})



# 이를 하나의 데이터프레임으로

importances = gbc_imp.merge(xgb_imp, on='Feature').merge(ran_imp, on='Feature').merge(ext_imp, on='Feature').merge(ada_imp, on='Feature')



# 항목당 평균 중요도

importances['Average'] = importances.mean(axis=1)



# 랭킹 정하기

importances = importances.sort_values(by='Average', ascending=False).reset_index(drop=True)



# 보기

importances
# 중요도를 다시 데이터 프레임에 넣기

fi = {'Features':importances['Feature'], 'Importance':importances['Average']}

importance = pd.DataFrame(fi, index=None).sort_values('Importance', ascending=False)
# 그래프 타이틀

titles = ['The most important features in predicting survival on the Titanic: 5 model average']



# 그래프 보기

importance_plotting(importance, 'Importance', 'Features', 'Reds_r', titles)
# 약한 느낌을 주는 것 3개를 뺍니다.

train = train.drop(['Class', 'Pclass', 'Embarked'], axis=1)

test = test.drop(['Class', 'Pclass', 'Embarked'], axis=1)



# 모델의 변수를 다시 정의하고

X_train = train

X_test = test



# 바꿉니다.

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train
X_test
# 모델 사용

ran = RandomForestClassifier(random_state=1)

knn = KNeighborsClassifier()

log = LogisticRegression()

xgb = XGBClassifier(random_state=1)

gbc = GradientBoostingClassifier(random_state=1)

svc = SVC(probability=True)

ext = ExtraTreesClassifier(random_state=1)

ada = AdaBoostClassifier(random_state=1)

gnb = GaussianNB()

gpc = GaussianProcessClassifier()

bag = BaggingClassifier(random_state=1)



# 리스트

models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         

scores_v2 = []



# Fit & cross validate

for mod in models:

    mod.fit(X_train, y_train)

    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 10)

    scores_v2.append(acc.mean())
# 테이블 만들어서 보기

results = pd.DataFrame({

    'Model': ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression', 'XGBoost', 'Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier'],

    'Original Score': scores,

    'Score with feature selection': scores_v2})



result_df = results.sort_values(by='Score with feature selection', ascending=False).reset_index(drop=True)

result_df.head(11)
# 결과

sns.barplot(x='Score with feature selection', y = 'Model', data = result_df, color = 'c')

plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')

plt.xlim(0.75, 0.85)
# 파라미터 서치

Cs = [0.001, 0.01, 0.1, 1, 5, 10, 15, 20, 50, 100]

gammas = [0.001, 0.01, 0.1, 1]



# 파라미터 그리드 셋팅

hyperparams = {'C': Cs, 'gamma' : gammas}



# 교차검증

gd=GridSearchCV(estimator = SVC(probability=True), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



# 모델 fiting 및 결과

gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
learning_rate = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

n_estimators = [100, 250, 500, 750, 1000, 1250, 1500]



hyperparams = {'learning_rate': learning_rate, 'n_estimators': n_estimators}



gd=GridSearchCV(estimator = GradientBoostingClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
penalty = ['l1', 'l2']

C = np.logspace(0, 4, 10)



hyperparams = {'penalty': penalty, 'C': C}



gd=GridSearchCV(estimator = LogisticRegression(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
learning_rate = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

n_estimators = [10, 25, 50, 75, 100, 250, 500, 750, 1000]



hyperparams = {'learning_rate': learning_rate, 'n_estimators': n_estimators}



gd=GridSearchCV(estimator = XGBClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
max_depth = [3, 4, 5, 6, 7, 8, 9, 10]

min_child_weight = [1, 2, 3, 4, 5, 6]



hyperparams = {'max_depth': max_depth, 'min_child_weight': min_child_weight}



gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.0001, n_estimators=10), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
gamma = [i*0.1 for i in range(0,5)]



hyperparams = {'gamma': gamma}



gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.0001, n_estimators=10, max_depth=3, 

                                          min_child_weight=1), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
subsample = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

colsample_bytree = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

    

hyperparams = {'subsample': subsample, 'colsample_bytree': colsample_bytree}



gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.0001, n_estimators=10, max_depth=3, 

                                          min_child_weight=1, gamma=0), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
reg_alpha = [1e-5, 1e-2, 0.1, 1, 100]

    

hyperparams = {'reg_alpha': reg_alpha}



gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.0001, n_estimators=10, max_depth=3, 

                                          min_child_weight=1, gamma=0, subsample=0.6, colsample_bytree=0.9),

                                         param_grid = hyperparams, verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
n_restarts_optimizer = [0, 1, 2, 3]

max_iter_predict = [1, 2, 5, 10, 20, 35, 50, 100]

warm_start = [True, False]



hyperparams = {'n_restarts_optimizer': n_restarts_optimizer, 'max_iter_predict': max_iter_predict, 'warm_start': warm_start}



gd=GridSearchCV(estimator = GaussianProcessClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
n_estimators = [10, 25, 50, 75, 100, 125, 150, 200]

learning_rate = [0.001, 0.01, 0.1, 0.5, 1, 1.5, 2]



hyperparams = {'n_estimators': n_estimators, 'learning_rate': learning_rate}



gd=GridSearchCV(estimator = AdaBoostClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

algorithm = ['auto']

weights = ['uniform', 'distance']

leaf_size = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]



hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 

               'n_neighbors': n_neighbors}



gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



# Fitting model and return results

gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
n_estimators = [10, 25, 50, 75, 100]

max_depth = [3, None]

max_features = [1, 3, 5, 7]

min_samples_split = [2, 4, 6, 8, 10]

min_samples_leaf = [2, 4, 6, 8, 10]



hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features,

               'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}



gd=GridSearchCV(estimator = RandomForestClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
n_estimators = [10, 25, 50, 75, 100]

max_depth = [3, None]

max_features = [1, 3, 5, 7]

min_samples_split = [2, 4, 6, 8, 10]

min_samples_leaf = [2, 4, 6, 8, 10]



hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features,

               'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}



gd=GridSearchCV(estimator = ExtraTreesClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
n_estimators = [10, 15, 20, 25, 50, 75, 100, 150]

max_samples = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 50]

max_features = [1, 3, 5, 7]



hyperparams = {'n_estimators': n_estimators, 'max_samples': max_samples, 'max_features': max_features}



gd=GridSearchCV(estimator = BaggingClassifier(), param_grid = hyperparams, 

                verbose=True, cv=5, scoring = "accuracy")



gd.fit(X_train, y_train)

print(gd.best_score_)

print(gd.best_estimator_)
# 튜닝 모델 시작

ran = RandomForestClassifier(n_estimators=25,

                             max_depth=3, 

                             max_features=3,

                             min_samples_leaf=2, 

                             min_samples_split=8,  

                             random_state=1)



knn = KNeighborsClassifier(algorithm='auto', 

                           leaf_size=1, 

                           n_neighbors=5, 

                           weights='uniform')



log = LogisticRegression(C=2.7825594022071245,

                         penalty='l2')



xgb = XGBClassifier(learning_rate=0.0001, 

                    n_estimators=10,

                    random_state=1)



gbc = GradientBoostingClassifier(learning_rate=0.0005,

                                 n_estimators=1250,

                                 random_state=1)



svc = SVC(probability=True)



ext = ExtraTreesClassifier(max_depth=None, 

                           max_features=3,

                           min_samples_leaf=2, 

                           min_samples_split=8,

                           n_estimators=10,

                           random_state=1)



ada = AdaBoostClassifier(learning_rate=0.1, 

                         n_estimators=50,

                         random_state=1)



gpc = GaussianProcessClassifier()



bag = BaggingClassifier(random_state=1)



# 리스트

models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         

scores_v3 = []



# Fit & 교차 검증

for mod in models:

    mod.fit(X_train, y_train)

    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 10)

    scores_v3.append(acc.mean())
# 랭킹 테이블 생성

results = pd.DataFrame({

    'Model': ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression', 'XGBoost', 'Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier'],

    'Original Score': scores,

    'Score with feature selection': scores_v2,

    'Score with tuned parameters': scores_v3})



result_df = results.sort_values(by='Score with tuned parameters', ascending=False).reset_index(drop=True)

result_df.head(11)
# 결과

sns.barplot(x=None, y = None, data = result_df, color = 'c')

plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')

plt.xlim(0.75, 0.86)
#튜닝한 파라미터로 하드보팅

grid_hard = VotingClassifier(estimators = [('Random Forest', ran), 

                                           ('Logistic Regression', log),

                                           ('XGBoost', xgb),

                                           ('Gradient Boosting', gbc),

                                           ('Extra Trees', ext),

                                           ('AdaBoost', ada),

                                           ('Gaussian Process', gpc),

                                           ('SVC', svc),

                                           ('K Nearest Neighbour', knn),

                                           ('Bagging Classifier', bag)], voting = 'hard')



grid_hard_cv = model_selection.cross_validate(grid_hard, X_train, y_train, cv = 10)

grid_hard.fit(X_train, y_train)



print("Hard voting on test set score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))
grid_soft = VotingClassifier(estimators = [('Random Forest', ran), 

                                           ('Logistic Regression', log),

                                           ('XGBoost', xgb),

                                           ('Gradient Boosting', gbc),

                                           ('Extra Trees', ext),

                                           ('AdaBoost', ada),

                                           ('Gaussian Process', gpc),

                                           ('SVC', svc),

                                           ('K Nearest Neighbour', knn),

                                           ('Bagging Classifier', bag)], voting = 'soft')



grid_soft_cv = model_selection.cross_validate(grid_soft, X_train, y_train, cv = 10)

grid_soft.fit(X_train, y_train)



print("Soft voting on test set score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))
# Final predictions

predictions = grid_soft.predict(X_test)



submission = pd.concat([pd.DataFrame(passId), pd.DataFrame(predictions)], axis = 'columns')



submission.columns = ["PassengerId", "Survived"]

submission.to_csv('titanic_submission.csv', header = True, index = False)
submission.head()