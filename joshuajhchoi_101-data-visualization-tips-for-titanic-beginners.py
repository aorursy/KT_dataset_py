# They are for data manipulation/ 기본 데이터 정리 및 처리
import pandas as pd
import numpy as np

# For Visualization / 시각화
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.style.use('seaborn-whitegrid')
import missingno

# Ignore warnings / 경고 제거 (Pandas often makes warnings)
import sys
import warnings

import warnings
warnings.filterwarnings('ignore')
test = pd.read_csv('../input/titanic/test.csv')
train = pd.read_csv('../input/titanic/train.csv')
missingno.matrix(train, figsize = (15,8))
# Correlation heatmap between numerical values (SibSp Parch Age and Fare values) and Survived 
g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True
                , fmt = ".2f", cmap = "coolwarm")
train.head()
corr = train.corr()
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
train.isnull().sum()
data = pd.concat((train, test))
data.columns
fig = plt.figure(figsize=(10,2))
sns.countplot(y='Survived', data=train)
print(train.Survived.value_counts())
# Plot for survived
fig = plt.figure(figsize = (10,5))
sns.countplot(x='Survived', data = train)
print(train['Survived'].value_counts())
f,ax=plt.subplots(1,2,figsize=(15,6))
train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=train,ax=ax[1])
ax[1].set_title('Survived')
plt.show()
train.groupby(['Pclass','Survived'])['Survived'].count()
sns.factorplot('Pclass','Survived',hue='Sex',data=data)
plt.show()
pd.crosstab(train.Pclass,train.Survived,margins=True).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(12,6))
train[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived per Pcalss')
sns.countplot('Pclass',hue='Survived',data=train,ax=ax[1])
ax[1].set_title('Pcalss Survived vs Not Survived')
plt.show()
# Bar chart of each Pclass type
fig = plt.figure(figsize = (10,10))
ax1 = plt.subplot(2,1,1)
ax1 = sns.countplot(x = 'Pclass', hue = 'Survived', data = train)
ax1.set_title('Ticket Class Survival Rate')
ax1.set_xticklabels(['1 Upper','2 Middle','3 Lower'])
ax1.set_ylim(0,400)
ax1.set_xlabel('Ticket Class')
ax1.set_ylabel('Count')
ax1.legend(['No','Yes'])

# Pointplot Pclass type
ax2 = plt.subplot(2,1,2)
sns.pointplot(x='Pclass', y='Survived', data=train)
ax2.set_xlabel('Ticket Class')
ax2.set_ylabel('Percent Survived')
ax2.set_title('Percentage Survived by Ticket Class')
train.groupby('Pclass').Survived.mean()
# Density plot
fig = plt.figure(figsize=(15,8),) 
ax=sns.kdeplot(train.Pclass[train.Survived == 0] , 
               color='gray',
               shade=True,
               label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Pclass'] , 
               color='g',
               shade=True, 
               label='survived', 
              )
plt.title('Passenger Class Distribution - Survived vs Non-Survived', fontsize = 25, pad = 40)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15, labelpad = 20)
plt.xlabel("Passenger Class", fontsize = 15,labelpad =20)
labels = ['1st Class', '2nd Class', '3rd Class']
plt.xticks(sorted(train.Pclass.unique()), labels);
plt.subplots(figsize = (15,10))
sns.barplot(x = "Pclass", 
            y = "Survived", 
            data=train, 
            linewidth=5,
            capsize = .1

           )
plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25, pad=40)
plt.xlabel("Survival Rate per Pclass", fontsize = 15);
plt.ylabel("% of Passenger Survived", fontsize = 15);
labels = ['1st Class', '2nd Class', '3rd Class']

val = [0,1,2] ## 임시 방면 temporaray measure
plt.xticks(val, labels);
sns.boxplot(x='Pclass', y='Survived', data=train)
sns.catplot(x="Pclass", y="Fare", kind="violin", data=train)
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 5))
women = train[train['Sex']=='female']
men = train[train['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=20, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=20, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=20, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=20, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')
f,ax=plt.subplots(1,2,figsize=(10,5))
data[data['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
data[data['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()
fig = plt.figure(figsize=(12,6),)
ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Age'] , color='gray',shade=True,label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Age'] , color='g',shade=True, label='survived')
plt.title('Age Distribution - Surviver V.S. Non Survivors', fontsize = 25, pad = 40)
plt.xlabel("Age", fontsize = 15, labelpad = 20)
plt.ylabel('Frequency', fontsize = 15, labelpad= 20);
data['Initial']=0
for i in data:
    data['Initial']=data.Name.str.extract('([A-Za-z]+)\.')
    
data.tail()
pd.crosstab(data.Initial,data.Sex).T.style.background_gradient(cmap='summer_r')
sns.factorplot('Pclass','Survived',col='Sex',data=data)
plt.show()
pal = {1:"seagreen", 0:"gray"}
g = sns.FacetGrid(train,size=5, col="Sex", row="Survived", margin_titles=True, hue = "Survived",
                  palette=pal)
g = g.map(plt.hist, "Age", edgecolor = 'white');
g.fig.suptitle("Survived by Sex and Age", size = 25)
plt.subplots_adjust(top=0.90)
f,ax=plt.subplots(1,2,figsize=(30,10))
train[['Age','Pclass']].groupby(['Pclass']).mean().plot.bar(ax=ax[0])
ax[0].set_title('')
sns.countplot('Age',hue='Pclass',data=train,ax=ax[1])
ax[1].set_title('')
plt.show()
sns.swarmplot(x=train['Survived'], y=train['Age'])
plt.xlabel("Survived")
plt.ylabel("Age")
plt.show()
plt.hist(x = [data[data['Survived']==1]['Age'], data[data['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()
pd.crosstab([train.Sex,train.Survived],train.Pclass,margins=True).style.background_gradient(cmap='summer_r')
plt.hist(data['Age'], bins=40)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Age')
plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age", hue="Survived", data=train,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age", hue="Survived", data=train,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()
data.isnull().sum()
data['LastName']=0
for i in data:
    data['LastName']=data.Name.str.extract('([A-Za-z]+)')
    
data.head()
data = data.reset_index(drop=True)
data['Age'] = data.groupby('Initial')['Age'].apply(lambda x: x.fillna(x.mean()))

data.head()
data.loc[(data.Initial.isnull()),'Initial']='Mrs'

data.tail()
data['Inicab'] = 0
for i in data:
    data['Inicab']=data.Cabin.str.extract('^([A-Za-z]+)')
    data.loc[((data.Cabin.isnull()) & (data.Pclass.values == 1 )),'Inicab']='X'
    data.loc[((data.Cabin.isnull()) & (data.Pclass.values == 2 )),'Inicab']='Y'
    data.loc[((data.Cabin.isnull()) & (data.Pclass.values == 3 )),'Inicab']='Z'
    
data.head()
data.loc[(data.Embarked.isnull())]
data.sort_values(['Ticket'], ascending = True)[55:70]
data.loc[(data.Embarked.isnull()),'Embarked']='S'
data.loc[(data.Fare.isnull())]
data = data.reset_index(drop=True)
data['Fare'] = data.groupby('Inicab')['Fare'].apply(lambda x: x.fillna(x.mean()))

data[1040:1045]
data.isnull().sum()
def facto(a):
    for i in data:
        data[a] = data[a].factorize()[0]    
        
facto('Embarked')
facto('Sex')
facto('Ticket')
facto('Initial')
facto('LastName')
facto('Inicab')

data.tail()
sns.jointplot(x="Initial", y="Fare", data=data, kind='scatter')
sns.relplot(x="Initial", y="LastName", hue="Survived", data=data)
sns.relplot(x="Initial", y="Age", hue="Survived", data=data)
sns.relplot(x="LastName", y="Fare", hue="Survived", data=data)
sns.relplot(x="LastName", y="Ticket", hue="Survived", data=data)
sns.jointplot(x="LastName", y="Fare", data=data, kind='scatter')
fig = plt.figure(figsize = (5,6))
ax = sns.countplot(x = 'SibSp', hue = 'Survived', data = train)
ax.set_title('Survival Rate with Total of Siblings and Spouse on Board')
ax.set_ylim(0,500)
ax.set_xlabel('# of Sibling and Spouse')
ax.set_ylabel('Count')
ax.legend(['No','Yes'],loc = 1)
fig = plt.figure(figsize = (5,6))
ax = sns.countplot(x = 'Parch', hue = 'Survived', data = train)
ax.set_title('Survival Rate with Total Parents and Children on Board')
ax.set_ylim(0,500)
ax.set_xlabel('# of Parents and Children')
ax.set_ylabel('Count')
ax.legend(['No','Yes'],loc = 1)
data['FamilySize']=0
data['FamilySize']= data['SibSp']+ data['Parch'] + 1

data.head()
plt.hist(x = [data[data['Survived']==1]['FamilySize'], data[data['Survived']==0]['FamilySize']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()
f,ax=plt.subplots(1,2,figsize=(20,8))
sns.barplot('SibSp','Survived',data=data,ax=ax[0])
ax[0].set_title('SibSp vs Survived')
sns.factorplot('SibSp','Survived',data=data,ax=ax[1])
ax[1].set_title('SibSp vs Survived')
plt.close(2)
plt.show()
sns.relplot(x="Pclass", y="FamilySize", hue="Survived", data=data)
plt.hist(x = [data[data['Survived']==1]['Fare'], data[data['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()
fig = plt.figure(figsize = (10,5))
sns.swarmplot(x="Pclass", y="Fare", data=train, hue='Survived')
plt.hist(data['Fare'], bins=40)
plt.xlabel('Fare')
plt.ylabel('Count')
plt.title('Distribution of fares')
plt.show()
sns.jointplot(x="Ticket", y="Fare", data=data, kind='scatter')
ax = sns.lineplot(x="Pclass", y="Fare", hue="Survived",data=data)
ax = sns.lineplot(x="Pclass", y="Ticket", hue="Survived",data=data)
sns.catplot(x="Ticket", y="Fare", kind="violin", data=data)
sns.relplot(x="Fare", y="Ticket", hue="Survived", data=data)
fig = plt.figure(figsize=(12,6),)
ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Fare'] , color='gray',shade=True,label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Fare'] , color='g',shade=True, label='survived')
plt.title('Fare Distribution - Surviver V.S. Non Survivors', fontsize = 25, pad = 40)
plt.xlabel("Fare", fontsize = 15, labelpad = 20)
plt.ylabel('Frequency', fontsize = 15, labelpad= 20);
fig = plt.figure(figsize=(12,6),)
ax=sns.kdeplot(data.loc[(data['Survived'] == 0),'Ticket'] , color='gray',shade=True,label='not survived')
ax=sns.kdeplot(data.loc[(data['Survived'] == 1),'Ticket'] , color='g',shade=True, label='survived')
plt.title('Ticket Distribution - Surviver V.S. Non Survivors', fontsize = 25, pad = 40)
plt.xlabel("Ticket", fontsize = 15, labelpad = 20)
plt.ylabel('Frequency', fontsize = 15, labelpad= 20);
f,ax=plt.subplots(2,2,figsize=(20,15))
sns.countplot('Embarked',data=data,ax=ax[0,0])
ax[0,0].set_title('No. Of Passengers Boarded')
sns.countplot('Embarked',hue='Sex',data=data,ax=ax[0,1])
ax[0,1].set_title('Male-Female Split for Embarked')
sns.countplot('Embarked',hue='Survived',data=data,ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked',hue='Pclass',data=data,ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()
sns.factorplot('Pclass','Survived',hue='Sex',col='Embarked',data=data)
plt.show()
sns.catplot(x="Embarked", y="Fare", kind="violin", data=data)
f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(data[data['Embarked']== 0].Fare,ax=ax[0])
ax[0].set_title('Fares in S')
sns.distplot(data[data['Embarked']== 1].Fare,ax=ax[1])
ax[1].set_title('Fares in C')
sns.distplot(data[data['Embarked']== 2].Fare,ax=ax[2])
ax[2].set_title('Fares in Q')
plt.show()
fig = plt.figure(figsize=(12,6),)
ax=sns.kdeplot(data.loc[(data['Survived'] == 0),'Ticket'] , color='gray',shade=True,label='not survived')
ax=sns.kdeplot(data.loc[(data['Survived'] == 1),'Ticket'] , color='g',shade=True, label='survived')
plt.title('Embarked Distribution - Surviver V.S. Non Survivors', fontsize = 25, pad = 40)
plt.xlabel("Embarked", fontsize = 15, labelpad = 20)
plt.ylabel('Frequency', fontsize = 15, labelpad= 20);
fig = plt.figure(figsize = (10,10))
ax1 = plt.subplot(2,1,1)
ax1 = sns.countplot(x = 'Embarked', hue = 'Survived', data = train)
ax1.set_title('Survival Rate per Embarked Place')
ax1.set_xticklabels(['1','2','3'])
ax1.set_ylim(0,400)
ax1.set_xlabel('Embarked')
ax1.set_ylabel('Count')
ax1.legend(['No','Yes'])


ax2 = plt.subplot(2,1,2)
sns.pointplot(x='Embarked', y='Survived', data=train)
ax2.set_xlabel('Embarked')
ax2.set_ylabel('Percent Survived')
ax2.set_title('Percentage Survived by Embarked')
sns.relplot(x="Embarked", y="Cabin", hue="Survived", data=data)
sns.jointplot(x="Embarked", y="Fare", data=data, kind='scatter')
fig = plt.figure(figsize = (10,5))
sns.swarmplot(x="Embarked", y="Fare", data=train, hue='Survived')
fig = plt.figure(figsize = (10,5))
sns.swarmplot(x="Pclass", y="Cabin", data=train, hue='Survived')
temp = pd.DataFrame()

def newdf(a,b):
    temp[a] = data[b]
    
newdf('Survived', 'Survived')
newdf('Pclass', 'Pclass')
newdf('Name', 'Initial')
newdf('Gender', 'Sex')
newdf('Age', 'Age')
newdf('Family', 'FamilySize')
newdf('Ticket', 'Ticket')
newdf('Fare', 'Fare')
newdf('Cabin', 'Inicab')
newdf('Embarked', 'Embarked')
newdf('LastName', 'LastName')


temp.tail()
g = sns.heatmap(temp.corr(),annot=True
                , fmt = ".2f", cmap = "coolwarm")
temp.isnull().sum()
temp.loc[(temp.Survived.isnull()) & (temp['Name'] == 0), 'Survived'] =  0.156673
temp.loc[(temp.Survived.isnull()) & (temp['Name'] == 1), 'Survived'] =  0.792000
temp.loc[(temp.Survived.isnull()) & (temp['Name'] == 2), 'Survived'] =  0.697802
temp.loc[(temp.Survived.isnull()) & (temp['Name'] == 3), 'Survived'] =  0.575000
temp.loc[(temp.Survived.isnull()) & (temp['Name'] == 4), 'Survived'] =  0.000000
temp.loc[(temp.Survived.isnull()) & (temp['Name'] == 5), 'Survived'] =  0.000000
temp.loc[(temp.Survived.isnull()) & (temp['Name'] == 6), 'Survived'] =  0.428571
temp.loc[(temp.Survived.isnull()) & (temp['Name'] == 7), 'Survived'] =  1.000000
temp.loc[(temp.Survived.isnull()) & (temp['Name'] == 8), 'Survived'] =  1.000000
temp.loc[(temp.Survived.isnull()) & (temp['Name'] == 9), 'Survived'] =  0.500000
temp.loc[(temp.Survived.isnull()) & (temp['Name'] == 10), 'Survived'] =  1.000000
temp.loc[(temp.Survived.isnull()) & (temp['Name'] == 11), 'Survived'] =  1.000000
temp.loc[(temp.Survived.isnull()) & (temp['Name'] == 12), 'Survived'] =  1.000000
temp.loc[(temp.Survived.isnull()) & (temp['Name'] == 13), 'Survived'] =  0.500000
temp.loc[(temp.Survived.isnull()) & (temp['Name'] == 14), 'Survived'] =  0.000000
temp.loc[(temp.Survived.isnull()) & (temp['Name'] == 15), 'Survived'] =  1.000000
temp.loc[(temp.Survived.isnull()) & (temp['Name'] == 16), 'Survived'] =  0.000000
temp.loc[(temp.Survived.isnull()) & (temp['Name'] == 17), 'Survived'] =  0.792000
temp.tail()
temp.groupby('Pclass').Survived.mean()
temp['Pclass'] = temp.groupby('Pclass')['Survived'].transform('mean')

temp.tail()
pd.crosstab(temp.Pclass,temp.Survived,margins=True).style.background_gradient(cmap='summer_r')
temp.Pclass.isnull().any()
temp.groupby('Name').Survived.mean()
temp['Name'] = temp.groupby('Name')['Survived'].transform('mean')

temp.tail()
temp.groupby('Name').Survived.mean()
temp.Name.isnull().any()
pd.crosstab(temp.Name,temp.Survived,margins=True).style.background_gradient(cmap='summer_r')
temp.groupby('Gender').Survived.mean()
temp['Gender'] = temp.groupby('Gender')['Survived'].transform('mean')

temp.head()
temp.Gender.isnull().any()
pd.crosstab(temp.Gender,temp.Survived,margins=True).style.background_gradient(cmap='summer_r')
temp.groupby('Age').Survived.mean()
temp['Age'] = temp.groupby('Age')['Survived'].transform('mean')

temp.head()
temp.Age.isnull().sum()
pd.crosstab(temp.Age,temp.Survived,margins=True).style.background_gradient(cmap='summer_r')
temp.groupby('Family').Survived.mean()
temp['Family'] = temp.groupby('Family')['Survived'].transform('mean')

temp.tail()
temp.Family.isnull().any()
pd.crosstab(temp.Family,temp.Survived,margins=True).style.background_gradient(cmap='summer_r')
temp.groupby('Ticket').Survived.mean()
temp['Ticket'] = temp.groupby('Ticket')['Survived'].transform('mean')

temp.tail()
temp.Ticket.isnull().sum()
pd.crosstab(temp.Ticket,temp.Survived,margins=True).style.background_gradient(cmap='summer_r')
temp.groupby('Fare').Survived.mean()
temp['Fare'] = temp.groupby('Fare')['Survived'].transform('mean')

temp.tail()
temp.Fare.isnull().sum()
pd.crosstab(temp.Fare,temp.Survived,margins=True).style.background_gradient(cmap='summer_r')
temp.groupby('Cabin').Survived.mean()

temp['Cabin'] = temp.groupby('Cabin')['Survived'].transform('mean')

temp.head()
temp.loc[(temp.Cabin.isnull())]
pd.crosstab(temp.Cabin,temp.Survived,margins=True).style.background_gradient(cmap='summer_r')
temp.groupby('Embarked').Survived.mean()
temp['Embarked'] = temp.groupby('Embarked')['Survived'].transform('mean')

temp
temp.loc[(temp.Embarked.isnull())]
pd.crosstab(temp.Embarked,temp.Survived,margins=True).style.background_gradient(cmap='summer_r')
temp.groupby('LastName').Survived.mean()
temp['LastName'] = temp.groupby('LastName')['Survived'].transform('mean')

temp
temp.isnull().any()
pd.crosstab(temp.LastName,temp.Survived,margins=True).style.background_gradient(cmap='summer_r')
sns.pairplot(temp)
g = sns.PairGrid(temp)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6);
df = pd.DataFrame()

def new(a):
    df[a] = temp[a]
    
new('Pclass')
new('Name')
new('Gender')
new('Age')
new('Family')
new('Ticket')
new('Fare')
new('Cabin')
new('Embarked')
new('LastName')


df.head()
df['Mean']=0

for i in df:
    df['Mean']=df.mean(axis=1)
    
df.head()
df['Highest']=0

for i in df:
    df['Highest']=df.max(axis=1)
    
df.head()
df['Lowest']=1

for i in df:
    df['Lowest']=df.min(axis=1)
    
df.head()
df['Social']=0

for i in df:
    df['Social']=(df['Name']+df['Gender']+df['Age']+df['Family'])/4
    
df.head(10)
df['Wealth']=0

for i in df:
    df['Wealth']=(df['Pclass']+df['Ticket']+df['Fare']+df['Cabin'])/4
    
df.head(10)
df['LowIndex'] = 1

for i in df:
    df['LowIndex']=(df['Lowest']+df['Mean'])/2
    
df.head(10)
df['HighIndex'] = 0

for i in df:
    df['HighIndex']=(df['Highest']+df['Mean'])/2
    
df.head(10)
df['Diff'] = 0

for i in df:
    df['Diff']=df['Highest']- df['Lowest']
    
df.head()
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(14, 8))
plt.title('Overall Correlation of Final Features', fontsize=18)
sns.heatmap(corr, mask=mask, annot=False,cmap='RdYlGn', linewidths=0.2, annot_kws={'size':20})
plt.show()
dfl = pd.DataFrame()

def new1(a):
    dfl[a] = data[a]
    
new1('Pclass')
new1('Initial')
new1('Sex')
new1('Age')
new1('FamilySize')
new1('Ticket')
new1('Fare')
new1('LastName')


def new2(a):
    dfl[a] = df[a]

new2('Highest')
new2('Social')
new2('Wealth')
new2('Lowest')
new2('Diff')
new2('Mean')
        
dfl
dfl.isnull().sum()
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train['Survived'].values
passId = test['PassengerId']
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


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn import model_selection
dfl_enc = dfl.apply(LabelEncoder().fit_transform)
                          
dfl_enc.head()
train = dfl_enc[:ntrain]
test = dfl_enc[ntrain:]
len(train)
len(test)
X_test = test
X_train = train
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
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
plt.xlim(0.85, 0.99)
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
# 아래는 Features라는 열에 트레인의 열들의 이름을 리스트로 만들어서 넣고 Importance에는 Logistic regression에는 coefficient를 바꾸어 넣어라는 넘파이 명령입니다.(즉 가로를 세로로)
fi = {'Features':train.columns.tolist(), 'Importance':np.transpose(log.coef_[0])}
importance = pd.DataFrame(fi, index=None).sort_values('Importance', ascending=False)


importance
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
# 약한 놈 탈락, 본래 좀 탈락시키는데 ...
# train = train.drop([ ], axis=1)
# test = test.drop([ ], axis=1)

# 모델의 변수를 다시 정의하고
X_train = train
X_test = test

# 바꿉니다.
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#  모델 재 트레이닝
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
result_df.head
# 결과
sns.barplot(x='Score with feature selection', y = 'Model', data = result_df, color = 'c')
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
plt.xlim(0.85, 1.00)
# SVC
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

# Gradient Boosting Classifier
learning_rate = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
n_estimators = [100, 250, 500, 750, 1000, 1250, 1500]

hyperparams = {'learning_rate': learning_rate, 'n_estimators': n_estimators}

gd=GridSearchCV(estimator = GradientBoostingClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)
# Logistic Regression
penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)

hyperparams = {'penalty': penalty, 'C': C}

gd=GridSearchCV(estimator = LogisticRegression(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)
# XGBoost Step 1.
learning_rate = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
n_estimators = [10, 25, 50, 75, 100, 250, 500, 750, 1000]

hyperparams = {'learning_rate': learning_rate, 'n_estimators': n_estimators}

gd=GridSearchCV(estimator = XGBClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)
# XGB Step 2.
max_depth = [3, 4, 5, 6, 7, 8, 9, 10]
min_child_weight = [1, 2, 3, 4, 5, 6]

hyperparams = {'max_depth': max_depth, 'min_child_weight': min_child_weight}

gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.0001, n_estimators=10), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)
# XGB Step 3.
gamma = [i*0.1 for i in range(0,5)]

hyperparams = {'gamma': gamma}

gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.0001, n_estimators=10, max_depth=3, 
                                          min_child_weight=1), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)

# XGB Step 4
subsample = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
colsample_bytree = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    
hyperparams = {'subsample': subsample, 'colsample_bytree': colsample_bytree}

gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.0001, n_estimators=10, max_depth=3, 
                                          min_child_weight=1, gamma=0), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)
# XGB Step 5
reg_alpha = [1e-5, 1e-2, 0.1, 1, 100]
    
hyperparams = {'reg_alpha': reg_alpha}

gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.0001, n_estimators=10, max_depth=3, 
                                          min_child_weight=1, gamma=0, subsample=0.6, colsample_bytree=0.9),
                                         param_grid = hyperparams, verbose=True, cv=5, scoring = "accuracy")

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)
# Gaussian Process
n_restarts_optimizer = [0, 1, 2, 3]
max_iter_predict = [1, 2, 5, 10, 20, 35, 50, 100]
warm_start = [True, False]

hyperparams = {'n_restarts_optimizer': n_restarts_optimizer, 'max_iter_predict': max_iter_predict, 'warm_start': warm_start}

gd=GridSearchCV(estimator = GaussianProcessClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)
# Adaboost.
n_estimators = [10, 25, 50, 75, 100, 125, 150, 200]
learning_rate = [0.001, 0.01, 0.1, 0.5, 1, 1.5, 2]

hyperparams = {'n_estimators': n_estimators, 'learning_rate': learning_rate}

gd=GridSearchCV(estimator = AdaBoostClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)
# KNN
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
# Random Forest.
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
# Extra Trees
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
# Bagging Classifier
n_estimators = [10, 15, 20, 25, 50, 75, 100, 150]
max_samples = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 50]
max_features = [1, 3, 5, 7]

hyperparams = {'n_estimators': n_estimators, 'max_samples': max_samples, 'max_features': max_features}

gd=GridSearchCV(estimator = BaggingClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)
# 모델 재 트레이닝
# 튜닝 모델 시작
# sample을 split하는 것은 전체데이터 80%를 트레인셋에 20%는 테스트셋에 줌  
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
plt.xlim(0.75, 1.00)
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
# Final predictions1
predictions = grid_soft.predict(X_test)

submission = pd.concat([pd.DataFrame(passId), pd.DataFrame(predictions)], axis = 'columns')

submission.columns = ["PassengerId", "Survived"]
submission.to_csv('titanic_submission133.csv', header = True, index = False)
# Final predictions2
predictions = grid_hard.predict(X_test)

submission = pd.concat([pd.DataFrame(passId), pd.DataFrame(predictions)], axis = 'columns')

submission.columns = ["PassengerId", "Survived"]
submission.to_csv('titanic_submission134.csv', header = True, index = False)