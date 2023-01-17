import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.model_selection import train_test_split

train=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")


print(train.shape)

print(test.shape)

train.head(10)
train.loc[:,["Age","Sex"]]
test.head(5)
ax = sns.countplot("Sex", data=train, palette="Set3")

train.describe(include="O")
pd.crosstab(train["Survived"],train["Sex"])

pd.crosstab(train["Sex"],train["Sex"])

gen_grp=train.groupby('Sex')

temp=gen_grp['Sex'].count()



k=temp.to_frame(name='Count')

k



pclass_grp=train.groupby('Pclass')

temp1=pclass_grp['Pclass'].count()

temp1.to_frame(name='count_check')
pclass_grp=train.groupby('Pclass')

l=pclass_grp['Pclass'].count()

l.to_frame(name='count')
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).count().sort_values(by='Survived', ascending=False)
train[['Pclass', 'Survived','Sex']].groupby(['Survived','Pclass'], as_index=False).count().sort_values(by='Survived', ascending=False)
sns.countplot('Pclass',data=train)
sns.countplot("Sex" , data = train , hue ='Survived' ,palette = 'hls')



sns.countplot(train['Embarked'], hue=train['Pclass'])

sns.countplot(train['Sex'],hue=train['Pclass'])
train['Embarked'].value_counts()
sns.countplot(train['Embarked'])
train.isnull().sum()
for col in train.columns:

    print(col, ":", train[col].unique().shape[0])
print(train.columns)
train['SibSp'].unique().shape[0]
#corr = train.corr()

train.corr().style.background_gradient(cmap='coolwarm')
train.isnull().sum()
train.isna().sum()
pd.crosstab(train.Pclass,train.Survived,margins=True).style.background_gradient(cmap='copper_r')
f,ax=plt.subplots(1,2,figsize=(18,8))

train['Survived'].value_counts().plot.pie(explode=[0.1,0.2],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('Pie chart')

sns.countplot(train['Survived'])

ax[1].set_title('Countplot')
pd.crosstab(train['Survived'],train['Pclass']).style.background_gradient(cmap='coolwarm')

#Most of the High Class passengers have been rescued
train.groupby(['Survived','Sex'])['Survived'].count()
train.isnull().sum()
train.describe(include='O')
pd.crosstab([train.Sex,train.Survived],train.Pclass,margins=True).style.background_gradient(cmap='summer_r')
sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
train.groupby(['Survived','Sex']).count()
train.isnull().sum()
for col in train.columns:

    print(col,':',train[col].unique().shape[0])
train['Sex'].value_counts()
train['Survived'].value_counts()
train['Pclass'].value_counts()
pd.crosstab([train.Survived,train.Pclass],[train.Sex],margins=True).style.background_gradient(cmap='summer_r')
sns.countplot('Pclass',hue='Sex',data=train)
train.groupby(['Survived','Sex'])['Age'].median()
#select embarked,count(*) from train group by 1

train['Embarked'].value_counts()
sns.countplot(train['Embarked'])
train['Age'].mean()
train.groupby(['Survived','Sex'])['Age'].mean()
sns.heatmap(train.corr(),annot=True,cmap='copper_r')

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
train['Age'].max()
train['Age_band']=0
train['Age_band'].head(20)
train['Age_band']=0

train.loc[train['Age']<=16,'Age_band']=0

train.loc[(train['Age']>16)&(train['Age']<=32),'Age_band']=1

train.loc[(train['Age']>32)&(train['Age']<=48),'Age_band']=2

train.loc[(train['Age']>48)&(train['Age']<=64),'Age_band']=3

train.loc[train['Age']>64,'Age_band']=4

train.head(2)
train.loc[0:2,['Sex','Ticket','Age']].head(5)
train.loc[[0,2,5],['Sex','Ticket','Age']].head(5)
train.iloc[[0,2,5],1:6].head(5)
train.head(6)
train['Age_band'].value_counts().to_frame().style.background_gradient(cmap='copper_r')
sns.heatmap(train.corr(),annot=True,cmap='summer_r',linewidth=0.2)

#fig=plt.gcf()

#fig.set_size_inches(15,8)

plt.gcf().set_size_inches(15,8)
sns.countplot('Sex',hue='Embarked',data=train)
f,ax=plt.subplots(2,2,figsize=(20,15))

sns.countplot('Embarked',data=train,ax=ax[0,0])

ax[0,0].set_title('No. Of Passengers Boarded')

sns.countplot('Embarked',hue='Sex',data=train,ax=ax[0,1])

ax[0,1].set_title('Male-Female Split for Embarked')

sns.countplot('Embarked',hue='Survived',data=train,ax=ax[1,0])

ax[1,0].set_title('Embarked vs Survived')

sns.countplot('Embarked',hue='Pclass',data=train,ax=ax[1,1])

ax[1,1].set_title('Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2,hspace=0.5)

plt.show()

train['Embarked'].fillna('S',inplace=True)



train['Embarked'].value_counts().to_frame().style.background_gradient(cmap='copper_r')
train.Age.isnull().any()
pd.crosstab(train.SibSp,train.Survived)
print('Highest fare was:',train.Fare.max())

print('Lowest Fare was:',train.Fare.min())

print('Average Fare was:',train.Fare.mean())
train.Age_band.value_counts().to_frame().style.background_gradient(cmap='summer')



#checking the number of passenegers in each band
train['Fare_cat']=0
train.loc[(train['Fare'])<=7.91,'Fare_cat']=0

train.loc[(train['Fare']>7.91)&(train['Fare']<=14.54),'Fare_cat']=1

train.loc[(train['Fare']>14.54)&(train['Fare']<=31),'Fare_cat']=2

train.loc[(train['Fare']>31),'Fare_cat']=3
#train['Fare_cat'].head(10).to_frame()
sns.factorplot('Fare_cat','Survived',data=train,hue='Sex')

plt.show()
train['Sex'].replace(['Male','Female'],[0,1],inplace=True)
train['Initial']=0

for i in train:

    train['Initial']=train.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
pd.crosstab(train.Initial,train.Sex).T.style.background_gradient(cmap='summer_r')
train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
train.groupby('Initial')['Age'].mean()
train.loc[(train.Age.isnull())&(train.Initial=='Mr'),'Age']=33

train.loc[(train.Age.isnull())&(train.Initial=='Mrs'),'Age']=36

train.loc[(train.Age.isnull())&(train.Initial=='Master'),'Age']=5

train.loc[(train.Age.isnull())&(train.Initial=='Miss'),'Age']=22

train.loc[(train.Age.isnull())&(train.Initial=='Other'),'Age']=46
train['Age'].isnull().any()
train['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

train['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)



##Initials to be changed
sns.factorplot('Pclass','Survived',col='Initial',data=train)

plt.show()
sns.factorplot('Embarked','Survived',data=train)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])

train_X=train[train.columns[1:]]

train_Y=train[train.columns[:1]]

test_X=test[test.columns[1:]]

test_Y=test[test.columns[:1]]

X=data[data.columns[1:]]

Y=data['Survived']
train_X,train_Y,test_X,test_Y=train_test_split(train,test_size=0.3,random_state=0,stratify=train['Survived'])
train.head(5)
X = train[["Age","Fare","Pclass","SibSp"]].values

Y = train[["Survived"]].values

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=120)
df.loc[:,['Age','Fare']]
df['change_age']=df.Age.diff()

df.head()
df['change_age_pct']=df.Age.pct_change()*100

df.head()
gen_mean=df.groupby(['Sex','Pclass','Survived'])['Survived'].count()

pd.DataFrame(gen_mean)
df.head()
agg_fun=df.groupby(['Sex','Pclass','Survived']).agg({'Survived':'count','Age':'max'})

agg_fun
gg=df.groupby(['Sex'])['Sex'].count()

gg
df.Sex.value_counts()
df['Age_diff']=df.Age.diff()
df.head()
df=df.drop(['Age_diff'],axis=1)

df.head()
df['Age_scale']=np.where(df.Age>30,np.where(df.Age>35,'E','G'),'L')

df.head()
pd.crosstab([train.Sex,train.Pclass],[train.Survived])
pd.crosstab([train.Survived,train.Pclass],[train.Sex],margins=True).style.background_gradient(cmap='summer_r')
train.groupby(['Pclass','Sex']).agg({'Age':['max','min'],'Survived':['mean']})
train.groupby(['Pclass']).agg(Age_grp=('Age','mean'))
train.groupby(['Pclass']).agg(Age_grp=('Age','max'),survival_mean=('Survived','mean'))
train.groupby('Pclass')['Age'].max()
train.groupby(['Pclass','Sex']).agg(Age_max=('Age','max'))
ctr1=(df.Age>63.0)

ctr2=(df.Age<70.0)

df[ctr1&ctr2]