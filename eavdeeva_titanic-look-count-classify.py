import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/train.csv')
df.head()
df.info()
sns.set_style('whitegrid')
sns.countplot(x='Pclass',hue='Survived',data=df[df['Sex']=='male'])
sns.countplot(x='Pclass',hue='Survived',data=df[df['Sex']=='female'])
g = sns.FacetGrid(data=df,col='Survived',row='Sex')
g.map(sns.distplot,'Age')
df['CabType']=df['Cabin'].apply(lambda s: str(s)[0])
df['CabType'].unique()
def CorrectTitle(s):
    if ((s=='Mr.') or (s=='Miss.') or (s=='Mrs.') or (s=='Master.') or (s=='Dr.') or (s=='Rev.')): return s
    return 'U.'

df['Title']=df['Name'].apply(lambda s: s.split()[1])
df['Title']=df['Title'].apply(lambda s: CorrectTitle(s))
df['Title'].value_counts()
df_F = df[(df['Sex']=='female')] # All females
df_F_cl1 = df[(df['Sex']=='female') & (df['Pclass']==1)] # females, class 1
df_F_cl2 = df[(df['Sex']=='female') & (df['Pclass']==2)] # females, class 2
df_F_cl3 = df[(df['Sex']=='female') & (df['Pclass']==3)] # females, class 3

df_M = df[(df['Sex']=='male')] # All males
df_M_cl1 = df[(df['Sex']=='male') & (df['Pclass']==1)] # males, class 1
df_M_cl2 = df[(df['Sex']=='male') & (df['Pclass']==2)] # males, class 2
df_M_cl3 = df[(df['Sex']=='male') & (df['Pclass']==3)] # males, class 3
def Difference(df):
    Nsurv = df[df['Survived']==1]['Survived'].count()
    Ndied = df[df['Survived']==0]['Survived'].count()
    Ndiff = Nsurv-Ndied
    survRate = Nsurv/(Nsurv+Ndied)
    dNsurv=Nsurv**0.5
    dNdied=Ndied**0.5
    dNdiff=(Nsurv+Ndied)**0.5
    dRateNum = (((Nsurv+Ndied)**2)*Nsurv + (Nsurv**2)*Ndied)**0.5
    dRateDen = (Nsurv+Ndied)**2
    dRate=dRateNum/dRateDen
    print('Nsurv={}+-{:.0f}, Ndied={}+-{:.0f}'.format(Nsurv,dNsurv,Ndied,dNdied))
    print('Nsurv-Ndied={}+-{:.0f}, survRate={:.2f}+-{:.2f}'.format(Ndiff,dNdiff,survRate,dRate))
Difference(df)
print('females, class 1:')
Difference(df_F_cl1)
print(' ')
print('females, class 2:')
Difference(df_F_cl2)
print(' ')
print('females, class 3:')
Difference(df_F_cl3)
df_F_cl1[df_F_cl1['Survived']==0]
df_F_cl2[df_F_cl2['Survived']==0]
print('females, class 2, adult:')
Difference(df_F_cl2[df_F_cl2['Age']>20])
print(' ')
print('females, class 2, embarked S:')
Difference(df_F_cl2[df_F_cl2['Embarked']=='S'])
print(' ')
print('females, class 2, adult & embarked S:')
Difference(df_F_cl2[(df_F_cl2['Age']>20) & (df_F_cl2['Embarked']=='S')])
print('males, class 1:')
Difference(df_M_cl1)
print(' ')
print('males, class 2:')
Difference(df_M_cl2)
print(' ')
print('males, class 3:')
Difference(df_M_cl3)
g = sns.FacetGrid(data=df_F_cl3,col='Survived')
g.map(sns.distplot,'Age')
print('females, 3rd class, Age<14')
Difference(df_F_cl3[df_F_cl3['Age']<14])
print(' ')
print('females, 3rd class, Age<6')
Difference(df_F_cl3[df_F_cl3['Age']<6])
print(' ')
print('females, 3rd class, Age>40')
Difference(df_F_cl3[df_F_cl3['Age']>40])
print(' ')
print('females, 3rd class, Age<40')
Difference(df_F_cl3[df_F_cl3['Age']<40])
sns.countplot(x='SibSp',hue='Survived',data=df_F_cl3)
sns.countplot(x='SibSp',hue='Survived',data=df)
sns.countplot(x='Parch',hue='Survived',data=df_F_cl3)
sns.countplot(x='Parch',hue='Survived',data=df)
print('females, 3rd class, SibSp==0:')
Difference(df_F_cl3[df_F_cl3['SibSp']==0])
print(' ')
print('females, 3rd class, Parch==0:')
Difference(df_F_cl3[df_F_cl3['Parch']==0])
print(' ')
print('females, 3rd class, SibSp==0 and Parch==0:')
cond1=(df_F_cl3['SibSp']==0)
cond2=(df_F_cl3['Parch']==0)
Difference(df_F_cl3[cond1 & cond2])
sns.countplot(x='Embarked',hue='Survived',data=df_F_cl3)
sns.countplot(x='Embarked',hue='Survived',data=df)
print('females, 3rd class, embarked at Q:')
Difference(df_F_cl3[df_F_cl3['Embarked']=='Q'])
print(' ')
print('females, 3rd class, embarked at C:')
Difference(df_F_cl3[df_F_cl3['Embarked']=='C'])
sns.countplot(x='CabType',hue='Survived',data=df_F_cl3)
Difference(df_F_cl3[(df_F_cl3['CabType']=='F')|(df_F_cl3['CabType']=='E')])
g = sns.FacetGrid(data=df_F_cl3,col='Survived')
g.map(sns.distplot,'Fare')
g = sns.FacetGrid(data=df_F_cl3,col='Survived')
g.map(sns.distplot,'Fare',bins=7)
plt.xlim(5,40)
print('females, 3rd class, fare<8:')
Difference(df_F_cl3[df_F_cl3['Fare']<8])
print(' ')
print('females, 3rd class, fare<10:')
Difference(df_F_cl3[df_F_cl3['Fare']<10])
print(' ')
print('females, 3rd class, fare<15:')
Difference(df_F_cl3[df_F_cl3['Fare']<15])
print(' ')
print('females, 3rd class, fare<20:')
Difference(df_F_cl3[df_F_cl3['Fare']<20])
sns.countplot(x='Title',hue='Survived',data=df_F_cl3)
g = sns.FacetGrid(data=df_M,col='Pclass',row='Survived')
g.map(sns.distplot,'Age')
df_M_cl12 = df_M[(df_M['Pclass']==1)|(df_M['Pclass']==2)]
print('males, 1st and 2nd class, Age<14:')
Difference(df_M_cl12[df_M_cl12['Age']<14])
print(' ')
print('males, 1st and 2nd class, Age<16:')
Difference(df_M_cl12[df_M_cl12['Age']<16])
print(' ')
print('males, 1st and 2nd class, Age<17:')
Difference(df_M_cl12[df_M_cl12['Age']<17])
print(' ')
print('males, 1st and 2nd class, Age<18:')
Difference(df_M_cl12[df_M_cl12['Age']<18])
print('males, 3rd class, Age<14')
Difference(df_M_cl3[df_M_cl3['Age']<14])
print(' ')
print('males, 3rd class, Age<5')
Difference(df_M_cl3[df_M_cl3['Age']<5])
df_M_cl12_Adult = df_M_cl12[df_M_cl12['Age']>14]# adult males, 1st and 2nd class
df_M_cl1_Adult = df_M_cl1[df_M_cl1['Age']>14]# adult males, 1st class
df_M_cl2_Adult = df_M_cl2[df_M_cl2['Age']>14]# adult males, 2nd class

# adult males from 1st and 2nd class and all males from 3rd class:
df_M_Further = pd.concat([df_M_cl12_Adult, df_M_cl3])
sns.countplot(x='SibSp',hue='Survived',data=df_M_Further)
sns.countplot(x='SibSp',hue='Survived',data=df_M_cl1_Adult)
sns.countplot(x='Parch',hue='Survived',data=df_M_cl1_Adult)
sns.countplot(x='Embarked',hue='Survived',data=df_M_cl1_Adult)
sns.countplot(x='Embarked',hue='Survived',data=df_M_cl3[df_M_cl3['Age']<14])
sns.countplot(x='CabType',hue='Survived',data=df_M_cl1_Adult)
Difference(df_M_cl1_Adult[(df_M_cl1_Adult['CabType']=='E')])
sns.countplot(x='CabType',hue='Survived',data=df_M_Further)
g = sns.FacetGrid(data=df_M_cl1_Adult,col='Survived')
g.map(sns.distplot,'Fare')
Difference(df_M_cl1_Adult[df_M_cl1_Adult['Fare']>400])
df_M_cl1_Adult[df_M_cl1_Adult['Fare']>400]
g = sns.FacetGrid(data=df_M_cl1_Adult[(df_M_cl1_Adult['SibSp']==0)&(df_M_cl1_Adult['Parch']==0)],col='Survived')
g.map(sns.distplot,'Fare')
g = sns.FacetGrid(data=df_M_cl1_Adult[(df_M_cl1_Adult['SibSp']==0)&(df_M_cl1_Adult['Parch']==0)],col='Survived')
g.map(sns.distplot,'Fare')
plt.xlim(0,100)
Difference(df_M_cl1[(df_M_cl1['SibSp']==0)&(df_M_cl1['Parch']==0)&(df_M_cl1['Fare']<40)&(df_M_cl1['Fare']>20)])
sns.countplot(x='Title',hue='Survived',data=df_M_cl1_Adult)
sns.countplot(x='Title',hue='Survived',data=df_M_Further)
Difference(df_M_cl1_Adult[df_M_cl1_Adult['Title']=='Dr.'])
dfNew=pd.DataFrame(columns=('PassengerId', 'Survived'))
dfTest = pd.read_csv('../input/test.csv')

for i in range(418):
    surv=0       
    PassId=dfTest.loc[i]['PassengerId']
    
    # females
    if (dfTest.loc[i]['Sex']=='female'): 
        if (dfTest.loc[i]['Pclass']==1): surv=1
        if (dfTest.loc[i]['Pclass']==2): surv=1
        if (dfTest.loc[i]['Pclass']==3): 
            if (dfTest.loc[i]['Embarked']=='Q'): surv=1
            if ((dfTest.loc[i]['Parch']==0) & (dfTest.loc[i]['SibSp']==0)): surv=1
            if (dfTest.loc[i]['Fare']<8): surv=1
    
    #males
    if (dfTest.loc[i]['Sex']=='male'):
        if (dfTest.loc[i]['Pclass']==1): 
            if (dfTest.loc[i]['Age']<14): surv=1
        if (dfTest.loc[i]['Pclass']==2):
            if (dfTest.loc[i]['Age']<14): surv=1
            
    dfNew.loc[i] = pd.Series({'PassengerId':PassId,'Survived':surv})

dfNew.to_csv('submitClass.csv',index=False)
