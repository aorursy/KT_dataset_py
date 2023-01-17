import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/train.csv')
df.head()
df.info()
df.describe()
sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap='Paired')
sns.boxplot(x='Pclass',y='Age',data=df).set_title('Age distribution by passenger class')
AgeFirstClass = df[df['Pclass']==1]['Age'].dropna()
AgeSecondClass = df[df['Pclass']==2]['Age'].dropna()
AgeThirdClass = df[df['Pclass']==3]['Age'].dropna()
from scipy.stats import norm
sns.distplot(AgeFirstClass, bins=30, fit=norm).set_title('First Class Age Distribution')
sns.distplot(AgeSecondClass, bins=30, fit=norm).set_title('Second Class Age Distribution')
sns.distplot(AgeThirdClass, bins=30, fit=norm).set_title('Third Class Age Distribution')
from scipy import stats
# 1st class age normality test
stats.normaltest(AgeFirstClass)
# 2nd class age normality test
stats.normaltest(AgeSecondClass)
# 3rd class age normality test
stats.normaltest(AgeThirdClass)
print('%.6f'%stats.normaltest(AgeThirdClass).pvalue)
def plot_norm_residuals(sample,title):
    Frequency = sample.value_counts()/len(sample)
    Residuals = pd.DataFrame(data=Frequency.values,index=Frequency.index,columns=['Frequency'])
    Residuals['Prediction'] = stats.norm.pdf((Residuals.index-np.mean(sample))/np.std(sample))/np.std(sample)
    Residuals['Residuals']=Residuals['Prediction']-Residuals['Frequency']
    if stats.normaltest(Residuals['Residuals']).pvalue >=0.05:
        print('Residuals follow normal distribution')
    else:
        print('Residuals do NOT follow normal distribution')
    sns.distplot(np.sort(Residuals['Residuals']),bins=30,fit=norm).set_title(title+' Residuals Distribution')
plot_norm_residuals(AgeFirstClass,'First Class Age')
plot_norm_residuals(AgeSecondClass,'Second Class Age')
plot_norm_residuals(AgeThirdClass,'Third Class Age')
print('1st Class Age Variance: %0.2f' %(np.var(AgeFirstClass)))
print('2nd Class Age Variance: %0.2f' %(np.var(AgeSecondClass)))
print('3rd Class Age Variance: %0.2f' %(np.var(AgeThirdClass)))
alpha=0.05
def compare_variances(sample1,sample2):
    p_value=stats.f.sf(np.var(sample1)/np.var(sample2),len(sample1)-1,len(sample2)-1)
    if p_value < alpha:
        return 'the variances are statistically different'
    else:
        return 'the variances are NOT statistically different'
print('1st X 2nd variance comparison:',compare_variances(AgeFirstClass,AgeSecondClass))
print('1st X 3rd variance comparison:',compare_variances(AgeFirstClass,AgeThirdClass))
print('2nd X 3rd variance comparison:',compare_variances(AgeSecondClass,AgeThirdClass))
AgeSecThiClass = np.append(AgeSecondClass,AgeThirdClass)
np.var(AgeSecThiClass)
compare_variances(AgeFirstClass,AgeSecThiClass)
def T_test(sample1,sample2,var):
    p_value = stats.ttest_ind(sample1,sample2,equal_var=var).pvalue
    if p_value >= 0.05:
        return 'the means are NOT statistically different'
    else:
        return 'the means are statistically different'
print('1st X 2nd mean comparison:',T_test(AgeFirstClass,AgeSecondClass,True))
print('1st X 3rd mean comparison:',T_test(AgeFirstClass,AgeThirdClass,False))
print('2nd X 3rd mean comparison:',T_test(AgeSecondClass,AgeThirdClass,False))
sns.boxplot(x='SibSp',y='Age',data=df).set_title('Age distribution by siblings and spouses')
AgeSibSp0 = df[df['SibSp']==0]['Age'].dropna()
AgeSibSp1 = df[df['SibSp']==1]['Age'].dropna()
AgeSibSp2 = df[df['SibSp']==2]['Age'].dropna()
AgeSibSp3 = df[df['SibSp']==3]['Age'].dropna()
AgeSibSp4 = df[df['SibSp']==4]['Age'].dropna()
AgeSibSp5 = df[df['SibSp']==5]['Age'].dropna()
sns.distplot(AgeSibSp3, bins=30, fit=norm).set_title('3 SibSp Age Distribution')
plot_norm_residuals(AgeSibSp3,'SibSp 3 Age')
print('%.15f'%stats.kruskal(AgeSibSp0,AgeSibSp1,AgeSibSp2,AgeSibSp3,AgeSibSp4,AgeSibSp5).pvalue)
AgeSibSp01 = df[df['SibSp']<=1]['Age'].dropna()
AgeSibSp2345 = df[df['SibSp']>=2]['Age'].dropna()
print('%.15f' % stats.kruskal(AgeSibSp01,AgeSibSp2345).pvalue)
df['Adult_Child']=df['SibSp'].map(lambda x: 'Adult' if x<=2 else 'Child')
Corr = df[['Pclass','Adult_Child','Age']]
Corr = Corr.groupby(['Pclass','Adult_Child'],as_index=False).median()
Corr = Corr.pivot(index='Pclass',columns='Adult_Child',values='Age')
sns.heatmap(Corr,annot=True)
def fill_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    Adult_Child = cols[2]
    if pd.isnull(Age):
        return Corr[Adult_Child][Pclass]
    else:
        return Age
df['Age'] = df[['Age','Pclass','Adult_Child']].apply(fill_age,axis=1)
sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap='Paired')
df.drop('Cabin',axis=1,inplace=True)
df.dropna(inplace=True)
df.head()
Names = pd.DataFrame(df['Name'].str.split(', ',1).tolist(),columns = ['Last Name','Name'])
Names.head()
Title = pd.DataFrame(Names['Name'].str.split('.',1).tolist(),columns=['Title','Name'])
Title['Title'].value_counts()
df = pd.concat([df,Title['Title']],axis=1)
Title_Prop = df[['Survived','Title']]
Title_Prop['Count']=1
Title_Prop = Title_Prop.groupby(['Survived','Title'],as_index=False).sum()
Title_Prop = Title_Prop.pivot(index='Survived',columns='Title',values='Count')
Title_Prop = Title_Prop.T
Title_Prop.fillna(0,inplace=True)
Title_Prop['Prop'] = Title_Prop[1]/(Title_Prop[0]+Title_Prop[1])
Title_Prop['Prop'].plot(kind='bar')
sex = pd.get_dummies(df['Sex'],drop_first=True)
embark = pd.get_dummies(df['Embarked'],drop_first=True)
adult = pd.get_dummies(df['Adult_Child'],drop_first=True)
title = pd.get_dummies(df['Title'],drop_first=True)
df.drop(['Sex','Embarked','Ticket','Adult_Child','Title'],axis=1,inplace=True)
df = pd.concat([df,sex,embark,adult,title],axis=1)
df.head()
df.dropna(inplace=True)
X_train = df.drop(['PassengerId','Survived','Name','Dr','Rev','Col','Major','Mlle','Mme','Sir','Don','Jonkheer','Lady','Ms','the Countess'],axis=1)
y_train = df['Survived']
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
X_test = pd.read_csv('../input/test.csv')
X_test.head()
sns.heatmap(X_test.isnull(),cbar=False,yticklabels=False,cmap='Paired')
X_test.drop('Cabin',axis=1,inplace=True)
X_test['Adult_Child']=X_test['SibSp'].map(lambda x: 'Adult' if x<=2 else 'Child')
X_test['Age'] = X_test[['Age','Pclass','Adult_Child']].apply(fill_age,axis=1)
sns.heatmap(X_test.isnull(),cbar=False,yticklabels=False,cmap='Paired')
Names_test = pd.DataFrame(X_test['Name'].str.split(', ',1).tolist(),columns = ['Last Name','Name'])
Title_test = pd.DataFrame(Names_test['Name'].str.split('.',1).tolist(),columns=['Title','Name'])
X_test = pd.concat([X_test,Title_test['Title']],axis=1)
sex_test = pd.get_dummies(X_test['Sex'],drop_first=True)
embark_test = pd.get_dummies(X_test['Embarked'],drop_first=True)
adult_test = pd.get_dummies(X_test['Adult_Child'],drop_first=True)
title_test = pd.get_dummies(X_test['Title'],drop_first=True)
X_test.drop(['Sex','Embarked','Ticket','Adult_Child','Title','Name'],axis=1,inplace=True)
X_test = pd.concat([X_test,sex_test,embark_test,adult_test,title_test],axis=1)
X_test.head()
X_test.info()
X_test.drop(['Dona','Dr','Ms','Rev'],axis=1,inplace=True)
X_test.set_index('PassengerId',inplace=True)
X_test.head()
X_test.isnull().any()
X_test[X_test.isnull().T.any().T]
ThirdClassFare = X_test[X_test['Pclass']==3]['Fare']
sns.distplot(ThirdClassFare.dropna())
X_test['Fare'] = X_test['Fare'].map(lambda x: ThirdClassFare.median() if pd.isnull(x) else x)
predictions = logmodel.predict(X_test)
predictions
X_test.index.values
Answer=pd.DataFrame(data=X_test.index.values)
Answer['Survived'] = predictions
Answer.columns = ['PassengerId','Survived']
Answer.head()
Answer = Answer.astype(int)
Answer.to_csv('Titanic_Answer.csv',index=False)
