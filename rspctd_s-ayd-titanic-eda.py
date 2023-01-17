# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')     # plt.style.available  komutu ile diger stilde ki plotlari da kullanabilirsin...

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings('ignore')




# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_df=pd.read_csv('/kaggle/input/titanic/train.csv')
test_df=pd.read_csv('/kaggle/input/titanic/test.csv')
test_PassengerId=test_df['PassengerId']
train_df.head()
train_df.describe()

train_df.info()
train_df.info()

def bar_plot(variable):
    """
    input: variable EX:"sex"
    output: barplot & value counts
    """
    
    var=train_df[variable]    #get feature
    varValue=var.value_counts()  #count number of categorical variable(value/sample)
    
    #visualization
    plt.figure(figsize=(9,5))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel('Frequency')
    plt.title(variable)
    plt.show()
    
    print(" {}:\n{}".format(variable,varValue))
    
    
category1=['Survived', 'Pclass', 'Sex','SibSp','Parch','Embarked']
for c in category1:
    bar_plot(c)

category2=['Cabin','Name', 'Ticket']
for c2 in category2:
    print("{}: \n".format(train_df[c2].value_counts()))
def hist_plot(variable):
    plt.figure(figsize=(9,7))
    plt.hist(train_df[variable],bins=10,color='g')
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.title("{} distribution with histogram".format(variable))
    plt.show()
    
    print("{}\n{}".format(train_df[variable].index,train_df[variable].value_counts()))
numeric_value=['PassengerId','Age','Fare']
for n in numeric_value:
    hist_plot(n)
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index= False).mean().sort_values(by='Survived',ascending=False)
train_df[["Sex","Survived"]].groupby(["Sex"], as_index= False).mean().sort_values(by='Survived',ascending=False)
train_df[["Age","Survived"]].groupby(["Age"], as_index= False).mean().sort_values(by='Survived',ascending=False)
train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index= False).mean().sort_values(by='Survived',ascending=False)
train_df[["Parch","Survived"]].groupby(["Parch"], as_index= False).mean().sort_values(by='Survived',ascending=False)
def detect_outliers(df,features):
    
    outlier_indices=[]
    
    for c in features:
        
        #1st quartile
        Q1=np.percentile(df[c],25)
        
        #3rd quartile
        Q3=np.percentile(df[c],75)
        
        #IQR
        IQR=Q3-Q1
        
        #Outlier Step
        outlier_step=1.5*IQR
        
        #Detect Outlier and their indices
        outlier_list_col=df[ (df[c]<Q1-outlier_step) | (df[c]>Q3+outlier_step)].index
        
        #Store indices
        
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices=Counter(outlier_indices)
        
    multiple_outliers= list(i for i,v in outlier_indices.items() if v>2 )
        
    return multiple_outliers
    
    
# Checking outliers that has been detected above:

train_df.iloc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]
# Drop Outliers
train_df=train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)

train_df1=pd.concat([train_df,test_df],axis=0).reset_index(drop=True) # I concatenated two data frames(train_df,test_df)   
                                                                      #and the reason is we will do machine learningand prediction later on with test_df.
                                                                      #If I keep the missing or non values on data,I will not be able to train the machine.
                                                                      #( Our model will not be able to read the data.)

train_df1.columns[train_df1.isnull().any()] # In the features of the train_df1, I am checking if there is any missing value.
                                            # and if there is missing value, which columns have those missing values.
train_df1.isnull() .sum()  # How many missing values on the data.
                           # We are seeing 418 missing value on Survived Feature. These missing values are coming from test_df.
                           # We will use test_df later on to predict whether who survived or died. That is why we see it as missing values on 'Survived'.
                           # In order to fill missing values on 'Age', I can not use basic statistic techniques. We are leaving that  as it is for now.
train_df1[train_df1.Embarked.isnull()]
train_df1.boxplot(column='Fare',by='Embarked')
plt.show()
train_df1["Embarked"]=train_df1.Embarked.fillna("C")
train_df1[train_df1.Embarked.isnull()] # checking if there is any missing value.
train_df1[train_df1['Fare'].isnull()]
train_df1["Fare"]=train_df1['Fare'].fillna(np.mean(train_df1[train_df1["Pclass"]==3]['Fare']))
train_df1[train_df1.Fare.isnull()]
list1=['SibSp','Parch', 'Age', 'Fare','Survived']
sns.heatmap(train_df1[list1].corr(), annot=True, fmt='.2f')
plt.show()
g=sns.factorplot(x="SibSp",y="Survived",data=train_df1, kind='bar',size=6)
g.set_ylabels('Survived Probability')
plt.show()
g=sns.factorplot(x="Parch",y="Survived",data=train_df1,kind='bar',size=6)
g.set_ylabels("Survived Probability")
plt.show()
g=sns.factorplot(x='Pclass',y='Survived',data=train_df1,kind='bar',size=6)
g.set_ylabels('Survived Probability')
plt.show()
g=sns.FacetGrid(train_df1, col='Survived',size=5)
g.map(sns.distplot,'Age',bins=25)
plt.show()
g=sns.FacetGrid(train_df1,col='Survived',row='Pclass',size=3)
g.map(plt.hist,"Age",bins=25)
g.add_legend()
plt.show()
g=sns.FacetGrid(train_df1,row='Embarked',size=3)
g.map(sns.pointplot,'Pclass','Survived','Sex')
g.add_legend()
plt.show()
g=sns.FacetGrid(train_df1,row='Embarked',col='Survived',size=3)
g.map(sns.barplot,"Sex","Fare")
g.add_legend()
plt.show()
train_df1[train_df1.Age.isnull()]
sns.factorplot(x='Sex',y='Age',data=train_df1,kind='box')
plt.show()
sns.factorplot(x='Sex',y='Age',hue='Pclass',data=train_df,kind='box')
plt.show()
sns.factorplot(x='Parch',y='Age',data=train_df1,kind='box',size=6)
sns.factorplot(x='SibSp',y='Age',data=train_df1,kind='box',size=6)
plt.show()
sns.heatmap(train_df1[['Age','SibSp','Parch','Pclass','Sex']].corr(),annot=True)
train_df1['Sex']=[1 if i=='male'else 0 for i in train_df1['Sex']]        #list comprehension
sns.heatmap(train_df1[['Age','Sex','Parch','SibSp','Pclass']].corr(),annot=True)
plt.show()

#finding indices for missing values.
index_nan_age=list(train_df1[train_df1["Age"].isnull()].index)

train_df1.head()
for i in index_nan_age:

    age_pred=train_df1['Age'][(train_df1['SibSp']== train_df1.iloc[i]['SibSp']) & (train_df1['Parch']== train_df1.iloc[i]['Parch']) & (train_df1['Pclass']== train_df1.iloc[i]['Pclass'])].median()
    age_med=train_df1['Age'].median()
    
    if not np.isnan(age_pred):
        train_df1['Age'].iloc[i]=age_pred
    else:
        train_df1['Age'].iloc[i]=age_med
        
train_df1.Name
name=train_df1.Name
train_df1['Title']=[i.split('.')[0].split(',')[-1].strip() for i in name]
train_df1
f=plt.figure(figsize=(8,8))
sns.countplot(x='Title',data=train_df1)
plt.xticks(rotation=45)
plt.xlabel('Title')
plt.ylabel("Number of People")
plt.show()
train_df1['Title']=train_df1['Title'].replace(['Dr','Rev','Col','Major','Lady','Don','Capt','the Countess','Dona','Sir','Jonkheer'],'Other')
train_df1['Title']=[0 if i=='Master' else 1 if i=='Miss' or i=='Ms' or i=='Mlle' or i=='Mrs' else 2 if i=='Mr' else 3 for i in train_df1['Title']]
train_df1['Title'].value_counts()
g=sns.factorplot(x='Title',y='Survived',data=train_df1,kind='bar')
g.set_xticklabels(["Master","Mrs","Mr","Other"])
g.set_ylabels(["Survival Probability"])
plt.show()
train_df1.drop(['Name'],axis=1,inplace=True)
train_df1
train_df1=pd.get_dummies(train_df1,columns=['Title'])
train_df1
train_df1.head()
train_df1['Family_Size']=train_df1['Parch']+train_df1['SibSp']+1

g=sns.factorplot(x='Family_Size',y='Survived',kind='bar',data=train_df1)
g.set_xlabels('Family Size')
g.set_ylabels('Survived Probability')
plt.show()
train_df1['Family_Size']=[ 1 if each<5 else 0 for each in train_df1['Family_Size']]
train_df1['Family_Size'].value_counts()
sns.countplot(x='Family_Size',data=train_df1)
plt.show()
train_df1['Family_Size'].value_counts()
g=sns.factorplot(x='Family_Size',y='Survived',kind='bar',data=train_df1)
g.set_xlabels("Family Size")
g.set_ylabels("Survived Probability")
plt.show()
train_df1=pd.get_dummies(train_df1,columns=["Family_Size"])
train_df1
train_df1.head()
sns.countplot(x='Embarked',data=train_df1)
plt.show()
train_df1.Embarked.value_counts()
train_df1=pd.get_dummies(train_df1,columns=['Embarked'])
train_df1
train_df1.head(10)
a='A/5. 21171'
a.replace('.','').replace('/','').strip().split()[0]
tickets=[]
for each in train_df1.Ticket:
    if not each.isdigit():
        tickets.append(each.replace('.','').replace('/','').strip().split()[0])
    else:
        tickets.append('X')
        
train_df1['Ticket']=tickets

train_df1=pd.get_dummies(train_df1,columns=['Ticket'])
train_df1
sns.countplot(x='Pclass',data=train_df1)
plt.show()
train_df1['Pclass']=train_df1['Pclass'].astype('category')
train_df1=pd.get_dummies(train_df1,columns=['Pclass'])
train_df1
train_df1.Sex=["male" if each==1 else "female" for each in train_df1.Sex]
sns.countplot(x="Sex",data=train_df1)
plt.show()
train_df1.Sex=train_df1.Sex.astype('category')
train_df1=pd.get_dummies(train_df1,columns=['Sex'])
train_df1