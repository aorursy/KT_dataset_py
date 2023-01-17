import numpy as np
import pandas as pd
import os
import os
print(os.listdir("../input"))
train_df=pd.read_csv("../input/train.csv",index_col='PassengerId')
test_df=pd.read_csv("../input/test.csv",index_col='PassengerId')
test_df['Survived']=-888
df=pd.concat((train_df,test_df),axis=0)
df.info()
df[df.Embarked.isnull()]
df.Embarked.value_counts()
#which emabarked point has higher survival count
pd.crosstab(df[df.Survived!=-888].Survived,df[df.Survived!=-888].Embarked)
df.groupby(['Pclass','Embarked']).Fare.median()
df.Embarked.fillna('C',inplace=True)
df[df.Embarked.isnull()]
df.info()
df[df.Fare.isnull()]
df.groupby(['Pclass','Fare']).Embarked.median()
median_fare=df.loc[(df.Pclass==3)& (df.Embarked=='S'),'Fare'].median()
median_fare
df.Fare.fillna(median_fare,inplace=True)
df[df.Fare.isnull()]
df.info()
pd.options.display.max_rows=15
df[df.Age.isnull()]
% %matplotlib inline
df.Age.plot(kind='hist',bins=20,color='c')
#mean of Age
df.Age.mean()
# check median of Ages
df.Age.median()
df.groupby('Sex').Age.median()
df[df.Age.notnull()].boxplot('Age','Sex')
df.groupby('Pclass').Age.median()
def GetTitle(name):
    title_group={'mr':'Mr',
            'mrs':'Mrs',
            'miss':'Miss',
            'master':'Master',
            'don':'Sir',
            'rev':'Sir',
            'dr':'Officer',
            'mme':'Mrs',
            'ms':'Mrs',
            'major':"Officer",
            'lady':'Lady',
            'sir':'Sir',
            'mlle':'Miss',
            'col':'Officer','capt':'Officer',
            'the countess':'Lady',
            'jonkheer':'Sir',
            'dona':'Lady'}
    
    first_name_with_title=name.split(',')[1]
    title=first_name_with_title.split('.')[0]
    title=title.strip().lower()
    return title_group[title]
df.Name.map(lambda x:GetTitle(x))

df.Name.map(lambda x:GetTitle(x)).unique()
df["Title"]=df.Name.map(lambda x: GetTitle(x))
df.info()
df[df.Age.notnull()].boxplot('Age','Title')
title_age_median=df.groupby('Title').Age.transform('median')
df.Age.fillna(title_age_median,inplace=True)
df.info()
df.head()
