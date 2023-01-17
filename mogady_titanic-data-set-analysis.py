import pandas as pa
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import *
from sklearn.ensemble import RandomForestClassifier
titanic_df=pa.read_csv("../input/train.csv")
test_df=pa.read_csv("../input/test.csv")
titanic_df.head()
import re
def extract_title(dataset):
    title=[]
    Name=[]
    for data in dataset:
        title.append(re.split('(\w+)\.',data)[1])
    return title
titanic_df['Title']=extract_title(titanic_df['Name'])
test_df['Title']=extract_title(test_df['Name'])
titanic_df.Title.unique()
titanic_df['Title'].value_counts()
def replace_titles(dataset):
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
replace_titles(titanic_df)
replace_titles(test_df)
titanic_df.Title.unique()
titanic_df['Title'].value_counts()
titanic_df['Cabin']=titanic_df.Cabin.str.extract('([A-Za-z])')    
test_df['Cabin']=test_df.Cabin.str.extract('([A-Za-z])')    
titanic_df.head()
test_df.head()
frames=[titanic_df,test_df]
all_data=pa.concat(frames)
all_data.head()
fig, ax = plt.subplots(figsize=(10,5))
Title_survival=sns.barplot(x='Title',y='Survived',data=all_data).set(ylabel='Survival_Probability',title='TITLE_SURVIVAL')
def values_null(dataset):
    columns={}
    for item in dataset.columns.values:
        if dataset[item].isnull().values.any():
            columns[item]=dataset[item].isnull().sum()
    return columns
values_null(all_data)
test_df['Fare'].isnull().values.any()
test_df['Fare']=test_df['Fare'].fillna(all_data.Fare.median())
test_df['Embarked'].isnull().values.any()
titanic_df['Embarked'].isnull().values.any()
all_data['Fare']=all_data['Fare'].fillna(all_data.Fare.median())
all_data['Embarked']=all_data['Embarked'].fillna('S')
all_data['Embarked'].value_counts()
titanic_df['Embarked']=titanic_df['Embarked'].fillna('S')
groups_fare=all_data.groupby(['Cabin','Pclass'])['Fare']
groups_fare.mean().plot(kind='bar')
groups_Pclass=all_data.groupby('Cabin')['Pclass']
pa.crosstab(all_data['Cabin'],all_data['Pclass']).plot(kind='bar')
pa.crosstab(all_data['Cabin'],all_data['Pclass'])
def filling_cabin(dataset):
    cabin=[]
    for index,data in dataset.iterrows():
        if data['Pclass']==1:
            if data['Fare']<40:
                cabin.append('A')
            elif data['Fare'] in range(40,60):
                cabin.append('D')
            elif data['Fare'] in range(60,90):
                cabin.append('C')
            else:
                cabin.append('B')
        elif data['Pclass']==2:
                if data['Fare']<10:
                    cabin.append('E')
                elif data['Fare'] in range(10,20):
                    cabin.append('D')
                else:
                    cabin.append('F')
        elif data['Pclass']==3:
                if data['Fare']<15:
                    cabin.append('F')
                elif data['Fare'] in range(15,25):
                    cabin.append('E')
                else:
                    cabin.append('G')
    return cabin
    
non_cabin=all_data[all_data['Cabin'].isnull()]
all_data.loc[all_data['Cabin'].isnull(),'Cabin']=filling_cabin(non_cabin)
pa.crosstab(all_data['Cabin'],all_data['Pclass']).plot(kind='bar')
non_cabin_train=titanic_df[titanic_df['Cabin'].isnull()]
non_cabin_test=test_df[test_df['Cabin'].isnull()]
titanic_df.loc[titanic_df['Cabin'].isnull(),'Cabin']=filling_cabin(non_cabin_train)
test_df.loc[test_df['Cabin'].isnull(),'Cabin']=filling_cabin(non_cabin_test)
pa.crosstab(all_data['Age'],all_data["Sex"]).plot(kind='kde')
pa.crosstab(all_data['Age'],all_data["Pclass"]).plot(kind='kde')
mapp_T={'Mr':0,'Mrs':1,'Miss':2,'Master':3,'Rare':4}
mapp_S={'male':0,'female':1}
mapp_E={'C':0,'Q':1,'S':2}
all_data=all_data.replace({"Title":mapp_T,'Sex':mapp_S,'Embarked':mapp_E})
titanic_df=titanic_df.replace({"Title":mapp_T,'Sex':mapp_S,'Embarked':mapp_E})
test_df=test_df.replace({"Title":mapp_T,'Sex':mapp_S,'Embarked':mapp_E})
copy_data=all_data
copy_data_ged=copy_data[~copy_data['Age'].isnull()]
copy_data_ged=copy_data_ged[copy_data_ged['Title']!=4]
copy_data_ged.loc[copy_data_ged['Age']<1,'Age']=1
x=copy_data_ged[['Sex','Parch','SibSp','Title','Pclass']]
y=copy_data_ged['Age']
ls=Lasso(alpha=0.1)
ls.fit(x,y)
aged3=ls.predict(x)
plt.scatter(y,aged3)
np.mean((y-aged3)**2)
non_aged=all_data[all_data['Age'].isnull()][['Sex','Parch','SibSp','Title','Pclass']]
non_aged=all_data[all_data['Age'].isnull()][['Sex','Parch','SibSp','Title','Pclass']]
aged=ls.predict(non_aged)
all_data.loc[all_data['Age'].isnull(),'Age']=aged
pa.crosstab(all_data['Age'],all_data["Sex"]).plot(kind='kde')
non_aged_train=titanic_df[titanic_df['Age'].isnull()][['Sex','Parch','SibSp','Title','Pclass']]
non_aged_train=titanic_df[titanic_df['Age'].isnull()][['Sex','Parch','SibSp','Title','Pclass']]
aged_train=ls.predict(non_aged_train)
titanic_df.loc[titanic_df['Age'].isnull(),'Age']=aged_train
non_aged_test=test_df[test_df['Age'].isnull()][['Sex','Parch','SibSp','Title','Pclass']]
non_aged_test=test_df[test_df['Age'].isnull()][['Sex','Parch','SibSp','Title','Pclass']]
aged_test=ls.predict(non_aged_test)
test_df.loc[test_df['Age'].isnull(),'Age']=aged_test
all_data['FSize']=all_data['Parch']+all_data['SibSp']+1
titanic_df['FSize']=titanic_df['Parch']+titanic_df['SibSp']+1
test_df['FSize']=test_df['Parch']+test_df['SibSp']+1
pa.crosstab(all_data['FSize'],all_data['Survived']).plot(kind='bar')
all_data['Mother']=0
all_data['Child']=0
titanic_df['Mother']=0
titanic_df['Child']=0
test_df['Mother']=0
test_df['Child']=0
all_data.loc[(all_data['Age']>18) & (all_data['Sex']==1) & (all_data['Title']!=2) & (all_data['Parch']>0),'Mother']=1
all_data.loc[(all_data['Age']<18),'Child']=1
titanic_df.loc[(titanic_df['Age']>18) & (titanic_df['Sex']==1) & (titanic_df['Title']!=2) & (titanic_df['Parch']>0),'Mother']=1
titanic_df.loc[(titanic_df['Age']<18),'Child']=1
test_df.loc[(test_df['Age']>18) & (test_df['Sex']==1) & (test_df['Title']!=2) & (test_df['Parch']>0),'Mother']=1
test_df.loc[(test_df['Age']<18),'Child']=1
pa.crosstab(all_data['Mother'],all_data['Survived']).plot(kind='bar')
pa.crosstab(all_data['Child'],all_data['Survived']).plot(kind='bar')
grouped_class=all_data.groupby('Pclass')
grouped_class.mean()['Fare']
all_data['Estate']=0
def fillin_Estat(dataset):
    Estate=[]
    for index,data in dataset.iterrows():
        if data['Fare']>=90:
            Estate.append(3)
        elif (data['Fare']>=40) and(data['Fare']<90):
            Estate.append(2)
        else:
            Estate.append(1)
    return Estate
all_data.loc[:,'Estate']=fillin_Estat(all_data)
titanic_df['Estate']=0
titanic_df.loc[:,'Estate']=fillin_Estat(titanic_df)
test_df['Estate']=0
test_df.loc[:,'Estate']=fillin_Estat(test_df)
g = sns.factorplot(x="Estate", y="Survived", hue="Sex", data=all_data,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")
g = sns.factorplot(x="Embarked", y="Survived", hue="Sex", data=all_data,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")
corr = all_data.corr()
_ , ax = plt.subplots( figsize =( 12 , 10 ) )
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
_ = sns.heatmap(
    corr, 
    cmap = cmap,
    ax=ax, 
    annot = True, 
)
sns.set(style="ticks")
sns.boxplot(x="Pclass", y="Age", hue="Survived", data=all_data, palette="PRGn")
sns.despine(offset=10, trim=True)
facet = sns.FacetGrid(all_data , hue='Survived' ,aspect=3 , row = 'Sex' )
facet.map(sns.kdeplot , 'Age' , shade= True )
facet.add_legend()
g = sns.FacetGrid(all_data, row="Sex", col="Embarked", margin_titles=True)
bins = np.linspace(0, 60, 13)
g.map(plt.hist, "Age", color="steelblue", bins=bins, lw=0)
sns.barplot(x='FSize',y='Survived',data=all_data).set(ylabel='Survival Probability')
features=['Pclass','Sex','Age','SibSp','Parch','Embarked','FSize','Title','Mother','Child','Estate']
forest=RandomForestClassifier()
model=forest.fit(titanic_df[features],titanic_df['Survived'])
importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
for f in range(titanic_df[features].shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
plt.figure()
plt.title("Feature importances")
plt.bar(range(titanic_df[features].shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(titanic_df[features].shape[1]), indices)
plt.xlim([-1, titanic_df[features].shape[1]])
plt.show()
y_pred=model.predict(test_df[features])
submission = pa.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": y_pred
    })

submission.to_csv('submission.csv', index=False)