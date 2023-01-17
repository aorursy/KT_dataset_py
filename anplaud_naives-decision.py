import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Visualisation library
from IPython.display import display
titanic = pd.read_csv("../input/titanic.csv")
titanic
titanic.hist(figsize=(14,14))
df_class = pd.DataFrame([titanic[titanic.Survived==1]['Pclass'].value_counts(),
                         titanic[titanic.Survived==0]['Pclass'].value_counts()])
df_class.index = ['Survived','Died']

display(df_class)
print("Percentage of Class 1 that survived:" ,round(df_class.iloc[0,0]/df_class.iloc[:,0].sum()*100),"%")
print("Percentage of Class 2 that survived:" ,round(df_class.iloc[0,1]/df_class.iloc[:,1].sum()*100), "%")
print("Percentage of Class 3 that survived:" ,round(df_class.iloc[0,2]/df_class.iloc[:,2].sum()*100), "%")

df_class.plot(kind='bar',figsize=(14,6))
df_sex = pd.DataFrame([titanic[titanic.Survived == 1]['Sex'].value_counts(), 
                       titanic[titanic.Survived == 0]['Sex'].value_counts()])
df_sex.index = ['Survived','Died']

display(df_sex) 

print("Percentage of female that survived:" ,round(df_sex.female[0]/df_sex.female.sum()*100), "%")
print("Percentage of male that survived:" ,round(df_sex.male[0]/df_sex.male.sum()*100), "%")

df_sex.plot(kind='bar', figsize=(14,6) )

titan = titanic[['Survived', 'Pclass', 'Age', 'Sex','Fare', 'Embarked']]
titan[titan.isnull().any(axis=1)]
titan.dropna(inplace=True)
titan
titan["Age_cat"]=pd.qcut(titan.Age,3,labels=["Age1","Age2","Age3"])
titan.head()
df_age = pd.DataFrame([titan[titan.Survived == 1]['Age_cat'].value_counts(), 
                       titan[titan.Survived == 0]['Age_cat'].value_counts()])
df_age.index = ['Survived','Died']

display(df_age) 

print("Percentage of Age1 that survived:" ,round(df_age.iloc[0,0]/df_age.iloc[:,0].sum()*100),"%")
print("Percentage of Age2 that survived:" ,round(df_age.iloc[0,1]/df_age.iloc[:,1].sum()*100), "%")
print("Percentage of Age3 that survived:" ,round(df_age.iloc[0,2]/df_age.iloc[:,2].sum()*100), "%")

df_age.plot(kind='bar', figsize=(14,6) )
titan["Fare_cat"]=pd.qcut(titan.Fare,3,labels=[0,1,2])
titan.Fare_cat = titan.Fare_cat.astype(int)
titan.head()
df_fare = pd.DataFrame([titan[titan.Survived == 1]['Fare_cat'].value_counts(), 
                       titan[titan.Survived == 0]['Fare_cat'].value_counts()])
df_fare.index = ['Survived','Died']

display(df_fare) 

print("Percentage of Fare1 that survived:" ,round(df_fare.iloc[0,0]/df_fare.iloc[:,0].sum()*100),"%")
print("Percentage of Fare that survived:" ,round(df_fare.iloc[0,1]/df_fare.iloc[:,1].sum()*100), "%")
print("Percentage of Fare3 that survived:" ,round(df_fare.iloc[0,2]/df_fare.iloc[:,2].sum()*100), "%")

df_fare.plot(kind='bar', figsize=(14,6) )
titan[['Fare_cat','Pclass']].corr()
tit = titan[['Survived','Pclass','Sex','Embarked']]
tit = pd.concat([tit,pd.get_dummies(tit.Sex)],axis=1).drop(['Sex','female'],axis=1)
tit
tit = pd.concat([tit,pd.get_dummies(tit.Embarked)],axis=1).drop(['Embarked','S'],axis=1)
percent = 0.7
in_train = np.random.binomial(1, percent, size=len(tit)).astype('bool')

tit_train = tit[in_train]
tit_test = tit[~in_train]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(tit_train[['Pclass','male','C','Q']],tit_train.Survived)
neigh.score(tit_test[['Pclass','male','C','Q']],tit_test.Survived)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(tit_train[['Pclass','male','C','Q']],tit_train.Survived)
clf.score(tit_test[['Pclass','male','C','Q']],tit_test.Survived)
from sklearn.tree import DecisionTreeClassifier
arb= DecisionTreeClassifier(random_state=0)
arb.fit(tit_train[['Pclass','male','C','Q']],tit_train.Survived)
arb.score(tit_test[['Pclass','male','C','Q']],tit_test.Survived)



