import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import kurtosis

from scipy.stats import skew
plt.rcParams['figure.figsize'] = (8,5) # set the default size of the plots

plt.style.use("ggplot")
train=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")
print(train.shape)   # TRAIN DataFrame has 891 rows & 12 columns

print(train.columns) 

train.sample(10)     # print 10 random observations => we can already see that there are missing values
print(test.shape)   # TEST DataFrame has 418 rows & 11 columns

print(test.columns) 

test.sample(10)     
train.info() 
test.info()
train.drop("PassengerId",axis=1).describe()
test.drop("PassengerId",axis=1).describe()
train.isna().sum()
test.isna().sum()
# Deleting the Cabin column from both datasets:



train.drop("Cabin",axis=1,inplace=True)

test.drop("Cabin",axis=1,inplace=True)
corrmatrix=train.drop("PassengerId",axis=1).corr()

sns.heatmap(corrmatrix,annot=True)
survived=(train.Survived.sum()/891)*100

not_survived=100-survived

print("{:.2f} % of people survived".format(survived))

print("{:.2f} % of people died".format(not_survived))
df=pd.DataFrame(train.groupby(["Pclass","Sex"]).sum().Survived)

df.reset_index(inplace=True)

df
sns.pointplot(x="Pclass",y="Survived",data=df,hue="Sex",palette={"female":"green","male":"purple"},

              markers=["*","^"],linestyles=["-","--"])

plt.title("Number of survivings grouped by sex & Pclass")
train["age_category"]=""



for i in range(891):

    if train.loc[i,"Age"]<1:

        train.loc[i,"age_category"]="Infants"

    elif 1<=train.loc[i,"Age"]<18:

        train.loc[i,"age_category"]="Children"

    elif 18<=train.loc[i,"Age"]<65:

        train.loc[i,"age_category"]="Adults"

    elif train.loc[i,"Age"]>=65:

        train.loc[i,"age_category"]="Elders"

    else:

        train.loc[i,"age_category"]="Unknown"

train.head()
df_2=pd.DataFrame(train.groupby(["age_category",]).sum().Survived)

df_2["Total"]=train["age_category"].value_counts()

df_2.reset_index(inplace=True)

df_2
for i in range(5):

    print("{:.2f} % of {} survived ".format((df_2.loc[i,"Survived"]/df_2.loc[i,"Total"])*100,df_2.loc[i,"age_category"]))
sns.barplot(x="age_category",y="Survived",data=df_2)

plt.title("Number of survivors grouped by  Age")
df_3=train.groupby("SibSp").sum().Survived

df_3
sns.barplot(y=df_3,x=df_3.index)

plt.title("Number of survivors grouped by number of Siblings/ Spouse")
df_4=train.groupby("Parch").sum().Survived

df_4
sns.barplot(y=df_4,x=df_4.index)

plt.title("Number of survivings grouped by number of Parents/ Children")
data={"Class":["Class1","Class2","Class3"],

      "Passengers":[train[train.Pclass==1].shape[0],train[train.Pclass==2].shape[0],train[train.Pclass==3].shape[0]]}
data={"Class":["Class1","Class2","Class3"],

      "Passengers":[train[train.Pclass==1].shape[0],train[train.Pclass==2].shape[0],train[train.Pclass==3].shape[0]]}

Class=pd.DataFrame(data,columns=["Class","Passengers"])

Class
sns.barplot(x=Class.Class,y=Class.Passengers)

plt.title("Number of passengers grouped by class")
status=[]

for i in range(891):

    status.append(train.Name[i].split()[1])

status[1:10]
train["status"]=""

for i in range(891):

    if "Mr." in (train.Name[i].split()):

        train.loc[i,"status"]="Mister"

    elif "Miss." in (train.Name[i].split()):

        train.loc[i,"status"]="Miss"

    elif "Mrs." in (train.Name[i].split()):

        train.loc[i,"status"]="Mistress"

    elif "Rev." in (train.Name[i].split()):

        train.loc[i,"status"]="Reverend"

    elif "Master." in (train.Name[i].split()):

        train.loc[i,"status"]="Master"

    elif "Mlle." in (train.Name[i].split()):

        train.loc[i,"status"]="Mademoiselle"

    elif "Dr." in (train.Name[i].split()):

        train.loc[i,"status"]="Doctor"

    else:

        train.loc[i,"status"]="Other" 
train.status.value_counts()
train[train.status=='Other']
print(train.Sex.value_counts())

print("{:.2f} % of people were males".format((train.Sex.value_counts()[0]/891)*100))

print("{:.2f} % of people were females".format((train.Sex.value_counts()[1]/891)*100))
print(train.groupby("Sex").sum().Survived)

print("{:.2f} % of females survived".format((train.groupby("Sex").sum().Survived[0]/314)*100))

print("{:.2f} % of males survived".format((train.groupby("Sex").sum().Survived[1]/577)*100))

sns.boxplot(y="Age",x="Sex",data=train,palette="gist_stern")

plt.title("Distribution of age grouped by sex")
sns.swarmplot(y="Age",x="Sex",hue="age_category",data=train,palette="Set1")

plt.title("Distribution of age grouped by sex")
plt.figure(figsize=(10,6))

sns.distplot(train.Age,kde=False)

plt.title("Distribution of age")
print("Skewness of Age distribution is {:.2f}".format(skew(train.dropna().Age)))

print("Kurtosis of Age distribution is {:.2f}".format(kurtosis(train.dropna().Age)))
train.SibSp.value_counts()
train.Parch.value_counts()
train.drop("Ticket",axis=1,inplace=True)

train.head()
train.groupby("Pclass").mean().Fare
min_3=train[train.Pclass==3].Fare.min()

max_3=train[train.Pclass==3].Fare.max()

min_2=train[train.Pclass==2].Fare.min()

max_2=train[train.Pclass==2].Fare.max()

min_1=train[train.Pclass==1].Fare.min()

max_1=train[train.Pclass==1].Fare.max()

print("Minimum price for class 3 is {}".format(min_3))

print("Maximum price for class 3 is {}".format(max_3))

print("Minimum price for class 2 is {}".format(min_2))

print("Maximum price for class 2 is {}".format(max_2))

print("Minimum price for class 1 is {}".format(min_1))

print("Maximum price for class 1 is {}".format(max_1))
sns.boxplot(train.Fare)
sns.swarmplot(y="Fare",data=train,color="purple")
sns.distplot(train.Fare,kde=False)

plt.xlim(0,300)

print(skew(train.Fare))

print(kurtosis(train.Fare))
train.Embarked.value_counts()
embarked=pd.DataFrame(train[["Pclass","Embarked"]].value_counts(sort=False))

embarked.reset_index(inplace=True)

embarked.columns=["Class","Embarked","No. of passengers"]

embarked
sns.pointplot(x="Embarked",y="No. of passengers",hue="Class",data=embarked,

               markers=["^","o","*"],linestyles=["-","--",":"])

position=(0,1,2)

labels=("Cherbourg","Queenstown","Southampton")

plt.xticks(position,labels)

plt.title("Number of passengers grouped by class and embarked place")

plt.show()
train[(train.status=="Master") & (train.Age.isna())]
mean_age_children=round(train[(train["age_category"]=="Children") | (train["age_category"]=="Infants")].Age.mean())

mean_age_children

train[(train.status=="Master") & (train.Age.isna())]=train[(train.status=="Master") & (train.Age.isna())].fillna(mean_age_children)

train[train.status=="Master"].Age.isna().sum() # Succesfully replaced the NaN values
train[(train.status=="Mistress") & (train.Age.isna())]
mean_age_mistress=round(train[train.status=="Mistress"].Age.mean())

mean_age_mistress
train[(train.status=="Mistress") & (train.Age.isna())]=train[(train.status=="Mistress") &(train.Age.isna())].fillna(mean_age_mistress)

train[train.status=="Mistress"].Age.isna().sum() 
train[(train.status=="Mister")].shape
train[(train.status=="Mister") & (train.Age.isna())]
mean_age_mister=round(train[train.status=="Mister"].Age.mean())

mean_age_mister
train[(train.status=="Mister") & (train.Age.isna())]=train[(train.status=="Mister") & (train.Age.isna())].fillna(mean_age_mister)

train[train.status=="Mister"].Age.isna().sum() 
for i in range(888):

    if train.loc[i,"Age"]<1:

        train.loc[i,"age_category"]="Infants"

    elif 1<=train.loc[i,"Age"]<18:

        train.loc[i,"age_category"]="Children"

    elif 18<=train.loc[i,"Age"]<65:

        train.loc[i,"age_category"]="Adults"

    elif train.loc[i,"Age"]>=65:

        train.loc[i,"age_category"]="Elders"

    else:

        train.loc[i,"age_category"]="Unknown"
print(train.Age.isna().sum())

train[train.Age.isna()]
mean_age_maleAdult=round(train[(train["age_category"]=="Adults") &(train.Sex=="male")].Age.mean())

mean_age_maleAdult
mean_age_femaleAdult=round(train[(train["age_category"]=="Adults") &(train.Sex=="female")].Age.mean())

print(mean_age_femaleAdult)

mean_age_femaleChildren=round(train[(train["age_category"]=="Children") &(train.Sex=="female")].Age.mean())

print(mean_age_femaleChildren)
train[(train.Age.isna()) & (train.Parch==0)]=train[(train.Age.isna()) & (train.Parch==0)].fillna(mean_age_femaleAdult)

train[train.Age.isna()] # these are the remaining records with NaN values
train[train.Age.isna()]=train[train.Age.isna()].fillna(mean_age_femaleChildren)
for i in range(891):

    if train.loc[i,"Age"]<1:

        train.loc[i,"age_category"]="Infants"

    elif 1<=train.loc[i,"Age"]<18:

        train.loc[i,"age_category"]="Children"

    elif 18<=train.loc[i,"Age"]<65:

        train.loc[i,"age_category"]="Adults"

    elif train.loc[i,"Age"]>=65:

        train.loc[i,"age_category"]="Elders"

    else:

        train.loc[i,"age_category"]="Unknown"
train.isna().sum()
train[train.Embarked.isna()]
train.fillna("S",inplace=True)

train.isna().sum()
test.drop("Ticket",axis=1,inplace=True)



test["age_category"]=""



for i in range(418):

    if test.loc[i,"Age"]<1:

        test.loc[i,"age_category"]="Infants"

    elif 1<=test.loc[i,"Age"]<18:

        test.loc[i,"age_category"]="Children"

    elif 18<=train.loc[i,"Age"]<65:

        test.loc[i,"age_category"]="Adults"

    elif test.loc[i,"Age"]>=65:

        test.loc[i,"age_category"]="Elders"

    else:

        test.loc[i,"age_category"]="Unknown"



test["status"]=""

for i in range(418):

    if "Mr." in (test.Name[i].split()):

        test.loc[i,"status"]="Mister"

    elif "Miss." in (test.Name[i].split()):

        test.loc[i,"status"]="Miss"

    elif "Mrs." in (test.Name[i].split()):

        test.loc[i,"status"]="Mistress"

    elif "Rev." in (test.Name[i].split()):

        test.loc[i,"status"]="Reverend"

    elif "Master." in (test.Name[i].split()):

        test.loc[i,"status"]="Master"

    elif "Dr." in (test.Name[i].split()):

        test.loc[i,"status"]="Doctor"

    else:

        test.loc[i,"status"]="Other" 

        

mean_age_children=round(test[(test["age_category"]=="Children") | (test["age_category"]=="Infants")].Age.mean())

test[(test.status=="Master") & (test.Age.isna())]=test[(test.status=="Master") & (test.Age.isna())].fillna(mean_age_children)



mean_age_mistress=round(test[test.status=="Mistress"].Age.mean())

test[(test.status=="Mistress") & (test.Age.isna())]=test[(test.status=="Mistress") &(test.Age.isna())].fillna(mean_age_mistress)



mean_age_mister=round(test[test.status=="Mister"].Age.mean())

test[(test.status=="Mister") & (test.Age.isna())]=test[(test.status=="Mister") & (test.Age.isna())].fillna(mean_age_mister)



for i in range(418):

    if test.loc[i,"Age"]<1:

        test.loc[i,"age_category"]="Infants"

    elif 1<=test.loc[i,"Age"]<18:

        test.loc[i,"age_category"]="Children"

    elif 18<=train.loc[i,"Age"]<65:

        test.loc[i,"age_category"]="Adults"

    elif test.loc[i,"Age"]>=65:

        test.loc[i,"age_category"]="Elders"

    else:

        test.loc[i,"age_category"]="Unknown"

        

mean_age_femaleAdult=round(test[(test["age_category"]=="Adults") &(test.Sex=="female")].Age.mean())

mean_age_femaleChildren=round(test[(test["age_category"]=="Children") &(test.Sex=="female")].Age.mean())



test[(test.Age.isna()) & (test.Parch==0)]=test[(test.Age.isna()) & (test.Parch==0)].fillna(mean_age_femaleAdult)

test[test.Age.isna()]=test[test.Age.isna()].fillna(mean_age_femaleChildren)



for i in range(418):

    if test.loc[i,"Age"]<1:

        test.loc[i,"age_category"]="Infants"

    elif 1<=test.loc[i,"Age"]<18:

        test.loc[i,"age_category"]="Children"

    elif 18<=test.loc[i,"Age"]<65:

        test.loc[i,"age_category"]="Adults"

    elif test.loc[i,"Age"]>=65:

        test.loc[i,"age_category"]="Elders"

    else:

        test.loc[i,"age_category"]="Unknown"

        

fare_mean=test[test.Pclass==3].Fare.mean()

test.fillna(fare_mean,inplace=True)



print(test.isna().sum())



from sklearn.preprocessing import LabelEncoder





label_encoder = LabelEncoder()



col=["Sex","Embarked","age_category"]

for i in col:

    train[i]=label_encoder.fit_transform(train[i])

    test[i]=label_encoder.fit_transform(test[i])

corrmatrix=train.drop("PassengerId",axis=1).corr()

sns.heatmap(corrmatrix,annot=True)
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

features1=["Pclass","Sex","Fare","age_category"]

X1=train[features1]

y1=train.Survived



X_train1, X_test1, y_train1,y_test1=train_test_split(X1,y1,test_size=0.22,random_state=0)





DT1=DecisionTreeClassifier(random_state=1)

DT1.fit(X_train1,y_train1)

y_pred1=DT1.predict(X_test1)

score=(round(accuracy_score(y_pred1,y_test1)*100,2))

score

features2=["Pclass","Sex","Fare"]

X2=train[features2]

y2=train.Survived



X_train2, X_test2, y_train2,y_test2=train_test_split(X2,y2,test_size=0.21,random_state=0)





DT2=DecisionTreeClassifier(random_state=1)

DT2.fit(X_train2,y_train2)

y_pred2=DT2.predict(X_test2)

score=round(accuracy_score(y_pred2,y_test2)*100,2)

score
from sklearn.ensemble import GradientBoostingClassifier



xgb = GradientBoostingClassifier(random_state=1)

xgb.fit(X_train1, y_train1)

y_pred = xgb.predict(X_test1)

score_xgb = round(accuracy_score(y_pred, y_test1) * 100, 2)

score_xgb
Jack=[3,1,7.89,0] # class=3, male=1, the median of price for class 3= 7.89, adult=0

Rose=[1,0,60,0]   



predictions=xgb.predict([Jack,Rose])

print("Did Jack survived?  {}".format(predictions[0]))

print("Did Rose survived?  {}".format(predictions[1]))
ids = test['PassengerId']

X_test=test[features1]

predictions = xgb.predict(X_test)



df = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

df.to_csv('submission.csv', index=False)