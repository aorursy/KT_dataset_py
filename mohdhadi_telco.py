import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df=pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head(15)
df.dtypes
df.drop("customerID",axis=1,inplace=True)
df.isnull().sum()

df.Dependents=df.Dependents.replace({"No":0, "Yes":1})

df.Partner=df.Partner.replace({"No":0, "Yes":1})

df.PhoneService=df.PhoneService.replace({"No":0, "Yes":1})

df.OnlineSecurity=df.OnlineSecurity.replace({"No":0,"No internet service":0, "Yes":1})

df.OnlineBackup=df.OnlineBackup.replace({"No":0,"No internet service":0, "Yes":1})

df.DeviceProtection=df.DeviceProtection.replace({"No":0,"No internet service":0, "Yes":1})

df.StreamingTV=df.StreamingTV.replace({"No":0,"No internet service":0,"Yes":1})

df.StreamingMovies=df.StreamingMovies.replace({"No":0,"No internet service":0, "Yes":1})

df.PaperlessBilling=df.PaperlessBilling.replace({"No":0, "Yes":1})

df.TechSupport=df.TechSupport.replace({"No":0,"No internet service":0, "Yes":1})

df.Churn=df.Churn.replace({"No":0, "Yes":1})

df.MultipleLines=df.MultipleLines.replace({"No":0,"No phone service":0,"Yes":1})
df.head(5)
sns.countplot("MultipleLines",data=df)

plt.show()

sns.countplot("Churn",data=df,hue="gender")

plt.show()

a=list(df["Contract"].value_counts().values)

def make_autopct(a):

    def my_autopct(pct):

        total = sum(a)

        val = int(round(pct*total/100.0))

        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)

    return my_autopct



plt.pie(a,labels=["Monthly","Two Year","One Year"],explode=(0.001,0.03,0.08), autopct=make_autopct(a))

fig = plt.gcf()

fig.set_size_inches(8,8)

plt.show()
df1=df[(df["InternetService"]=="DSL") | (df["InternetService"]== "Fiber optic")]

sns.countplot("InternetService",data=df1,hue="StreamingMovies")

fig = plt.gcf()

fig.set_size_inches(10,5)

plt.show()

group_size=list(df.InternetService.value_counts().values)

group_names=list(df.InternetService.value_counts().index)

a=df.groupby([df.InternetService,df.Contract],as_index=False).count()

ddf = pd.DataFrame({"service":a.InternetService,"contract":a.Contract,"value":a.gender})

subgroup_size=list(list(ddf.value.values))

subgroup_names=list(ddf.contract.values)



temp=group_size[0]

group_size[0]=group_size[1]

group_size[1]=temp

temp1=group_names[0]

group_names[0]=group_names[1]

group_names[1]=temp1



fig, ax = plt.subplots()

ax.axis('equal')

mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names,startangle=90)

plt.setp( mypie, width=0.5, edgecolor='white')

fig.set_size_inches(8,8)

plt.title('Distribution of type of service & type of contract')

mypie2, _ = ax.pie(subgroup_size, radius=1.3-0.3, labels=subgroup_names, labeldistance=0.3,startangle=90,autopct='%.1f%%')

plt.setp( mypie2, width=0.4, edgecolor='white')



plt.margins(0,0)



plt.show()

a=df.groupby([df.InternetService,df.Contract],as_index=False).count()

dsl=list(ddf[ddf.service=="DSL"].value)

fib=list(ddf[ddf.service=="Fiber optic"].value)



def make_autopct(dsl):

    def my_autopct(pct):

        total = sum(dsl)

        val = int(round(pct*total/100.0))

        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)

    return my_autopct





plt.pie(dsl,labels=["Monthly","One Year","Two Year"],autopct=make_autopct(dsl),explode=(0.02,0.02,0.02),)

plt.title("Distribution of DSL")

fig = plt.gcf()

fig.set_size_inches(5,5)

plt.show()



plt.pie(fib,labels=["Monthly","One Year","Two Year"],autopct=make_autopct(fib),explode=(0.02,0.02,0.02),)

plt.title("Distribution of Fiber optic")

fig = plt.gcf()

fig.set_size_inches(5,5)

plt.show()

sns.catplot(x="InternetService", y="tenure", hue="Contract", kind="bar", data=df)

plt.show()
sns.distplot(df["tenure"],hist=True,vertical=True)
sns.boxplot(x="Churn", y="tenure", data=df)

plt.show()
d=pd.get_dummies(df.InternetService)

df=pd.concat([df, d], axis=1)

e=pd.get_dummies(df.Contract)

df=pd.concat([df, e], axis=1)

df=df.drop(["InternetService","Contract","PaymentMethod","TotalCharges","MonthlyCharges"],axis=1)

df.gender=df.gender.replace({"Female":0, "Male":1})
X=df.loc[:,df.columns!="Churn"].values

y=df.loc[:,df.columns=="Churn"].values.flatten()

from sklearn.model_selection import train_test_split

X_train , X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#Model 1 - Logistic Regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred)*100)
#Model 2 - Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

modelrf = RandomForestClassifier(n_estimators=2,oob_score=True,min_samples_leaf = 40)

modelrf.fit(X_train,y_train)

y_predrf=modelrf.predict(X_test)

print(accuracy_score(y_test,y_predrf)*100)
#Model 3 - Support Vector Machine

from sklearn import svm

modelsvm=svm.SVC(kernel="linear")

modelsvm.fit(X_train,y_train)

y_predsvm=modelsvm.predict(X_test)

print(accuracy_score(y_test,y_predsvm)*100)