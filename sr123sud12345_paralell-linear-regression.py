from pandas import Series,DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")
import statistics
from scipy import stats
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures 
df=pd.read_csv("responses.csv")
df.head(1)
def replacer_mean(dff):
    r0=np.mean(df1)
    r1=r0.index
    r2=r0.values
    for i in np.arange(len(r1)):
          ri=r1[i]
          rv=r2[i]
          dff[ri].fillna(value=rv)
          dff.replace([np.inf,-np.inf],rv)
    return(dff)

def nan_remover(v,vm):
    vr=[]
    for i in np.arange(len(v)):
        if str(v[i])=="nan":
            vr=np.append(vr,vm)
        else:
            vr=np.append(vr,v[i])
    return(vr)

def replacer_mean(dff):
    dff1=dff
    r0=np.mean(dff)
    r1=r0.index
    r2=r0.values
    for i in np.arange(len(r1)):
          ri=r1[i]
          rv=r2[i]
          dff1[ri].fillna(value=rv)
          dff1[ri]=(nan_remover(dff[ri].values,rv))
    return(dff1)

def numriser(a):
    a1=[]
    for i in np.arange(len(a)):
        a1=np.append(a1,round(a[i]))
    return(a1)

def pie_plotter(var):
    dfm=pd.get_dummies(df[var])
    sm=(dfm.sum())
    #plt.subplot(2, 1, 1)
    #sm.plot(kind="pie",figsize=(10,10),fontsize=10,title=var,table=True)
    #plt.subplot(2, 1, 2)
    df1[var].plot(figsize=(10,10),kind="kde")

def bi_var(var1,var2,dff):

    sns.lmplot(var1,var2,dff,order=1,scatter=False,
           scatter_kws={"marker":"o","color":"red"},
              line_kws={"linewidth":1,"color":"blue"})
    r=np.corrcoef(dff[var1],dff[var2])[0,1]
    print("correlation_coef between",var1,"and",var2,"is",r)

def counter(m):
    u=[1,2,3,4,5]
    n=len(m)
    s1=[]
    for i in u:
        s=0
        for j in np.arange(n):
            if i==m[j]:
                s=s+1
        s1=np.append(s1,s)
    return(s1)

def counter2(m):
    u=np.unique(m)
    n=len(m)
    s1=[]
    for i in u:
        s=0
        for j in np.arange(n):
            if i==m[j]:
                s=s+1
        s1=np.append(s1,s)
    s11=DataFrame(s1,index=u).T
    return(s11)

def dpie_plotter(var,al):
    labels1=np.unique(df_r[al])
    labels=labels1[::-1] 
    dfm=pd.get_dummies(df1[var],columns=labels)
    sm=(dfm.sum())
    plt.subplot(2, 1, 1)
    sm.plot(kind="pie",figsize=(10,10),fontsize=10,title="Training data set",table=True,autopct='%1.1f%%')
    plt.subplot(2, 1, 2)
    dfm=pd.get_dummies(df_r2[var],columns=labels)
    sm1=(dfm.sum())
    sm1.plot(kind="pie",figsize=(10,10),fontsize=10,title="Response data set",table=True,autopct='%1.1f%%')
    plt.legend(labels, loc="best")

def happiness(n):
    for i in np.arange(len(dfr_pre["Name"].values)):
        if dfr_pre["Name"][i]==n:
            hpp=dfr_pre["index of happiness"][i]
        if dfr_pre["Name"][i]==n:
            hpi=dfi_pre["index of happiness"][i]
    if hpp==1:
        hpr="you are really not happy and your index of happiness is "
    elif hpp==2:
        hpr="you are not very much happy and your index of happiness is "
    else:
        hpr="you are very happy and your index of happiness is "
    print(n,hpr,hpi,".")
y=df["Happiness in life"]
df1=df[["Dance","Shopping","Fun with friends","Parents' advice","Eating to survive",
    "Pets","Darkness","Fear of public speaking","Smoking","Alcohol","Economy Management","Healthy eating",
        "Decision making","Workaholism","Friends versus money","Loneliness",
   "God","Dreams","Number of friends","Socializing","Entertainment spending","Age","Height","Weight",
   "Gender","Only child","Village - town","Internet usage"]]
df_music=(df[df.columns[0:19]])
music=numriser(np.mean(df_music.T))
df_movies=(df[df.columns[19:33]])
movies=numriser(np.mean(df_movies.T))

edu=[]
for i in (df["Education"].values):
    if i=="secondary school":
        edu=np.append(edu,1)
    elif i=="masters degree":
        edu=np.append(edu,2)
    elif i=="college/bachelor degree":
        edu=np.append(edu,3)
    elif i=="doctorate degree":
        edu=np.append(edu,4)
    else:
        edu=np.append(edu,5)
a7=[]
for i in np.arange(len(df1["Height"])):
    a17=df1["Height"][i]/df1["Weight"][i]
    a7=np.append(a7,a17)

from sklearn import preprocessing
sc=preprocessing.MinMaxScaler(feature_range=(1,5))
a8=sc.fit_transform(df1["Age"].values.reshape(-1,1))

a1=[]
for i in df1["Gender"].values:
    if i=="female":
        a1=np.append(a1,0)
    else:
        a1=np.append(a1,1)

a2=[]
for i in df1["Only child"].values:
    if i=="no":
        a2=np.append(a2,0)
    else :
        a2=np.append(a2,1)

a3=[]
for i in df1["Village - town"].values:
    if i=="village":
        a3=np.append(a3,1)
    else:
        a3=np.append(a3,0)

a4=[]
for i in df1["Internet usage"].values:
    if i=="few hours a day":
        a4=np.append(a4,0)
    elif i=="less than an hour a day":
        a4=np.append(a4,1)
    else :
        a4=np.append(a4,2)

a5=[]
for i in df1["Smoking"].values:
    if i=="never smoked":
        a5=np.append(a5,0)
    elif i=="tried smoking":
        a5=np.append(a5,1)
    elif i=="former smoker":
        a5=np.append(a5,2)
    else:
        a5=np.append(a5,3)

a6=[]
for i in df1["Alcohol"]:
    if i=="drink a lot":
        a6=np.append(a6,0)
    elif i=="social drinker":
        a6=np.append(a6,1)
    else:
        a6=np.append(a6,2)

df1["hw_rat"]=a7
df1["age"]=a8
df1["gender"]=a1
df1["alcohol"]=a6
df1["smoke"]=a5
df1["only_child"]=a2
df1["area"]=a3
df1["internet"]=a4
df1["music"]=music
df1["movie"]=movies
df1["education"]=edu

df1=df1.drop(["Height","Weight","Age","Gender","Only child","Internet usage","Smoking","Village - town","Alcohol"],axis=1)

dfy=DataFrame(y)
dfy=dfy.fillna(value=np.mean(y))
dfy=DataFrame(nan_remover(dfy["Happiness in life"].values,np.mean(y)))
r0=(np.mean(df1))
r1=r0.index
r2=r0.values
df2=df1
df1=replacer_mean(df2)
y11=dfy.values
y22=[]
for i in np.arange(len(y11)):
    if (y11[i]<2.5):
        y22=np.append(y22,1)
    elif (y11[i]>3.5):
        y22=np.append(y22,3)
    else:
        y22=np.append(y22,2)
col=df1.columns
df_pre=DataFrame([],index=np.arange(len(df1["Dance"].values)))
for i in np.arange(len(col)):
    c1=col[i]
    model_le=LinearRegression()
    model_le.fit((df1[c1].values).reshape(-1,1),y22)
    y_pre=model_le.predict((df1[c1].values).reshape(-1,1))
    df_pre[c1]=y_pre   
p=0
p1=[]
for i in np.arange(len(df["Age"].values)):
    p2=(np.mean((df_pre[i:i+1]).values))
    #if p2>2.53:
        #p3=3
    #elif p2<2.52:
        #p3=1
    #else:
        #p3=2
    p1=np.append(p1,p2)
#((np.mean(df_pre.T)[1:20])).plot()
plt.plot(y22[0:20])
#plt.plot(p1[0:20])
plt.plot(p1[0:20])
df1["y"]=y22
t44=(df1.corr()["y"].values[:-1])
df1=df1.drop("y",axis=1)
df1=t44*df1
df_pre=DataFrame([],index=np.arange(len(df1["Dance"].values)))
for i in np.arange(len(col)):
    c1=col[i]
    model_le=LinearRegression()
    model_le.fit((df1[c1].values).reshape(-1,1),y22)
    y_pre=model_le.predict((df1[c1].values).reshape(-1,1))
    df_pre[c1]=y_pre   
p=0
p1=[]
for i in np.arange(len(df["Age"].values)):
    p2=(np.mean((df_pre[i:i+1]).values))
    #if p2>2.53:
        #p3=3
    #elif p2<2.52:
        #p3=1
    #else:
        #p3=2
    p1=np.append(p1,p2)
#((np.mean(df_pre.T)[1:20])).plot()
plt.plot(y22[0:20])
#plt.plot(p1[0:20]*np.mean(t44)/3)
plt.plot(p1[0:20])
df_r=pd.read_csv("Responses11.csv")
#df_r=df_r.drop("Unnamed: 35",axis=1)
df_r.tail(2)
df_r1=df_r[["Timestamp","Name","Comments about yourself (Hostel room number, Class, Department etc.) ","Your email id"]]
df_r2=df_r.drop(["Timestamp","Name","Comments about yourself (Hostel room number, Class, Department etc.) ","Your email id"],axis=1)
h=[]
for i in np.arange(len(df_r2["Height (in feet)"])):
    h=np.append(h,(df_r2["Height (in feet)"][i]*30.48))
edu=[]
for i in (df_r2["Highest level of education attained?"].values):
    if i=="Secondary school":
        edu=np.append(edu,1)
    elif i=="Masters degree":
        edu=np.append(edu,2)
    elif i=="Secondary school":
        edu=np.append(edu,3)
    elif i=="Doctorate degree":
        edu=np.append(edu,4)
    else:
        edu=np.append(edu,5)
a7=[]
for i in np.arange(len(df_r2["Weight (in kgs)"])):
    a17=h[i]/df_r2["Weight (in kgs)"][i]
    a7=np.append(a7,a17)

from sklearn import preprocessing
sc=preprocessing.MinMaxScaler(feature_range=(1,5))
a8=sc.fit_transform(df_r2["Age"].values.reshape(-1,1))

a1=[]
for i in df_r2["Sex"].values:
    if i=="Female":
        a1=np.append(a1,0)
    else:
        a1=np.append(a1,1)

a2=[]
for i in df_r2["Are you the only child of your parents?"].values:
    if i=="No":
        a2=np.append(a2,0)
    else :
        a2=np.append(a2,1)

a3=[]
for i in df_r2["Where did you spend most of you childhood?"].values:
    if i=="Village":
        a3=np.append(a3,1)
    else:
        a3=np.append(a3,0)

a4=[]
for i in df_r2["Internet usage in a day."].values:
    if i=="More than an hour but less than 4 hours":
        a4=np.append(a4,0)
    elif i=="More than 4 hours":
        a4=np.append(a4,2)
    else :
        a4=np.append(a4,1)

a5=[]
for i in df_r2["Do you smoke?"].values:
    if i=="Never smoked":
        a5=np.append(a5,0)
    elif i=="Tried smoking":
        a5=np.append(a5,1)
    elif i=="Former smoker":
        a5=np.append(a5,2)
    else:
        a5=np.append(a5,3)

a6=[]
for i in df_r2["Level of alcohol consumption."]:
    if i=="Drink a lot":
        a6=np.append(a6,0)
    elif i=="Social drinker":
        a6=np.append(a6,1)
    else:
        a6=np.append(a6,2)

df_r2["hw_rat"]=a7
df_r2["age"]=a8
df_r2["gender"]=a1
df_r2["alcohol"]=a6
df_r2["smoke"]=a5
df_r2["only_child"]=a2
df_r2["area"]=a3
df_r2["internet"]=a4
df_r2["education"]=edu

df_r2=df_r2.drop(["Weight (in kgs)","Age","Sex","Are you the only child of your parents?","Internet usage in a day.","Do you smoke?","Where did you spend most of you childhood?","Level of alcohol consumption."],axis=1)
df_r2=df_r2.drop(["Highest level of education attained?"],axis=1)
df_r2=df_r2.drop(["Height (in feet)"],axis=1)
r0=(np.mean(df_r2))
r1=r0.index
r2=r0.values
df_r21=df_r2
df_r2=replacer_mean(df_r21)
y22
for i in np.arange(len(col)):
    c1=col[i]
    model_le=LinearRegression()
    model_le.fit((df1[c1].values).reshape(-1,1),y22)
    y_pre=model_le.predict((df1[c1].values).reshape(-1,1))
    df_pre[c1]=y_pre   
col1=df_r2.columns
for i in np.arange(len(col1)):
    c1=col1[i]
    #model_le=LinearRegression()
    #model_le.fit((df_r2[c1].values).reshape(-1,1),y22)
    y_pre=model_le.predict(((DataFrame(x22))[0].values).reshape(-1,1))
    df_pre[c1]=y_pre   
import random
name=["aparna","aliva","sr","shivam","nancy"]
turn=0
df=DataFrame((np.zeros(500)).reshape(100,5),columns=name)
i=random.randint(0,len(name)-1)
nm=name[i]
if i==0:
    print(nm," your turn")
if i==1:
    print(nm," your turn")
if i==2:
    print(nm," your turn")
if i==3:
    print(nm,"your turn")
if i==4:
    print(nm,"your turn")
win=nm
df[win][turn]=1
turn=turn+1
#df["aliva"][0]=1
#df["sr"][1]=1
df[0:turn+1]
DataFrame(sum(df.values),index=name)




num=[0,1,2,3,4,6,7,8]
n=len(num)
p=3
c(n,p)
import math
from itertools import permutations
perm = permutations(num,p) 
  
# Print the obtained permutations 
for i in list(perm): 
    if sum(i)==19:
        print (i) 












