import os
for dirname, _, filenames in os.walk('/kaggle/input/titanic-dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
train = pd.read_csv("../input/train.csv")
train.head()
train.describe()
train.info()
for col in train.columns:
    print(col,train[col].isnull().sum())
train.shape
survived = train['Survived']
survived.value_counts()
survived.head()
data = survived.value_counts()
print(data)
labels = ['no','yes']
colors = ['blue','yellow']
#MAKING A PIE CHART FOR VISUALIZATION
plt.pie(data,labels = labels,colors=colors,autopct='%1.1f%%')
plt.show()
survived = pd.DataFrame(survived)
survived.shape
survived.info()
survived.describe()
import pandas_profiling
pandas_profiling.ProfileReport(train)
'''import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot'''
'''col = "Sex"
grouped = train[col].value_counts().reset_index()
grouped = grouped.rename(columns = {col : "count", "index" : col})

## plot
trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0])
layout = {'title': 'Sex(male, female)'}
fig = go.Figure(data = [trace], layout = layout)
fig.layout.template='presentation'
iplot(fig)'''
train.head()
data = train['Gender']
plt.pie(data.value_counts(),labels=data.unique(),colors = ['orange','purple'],autopct='%1.1f%%')
plt.show()
'''from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
le = LabelEncoder()'''
#train['Sex'] = le.fit_transform(train['Sex'])
#train['Sex'] = to_categorical(train['Sex'])
survived = train[train['Survived'] == 1]
dead = train[train['Survived'] == 0]
plt.pie(survived['Gender'].value_counts(),labels=survived['Gender'].unique(),colors=['orange','pink'],autopct='%1.1f%%')
print("Survived")
plt.show()
plt.pie(dead['Gender'].value_counts(),labels=dead['Gender'].unique(),colors=['yellow','blue'],autopct='%1.1f%%')
print("Dead")
plt.show()
train.hist(figsize=(12,8))
plt.figure()
train.head()
train.info()
train['Pclass'].value_counts()
plt.pie(train['Pclass'].value_counts(),labels = train['Pclass'].unique(),colors = ['orange','green','pink'],autopct = '%1.1f%%')
plt.show()
class_1 = train[train['Pclass']==1]
class_2 = train[train['Pclass']==2]
class_3 = train[train['Pclass']==3]
class_1.shape
class_1.head()
class_2.shape
class_3.shape
class_1['Survived'].value_counts()
class_2['Survived'].value_counts()
class_3['Survived'].value_counts()
n_groups = 2
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 6
opacity = 0.8

plt1 = plt.bar(index,class_1['Survived'].value_counts(),align='center',color='b',label='Class 1',alpha = opacity)
#plt.bar(survived['Sex'].unique(),survived['Sex'].value_counts(),alpha=0.5,align='center')
#print("Survived")
#plt.show()

plt2 = plt.bar(index + 3,class_2['Survived'].value_counts(),align='center',color = 'g',label='Class 2',alpha = opacity)
plt3 = plt.bar(index +bar_width,class_3['Survived'].value_counts(),align = 'center',color = 'r',label ='Class 3',alpha=opacity)
plt.xlabel('Survived')
plt.ylabel('Person')
#plt.xticks(index + bar_width, ('alive','dead'))
#print("Dead")
plt.legend()

plt.tight_layout()
plt.show()
train.head()
sur = train[train['Survived'] == 1]
ns = train[train['Survived'] == 0]
len(sur),len(ns)
col = 'Pclass'
v1=sur[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
v2 =ns[col].value_counts().reset_index()
v2 = v2.rename(columns={col:'count','index':col})
v2['percent'] = v2['count'].apply(lambda x:100*x/sum(v2['count']))
index = 0
v1=v1.rename(columns={col:'count','index':col})
v1
train.head()
data =  train.groupby(['Pclass'])
data.get_group(2)
pclass1 = data.get_group(1)
pclass2 = data.get_group(2)
pclass3 = data.get_group(3)
survived = pclass1.groupby(['Survived'])
survive1 = survived .get_group(1)
notsurvive1 = survived.get_group(0)
survived = pclass2.groupby(['Survived'])
survive2 = survived .get_group(1)
notsurvive2 = survived.get_group(0)
survived = pclass3.groupby(['Survived'])
survive3 = survived .get_group(1)
notsurvive3 = survived.get_group(0)
pclass1['Survived'].value_counts()
pclass2['Survived'].value_counts()
pclass3['Survived'].value_counts()
n_groups =3
fig, ax = plt.subplots()
x = np.arange(n_groups)
width = 0.35
plt.bar(x-width/2,sur['Pclass'].value_counts(),width,color='red',alpha=0.5)
plt.bar(x+width/2,ns['Pclass'].value_counts(),width,color= 'blue',alpha= 0.5)
plt.show()

col = 'Pclass'
v1 =  sur[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
col = 'Pclass'
v2 =  ns[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
v2
train.head()
train['Age'].value_counts()
train['Age'].unique()
train['Pclass'].value_counts().reset_index()
train['Embarked'].value_counts()
sur['Embarked'].value_counts()
data = train
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Gender'] = to_categorical(data['Gender'])
data.head()
survival =  data[data['Survived']== 1]
not_survival = data[data['Survived'] == 0]
survival.head()
#bar chart of people who survived
plt.bar(data['Survived'].unique(),data['Survived'].value_counts(),color = 'blue',alpha =0.2)
plt.show()
group = survival['Gender'].value_counts().reset_index()
group = group.rename(columns = {"index" : "Gender" , "Gender" : "Count"})
group = group.sort_values('Gender')
group
group
group1 = not_survival['Gender'].value_counts().reset_index()
group1 = group1.rename(columns = {"index" : "Gender", "Gender" : "Count"})
group1 = group1.sort_values('Gender')
group1.head()
group.head()
len(survival)
not_survival['Gender'].value_counts().reset_index()
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
#First we will be looking at the gender
x = np.arange(2)
width = 0.35
f,ax= plt.subplots()
ax1 = plt.bar(x - width/2,group['Count'],width,color = 'blue',alpha = 0.5,label ='Survived')
ax2= plt.bar(x + width/2,group1['Count'],width,color = 'yellow',alpha =0.5,label = 'Dead')
autolabel(ax1)
autolabel(ax2)
ax.set_xticks(x)
ax.set_xticklabels(['Female','Male'])
ax.legend()
f.tight_layout()
plt.show()
#Now we will be seeing according to class
Survive_class = survival['Pclass'].value_counts().reset_index()
Survive_class = Survive_class.rename(columns={'index':'Pclass','Pclass':'count'})
Survive_class = Survive_class.sort_values('Pclass')
Survive_class
Not_Survive_class = not_survival['Pclass'].value_counts().reset_index()
Not_Survive_class =Not_Survive_class.rename(columns={'index':'Pclass','Pclass':'count'})
Not_Survive_class = Not_Survive_class.sort_values('Pclass')
Not_Survive_class
col = 'Pclass'
fig,ax = plt.subplots()
x = np.arange(3)
width =0.35
ax1 = plt.bar(x-width/2,Survive_class['count'],width,alpha=0.5,color = ['red'],label = 'Survive')
ax2 = plt.bar(x+width/2,Not_Survive_class['count'],width,alpha=0.5,color = ['yellow'],label = 'Not_Survive')
ax.set_xticks(x)
ax.set_xticklabels(['Class_1','Class_2','Class_3'])
ax.legend()
f.tight_layout()
plt.show()
Survive_class
Not_Survive_class
data.head()
#Now we will be seeing according to the embarked 
data['Embarked'].unique()
#So we have null values 
data['Embarked'].isnull().value_counts()
len(data)
embr = data
embr.head()
embr = embr.dropna(subset = ['Embarked'])
len(embr)
embr['Embarked'].unique()
embr['Embarked'].value_counts()
survived = embr[embr['Survived'] == 1]
not_survived = embr[embr['Survived'] == 0]
survived = survived['Embarked'].value_counts().reset_index()
survived = survived.rename(columns = {'index':'Embarked','Embarked':'count'})
survived = survived.sort_values('Embarked')
not_survived = not_survived['Embarked'].value_counts().reset_index()
not_survived = not_survived.rename(columns = {'index':'Embarked','Embarked':'count'})
not_survived = not_survived.sort_values('Embarked')
not_survived
f,ax = plt.subplots()
x = np.arange(3)
width = 0.35
ax1 = plt.bar(x-width/2,survived['count'],width,alpha = 0.5,color = 'blue',label = 'Survived')
ax2 = plt.bar(x+width/2,not_survived['count'],width,alpha = 0.5,color = 'yellow',label = 'Dead')
ax.set_xticks(x)
ax.set_xticklabels(['C','Q','S'])
ax.legend()
f.tight_layout()
plt.show()
#Now we will be dealing with SibSp variable
data['SibSp'].value_counts()
min = data['SibSp'].min()
max = data['SibSp'].max()
sibsp = []
sibsb = pd.DataFrame(sibsp)
#Now we will be dealing with SibSp variable
sibsp = pd.DataFrame(sibsp,columns = ['Index','Survive','Dead'])
sibsp['Index'] = np.arange(min,max+1)
Sibsp_ns = not_survival['SibSp'].value_counts().reset_index()
Sibsp_ns = Sibsp_ns.rename(columns = {'index':'SibSp','SibSp': 'count'})
Sibsp_ns = Sibsp_ns.sort_values('SibSp')
Sibsp_s = survival['SibSp'].value_counts().reset_index()
Sibsp_s = Sibsp_s.rename(columns = {'index':'SibSp','SibSp': 'count'})
#Sibsp_s['Index'] = np.arange(min,max+1)
Sibsp_s = Sibsp_s.sort_values('SibSp')
Sibsp_s
survival['SibSp'].unique()
class Sibsp:
    def __init__(self,Sibsp,count):
        self.Sibsp = Sibsp
        self.count = count
    def Sibsp():
        return(self.Sibsp)
    def Count():
        return(self.count)
Sibsp_s['count'][0]
Sibsp_ns
ns = Sibsp(Sibsp_ns['SibSp'],Sibsp_ns['count'])
s = Sibsp(Sibsp_s['SibSp'],Sibsp_s['count'])
print(s.count[3])
sc = 0
nsc = 0
for ind in range(len(sibsp)):
    try:
        if(sibsp['Index'][ind] == s.Sibsp[sc]):
            #print('here')
            sibsp['Survive'][ind] = int(s.count[sc])
            sc +=1
    except:
        pass
    try:
        if(sibsp['Index'][ind] == ns.Sibsp[nsc]):
            sibsp['Dead'][ind] = int(ns.count[nsc])
            nsc +=1
    except:
        pass
    #print(sc)
    #print(ind)
sibsp
data.head()
Sibsp_ns = not_survival['SibSp'].value_counts().reset_index()
Sibsp_ns = Sibsp_ns.rename(columns = {'index':'SibSp','SibSp': 'count'})
Sibsp_ns = Sibsp_ns.sort_values('SibSp')
len(Sibsp_ns)
f,ax = plt.subplots()
width = 0.35
x = np.arange(len(sibsp['Index']))
ax1 = plt.bar(x-width/2,sibsp['Survive'],width,label = 'Survive',alpha =0.5,color = 'blue')
ax2 = plt.bar(x+width/2,sibsp['Dead'],width,label = 'Dead',alpha =0.5,color = 'yellow')
ax.set_xticks(np.arange(min,max+1))
ax.set_xticklabels(np.arange(min,max+1))
ax.legend()
f.tight_layout()
plt.show()
survival.head()
#will do same as we did for spsib
#first filter out null values
train['Age'].isnull().value_counts()
age_filter = train.dropna(subset = ['Age'])
age_filter['Age'].isnull().value_counts()
agesurvival = age_filter[age_filter['Survived']==1]
agesurvival = agesurvival['Age'].value_counts().reset_index()
agesurvival = agesurvival.rename(columns = {'index':'Age','Age':'count'})
agesurvival = agesurvival.sort_values('Age').reset_index()
agesurvival['alive_per'] = agesurvival['count'].apply(lambda x :x/sum(agesurvival['count'])*100)
agesurvival = agesurvival.drop(['index'],axis =1)
agesurvival.head()
agesurvival.info()
agesurvivalnot = age_filter[age_filter['Survived']==0]
agesurvivalnot = agesurvivalnot['Age'].value_counts().reset_index()
agesurvivalnot = agesurvivalnot.rename(columns = {'index':'Age','Age':'count'})
agesurvivalnot = agesurvivalnot.sort_values('Age').reset_index()
agesurvivalnot['dead_per'] = agesurvivalnot['count'].apply(lambda x: x/sum(agesurvivalnot['count'])*100)
agesurvivalnot = agesurvivalnot.drop(['index'],axis =1)
agesurvivalnot.head()
ages = age_filter['Age'].value_counts().reset_index()
ages = ages.rename(columns = {'index':'Age','Age': 'count'})
ages = ages.sort_values('Age').reset_index()
ages['percent'] = ages['count'].apply(lambda x: 100*x /sum(ages['count']))
ages = ages.drop(['index'],axis =1)
ages = pd.DataFrame(ages,columns = ['Age','count','percent','alive','aliveper','dead','deadper'])
ages.head()
class Entity:
    def __init__(self,main_entity,count,per):
        self.main_entity = main_entity
        self.count = count
        self.per = per
    def Main_entity(self):
        return(self.main_entity)
    def Count(self):
        return(self.count)
    def Percent(self):
        return(self.per)
class Age(Entity):
    def __init__(self,numage,count,per):
        super().__init__(numage,count)
        self.per = per
    def Percent(self):
        return(self.per)
ages.head()
agesurvival.head()
sur_age = Entity(agesurvival['Age'],agesurvival['count'],agesurvival['alive_per'])
not_sur_age = Entity(agesurvivalnot['Age'],agesurvivalnot['count'],agesurvivalnot['dead_per'])
ages['Age'][0]
sur_age.Main_entity()[9]
sc = 0
nsc = 0
for ind in range(len(ages)):
    try:
        if(ages['Age'][ind] == sur_age.Main_entity()[sc]):
            ages['alive'][ind] = sur_age.Count()[sc]
            ages['aliveper'][ind] = sur_age.Percent()[sc]
            sc = sc+1
    except:
        pass
    try:
        if(ages['Age'][ind] == not_sur_age.Main_entity()[nsc]):
            ages['dead'][ind] = not_sur_age.Count()[nsc]
            ages['deadper'][ind] = not_sur_age.Percent()[nsc]
            nsc +=1
    except:
        pass
ages
#Overall people of diffrent ages
plt.plot(ages['Age'],ages['percent'],color = 'blue',alpha =0.5)
#plt.plot(ages['Age'],ages['aliveper'],color = 'red',alpha =0.5)
#plt.plot(ages['Age'],ages['deadper'],color = 'green',alpha =0.5)
#plt.plot(not_survival_age['Age'].unique(),not_survival_age['Age'].value_counts(),color = 'red',alpha =0.5)
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
#axes.set_ylim([survival_age['Age'].value_counts().min(),survival_age['Age'].value_counts().max()])
#axes.set_xlim(0,50)
#axes.set_ylim(0,90)
plt.figure(figsize=(8000, 15000))
plt.show()
#plt.plot(ages['Age'],ages['percent'],color = 'blue',alpha =0.5)
plt.plot(ages['Age'],ages['aliveper'],color = 'red')
plt.plot(ages['Age'],ages['deadper'],color = 'yellow')
#plt.plot(not_survival_age['Age'].unique(),not_survival_age['Age'].value_counts(),color = 'red',alpha =0.5)
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
#axes.set_ylim([survival_age['Age'].value_counts().min(),survival_age['Age'].value_counts().max()])
axes.set_xlim(0,90)
#axes.set_ylim(0,90)
plt.figure(figsize=(8000, 15000))
plt.show()
train.head()
#Checking the null values
train['Parch'].isnull().value_counts()
#so there are no null values 
#Total Parch visualization
plt.bar(train['Parch'].unique(),train['Parch'].value_counts(),width = 0.35)
plt.show()
#survived and unsurvived parch values
parch_s = survival['Parch'].value_counts().reset_index()
parch_s = parch_s.rename(columns = {"index":'Parch','Parch':'count'})
parch_s
parch_ns = not_survival['Parch'].value_counts().reset_index()
parch_ns = parch_ns.rename(columns = {"index":'Parch','Parch':'count'})
parch_ns = parch_ns.sort_values('Parch').reset_index()
parch_ns = parch_ns.drop(['index'],axis = 1)
parch_ns
cparch = np.arange(train['Parch'].min(),train['Parch'].max()+1)
cparch = pd.DataFrame(cparch,columns = ['Parch'])
cparch = pd.DataFrame(cparch , columns = ['Parch','Survived','Dead'])
s = 0
ns = 0
for ind in range(len(cparch)):
    try:
        if(cparch['Parch'][ind] == parch_s['Parch'][s]):
            cparch['Survived'][ind] = parch_s['count'][s]
            s +=1
    except:
        pass
    try:
        if(cparch['Parch'][ind] == parch_ns['Parch'][ns]):
            cparch['Dead'][ind] = parch_ns['count'][ns]
            ns +=1
    except:
        pass
cparch
f,ax = plt.subplots()
width = 0.35
x = np.arange(len(cparch))
ax1 = plt.bar(x-width/2,cparch['Survived'],width,alpha =0.5,color = 'blue',label = 'Survived')
ax2 = plt.bar(x+width/2,cparch['Dead'],width,alpha =0.5,color = 'yellow',label = 'Dead')
ax.set_xticks(cparch['Parch'])
ax.set_xticklabels(x)
ax.legend(loc = 'upper right')
f.tight_layout()
plt.show()
train.corr()
train = pd.read_csv("../input/train.csv")
train.head()
#First we have to see if there are null values in that we have to handle them
print(train['Pclass'].isnull().value_counts())
print(train['Gender'].isnull().value_counts())
print(train['Age'].isnull().value_counts())
print(train['SibSp'].isnull().value_counts())
print(train['Parch'].isnull().value_counts())
print(train['Embarked'].isnull().value_counts())
#We will be filling the data of age with mean and embarked with mode
train['Age'].fillna(train['Age'].mean(),inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)
print(train['Age'].isnull().value_counts())
print(train['Embarked'].isnull().value_counts())
train.head()
#Now convert these values to categorical
tr = train.sort_values('Gender')
tr.head()
tr = tr.sort_values('Embarked')
tr['Gender'] = le.fit_transform(tr['Gender'])
tr['Gender'] = to_categorical(tr['Gender'])
tr['Embarked'] = le.fit_transform(tr['Embarked'])
tr['Embarked'] = to_categorical(tr['Embarked'])
x = tr[['Pclass','Gender','Age','SibSp','Parch','Embarked']]
x = pd.DataFrame(x)
y = tr['Survived']
tr['Embarked'].unique()
x.info()
#At first we have to convert embarked variable to categorical
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
le = LabelEncoder()
x['Embarked'] = le.fit_transform(x['Embarked'])
x['Embarked'] = to_categorical(x['Embarked'])
x.info()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state=50)
model  =  LogisticRegression(n_jobs = 10000)
model.fit(xtrain,ytrain)
ytemp = model.predict(xtest)
print(ytemp)
y_pr = model.predict(xtest)
xtest.shape
ytest = ytest.values.reshape(-1,1)
y_pr = pd.DataFrame(y_pr)
y_pr =  y_pr.values.reshape(-1,1)
ytest.shape , y_pr.shape
from sklearn.metrics import accuracy_score
lr_ac=accuracy_score(ytest, y_pr)
print(lr_ac)
len(xtest),len(ytest)
ax,f = plt.subplots()
width = 20
xval = np.arange(len(xtest))
ax1 = plt.scatter(xval+width/2,ytest,width,color = 'orange',label = 'Ytest',alpha = 0.9)
ax2 = plt.scatter(xval-width/2,y_pr,width,color = 'green',label = 'Ypredict',alpha = 0.6)
axes = plt.gca()
axes.set_ylim(-1,2)
plt.show()
#Now do it for test dataset
test = pd.read_csv("D:\\Datasets\\Logistic Regression\\test.csv")
train.head()
test.info()
test = test.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis = 1)
test.head()
ts = test.sort_values('Gender')
ts = ts.sort_values('Embarked')
ts['Gender'] = le.fit_transform(ts['Gender'])
ts['Gender'] = to_categorical(ts['Gender'])
ts['Embarked'] = le.fit_transform(ts['Embarked'])
ts['Embarked'] = to_categorical(ts['Embarked'])
xts = ts[['Pclass','Gender','Age','SibSp','Parch','Embarked']]
xts = pd.DataFrame(xts)
#Now we will check for null values 
print(xts['Pclass'].isnull().value_counts())
print(xts['Gender'].isnull().value_counts())
print(xts['Age'].isnull().value_counts())
print(xts['SibSp'].isnull().value_counts())
print(xts['Parch'].isnull().value_counts())
print(xts['Embarked'].isnull().value_counts())
#We only have age type null so we will replace them with Mean
xts['Age'].fillna(xts['Age'].mean(),inplace = True)
print(xts['Age'].isnull().value_counts())
xts.info()
yts = model.predict(xts)
#Thats how we predict test set with accuracy of approx 81%