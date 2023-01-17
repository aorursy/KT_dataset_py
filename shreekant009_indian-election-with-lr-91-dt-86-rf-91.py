import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px
df = pd.read_csv("../input/indian-candidates-for-general-election-2019/LS_2.0.csv")
pd.set_option('display.max_rows',20000, 'display.max_columns',100)
df.shape
df.info()
df.head(5)
df.isnull().sum().sort_values(ascending=False).head(30)
df[df['NAME']=='NOTA'].head()
df = df.fillna({'SYMBOL':'NO SYMBOL',

                'GENDER':'NOT APPLICABLE',

                'CRIMINAL\nCASES':0,'AGE':0.0,

                'CATEGORY':'NOT APPLICABLE',

                'EDUCATION':'NOT APPLICABLE',

                'ASSETS':'Rs 0',

                'LIABILITIES':'Rs 0',})
df.isnull().sum().sort_values(ascending=False).head(30)
(df['PARTY'].value_counts()/len(df['PARTY']))*100
df.head()
(df['WINNER'].value_counts()/len(df['WINNER']))*100
plt.rcParams['figure.figsize'] = (12,8)

labels = 'LOST', 'WON'

sizes = [76,24]

explode = (0, 0.1)  



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%0.0f%%',

        shadow=True, startangle=90,center=(0, 0))

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('2019 GENERAL ELECTION WINNING-LOSING PERCENTAGE')

plt.show()
plt.figure(figsize=(10,8))

sns.countplot(y='EDUCATION',hue='WINNER',data=df,palette="GnBu")
plt.figure(figsize=(10,8))

sns.countplot(x='WINNER',hue='GENDER',data=df)
#Filling Nan Values in CRIMINAL CASE COLUMN

df['CRIMINAL\nCASES']=df['CRIMINAL\nCASES'].replace(to_replace ="Not Available", 

                 value ="0")

df['CRIMINAL\nCASES'] = df['CRIMINAL\nCASES'].astype(int)
#Filling Nan Values in Post Graduate

df['EDUCATION']=df['EDUCATION'].replace(to_replace ="Post Graduate\n", 

                 value ="Post Graduate")
df.head()
#Checking if the candidate is criminal/ncase or not

for i in range(len(df)):

    if df.iloc[i,7]>0:

        df.iloc[i,7]='HAS CASE'

    else:

        df.iloc[i,7]='NO CASE'
plt.figure(figsize=(8,4))

sns.countplot(x='WINNER',hue='CRIMINAL\nCASES',data=df)
sns.countplot(y='GENDER',hue='CRIMINAL\nCASES',data=df,palette='hot')
df.head()
df[df['ASSETS']=='Not Available']
for i in range(len(df)):

    if df.iloc[i,11]=='Not Available':

        df.iloc[i,11] = "-1"    
#Removing '\n ~' from ASSETS

df['ASSETS'] = df['ASSETS'].str.split('\n ~', 1, expand=True)[0]

#Removing 'Rs' from ASSETS

df['ASSETS'] = df['ASSETS'].str.split(expand=True)[1]

#Removing special charater "," and then joinig it

df['ASSETS']=df['ASSETS'].str.split(",").str.join(" ")

#Removing the space between the join

df['ASSETS']=df['ASSETS'].str.replace(' ', '')

#Converting ASSETS column into flaot as int values are very high

df['ASSETS'] = df['ASSETS'].astype(float)
df['ASSETS'] = df['ASSETS'].fillna(value = 0.0)
#ASSETS can be converted into Different Economic Classes

STATUS = []

for i in df['ASSETS']:

    if i >0.0 and i < 500000.0:

        STATUS.append('NEAR TO BPL')

    if i >= 500000.0 and i <= 1000000.0:

        STATUS.append('LOWER CLASS')

    if i >= 1000001.0 and i <= 2500000.0:

        STATUS.append('LOWER MIDDLE CLASS')

    elif i >= 2500001.0 and i <= 10000000.0:

        STATUS.append('MIDDLE CLASS')

    elif i >= 10000001.0 and i <= 100000000.0:

        STATUS.append('UPPER MIDDLE CLASS')

    elif i >= 100000000.0 and i <= 250000000.0:

        STATUS.append('ELITE CLASS')

    elif i >= 250000001.0 and i <= 1000000000.0:

        STATUS.append('SUPER RICH')

    elif i >= 1000000001.0:

        STATUS.append('RICHEST OF RICH')

    elif i == -1.0:

        STATUS.append('ASSETS NOT MENTIONED')

    elif i == 0.0:

        STATUS.append('NO ASSETS')

df['STATUS'] = STATUS
sns.countplot(y='STATUS',hue='CRIMINAL\nCASES',data=df,palette='hot')
print(df['PARTY'].nunique(),df['PARTY'].unique())
#Using Groupby on Party with Total Votes(TO CHECK HIGHEST PERCENTAGE OF VOTES FOR TOP 10 PARTIES)

p = df.groupby('PARTY')['TOTAL\nVOTES']

X = ((p.sum()/df['TOTAL\nVOTES'].sum())*100).sort_values(ascending = False).head(10)

X
plt.rcParams['figure.figsize'] = (12,8)

labels = 'BJP', 'INC','AITC','BSP','SP','YSRCP','CPI(M)','DMK','SHS','TDP','OTHERS'

sizes = [38,20,4,3,2,2,2,2,2,2,23]

 

colors=('orange', 'green', 'deeppink', 'blue', 'red', 'yellow', 'crimson', 'brown','darkorange','pink','gray')



my_circle = plt.Circle((0, 0), 0.7, color='white')



d = plt.pie(sizes, labels=labels, autopct='%0.0f%%',

            startangle=90,colors=colors, labeldistance=1.05)

plt.axis('equal')

plt.gca().add_artist(my_circle)

plt.title('VOTE SHARE OF PARTY')            

plt.show()
df.head()
df['AGE'].nunique()
df.head()
plt.rcParams['figure.figsize'] = (10,8)

labels = 'BJP','INC','DMK','YSRCP','AITC','SHS','JD(U)','BSP','BJD','TRS','OTHER'

sizes = [300, 52, 23, 22, 22, 18, 16, 11, 11, 9,55]

 

colors=('orange', 'green', 'deeppink', 'blue', 'red', 'yellow', 'crimson', 'brown','darkorange','pink','gray')



my_circle = plt.Circle((0, 0), 0.7, color='white')



d = plt.pie(sizes, labels=labels, autopct='%0.0f%%',

            startangle=90,colors=colors, labeldistance=1.05)

plt.axis('equal')

plt.gca().add_artist(my_circle)

plt.title('OUT OF 539 SEATS WINNER DISTRIBUTION')            

plt.show()
df.head()
p = df.groupby('STATE')['CONSTITUENCY'].nunique().reset_index()

plt.figure(figsize=(15,10))

sns.barplot(y='STATE',x='CONSTITUENCY',data=p,palette='spring')
p = df.groupby(['PARTY','EDUCATION','GENDER'])['WINNER'].sum().reset_index().sort_values('WINNER',ascending = False)

p = p[p['WINNER']!=0]

fig = px.bar(p, x='EDUCATION', y='WINNER',hover_data =['PARTY'], color='GENDER', height=650)

fig.show()
p = df.groupby(['PARTY','STATE']).sum().reset_index().sort_values('PARTY',ascending =True)

fig = px.bar(p, x='STATE', y='WINNER',hover_data =['PARTY'], color='PARTY', height=650)

fig.show()
#Creating Age_Group from Age column

AGE_GROUP = []

for i in df['AGE']:

    if i >= 24 and i <=35:

        AGE_GROUP.append('YOUNG AGE')

    elif i >= 36 and i<=60:

        AGE_GROUP.append('MIDDLE AGE')

    elif i >=60:

        AGE_GROUP.append('OLD AGE')

    else:

        AGE_GROUP.append('NOT KNOWN')

df['AGE_GROUP'] = AGE_GROUP
p = df.groupby(['PARTY','AGE_GROUP'])['WINNER'].sum().reset_index().sort_values('WINNER',ascending = False)

p = p[p['WINNER']!=0]

fig = px.bar(p, x='PARTY',y='WINNER',hover_data =['AGE_GROUP'], color='AGE_GROUP', height=650)

fig.show()
p = df.groupby(['PARTY','GENDER'])['WINNER'].count().reset_index().sort_values('WINNER',ascending = False)

#p = p[p['WINNER']!=0]

fig = px.bar(p, x='PARTY',y='WINNER',hover_data =['GENDER'], color='GENDER', height=700)

fig.show()
p = df.groupby(['PARTY','CRIMINAL\nCASES'])['WINNER'].sum().reset_index().sort_values('WINNER',ascending = False)

p = p[p['WINNER']!=0]

fig = px.bar(p, x='PARTY',y='WINNER',hover_data =['CRIMINAL\nCASES'], color='CRIMINAL\nCASES', height=750)

fig.show()
from sklearn.preprocessing import LabelEncoder

labelEncoder_X = LabelEncoder()

df['STATE'] = labelEncoder_X.fit_transform(df['STATE'])

df['CONSTITUENCY'] = labelEncoder_X.fit_transform(df['CONSTITUENCY'])

df['NAME'] = labelEncoder_X.fit_transform(df['NAME'])

df['PARTY'] = labelEncoder_X.fit_transform(df['PARTY'])

df['SYMBOL'] = labelEncoder_X.fit_transform(df['SYMBOL'])

df['GENDER'] = labelEncoder_X.fit_transform(df['GENDER'])

df['CRIMINAL\nCASES'] = labelEncoder_X.fit_transform(df['CRIMINAL\nCASES'])

df['CATEGORY'] = labelEncoder_X.fit_transform(df['CATEGORY'])

df['EDUCATION'] = labelEncoder_X.fit_transform(df['EDUCATION'])

df['STATUS'] = labelEncoder_X.fit_transform(df['STATUS'])

df['AGE_GROUP'] = labelEncoder_X.fit_transform(df['AGE_GROUP'])
X=df.drop(['WINNER','ASSETS','LIABILITIES','GENERAL\nVOTES','POSTAL\nVOTES','AGE','OVER TOTAL ELECTORS \nIN CONSTITUENCY','OVER TOTAL VOTES POLLED \nIN CONSTITUENCY'],axis=1)

y=df['WINNER']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
from sklearn.preprocessing import StandardScaler

Scaler_X = StandardScaler()

X_train = Scaler_X.fit_transform(X_train)

X_test = Scaler_X.transform(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

#Logistic Regression

from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)



print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))
#Decision Tree

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)

print(accuracy_score(y_test,predictions ))

print(confusion_matrix(y_test,predictions ))
#Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train,y_train)

rfc__pred = rfc.predict(X_test)

print(accuracy_score(y_test,rfc__pred))

print(confusion_matrix(y_test,rfc__pred))