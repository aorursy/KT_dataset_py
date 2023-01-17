import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt

%matplotlib inline





import warnings

warnings.filterwarnings("ignore")



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df=pd.read_csv("../input/titanic/train.csv")

df_test=pd.read_csv("../input/titanic/test.csv")

df_gender=pd.read_csv("../input/titanic/gender_submission.csv")
df.head()
df.shape
df.describe()
df.info()
df.columns
df.drop(['PassengerId'],axis=1,inplace=True)

df.drop(['Ticket'],axis=1,inplace=True)

df.head()
#clean data

def clean_data(df):

    df["fare"]=df["fare"].fillna(df["fare"].dropna().median())

    df["age"]=df["age"].fillna(df["age"].dropna().median())

    

    df.loc[df["gender"]=="male","gender"]=0

    df.loc[df["gender"]=="female","gender"]=1

    

    df["embarked"]=df["embarked"].fillna("S")

    df.loc[df["embarked"]=="S","embarked"]=0

    df.loc[df["embarked"]=="C","embarked"]=1

    df.loc[df["embarked"]=="Q","embarked"]=2


df.columns=["survived","pclass","name","gender","age","sibsp","parch","fare","cabin","embarked"]
df.columns
#count the number of missing values in the dataframe

df.isnull().sum()
print(df.age.median())
def impute_median(series):

    return series.fillna(series.median())
df.age=df.age.transform(impute_median)
df.embarked.unique()
print(df["embarked"].mode())
#fill the missing embarked values with mode

df["embarked"].fillna(str(df["embarked"].mode().values[0]),inplace=True)
print(df.cabin.mode())
#fill the missing cabin values with mode

df["cabin"].fillna(str(df["cabin"].mode().values[0]),inplace=True)
df["cabin"]=df["cabin"].apply(lambda x:str(x).replace(' ','')if ' ' in str(x) else str(x))
df["cabin_data"] = df["cabin"].isnull().apply(lambda x: not x)

df["deck"] = df["cabin"].str.slice(0,1)

df["room"] = df["cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")

df[df["cabin_data"]].head()
# Extract titles from name

df['title']=0

for i in df:

    df['title']=df['name'].str.extract('([A-Za-z]+)\.', expand=False)  # Use REGEX to define a search pattern

df.head()
df.drop(["name"],axis=1,inplace=True)

df.drop(["cabin"],axis=1,inplace=True)

df.drop(["cabin_data"],axis=1,inplace=True)

df.drop(['room'],axis=1,inplace=True)
df.head()
df["family_size"]=df.sibsp+df.parch

df.head()
#Gender show bar plot

sns.set(style='whitegrid')

ax=sns.barplot(x=df['gender'].value_counts().index,y=df['gender'].value_counts().values,palette="Blues_d",hue=['female','male'])

plt.legend(loc=8)

plt.xlabel('Gender')

plt.ylabel('Frequency')

plt.title('Show of Gender Bar Plot')

plt.show()
plt.figure(figsize=(18,5))

sns.barplot(x=df['title'].value_counts().index,

              y=df['title'].value_counts().values)

plt.xlabel('Title')

plt.ylabel('Frequency')

plt.title('Show of Title Bar Plot')

plt.show()
#title - age - gender

plt.figure(figsize=(10,7))

sns.barplot(x = "title", y = "age", hue = "gender", data = df)

plt.xticks(rotation=45)

plt.show()
#title - age - survived

plt.figure(figsize=(10,7))

sns.barplot(x = "title", y = "age", hue = "survived", data = df)

plt.xticks(rotation=45)

plt.show()
#title - age -embarked

plt.figure(figsize=(10,7))

sns.barplot(x = "title", y = "age", hue = "embarked", data = df)

plt.xticks(rotation=45)

plt.show()
#title - age - pclass

plt.figure(figsize=(10,7))

sns.barplot(x = "title", y = "age", hue = "pclass", data = df)

plt.xticks(rotation=45)

plt.show()
#title - age - gender

plt.figure(figsize=(18,7))

sns.barplot(x = "title", y = "age", hue = "family_size", data = df)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(12,7))

sns.catplot(y="gender", x="family_size",

                 hue="survived",

                 data=df, kind="bar")

plt.title('for Survived Gender & Family Size')

plt.show()
plt.figure(figsize=(18,7))

sns.catplot(y="gender", x="age",

                 hue="survived",

                 data=df, kind="bar")

plt.title('for Survived Age & Family Size')

plt.show()
plt.figure(figsize=(18,7))

sns.catplot(y="title", x="age",

                 hue="survived",

                 data=df, kind="bar")

plt.title('for Title Age & Family Size')

plt.show()
plt.figure(figsize=(18,7))

sns.catplot(y="title", x="family_size",

                 hue="survived",

                 data=df, kind="bar",height=6.27, aspect=11.7/8.27)

plt.title('for Title Family Size & Family Size')

plt.show()
labels=df['title'].value_counts().index

colors=['blue','red',"green","pink","orange"]

explode=[0.2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

values=df['title'].value_counts().values



#visualization

plt.figure(figsize=(7,7))

plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('Title According Analysis',color='black',fontsize=10)

plt.show()
plt.figure(figsize=(10,10))

sns.catplot(x="gender", y="age",

                 hue="survived",

                 data=df, kind="bar")

plt.title('for Survived Gender & Age')

plt.show()
labels=df['gender'].value_counts().index

colors=['blue','red']

explode=[0.2,0]

values=df['gender'].value_counts().values



#visualization

plt.figure(figsize=(7,7))

plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('Gender According Analysis',color='black',fontsize=10)

plt.show()
df['age']=df['age']

bins=[0,14,24,34,54,74,89]

labels=["child","youth","young adult","early adult","adult","senior"]

df['age_group']=pd.cut(df['age'],bins,labels=labels)

plt.figure(figsize=(16,5))

sns.barplot(x='age_group',y='survived',data=df)

plt.show()

labels=df['age_group'].value_counts().index

colors=['blue','red','yellow','green','brown',"orange"]

explode=[0.2,0,0,0,0,0]

values=df['age_group'].value_counts().values



#visualization

plt.figure(figsize=(7,7))

plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('Age Group According Analysis',color='black',fontsize=10)

plt.show()
plt.figure(figsize=(10,10))

sns.catplot(x="gender", y="survived",

                 hue="deck",

                 data=df, kind="bar",height=5.27, aspect=11.7/8.27)

plt.title('Survived Gender & Deck')

plt.show()
labels=df['deck'].value_counts().index

colors=['blue','red','yellow','green','brown',"orange","pink"]

explode=[0.2,0,0,0,0,0,0,0]

values=df['deck'].value_counts().values



#visualization

plt.figure(figsize=(7,7))

plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('Deck According Analysis',color='black',fontsize=10)

plt.show()
sns.barplot(x='embarked',y='survived',data=df)

plt.show()
plt.figure(figsize=(10,10))

sns.catplot(x="gender", y="survived",

                 hue="embarked",

                 data=df, kind="bar",height=5.27, aspect=11.7/8.27)

plt.title('Survived Gender & Embarked')

plt.show()
labels=df['embarked'].value_counts().index

colors=['blue','red','green']

explode=[0.2,0,0]

values=df['embarked'].value_counts().values



#visualization

plt.figure(figsize=(7,7))

plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('Embarked According Analysis',color='black',fontsize=10)

plt.show()
f,ax=plt.subplots(figsize=(9,10))

sns.barplot(x=df['gender'].value_counts().values,y=df['gender'].value_counts().index,alpha=0.5,color='red',label='gender')

sns.barplot(x=df['age_group'].value_counts().values,y=df['age_group'].value_counts().index,color='blue',alpha=0.7,label='age_group')

ax.legend(loc='upper right',frameon=True)

ax.set(xlabel='gender , age_group',ylabel='Groups',title="Gender vs Age Group ")

plt.show()
#Gender show point plot

df['age_group'].unique()

len(df[(df['age_group']=='youth')].survived)

f,ax1=plt.subplots(figsize=(25,10))

sns.pointplot(x=np.arange(1,201),y=df[(df['age_group']=='youth')].survived,color='lime',alpha=0.8)

sns.pointplot(x=np.arange(1,201),y=df[(df['age_group']=='youth')].family_size,color='red',alpha=0.5)

plt.xlabel('Youth index State')

plt.ylabel('Frequency')

plt.title('Youth Survived & Family_Size')

plt.xticks(rotation=90)

plt.grid()

plt.show()
labels=df['family_size'].value_counts().index

colors=['blue','red','green',"orange","pink","brown"]

explode=[0.2,0,0,0,0,0,0,0,0]

values=df['family_size'].value_counts().values



#visualization

plt.figure(figsize=(7,7))

plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('Family Size According Analysis',color='black',fontsize=10)

plt.show()
plt.figure(figsize=(15,7))

ax = sns.pointplot(x="age", y="family_size", hue="survived",data=df)

plt.xticks(rotation=75)

plt.show()
sns.set(style="whitegrid", palette="muted")

sns.swarmplot(x="pclass", y="age", hue="survived",

              palette=["r", "c", "y"], data=df)

plt.show()
sns.swarmplot(x="deck", y="age", hue="survived",

              palette=["r", "c", "y"], data=df)

plt.show()
sns.swarmplot(x="gender", y="age", hue="survived",

              palette=["r", "c", "y"], data=df)

plt.show()
sns.swarmplot(x="embarked", y="age", hue="survived",

              palette=["r", "c", "y"], data=df)

plt.show()
sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.swarmplot(x="age_group", y="age", hue="survived",

              palette=["r", "c", "y"], data=df)

plt.show()
sns.set(style="whitegrid", palette="muted")

sns.lineplot(x="pclass", y="age",

             hue="gender", style="embarked",

             data=df)

plt.show()
sns.set(style="whitegrid", palette="muted")

sns.lineplot(x="pclass", y="age",

             hue="gender", style="survived",

             data=df)

plt.show()
print(df['gender'].value_counts(dropna=False))
fig=plt.figure(figsize=(18,6))

plt.subplot2grid((3,4),(0,1))

df.survived[df.gender=='male'].value_counts(normalize=True).plot(kind='bar',alpha=1)

plt.title("Men Survived")

plt.show()
female_color="#FA0000"

fig=plt.figure(figsize=(18,6))

plt.subplot2grid((3,4),(0,2))

df.survived[df.gender=='female'].value_counts(normalize=True).plot(kind='bar',alpha=1,color=female_color)

plt.title("Women Survived")

plt.show()
female_color="#FA0000"

fig=plt.figure(figsize=(18,6))

plt.subplot2grid((3,4),(0,3))

df.gender[df.survived==1].value_counts(normalize=True).plot(kind='bar',alpha=1,color=[female_color,'b'])

plt.title("Gender of Survived")

plt.show()
fig=plt.figure(figsize=(18,6))

plt.subplot2grid((2,3),(1,0),colspan=2)

for x in [1,2,3]:

    df.age[df.pclass==x].plot(kind='kde')

    

plt.title('Class wrt Age')

plt.legend(('1st','2nd','3rd'))

plt.show()
fig=plt.figure(figsize=(18,6))



#rich man

plt.subplot2grid((4,4),(0,0))

df.survived[(df.gender=='male') & (df.pclass==1)].value_counts(normalize=True).plot(kind='bar',alpha=1,color='b')

plt.title('Rich Man Survived')





#poor man

plt.subplot2grid((4,4),(0,1))

df.survived[(df.gender=='male') & (df.pclass==3)].value_counts(normalize=True).plot(kind='bar',alpha=1,color='r')

plt.title('Poor Man Survived')



#rich women

plt.subplot2grid((4,4),(0,2))

df.survived[(df.gender=='female') & (df.pclass==1)].value_counts(normalize=True).plot(kind='bar',alpha=1,color='b')

plt.title('Rich Women Survived')



#poor  woman

plt.subplot2grid((4,4),(0,3))

df.survived[(df.gender=='female') & (df.pclass==3)].value_counts(normalize=True).plot(kind='bar',alpha=1,color='g')

plt.title('Poor Women Survived')





plt.show()
plt.figure(figsize=(16,5))

sns.barplot(x='pclass',y='age',data=df)

plt.show()
fig=plt.figure(figsize=(18,6))

plt.subplot2grid((2,3),(0,2))

df.pclass.value_counts(normalize=True).plot(kind='bar',alpha=1)

plt.title("Class")

plt.show()
plt.figure(figsize=(16,5))

sns.barplot(x='pclass',y='fare',data=df)

plt.show()
plt.figure(figsize=(10,10))

sns.jointplot(x=np.arange(1,217),y=df[(df['pclass']==1)].age,color='lime',alpha=0.8)

plt.xlabel('First Class index State')

plt.ylabel('Frequency')

plt.title('First Class Frequency Age')

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
plt.figure(figsize=(10,10))

sns.jointplot(x=np.arange(1,185),y=df[(df['pclass']==2)].age,color='lime',kind='hex',alpha=0.8)

plt.xlabel('Second Class index State')

plt.ylabel('Frequency')

plt.title('Second Class Frequency Age')

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
plt.figure(figsize=(15,10))

sns.jointplot(x=np.arange(1,492),y=df[(df['pclass']==3)].age,color='lime',space=0,kind='kde')

plt.xlabel('Third Class index State')

plt.ylabel('Frequency')

plt.title('Third Class Frequency Age')

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
df.sibsp.unique()
sns.barplot(x='sibsp',y='age',data=df)

plt.show()
plt.figure(figsize=(16,5))

sns.barplot(x='gender',y='sibsp',data=df)

plt.show()
df.parch.unique()
sns.barplot(x='parch',y='age',data=df)

plt.show()
sns.barplot(x='parch',y='sibsp',data=df)

plt.show()
plt.figure(figsize=(16,5))

sns.barplot(x='deck',y='pclass',data=df)

plt.show()
sum(df.pclass==3)
sum(df.pclass==2)
sum(df.pclass==1)
sns.barplot(x='pclass',y='survived',data=df)

plt.show()
plt.figure(figsize=(16,5))

result = df.groupby(["deck"])['survived'].aggregate(np.median).reset_index().sort_values('survived')

sns.barplot(x='deck', y="survived", data=df, order=result['deck'])

plt.title('deck-survived')

plt.show()
df.corr()
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(df.corr(),annot=True,linewidths=.5,fmt='.0%',ax=ax)

plt.show()
df.head(2)
df['gender_factor'] = pd.factorize(df.gender)[0]

df.head(3)
color_list = ['red' if i==1 else 'green' for i in df.loc[:,'gender_factor']]
fig=plt.figure(figsize=(18,6))

plt.subplot2grid((2,3),(1,0),colspan=2)

for x in [0,1]:

    df.age[df.gender_factor==x].plot(kind='kde')

    

plt.title('Class wrt Age')

plt.legend(('Male','Female'))

plt.show()
df.survived.dropna(inplace = True)

labels = df.survived.value_counts().index

colors = ["red","lightblue"]

explode = [0,0]

sizes = df.survived.value_counts().values



# visual cp

plt.figure(0,figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('People According to Survived',color = 'blue',fontsize = 15)

plt.show()
df.age[(df.gender_factor==x) & (df.survived==1) & (df.embarked=='C') & (df.pclass==1)].mean()
df_test['gender_factor'] = pd.factorize(df_test.Sex)[0]

df.head(3)
df.head()