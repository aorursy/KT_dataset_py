import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from collections import Counter

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df_train=pd.read_csv("../input/titanic/train.csv")

df_test=pd.read_csv("../input/titanic/test.csv")

df_gender=pd.read_csv("../input/titanic/gender_submission.csv")
print("================TRAIN INFO=========================")

df_train.info()
print("============TRAIN COLUMNS==================")

df_train.columns
df_train.columns=["passid","survived","pclass","name","gender","age","sibsp","parch","ticket","fare","cabin","embarked"]
df_train.columns
df_train.shape
print("=============TRAIN HEAD===================")

df_train.head()
print("=============TRAIN TAIL====================")

df_train.tail()
df_train.dtypes
df_train.corr()
print("================TEST INFO============================")

df_test.info()

print("================TEST COLUMNS====================")

df_test.columns
df_test.columns=["passid","pclass","name","gender","age","sibsp","parch","ticket","fare","cabin","embarked"]
df_test.columns
df_test.shape
print("=================TEST HEAD=================")

df_test.head()
print("===============TEST TAIL====================")

df_test.tail()
df_test.dtypes
print("================GENDER SUBMISSION INFO============================")

df_gender.info()
print("================GENDER SUBMISSION COLUMNS======================")

df_gender.columns
df_gender.columns=["passid","survived"]
df_gender.columns
df_gender.shape
print("===========GENDER SUBMISSION HEAD====================")

df_gender.head()
print("================GENDER SUBMISSION TAIL=======================")

df_gender.tail()
df_gender.dtypes
print("===================TRAIN NAN VALUES====================")

print("=========AGE NAN==============")

print(df_train["age"].value_counts(dropna=False)) #177 NaN 

print("=========CABIN NAN=====================")

print(df_train["cabin"].value_counts(dropna=False)) #687 NaN

print("========EMBARKED NAN====================")

print(df_train["embarked"].value_counts(dropna=False)) #2 NaN
print("======================TEST NaN VALUES=======================")

print("========================AGE NaN VALUES======================")

print(df_test["age"].value_counts(dropna=False))#86 NaN Values

print("===================FARE NaN VALUES==========================")

print(df_test["fare"].value_counts(dropna=False))#1 NaN Value

print("====================CABIN NaN VALUES========================")

print(df_test["cabin"].value_counts(dropna=False)) #327 NaN Value
df_train.describe()
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(df_train.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)

plt.show()
fig,axes=plt.subplots(nrows=2,ncols=1)

df_train.plot(kind='hist',y='age',bins=50,range=(0,100),normed=True,ax=axes[0])

df_train.plot(kind='hist',y='age',bins=50,range=(0,100),normed=True,ax=axes[1],cumulative=True)

plt.show()
print(df_train['gender'].value_counts(dropna=False))
sns.barplot(x='gender',y='age',data=df_train)

plt.show()
df_train['age']=df_train['age']

bins=[29,45,59,np.inf]

labels=["Young Adult","Middle-Aged Adults","Old Adults"]

df_train['age_group']=pd.cut(df_train['age'],bins,labels=labels)

sns.barplot(x='age_group',y='survived',data=df_train)

plt.show()
sns.barplot(x='gender',y='survived',data=df_train)

plt.show()
df_train.pclass.unique()
sns.barplot(x='pclass',y='age',data=df_train)

plt.show()
sns.barplot(x='survived',y='age',data=df_train)

plt.show()
df_train.embarked.unique()
sns.barplot(x='embarked',y='survived',data=df_train)

plt.show()
df_train.sibsp.unique()
sns.barplot(x='sibsp',y='age',data=df_train)

plt.show()
sns.barplot(x='gender',y='sibsp',data=df_train)

plt.show()
df_train.parch.unique()
sns.barplot(x='parch',y='age',data=df_train)

plt.show()
sns.barplot(x='parch',y='sibsp',data=df_train)

plt.show()
df_train.cabin.unique()
plt.figure(figsize=(15,30))

sns.barplot(x='pclass',y='cabin',data=df_train)

plt.show()
sum(df_train.pclass==3)



sum(df_train.pclass==2)

sum(df_train.pclass==1)
sns.barplot(x='pclass',y='survived',data=df_train)

plt.show()
plt.figure(figsize=(15,30))

result = df_train.groupby(["cabin"])['survived'].aggregate(np.median).reset_index().sort_values('survived')

sns.barplot(x='survived', y="cabin", data=df_train, order=result['cabin'])

plt.title('cabin-survived')

plt.show()
df_train.head()
# Most common 15 Name or Surname of dying people

separate = df_train.name.str.split() 

a,b,c = zip(*separate)                    

name_list = a+b+c

name_count = Counter(name_list)         

most_common_names = name_count.most_common(15)  

x,y = zip(*most_common_names)

x,y = list(x),list(y)

    

plt.figure(figsize=(15,10))

ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Name or Surname of dying people')

plt.ylabel('Frequency')

plt.title('Most common 15 Name or Surname of dying people')

plt.show()
name_list1 = list(df_train['name'])

pclass = []

survived =[]

age=[]

parch=[]

sibsp=[]







for i in name_list1:

    x = df_train[df_train['name']==i]

    pclass.append(sum(x.pclass)/len(x))

    survived.append(sum(x.survived) / len(x))

    age.append(sum(x.age) / len(x))

    parch.append(sum(x.parch) / len(x))

    sibsp.append(sum(x.sibsp) / len(x))

    

# visualization



f,ax = plt.subplots(figsize = (15,220))



sns.barplot(x=age,y=name_list1,color='c',alpha = 0.6,label='age')

sns.barplot(x=survived,y=name_list1,color='m',alpha = 0.7,label='survived')

sns.barplot(x=pclass,y=name_list1,color='g',alpha = 0.5,label='pclass' )

sns.barplot(x=parch,y=name_list1,color='y',alpha = 0.6,label='parch')

sns.barplot(x=sibsp,y=name_list1,color='r',alpha = 0.6,label='sibsp')



ax.legend(loc='lower right',frameon = True)

ax.set(xlabel='Percentage of pclass, survived, age, parch, sibsp', ylabel='name',

       title = "Percentage of Name's According to pclass, survived, age, parch, sibsp ")

plt.show()
#missing values pclass

sum(df_train.pclass.isna())
#missing values survived

sum(df_train.survived.isna())
#missing values parch

sum(df_train.age.isna())
df_train.columns
df_train.dtypes
df_train['age']=df_train['age'].fillna(-0.5)
x=df_train.iloc[:,[2,5]].values#pclass,parch
y=df_train.survived.values.reshape(-1,1)
multiple_linear_regression=LinearRegression()

multiple_linear_regression.fit(x,y)
print("b0:",multiple_linear_regression.intercept_)
print("b1,b2:",multiple_linear_regression.coef_)
df_train.age.max()
#pclass:1 and age 80 vs. plcass:2 and age 80

multiple_linear_regression.predict(np.array([[1,80],[2,80]]))
#pclass:1 and age 80 vs. plcass:3 and age 80

multiple_linear_regression.predict(np.array([[1,80],[3,80]]))
#pclass:2 and age 80 vs. plcass:3 and age 80

multiple_linear_regression.predict(np.array([[2,80],[3,80]]))
df_train.age.mean()
#pclass:1 and age 23 vs. plcass:2 and age 23

multiple_linear_regression.predict(np.array([[1,23],[2,23]]))
#pclass:1 and age 23 vs. plcass:3 and age 23

multiple_linear_regression.predict(np.array([[1,23],[3,23]]))
#pclass:2 and age 23 vs. plcass:3 and age 23

multiple_linear_regression.predict(np.array([[2,23],[3,23]]))
print('end')
