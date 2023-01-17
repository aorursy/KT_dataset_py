# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings
warnings.filterwarnings('ignore') 

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_PassengerId = test_df["PassengerId"]
train_df.head()

train_df.describe()

train_df.info()

import matplotlib.pyplot as plt


def bar_plot(variable):
    var=train_df[variable]
    varvalue=var.value_counts()
    plt.figure(figsize=(9,3))
    plt.bar(varvalue.index,varvalue)
    plt.xticks(varvalue.index,varvalue)
    plt.title(variable)
    plt.ylabel("frequenty")
    plt.show()
    print("{}\n {}".format(variable,varvalue))
kategori=["Survived","Pclass","Embarked","SibSp","Parch"]

for i in kategori:
    
    bar_plot(i)
    
def bar_plot(variable):
    
    var=train_df[variable]
    varvalue=var.value_counts()
    
    plt.figure(figsize=(9,3))
    plt.bar(varvalue.index,varvalue)
    
    plt.xticks(varvalue.index,varvalue.index.values)
    plt.ylabel("Frequency")
    
    plt.title(variable)
    
    plt.show()
    
    print("{}: \n {}".format(variable,varvalue))

kategori=["Survived", "Pclass","Embarked","SibSp","Parch"]

for k in kategori:
    bar_plot(k)
def plot_hist(variable):
    plt.figure(figsize=(6,3))
    plt.hist(train_df[variable],bins=100)
    plt.xlabel(variable)
    plt.ylabel("frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
num=["Fare","Age","PassengerId"]

for n in num:
    plot_hist(n)
def plot_hist(variable):
    
    plt.figure(figsize=(9,3))
    
    plt.hist(train_df[variable], bins=50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{}  distribution with hist".format(variable))
    plt.show()
    
    
numericVar = ["Fare", "Age","PassengerId"]
for n in numericVar:
    plot_hist(n)
train_df[["Pclass","Survived"]].groupby(["Survived"],as_index=False).mean().sort_values(by="Survived", ascending=True)
train_df[["Pclass","Survived"]].groupby(["Survived"],as_index=False).mean().sort_value(by="Survived",ascending=True)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter
def outlier_hesaplama(df,feature):
    outlier_lists=[]
    for c in feature:
        
        Q1=np.percentile(df[c],25)
        
        Q3=np.percentile(df[c],75)
        
        IQR=Q3-Q1
        
        outlier_step=IQR * 1.5
        
        outlier_list=df[(df[c] < Q1-outlier_step) | (df[c] > Q3 + outlier_step) ].index
        print(outlier_list)
        outlier_lists.extend(outlier_list)
    
    outlier_lists = Counter(outlier_lists)
    multiple_outliers = list(i for i, v in outlier_lists.items() if v > 2)
    return multiple_outliers
    
train_df.loc[outlier_hesaplama(train_df,["Age","SibSp","Parch","Fare"])]
from collections import Counter

train_df=train_df.drop(outlier_hesaplama(train_df,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)
train_df=train_df.drop(outlier_hesaplama(train_df,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)
train_df_len=len(train_df)
train_df=pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)
train_df_len=len(train_df)
train_df=pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
train_df.head()
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column="Fare",by="Embarked")
plt.show()
train_df.boxplot(column="Fare",by = "Embarked")
plt.show()
train_df["Embarked"]=train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]
train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]
train_df.iloc[61:62,:]
train_df[train_df["Fare"].isnull()]
train_df["Fare"]=train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))
train_df[train_df["Fare"].isnull()]
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived",ascending=False)
median_house_hold_in_come = pd.read_csv('../input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding="windows-1252")
percentage_people_below_poverty_level = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")
percent_over_25_completed_highSchool = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")
share_race_city = pd.read_csv('../input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv', encoding="windows-1252")
kill = pd.read_csv('../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()

percentage_people_below_poverty_level.info()

percentage_people_below_poverty_level["Geographic Area"].unique()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level.poverty_rate.fillna('empty',inplace = True)
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level.poverty_rate.replace(['_'],0.0,inplace=True)
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level.loc[percentage_people_below_poverty_level.poverty_rate==0.0 ]=0
percentage_people_below_poverty_level.poverty_rate=percentage_people_below_poverty_level.poverty_rate.astype(int)
percentage_people_below_poverty_level.poverty_rate.replace(['0.0'],0,inplace = True)
percentage_people_below_poverty_level.poverty_rate.value_counts()
area_list=list(percentage_people_below_poverty_level["Geographic Area"].unique())
area_list
percentage_people_below_poverty_level.poverty_rate=percentage_people_below_poverty_level.poverty_rate.astype(float)
area_list=list(percentage_people_below_poverty_level["Geographic Area"].unique())
area_poverty_ratio=[]

for i in area_list:
    x=percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]
    area_poverty_rate=sum(x.poverty_rate)/len(x)
    area_poverty_ratio.append(area_poverty_rate)

data=pd.DataFrame({'area_list':area_list,'area_poverty_ratio':area_poverty_ratio})
new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values    

sorted_data=data.reindex(new_index)

plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['area_list'],y=sorted_data['area_poverty_ratio'])
plt.xticks(rotation=90)
plt.xlabel('States')
plt.ylabel('poverty rate')
plt.title('poverty rate Given State')


percentage_people_below_poverty_level.poverty_rate=percentage_people_below_poverty_level.poverty_rate.astype(float)
area_list=list(percentage_people_below_poverty_level['Geographic Area'].unique())

area_poverty_ratio=[]

for i in area_list:
    
    x=percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]
    
    area_poverty_rate=sum(x.poverty_rate)/len(x)
    area_poverty_ratio.append(area_poverty_rate)
    
data=pd.DataFrame({'area_list':area_list,'area_poverty_ratio':area_poverty_ratio})
new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values


sorted_data=data.reindex(new_index)

#visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])
plt.xticks(rotation=45)
plt.xlabel('States')
plt.ylabel('Poverty Rate')
plt.title('Poverty Rate Given States')
percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace = True)
percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)
area_list = list(percentage_people_below_poverty_level['Geographic Area'].unique())
area_poverty_ratio = []
for i in area_list:
    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]
    area_poverty_rate = sum(x.poverty_rate)/len(x)
    area_poverty_ratio.append(area_poverty_rate)
data = pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})
new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])
plt.xticks(rotation= 45)
plt.xlabel('States')
plt.ylabel('Poverty Rate')
plt.title('Poverty Rate Given States')
percentage_people_below_poverty_level.poverty_rate.value_counts()
sr=pd.Series([100,None,None,18,65,None,32,10,5,24,None,None])
index_=pd.date_range('2010-10-09',periods=12, freq='M')
sr.index=index_
print(sr)
result=sr.fillna(method='ffill')
print(result)

# importing pandas as pd 
import pandas as pd 
  
# Creating the Series 
sr = pd.Series([100, None, None, 18, 65, None, 32, 10, 5, 24, None,None]) 
  
# Create the Index 
index_ = pd.date_range('2010-10-09', periods = 12, freq ='M') 
  
# set the index 
sr.index = index_ 
  
# Print the series 
print(sr)
result = sr.fillna(method = 'ffill') 
  
# Print the result 
print(result) 
kill.info()
kill.head()
kill.name.value_counts()
seperate=kill.name[kill.name!='TK TK'].str.split()
a,b=zip(*seperate)
name_list=a+b
count_name=Counter(name_list)
most_common=count_name.most_common(15)
x,y=zip(*most_common)
x,y= list(x),list(y)

plt.figure(figsize=(25,10))
sns.barplot(x=x,y=y,palette = sns.cubehelix_palette(len(x)))
plt.xlabel('Name or Surname of killed people')
plt.ylabel('Frequency')
plt.title('Most common 15 Name or Surname of killed people')
seperate=kill.name[kill.name != 'TK TK'].str.split()
a,b=zip(*seperate)
name_list=a+b

count_name=Counter(name_list)
most_common=count_name.most_common(15)
x,y=zip(*most_common)

x,y =list(x),list(y)

plt.figure(figsize=(6,3))

ax=sns.barplot(x=x,y=y,palette = sns.cubehelix_palette(len(x)))
plt.xlabel('Name or Surname of killed people')
plt.ylabel('Frequency')
plt.title('Most common 15 Name or Surname of killed people')
seperate=kill.name[kill.name!='TK TK'].str.split()
a,b=zip(*seperate)
name_list=a+b

Count_name=Counter(name_list)
most_common=Count_name.most_common(15)
x,y=zip(*most_common)

x,y = list(x),list(y)
# 
plt.figure(figsize=(15,10))
ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))
plt.xlabel('Name or Surname of killed people')
plt.ylabel('Frequency')
plt.title('Most common 15 Name or Surname of killed people')
percent_over_25_completed_highSchool.head()

percent_over_25_completed_highSchool.info()

percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace = True)
percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)
area_list = list(percent_over_25_completed_highSchool['Geographic Area'].unique())
area_highschool = []
for i in area_list:
    x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area']==i]
    area_highschool_rate = sum(x.percent_completed_hs)/len(x)
    area_highschool.append(area_highschool_rate)
# sorting
data = pd.DataFrame({'area_list': area_list,'area_highschool_ratio':area_highschool})
new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values
sorted_data2 = data.reindex(new_index)
# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_highschool_ratio'])
plt.xticks(rotation= 90)
plt.xlabel('States')
plt.ylabel('High School Graduate Rate')
plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
share_race_city.head()
share_race_city.info()
share_race_city.share_hispanic.value_counts(1)
share_race_city.replace(['_'],0.0,inplace=True)
share_race_city.replace(['(X)'],0.0,inplace=True)
area_list=list(share_race_city['Geographic area'].unique())
share_race_city.iloc[:,2:]= share_race_city.iloc[:,2:].astype(float)
share_white=[]
share_black=[]
share_native_american=[]
share_asian=[]
share_hispanic=[]

for i in area_list:
    x=share_race_city[share_race_city['Geographic area']==i]
    share_white.append(sum(x.share_white)/len(x))
    share_black.append(sum(x.share_black)/len(x))
    share_native_american.append(sum(x.share_native_american)/len(x))
    share_asian.append(sum(x.share_asian)/len(x))
    share_hispanic.append(sum(x.share_hispanic)/len(x))
                       
                       
f,ax =plt.subplots(figsize=(9,13))
sns.barplot(x=share_white,y=area_list,color='green',alpha=0.5,label='white')
sns.barplot(x=share_black,y=area_list,color='blue',alpha=0.5,label='black')
sns.barplot(x=share_native_american,y=area_list,color='yellow',alpha=0.5,label='native american')
sns.barplot(x=share_asian,y=area_list,color='red',alpha=0.5,label='asian')
sns.barplot(x=share_hispanic,y=area_list,color='cyan',alpha=0.5,label='hispanic')

ax.legend(loc='best',fontsize= 'xx-large',numpoints=5 ,frameon=True ,mode='expand',borderaxespad=5.5  )

ax.set(title='bilmiyourm net ne yazılmalı',xlabel=' percentage',ylabel='race percentage')
share_race_city.replace(['_'],0.0,inplace=True)
share_race_city.replace(['(X)'],0.0,inplace=True)
area_list = list(share_race_city['Geographic area'].unique())

share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] =share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)


share_white=[]
share_black=[]
share_native_american=[]
share_asian=[]
share_hispanic=[]
               
for i in area_list:
               
             x=share_race_city[share_race_city['Geographic area']==i]
             share_white.append(sum(x.share_white)/len(x))
          
             share_black.append(sum(x.share_black)/len(x))
             share_native_american.append(sum(x.share_native_american)/len(x))
             share_asian.append(sum(x.share_asian)/len(x))
             share_hispanic.append(sum(x.share_hispanic)/len(x))
                
                
f,ax=plt.subplots(figsize=(9,13))
                
sns.barplot(x=share_white,y=area_list,color='green', alpha=0.5, label='White')
                
sns.barplot(x=share_black,y=area_list,color='blue', alpha=0.6, label='African_America')
        
sns.barplot(x=share_native_american, y=area_list, color='red', alpha=0.7, label='Native-American')
                
sns.barplot(x=share_asian, y= area_list, color='yellow',alpha=0.6,label='Asian')
sns.barplot(x=share_hispanic, y= area_list, color='cyan', alpha=0.6, label='Hispanic')
                

ax.legend(loc='upper center', shadow=True, fontsize='x-large' ,frameon = True) 

ax.set(xlabel='Percentage of Races' ,ylabel='States',title = "Percentage of State's Population According to Races ")
 

percent_over_25_completed_highSchool.info()
percent_over_25_completed_highSchool.head()
percent_over_25_completed_highSchool.tail()
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0,inplace=True)
percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)
area_list=list(percent_over_25_completed_highSchool['Geographic Area'].unique())
ortalama=[]


for i in area_list:
    
    x=percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area']==i]
    ortalama.append(sum(x.percent_completed_hs)/len(x))
    
        
data = pd.DataFrame({'area_list': area_list,'area_highschool_ratio':ortalama})    
new_index=(data['area_highschool_ratio'].sort_values(ascending=True)).index.values
print(new_index,end=' new indexli ')
sorted_data2=data.reindex(new_index)
print(sorted_data2)
plt.figure(figsize=[16,8])
sns.barplot(x=sorted_data2['area_list'],y=sorted_data2['area_highschool_ratio'])
plt.xticks(rotation=45)
plt.xlabel('area highschool ration')
plt.ylabel(' area_list')
plt.title('ercentage of Given States Population Above 25 that Has Graduated High School')
sorted_data.head()
sorted_data2.head()
data=pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1 )
data=data.sort_values('area_poverty_ratio',ascending=True)
data
sorted_data['area_poverty_ratio']=sorted_data['area_poverty_ratio']/max(sorted_data['area_poverty_ratio'])
sorted_data2['area_highschool_ratio']=sorted_data2['area_highschool_ratio' ]/max(sorted_data2['area_highschool_ratio'])
data=pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1 )
data=data.sort_values('area_poverty_ratio',ascending=True)
f,ax1=plt.subplots(figsize=(18,10))
sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='lime',alpha=0.8)
sns.pointplot(x='area_list',y='area_highschool_ratio',data=data ,color='red',alpha=0.8)
plt.text(0.5,0.5,'area_poverty_ratio',color='red',fontsize = 17,style = 'italic')
plt.text(0.7,0.7,'area_highschool_ratio',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('States',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')
plt.grid()
l=sns.jointplot(data.area_poverty_ratio,data.area_highschool_ratio, kind="reg", size=7,dropna=True)
g = sns.jointplot("area_poverty_ratio", "area_highschool_ratio", data=data,size=5, ratio=3, color="r")
kill.race.head(15)

print(baska)
kill.race.dropna(inplace=True)
labels=kill.race.value_counts().index
baska=kill.race.index.values
colors = ['grey','blue','red','yellow','green','brown']
explode = [0,0,0,2,1,0]
sizes = kill.race.value_counts().values
explode = [0,0,0,2,1,0]

plt.figure(figsize=(7,7))
plt.pie(sizes,labels=labels,colors=colors, autopct='%1.1f%%')

plt.legend( labels, loc="upper right",title='kille peple according to Races')
plt.title('killed people According to Races',color='red',fontsize=15)
kill.race.dropna(inplace=True)
labels = kill.race.value_counts().index
baska = kill.race.index.values
print(baska,end=' index values olan deger')
print(labels,end='label bu')
colors = ['grey','blue','red','yellow','green','brown']
explode = [0,0,0,2,1,0]
sizes = kill.race.value_counts().values
print(sizes,end='value_counts().values')

# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Killed People According to Races',color = 'blue',fontsize = 15)
sns.lmplot(x="area_poverty_ratio", y="area_highschool_ratio", data=data)
plt.show()
sns.kdeplot(data.area_poverty_ratio,data.area_highschool_ratio,shade=True,cut=3)
plt.show()
s=sns.cubehelix_palette(8, start=.5, rot=-.75)

sns.violinplot(data=data,paletta=s,inner="points")
plt.show()
data.corr()
f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(),annot=True,linewidth=0.5, fmt='.1f', linecolor='red',ax=ax)
kill.head()

kill.manner_of_death.unique()

sns.boxplot(x="gender", y="age", hue="manner_of_death", data=kill, palette="PRGn")
plt.show()
sns.swarmplot(x="gender", y="age",hue="manner_of_death", data=kill)
plt.show()
sns.pairplot(data)
plt.show()
sns.countplot(kill.gender)
#sns.countplot(kill.manner_of_death)
plt.title("gender",color = 'blue',fontsize=15)
armed=kill.armed.value_counts()
plt.figure(figsize=(6,10))
sns.barplot(x=armed[:10].index,y=armed[:10].values)
plt.ylabel('Number of wepeon')
plt.xlabel('Weopen type')
plt.title('kill weapon', color='blue', fontsize=15)
armed = kill.armed.value_counts()
#print(armed)
plt.figure(figsize=(10,7))
sns.barplot(x=armed[:15].index,y=armed[:15].values)
plt.ylabel('Number of Weapon')
plt.xlabel('Weapon Types')
plt.title('Kill weapon',color = 'blue',fontsize=15)
above2=["yuksek" if i>25 else "dusuk" for i in kill.age]
df=pd.DataFrame({'above2':above2})

sns.countplot(x=df.above2)
plt.ylabel("yas")
plt.title("age of killed people", color="red")
sns.countplot(x=kill.race)
plt.title('According to race killing people')
city = kill.city.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=city[:15].index,y=city[:15].values)
plt.xticks(rotation=45)
plt.title('Most dangerous cities',color = 'blue',fontsize=15)
sta = kill.state.value_counts().index[:10]
sns.barplot(x=sta,y = kill.state.value_counts().values[:10])
plt.title('Kill Numbers from States',color = 'blue',fontsize=15)
g = sns.FacetGrid(train_df,col="Survived",row="Pclass",size=10)
g.map(plt.hist,"Age",bins=25)
plt.show()

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,10)) 

list1=["SibSp","Parch","Age","Fare","Survived"]
sns.heatmap(train_df[list1].corr(),annot=True, fmt="1.5f",cmap="YlGnBu")
plt.show()


g=sns.factorplot(x="SibSp",y="Survived",data=train_df,kind="bar",size=6)
g.get_ylabels("survived probability")
plt.show()
g=sns.factorplot(x="SibSp",y="Survived",data=train_df,kind="bar",size=6)
g.get_ylabels("survived probability")
plt.show()
g=sns.factorplot(x="Parch",y="Survived",kind="bar",data=train_df,size=6)
g.set_ylabels("Survived Probability")
plt.show()

g = sns.factorplot(x = "Pclass", y = "Survived", data = train_df, kind = "bar", size = 6)
g.set_ylabels("Survived Probability")
plt.show()


g=sns.FacetGrid(train_df, row="Embarked",size=5)
g.map(sns.pointplot,"Pclass","Survived","Sex")
g.add_legend()
plt.show()
train_df[train_df["Age"].isnull()]


sns.factorplot(x = "Sex", y = "Age", data = train_df, kind = "box")
plt.show()
sns.factorplot(x = "Sex", y = "Age", hue = "Pclass",data = train_df, kind = "box")
plt.show()
sns.factorplot(x = "Parch", y = "Age", data = train_df, kind = "box")
sns.factorplot(x = "SibSp", y = "Age", data = train_df, kind = "bar")
plt.show()
sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(), annot = True)
plt.show()