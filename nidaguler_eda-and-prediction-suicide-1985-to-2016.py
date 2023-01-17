import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.basemap import Basemap

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator





import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#pip install plotly==3.10.0
df=pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")

countries = pd.read_csv("../input/counties-geographic-coordinates/countries.csv")
df.head()
df.columns=["country","year","gender","age_group","suicide_number","population","sui_pop","country_year","hdi","gdp_for_year","gdp_per_capita","generation"]
df.drop(["country_year","gdp_for_year","gdp_per_capita","hdi"],axis=1,inplace=True)

df.head()
df.age_group.unique()
df["age_group"]=df["age_group"].apply(lambda x: str(x).replace('5-14 years','child') if '5-14 years' in str(x) else str(x))

df["age_group"]=df["age_group"].apply(lambda x: str(x).replace('15-24 years','youth') if '15-24 years' in str(x) else str(x))

df["age_group"]=df["age_group"].apply(lambda x: str(x).replace('25-34 years','young adult') if '25-34 years' in str(x) else str(x))

df["age_group"]=df["age_group"].apply(lambda x: str(x).replace('35-54 years','early adult') if '35-54 years' in str(x) else str(x))

df["age_group"]=df["age_group"].apply(lambda x: str(x).replace('55-74 years','adult') if '55-74 years' in str(x) else str(x))

df["age_group"]=df["age_group"].apply(lambda x: str(x).replace('75+ years','senior') if '75+ years' in str(x) else str(x))
df.head()
df.tail()
df.info()
df.shape
df.describe()
df.isnull().sum()
df.isnull().sum()
df.head()
df = df.sort_values(by=["suicide_number","age_group"], ascending=False)

df['rank']=tuple(zip(df.suicide_number,df.age_group))

df['rank']=df.groupby('suicide_number',sort=False)['rank'].apply(lambda x : pd.Series(pd.factorize(x)[0])).values

df.head()
df.reset_index(inplace=True,drop=True)

df.head()
df["gender"].value_counts()
#Gender show bar plot

sns.set(style='whitegrid')

ax=sns.barplot(x=df['gender'].value_counts().index,y=df['gender'].value_counts().values,palette="Blues_d",hue=['female','male'])

plt.legend(loc=8)

plt.xlabel('Gender')

plt.ylabel('Frequency')

plt.title('Show of Gender Bar Plot')

plt.show()
df["age_group"].value_counts()
plt.figure(figsize=(7,3))

sns.barplot(x=df['age_group'].value_counts().index,

              y=df['age_group'].value_counts().values)

plt.xlabel('age_group')

plt.ylabel('Frequency')

plt.title('Show of age_group Bar Plot')

plt.show()
df["generation"].value_counts()
plt.figure(figsize=(15,2))

sns.barplot(x=df['generation'].value_counts().index,

              y=df['generation'].value_counts().values)

plt.xlabel('generation')

plt.ylabel('Frequency')

plt.title('Show of generation Bar Plot')

plt.show()
df.nunique()
plt.figure(figsize=(10,3))

sns.barplot(x = "generation", y = "suicide_number", hue = "gender", data = df)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(10,3))

sns.barplot(x = "age_group", y = "suicide_number", hue = "gender", data = df)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(12,7))

sns.catplot(y="gender", x="suicide_number",

                 hue="age_group",

                 data=df, kind="bar")

plt.title('for age group & suicide_number')

plt.show()
plt.figure(figsize=(10,10))

sns.catplot(x="gender", y="suicide_number",

                 hue="generation",

                 data=df, kind="bar")

plt.title('for generation & suicide number')

plt.show()
f,ax=plt.subplots(figsize=(9,3))

sns.barplot(x=df['gender'].value_counts().values,y=df['gender'].value_counts().index,alpha=0.5,color='red',label='Gender')

sns.barplot(x=df['age_group'].value_counts().values,y=df['age_group'].value_counts().index,color='blue',alpha=0.7,label='Age Group')

ax.legend(loc='upper right',frameon=True)

ax.set(xlabel='Gender , Age Group',ylabel='Groups',title="Gender vs Age Group ")

plt.show()
plt.figure(figsize=(10,5))

sns.barplot(df.suicide_number[:600],df.country[:600])

plt.show()
g = sns.jointplot("population", "suicide_number", data=df, kind="reg",

                  xlim=(260, 43805220), ylim=(0, 22340), color="m", height=7)
sns.kdeplot(df['suicide_number'])

plt.xlabel('Values')

plt.ylabel('Frequency')

plt.title('Suicide Number Kde Plot System Analysis')

plt.show()
sns.violinplot(df['suicide_number'])

plt.xlabel('suicide_number')

plt.ylabel('Frequency')

plt.title('Violin suicide_number Show')

plt.show()
df.head()
sns.heatmap(df.corr())

plt.show()
sns.boxenplot(x="age_group", y="suicide_number",

              color="b",

              scale="linear", data=df)

plt.show()
plt.figure(figsize=(16,5))

sns.barplot(df.suicide_number,df.gender)

plt.show()
plt.figure(figsize=(16,5))

sns.barplot(df.suicide_number,df.age_group)

plt.show()
df.age_group.dropna(inplace = True)

labels = df.age_group.value_counts().index

colors = ['b','r','g','orange','pink','y']

explode = [0,0,0,0,0,0]

sizes = df.age_group.value_counts().values



# visual 

plt.figure(0,figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Suicide According to Age Group',color = 'blue',fontsize = 15)

plt.show()
plt.figure(figsize=(16,5))

sns.barplot(df.suicide_number,df.generation)

plt.show()
df.generation.dropna(inplace = True)

labels = df.generation.value_counts().index

colors = ['b','r','g','orange','pink','y']

explode = [0,0,0,0,0,0]

sizes = df.generation.value_counts().values



# visual 

plt.figure(0,figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Suicide According to Generation',color = 'blue',fontsize = 15)

plt.show()
f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='year',y='suicide_number',data=df,color='lime',alpha=0.8)

plt.xlabel('year',fontsize = 15,color='blue')

plt.ylabel('values',fontsize = 15,color='blue')

plt.title('year - suicide number',fontsize = 20,color='blue')

plt.grid()
x1985 = df.country[df.year == 1985]

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(x1985))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
x1995 = df.country[df.year == 1995]

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(x1995))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
x2005 = df.country[df.year == 2005]

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(x2005))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
x2015 = df.country[df.year == 2015]

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(x2015))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
df.columns
df.corr()
#create a new data frame

data=pd.DataFrame(df.iloc[:,4:6])



data.head(3)
plt.figure(figsize=(20,5))

plt.scatter(data.population*0.1,data.suicide_number)

plt.xlabel("Population")

plt.ylabel("Suicide Number")

plt.show()
from sklearn.linear_model import LinearRegression



linear_reg=LinearRegression()



x=data.population.values.reshape(-1,1)

y=data.suicide_number.values.reshape(-1,1)



linear_reg.fit(x,y)



b0 =linear_reg.intercept_

b1=linear_reg.coef_

print("b0:",b0)

print("b1:",b1)

print("Prediction 5M:",linear_reg.predict([[5000000]]))

print("Prediction 10M:",linear_reg.predict([[10000000]]))

print("Prediction 15M:",linear_reg.predict([[15000000]]))

df.population.min()
df.population.max()
array=np.array([278,5000000,10000000,15000000,20000000,45000000,80000000]).reshape(-1,1)

y_head=linear_reg.predict(array)

print("y_head:",y_head)
plt.figure(figsize=(20,5))

plt.scatter(x,y)

plt.plot(array,y_head,color='r')

plt.show()