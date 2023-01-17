import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

from wordcloud import WordCloud, STOPWORDS



%matplotlib inline
df = pd.read_csv('../input/windows-store/msft.csv')
df.info()
df.head()
def findNull(dataFrame):

    for col in dataFrame.columns:

        null_sum = df[col].isna().sum()

        print(f'"{col}": {null_sum} null values')
findNull(df)
mask=df.isna().any(axis=1)==True

df[mask]
df = df.drop(index=5321)
findNull(df)
# check the Price data

df['Price'].unique()
# delete ','

df['Price'] = df['Price'].str.replace(',','')



# replace'Free' with 0

df.loc[df['Price']=='Free','Price']=0



# delete '₹ ' 

df.loc[df['Price']!=0,'Price']=df['Price'].str[2:]



# convert string to float

df['Price'] = df['Price'].astype(float)



#result

df['Price'].unique()
df['year'] = df['Date'].str[6:10]

df['month'] = df['Date'].str[3:5]

df['year_month'] = df['Date'].str[6:10]+'-'+df['Date'].str[3:5]
df['Price'].unique()
Ccount = df['Category']

Ccount = Ccount.reset_index()

Ccount = Ccount.groupby(['Category']).count()

Ccount = Ccount.sort_values('index',ascending=False)





color = ("#55efc4","#81ecec","#74b9ff","#a29bfe","#dfe6e9",

         "#ffeaa7","#e17055","#d63031","#e84393","#2d3436",

         "#00b894","#0984e3","#ffeaa7","#fab1a0","#fd79a8")



label = Ccount.index



plt.figure(figsize=(15,10))

plt.title("App category ratio",fontsize=20)



plt.pie(Ccount, labels=label,colors=color,counterclock=False, startangle=90,autopct="%1.1f%%", pctdistance=0.7)

plt.show()
tmp1 = df[['Price']]

tmp1 = tmp1.reset_index()

tmp1 = tmp1.groupby(['Price']).count()

tmp1 = tmp1.reset_index()

tmp1 = tmp1.rename(columns={'index': 'count'})

# Pcount



tmp2 = tmp1[tmp1["Price"]!=0]

tmp2 = tmp2.reset_index()

tmp2 = tmp2.groupby(['Price']).count()

tmp2 = tmp2.reset_index()

sum_not_free = tmp2['count'].sum()



tmp1 = tmp1[:1]

Pcount = tmp1.append({'Price': 'Not Free Apps', 'count':sum_not_free }, ignore_index=True)

Pcount.loc[Pcount['Price']==0, 'Price'] = "Free Apps"



label = Pcount['Price']

color = ("#4bcffa","#ff5e57")



plt.figure(figsize=(15,10))

plt.title("Free apps ratio",fontsize=20)



plt.pie(Pcount['count'],labels=label,colors=color,counterclock=False, startangle=90,autopct="%1.1f%%", pctdistance=0.7)

plt.show()
plt.subplots(figsize=(12,6))

plt.hist(df['Rating'],color="#ff6b6b")



plt.title("Rating distribution",fontsize=20)

plt.xlabel("Rating", fontsize=15)

plt.ylabel("Count", fontsize=15)

plt.show()
fig = px.scatter(df, x='Price', y='Rating')

fig.update_layout(title_text="Relationship between Rating and Price")

fig.show()
fig = px.scatter(df, x='year_month', y='Rating',color='No of people Rated')

fig.update_layout(title_text="Relationship among year_month, Rating and No of people Rated")

fig.show()
Ycount = df[['year']]

Ycount = Ycount.reset_index()

Ycount = Ycount.groupby(['year']).count()

Ycount = Ycount.reset_index()

Ycount = Ycount.rename(columns={'index': 'count'})



plt.subplots(figsize=(10,5))

plt.title('Release Year distribution',fontsize=20)

plt.xlabel('Year',fontsize=15)

plt.ylabel('Count',fontsize=15)

plt.bar(Ycount['year'], Ycount['count'],color="#808e9b")

plt.show()
sample1 = df[['year','Rating']]

sample1 = sample1.groupby(['year'])['Rating'].mean()

sample1 = sample1.reset_index()



plt.subplots(figsize=(10,5))

plt.title('Avg. rating by release year',fontsize=20)

plt.xlabel('Year',fontsize=15)

plt.ylabel('Avg. rating',fontsize=15)

plt.plot(sample1['year'],sample1['Rating'],color="#3c40c6")

plt.show()
sample2 = df[['month','Rating']]

sample2 = sample2.groupby(['month'])['Rating'].mean()

sample2 = sample2.reset_index()



plt.subplots(figsize=(10,5))

plt.title('Avg. rating by release month',fontsize=20)

plt.xlabel('Month',fontsize=15)

plt.ylabel('Avg. rating',fontsize=15)



plt.plot(sample2['month'],sample2['Rating'],color="#05c46b")

plt.title

plt.show()
sample3 = df[['year_month','Rating']]

sample3 = sample3.groupby(['year_month'])['Rating'].mean()

sample3 = sample3.reset_index()



plt.subplots(figsize=(20,10))

plt.title('Avg. rating by release year_month',fontsize=20)

plt.xlabel('Year_Month',fontsize=15)

plt.ylabel('Avg. rating',fontsize=15)



plt.plot(sample3['year_month'],sample3['Rating'],color="#ffa801")

plt.xticks(rotation=90)



plt.tight_layout()

plt.show()
plt.subplots(figsize=(10,5))

plt.title('Number of Apps by category',fontsize=20)

plt.xlabel('Category',fontsize=15)

plt.ylabel('count',fontsize=15)

df['Category'].value_counts().plot(kind="bar",color="#ff6d69")
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='#c8d6e5',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=50, 

                          random_state=42

                         ).generate(str(df['Name']))



plt.subplots(figsize=(12,6))

plt.axis('off')

plt.imshow(wordcloud)