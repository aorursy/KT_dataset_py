import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS



plt.style.use('ggplot')
df = pd.read_csv('/kaggle/input/ramen-ratings/ramen-ratings.csv')
df.info()
print(f'There are {df.shape[0]} rows and {df.shape[1]} columns')
null_df = pd.DataFrame(df.isnull().sum())

nullpct_df = pd.DataFrame(df.isnull().sum()/len(df))



na_df_stats = null_df.merge(nullpct_df, left_index= True, right_index= True)

na_df_stats.columns = ['NA count', '%']
na_df_stats
df = df.loc[~df['Style'].isna(), :]

df.drop(columns='Review #', inplace= True, axis= 1)
df['Stars'] = pd.to_numeric(df.Stars, errors= 'coerce')
print(f'The mean of ramen ratings is: {np.mean(df.Stars)}')

print(f'The standard deviation of ramen ratings is: {np.std(df.Stars)}')
plt.figure(figsize=(15,8))

plt.title('Count of Different Unique Values in Object Columns')

objects = df.select_dtypes(object).drop(columns='Variety', axis=1).apply(pd.Series.nunique)

objects.plot(kind='bar')
df.groupby('Country', as_index= False).agg({'Stars':'mean'}).sort_values(by= 'Stars', ascending= False)
df.groupby('Style', as_index= False)['Stars'].mean()
df.Country.value_counts().tail()
df.Brand.value_counts().tail()
brand_counts = df.Brand.value_counts()

brand_counts = brand_counts[brand_counts > 50]



list1 = []

for i in brand_counts.index:

    list1.append(i)
df1_1 = df.copy()



#This will convert brands with less than 50 counts into 'other'

df1_1['Brand'] = df1_1['Brand'].apply(lambda x: x if x in list1 else 'other')
df1_1_groupby = df1_1.groupby('Brand', as_index= False)['Stars'].mean().sort_values(ascending=False, by= 'Stars')



plt.figure(figsize=(15,8))

plt.title('Average Brand Rating')

plt.xticks(rotation= 25, fontsize=12)

sns.barplot(data=df1_1_groupby, x='Brand', y='Stars', palette= 'rocket')
df3 = df1_1.copy()
print('Here we see that the style of ramen is mainly divided into 4 types. Box, Bar, and Can have very little representation in the data. For now, we will ignore')

df3.Style.value_counts()
plt.figure(figsize=(15,8))

plt.title('Ramen Style Rating Histogram')

sns.kdeplot(df3.loc[df['Style'] == 'Pack', 'Stars'], color= 'yellow', label='Pack')

sns.kdeplot(df3.loc[df['Style'] == 'Bowl', 'Stars'], color= 'red', label='Bowl')

sns.kdeplot(df3.loc[df['Style'] == 'Cup', 'Stars'], color= 'blue', label='Cup')

sns.kdeplot(df3.loc[df['Style'] == 'Tray', 'Stars'], color= 'green', label='Tray')
#Convert 'United States' value to 'USA'

df1_1.loc[df1_1.Country == 'United States', 'Country'] = 'USA'
#Find list that has countries with more than 40 reviews

df_country = df1_1.Country.value_counts()

df_country = df_country[df_country > 40]

df_country = df_country.index



df2 = df1_1.loc[df1_1['Country'].isin(df_country), :]



styles = ['Pack', 'Bowl', 'Cup','Tray']

df2 = df2.loc[df2.Style.isin(styles), :]
df2.Country.value_counts()
df2.groupby(['Country', 'Style'])['Stars'].mean().unstack().plot(kind= 'bar', stacked= True, figsize=(15,10))

plt.xticks(rotation= 25, fontsize=12)

plt.title('Average Star Rating By Style Per Country')
df3_1 = df2.copy()



df3_1 = df2.loc[df2['Top Ten'].notnull(), :]

df3_1 = df3_1.loc[df3_1['Top Ten'] != '\n', :]
df3_1['Year'] = df3_1['Top Ten'].apply(lambda x: x.split('#')[0])

df3_1['Ranking'] = df3_1['Top Ten'].apply(lambda x: x.split('#')[1])
df3_1
pie = df3_1.loc[df3_1['Ranking'] == '1', :]

pie_grp = pie.Country.value_counts()



plt.figure(figsize=(15,10))

plt.title('Where #1 Rated Ramen Brands Are From')

pie_grp.plot.pie(textprops={'fontsize': 13}, shadow= True)
pie2 = df3_1.Country.value_counts()



plt.figure(figsize=(15,10))

plt.title('Where Top Ten Rated Ramen Brands Are From')

pie2.plot.pie(textprops={'fontsize': 13}, shadow= True)
plt.figure(figsize=(20,10))

plt.yticks(fontsize=18)

plt.xticks(fontsize=18)

plt.title('Count of Ramen Brands in Top Ten')

sns.countplot(data= df3_1, y= 'Brand', order=df3_1.Brand.value_counts().index, palette= 'rocket')
plt.figure(figsize=(15,5))

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)

plt.title('Count of Ramen Styles in Top Ten')

sns.countplot(data= df3_1, y= 'Style', palette='rocket')
#Custom function to extract text from variety column

def get_text(column):

    words = ''

    for text in column:

        words += text

    return words
text1 = get_text(df3_1['Variety'])



stopwords = set(STOPWORDS)

wc = WordCloud(background_color= 'black', stopwords= stopwords,

              width=1600, height=800)



wc.generate(text1)

plt.figure(figsize=(20,10), facecolor='k')

plt.axis('off')

plt.tight_layout(pad=0)

plt.imshow(wc)

plt.show()
text2 = get_text(df2['Variety'])



stopwords = set(STOPWORDS)

wc = WordCloud(background_color= 'black', stopwords= stopwords,

              width=1600, height=800)



wc.generate(text1)

plt.figure(figsize=(20,10), facecolor='k')

plt.axis('off')

plt.tight_layout(pad=0)

plt.imshow(wc)

plt.show()