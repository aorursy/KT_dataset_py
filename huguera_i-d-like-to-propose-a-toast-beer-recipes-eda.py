import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

sns.set()
pd.options.display.max_columns = 30
df = pd.read_csv('../input/recipeData.csv', encoding='latin1')
df.head()
print('Data set shape:{}'.format(df.shape))
print('Unique styles: {}'.format(df['Style'].nunique()))
df_missing = df.copy()
df_missing = df_missing.T
true = df_missing.isnull().sum(axis=1)
false = (len(df_missing.columns) - true)
df_missing['Valid Count'] = false / len(df_missing.columns)
df_missing['NA Count'] = true / len(df_missing.columns)

df_missing[['NA Count','Valid Count']].sort_values(
    'NA Count', ascending=False).plot.bar(
    stacked=True,figsize=(12,6))
plt.legend(loc=9)
plt.ylim(0,1.15)
plt.title('Normed Missing Values Count', fontsize=20)
plt.xlabel('Normed (%) count', fontsize=20)
plt.ylabel('Column name', fontsize=20)
plt.xticks(rotation=60)
plt.show()

df = df[pd.notnull(df['Style'])] #use only samples with valid Style col
gb_style = df.groupby(['Style']).count().sort_values(['BeerID'], ascending=False).reset_index()[:20]
gb_style['BeerID'] = (gb_style['BeerID'] / len(df)) * 100

plt.figure(figsize=(12,8))
g = sns.barplot(x=gb_style['BeerID'], y=gb_style['Style'], orient='h')
plt.title('Normed Style Popularity (%) for 20 most popular Styles', fontsize=22)
plt.ylabel('Style Name', fontsize=20)
plt.xlabel('Normed Style Popularity (%)', fontsize=20)

plt.xlim(0,18)

for index, row in gb_style.iterrows():
    g.text(y=index+0.2,x=row['BeerID']+1,s='{:.2f}%'.format(row['BeerID']),
           color='black', ha="center", fontsize=16)

plt.show()
general_styles = ['Amber Ale','Pale Ale','Red Ale','Cider','Spice Beer',
                  'IPA','Lager','Specialty','Porter','Wheat Beer']
general_styles_dict = {'Brown':'Red','Fruit':'Spice', 'Stout':'Porter'}

df_general_styles = df.copy()
df_general_styles['Style_aux'] = 'Other'
for style in general_styles:
    df_general_styles.loc[df_general_styles['Style'].str.contains(style), 'Style_aux'] = style
for key in general_styles_dict:
    df_general_styles.loc[df_general_styles['Style'].str.contains('{} Ale'.format(key)), 'Style_aux'] = '{} Ale'.format(general_styles_dict[key])

df_general_styles = df_general_styles[df_general_styles['Style_aux']!='Other']
gb_style = df_general_styles.groupby(['Style_aux']).count().sort_values(['BeerID'], ascending=False).reset_index()[:20]
gb_style['BeerID'] = (gb_style['BeerID'] / len(df)) * 100

plt.figure(figsize=(12,6))
g=sns.barplot(x=gb_style['BeerID'], y=gb_style['Style_aux'], orient='h')
plt.title('Normed Style Popularity (%) for GENERAL Styles', fontsize=25)
plt.ylabel('Style Name', fontsize=20)
plt.xlabel('Normed Style Popularity (%)', fontsize=20)
plt.xlim(0,22.5)

for index, row in gb_style.iterrows():
    g.text(y=index+0.1,x=row['BeerID']+1,s='{:.2f}%'.format(row['BeerID']),
           color='black', ha="center", fontsize=16)

plt.show()
plt.figure(figsize=(12,12))
count=0
for col, color in zip(['OG', 'FG', 'ABV', 'IBU','Color'],['b','y','c','m','g']):
    count+=1
    if(count==5):
        plt.subplot(3,2,(5,6))
    else:
        plt.subplot(3,2,count)
    sns.distplot(df[col], bins=100, label=col, color=color)
    plt.title('{} Distribution'.format(col), fontsize=15)
    plt.legend()
    plt.ylabel('Normed Frequency', fontsize=15)
    plt.xlabel(col, fontsize=15)

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
plt.figure(figsize=(12,12))
count=0
for col, color in zip(['OG', 'FG', 'ABV', 'IBU','Color'],['b','y','c','m','g']):
    count+=1
    if(count==5):
        plt.subplot(3,2,(5,6))
    else:
        plt.subplot(3,2,count)
    sns.distplot(np.log1p(df[col]), bins=100, label=col, color=color)
    plt.title('Log(1 + {}) Distribution'.format(col), fontsize=15)
    plt.legend()
    plt.ylabel('Normed Frequency', fontsize=15)
    plt.xlabel('Log(1 + {})'.format(col), fontsize=15)

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
general_styles = ['Amber Ale','Pale Ale','Red Ale','Cider','Spice Beer',
                  'IPA','Lager','Specialty','Porter','Wheat Beer']
general_styles_dict = {'Brown':'Red','Fruit':'Spice', 'Stout':'Porter'}

df_general_styles = df.copy()
df_general_styles['Style_aux'] = 'Other'
for style in general_styles:
    df_general_styles.loc[df_general_styles['Style'].str.contains(style), 'Style_aux'] = style
for key in general_styles_dict:
    df_general_styles.loc[df_general_styles['Style'].str.contains('{} Ale'.format(key)), 'Style_aux'] = '{} Ale'.format(general_styles_dict[key])

plt.figure(figsize=(12,6))
sns.boxplot(df_general_styles['Style_aux'], df_general_styles['ABV'])
plt.xticks(rotation=45)
plt.ylim(0,25)
plt.title('ABV by GENERAL Styles', fontsize=22)
plt.xlabel('Style', fontsize=20)
plt.ylabel('ABV', fontsize=20)
plt.show()
order = df_general_styles.groupby('Style_aux')['Color'].median().fillna(0).sort_values()[::-1].index

plt.figure(figsize=(12,6))
sns.boxplot(df_general_styles['Style_aux'], df_general_styles['Color'])
plt.xticks(rotation=45)
plt.ylim(0,55)
plt.title('Color by GENERAL Styles', fontsize=22)
plt.xlabel('Style', fontsize=20)
plt.ylabel('Color', fontsize=20)
plt.show()
df_abv_color = df[(df['ABV']<=20) & (df['Color']<=50)]
df_abv_color = df_abv_color.sample(int(len(df_abv_color)/10), random_state=42)

plt.figure(figsize=(12,6))
sns.regplot(df_abv_color['ABV'],df_abv_color['Color'])
plt.title('ABV and Color relation', fontsize=22)
plt.xlabel('ABV', fontsize=20)
plt.ylabel('Color', fontsize=20)
plt.show()
plt.figure(figsize=(12,14))
count=0    
for col in ['OG', 'FG', 'IBU']:
    for i in range(1,3):
        count+=1
        plt.subplot(3,2,count)

        if (i==1):
            sns.boxplot(df_general_styles['Style_aux'], np.log1p(df_general_styles[col]))
        else:
            sns.violinplot(df_general_styles['Style_aux'], np.log1p(df_general_styles[col]))
        plt.xticks(rotation=45)
        plt.title('Log (1+{}) by GENERAL Styles'.format(col), fontsize=14)
        plt.xlabel(' ')
        plt.ylabel('Log (1+{})'.format(col), fontsize=14)

plt.subplots_adjust(hspace=0.4)
plt.show()
col= 'BrewMethod'
gb_brew_method = df.groupby([col]).count().sort_values(['BeerID'], ascending=False).reset_index()[:20]
gb_brew_method['BeerID'] = (gb_brew_method['BeerID'] / len(df)) * 100

plt.figure(figsize=(8,6))
g=sns.barplot(gb_brew_method[col], gb_brew_method['BeerID'])
plt.title('{} Distribution'.format(col), fontsize=15)
plt.legend()
plt.ylabel('Normed Frequency (%)', fontsize=15)
plt.xlabel(col, fontsize=15)

for index, row in gb_brew_method.iterrows():
    g.text(x=index,y=row['BeerID']+1,s='{:.2f}%'.format(row['BeerID']), 
           color='black', ha="center", fontsize=16)

plt.show()
col = 'SugarScale'
gb_brew_method = df.groupby([col]).count().sort_values(['BeerID'], ascending=False).reset_index()[:20]
gb_brew_method['BeerID'] = (gb_brew_method['BeerID'] / len(df)) * 100

plt.figure(figsize=(8,6))
g=sns.barplot(gb_brew_method[col], gb_brew_method['BeerID'])
plt.title('{} Distribution'.format(col), fontsize=15)
plt.legend()
plt.ylabel('Normed Frequency (%)', fontsize=15)
plt.xlabel(col, fontsize=15)

for index, row in gb_brew_method.iterrows():
    g.text(x=index,y=row['BeerID']+1,s='{:.2f}%'.format(row['BeerID']), 
           color='black', ha="center", fontsize=16)

plt.show()
word_cloud = WordCloud(width=1440, height=1080, 
                 stopwords=set(STOPWORDS)).generate(" ".join(df['Name'].astype(str)))
fig = plt.figure(figsize=(12, 12))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()
fig.savefig("homemade_beer_word_cloud.png", dpi=900)