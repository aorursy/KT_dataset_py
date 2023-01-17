import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud,STOPWORDS



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)     



df=pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv',index_col='Id')
df.head(3)
df_row,df_column=df.shape

print('Amazon Fine Food review dataset has {} rows and {} columns.'.format(df_row,df_column))

df.info()
column_list=df.columns.to_list()

print('Unique values in each column')

for column in column_list:

    print(column,": ",df[column].nunique())
plt.style.use('tableau-colorblind10')

df['Score'].value_counts().plot.bar(title='Number of reviews per rating')

plt.xlabel('Rating')

plt.ylabel('Number of reviews')

print((df['Score'].value_counts(normalize=True)*100).round(2))
pos_rev=df[df['Score']>3]

pos_rev_text=[]

for x in pos_rev['Summary']:

     pos_rev_text.append(x)

pos_rev_text=(' ').join(str(x) for x in pos_rev_text).lower()

plt.figure(figsize=(10,10))

pos_rev_cloud=WordCloud(max_words=100,stopwords=set(STOPWORDS),random_state=None,background_color='#aaffaa').generate(pos_rev_text)

plt.imshow(pos_rev_cloud,interpolation='bilinear')

plt.axis('off')

plt.show()
neg_rev=df[df['Score']<3]

neg_rev_text=[]

for x in neg_rev['Summary']:

    neg_rev_text.append(x)

neg_rev_text=(' ').join(str(x) for x in neg_rev_text).lower()

plt.figure(figsize=(10,10))

neg_rev_cloud=WordCloud(max_words=100,stopwords=set(STOPWORDS),random_state=None,background_color='black').generate(neg_rev_text)

plt.imshow(neg_rev_cloud,interpolation='bilinear')

plt.axis('off')

plt.show()
bins=[-0.5,0.5,2,100,1000]

group=['Not Voted','Useful','Highly Useful', 'Extremly Useful']

df['helpful']=pd.cut(df['HelpfulnessNumerator'],bins,labels=group)

helpfulness=pd.crosstab(df['Score'],df['helpful'])

helpfulness.div(helpfulness.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
df['WordCount']=df['Text'].str.split(' ').apply(lambda x: len(x))

df['WordCount'].groupby(df['Score']).describe()
pd.DataFrame(df[df['Score']>3]['ProductId'].value_counts().head(5))
pd.DataFrame(df[df['Score']<3]['ProductId'].value_counts().head(5))
top1_rev_df=df[df['ProductId']=='B007JFMH8M']

top1_rev_df['helpful'].value_counts()
print(top1_rev_df[top1_rev_df['helpful']=='Highly Useful']['Text'])
df[df['HelpfulnessNumerator']==df['HelpfulnessNumerator'].max()]
top_df=df[df['HelpfulnessNumerator']>100].sort_values(by='HelpfulnessNumerator',ascending=False)
top_df_copy=top_df.drop(columns=['ProductId','UserId','ProfileName'],axis=1)

top_df_no_duplicates=top_df_copy.drop_duplicates()

top_df_useful=top_df.loc[top_df_no_duplicates.index]
top_df_useful.groupby('Score')['WordCount'].describe()