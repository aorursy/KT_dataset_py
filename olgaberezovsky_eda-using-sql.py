#Getting all the packages we need: 



import numpy as np # linear algebra

import pandas as pd # data processing



import seaborn as sns #statist graph package

import matplotlib.pyplot as plt #plot package

import pandasql as ps #sql package

import wordcloud #will use for the word cloud plot

from wordcloud import WordCloud, STOPWORDS # optional to filter out the stopwords



#Optional helpful plot stypes:

plt.style.use('bmh') #setting up 'bmh' as "Bayesian Methods for Hackers" style sheet

#plt.style.use('ggplot') #R ggplot stype

#print(plt.style.available) #pick another style

df = pd.read_csv('/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv')

df.sample(5)
df.tail(5)
print("Data shape :",df.shape)
df.info()

df.describe()
#Empty values:



df.isnull().sum().sort_values(ascending = False)
q1 = """SELECT removed_by, count(distinct id)as number_of_removed_posts

FROM df 

where removed_by is not null 

group by removed_by 

order by 2 desc """



grouped_df = ps.sqldf(q1, locals())

grouped_df
#Visualizing bar chart based of SQL output:



removed_by = grouped_df['removed_by'].tolist()

number_of_removed_posts = grouped_df['number_of_removed_posts'].tolist()



plt.figure(figsize=(12,8))

plt.ylabel("Number of deleted reddits")

plt.bar(removed_by, number_of_removed_posts)



plt.show()
q2 = """SELECT author, count(id) as number_of_removed_posts 

FROM df 

where removed_by = 'moderator' 

group by author 

order by 2 desc 

limit 3"""

print(ps.sqldf(q2, locals()))
#Step 1: Getting proportion of all removed posts / removed "virus" posts

q3 = """

with Virus as (

SELECT id 

FROM df 

where removed_by = 'moderator' 

and title like '%virus%'

)



SELECT count(v.id) as virus_removed, count(d.id) as all_removed

FROM df d 

left join virus v on v.id = d.id 

where d.removed_by = 'moderator';"""



removed_moderator_df = ps.sqldf(q3, locals())



#print(type(removed_moderator_df))

print(removed_moderator_df.values)

print(removed_moderator_df.values[0])
#Step 2: getting % virus reddits from all removed posts:



virus_removed_id = removed_moderator_df.values[0][0]

all_removed_id = removed_moderator_df.values[0][1]





print(virus_removed_id/all_removed_id)
#Top 10 reddits with the most number of comments:



q4 = """SELECT title, num_comments as number_of_comments 

FROM df  

where title != 'data_irl'

order by 2 desc 

limit 10"""

print(ps.sqldf(q4, locals()))
#To build a wordcloud, we have to remove NULL values first:

df["title"] = df["title"].fillna(value="")
#Now let's add a string value instead to make our Series clean:

word_string=" ".join(df['title'].str.lower())



#word_string
#And - plotting:



plt.figure(figsize=(15,15))

wc = WordCloud(background_color="purple", stopwords = STOPWORDS, max_words=2000, max_font_size= 300,  width=1600, height=800)

wc.generate(word_string)



plt.imshow(wc.recolor( colormap= 'viridis' , random_state=17), interpolation="bilinear")

plt.axis('off')
#Comments distribution plot:



fig, ax = plt.subplots()

_ = sns.distplot(df[df["num_comments"] < 25]["num_comments"], kde=False, rug=False, hist_kws={'alpha': 1}, ax=ax)

_ = ax.set(xlabel="num_comments", ylabel="id")



plt.ylabel("Number of reddits")

plt.xlabel("Comments")



plt.show()
df.corr()
h_labels = [x.replace('_', ' ').title() for x in 

            list(df.select_dtypes(include=['number', 'bool']).columns.values)]



fig, ax = plt.subplots(figsize=(10,6))

_ = sns.heatmap(df.corr(), annot=True, xticklabels=h_labels, yticklabels=h_labels, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)
df.score.describe()
df.score.median()
#Score distribution: 



fig, ax = plt.subplots()

_ = sns.distplot(df[df["score"] < 22]["score"], kde=False, hist_kws={'alpha': 1}, ax=ax)

_ = ax.set(xlabel="score", ylabel="No. of reddits")