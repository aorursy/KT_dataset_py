import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import os 

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/dataisbeautiful/r_dataisbeautiful_posts.csv')
data.head()
data['title'][55]
data.score.nunique()
(data.isnull().sum() / len(data)) * 100
data.head()
del data['id']

del data['author_flair_text']

del data ['removed_by']

del data ['total_awards_received']

del data['awarders']

del data ['created_utc']

del data ['full_link']
data.head(100)
data.describe().T
len(data)
print(len(data[data['score'] < 20]), 'Posts with less than 20 votes')

print(len(data[data['score'] > 20]), 'Posts with more than 20 votes')
print(len(data[data['num_comments'] < 20]), 'Posts with less than 20 comments')

print(len(data[data['num_comments'] > 20]), 'Posts with more than 20 comments')
data[data['score'] == data['score'].max()]['title'].iloc[0]
from wordcloud import WordCloud, STOPWORDS
words = data['title'].values
type(words[25])
ls = []



for i in words:

    ls.append(str(i))
ls[325]
type(words[325])
ls[0]
# The wordcloud of Cthulhu/squidy thing for HP Lovecraft

plt.figure(figsize=(16,13))

wc = WordCloud(background_color="red", stopwords = STOPWORDS, max_words=2000, max_font_size= 300,  width=1600, height=800)

wc.generate(" ".join(ls))

plt.title("Most Discussed Terms", fontsize=20)

plt.imshow(wc.recolor( colormap= 'viridis' , random_state=17), alpha=0.98, interpolation="bilinear", )

plt.axis('off')
most_pop = data.sort_values('score', ascending  = False )[['title','score']].head(12)

most_pop['score1'] = most_pop['score']/1000
import matplotlib.style as style

style.available 
style.use('seaborn-notebook')
plt.figure(figsize = (20, 15))

sns.barplot(data = most_pop, y = 'title', x = 'score1', color = 'C')

plt.xticks(fontsize=20, rotation=0)

plt.yticks(fontsize=21, rotation=0)

plt.xlabel('Votes in Thousands', fontsize = 21)

plt.ylabel('')

plt.title('Most popular posts', fontsize = 30)
data.head()