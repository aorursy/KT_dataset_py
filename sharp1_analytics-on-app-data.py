import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
print(os.listdir("../input"))
df=pd.read_csv("../input/AppleStore.csv")

mycol=list(df.columns)
mycol.remove("Unnamed: 0")
df=df[mycol]
df.head()
df.describe()
df.info()
df.shape[0]
df1=df[df['price']==0]
df1.shape[0]
df1.head()
len(list(set(df1['prime_genre'])))
plt.figure(figsize=(20,10))
sns.countplot(df1['prime_genre'])
plt.xticks(rotation=90)

plt.figure(figsize=(20,10))
sns.countplot(df1['prime_genre'],hue='cont_rating',data=df1)
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1, 1))
plt.figure(figsize=(20,10))
sns.countplot(df1['prime_genre'],hue='user_rating',data=df1)
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1, 1))
plt.figure(figsize=(20,10))
sns.countplot(df1['prime_genre'],hue='user_rating_ver',data=df1)
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1, 1))
sns.violinplot(x='user_rating',data=df1)
sns.violinplot(x='user_rating_ver',data=df1,color='M')
plt.figure(figsize=(25,10))
sns.violinplot(x='user_rating_ver',y='prime_genre',data=df1,palette='dark')
plt.figure(figsize=(25,10))
sns.violinplot(y='user_rating_ver',x='prime_genre',data=df1,palette='dark')
plt.figure(figsize=(20,10))
df1['prime_genre'].value_counts().plot.bar()
import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

from subprocess import check_output
import wordcloud
from wordcloud import WordCloud, STOPWORDS


mpl.rcParams['font.size']=12                #10 
mpl.rcParams['savefig.dpi']=100           #72 
mpl.rcParams['figure.subplot.bottom']=.1 


stopwords = set(STOPWORDS)
stopwords.add('dtype')
stopwords.add('object')
stopwords.add('prime_genre')
stopwords.add('Name')
stopwords.add('Length')
data = pd.read_csv("../input/AppleStore.csv")

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(data['prime_genre']))

print(wordcloud)
fig = plt.figure(1)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

plt.figure(figsize=(20,10))
sns.jointplot(x='user_rating', y='rating_count_tot',data=df1,kind='hex')
plt.figure(figsize=(20,10))
df['price'].value_counts().plot.bar()
plt.figure(figsize=(20,10))
df['price'].value_counts().plot.barh()
plt.figure(figsize=(20,10))
df['price'].value_counts().plot.area()
plt.figure(figsize=(20,10))
df['price'].value_counts().plot()
plt.figure(figsize=(20,10))
df['price'].value_counts().plot.pie()
plt.figure(figsize=(20,10))
sns.countplot(df['price'],hue='prime_genre',data=df)
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1, 1))
import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

from subprocess import check_output
import wordcloud
from wordcloud import WordCloud, STOPWORDS


mpl.rcParams['font.size']=12                #10 
mpl.rcParams['savefig.dpi']=100           #72 
mpl.rcParams['figure.subplot.bottom']=.1 


stopwords = set(STOPWORDS)
stopwords.add('dtype')
stopwords.add('object')
stopwords.add('prime_genre')
stopwords.add('Name')
stopwords.add('Length')
data = pd.read_csv("../input/appleStore_description.csv")

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(data['app_desc']))

print(wordcloud)
fig = plt.figure(1)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

plt.figure(figsize=(40,20))
sns.pairplot(df.drop("id", axis=1), hue="prime_genre", size=3)