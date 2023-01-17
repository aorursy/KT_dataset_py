import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import os
from wordcloud import WordCloud, STOPWORDS

df = pd.read_csv('../input/NYC_Dog_Licensing_Dataset.csv',engine = 'python', sep = '\t' , thousands=",")
df.head()
df['BreedName']=df['BreedName'].str.lower()
df['AnimalName']=df['AnimalName'].str.lower()
df['Borough']=df['Borough'].str.lower()
plt.title("Top 5 boroughs with the most dogs",size=20)
sns.countplot(y=df['Borough'],order=df['Borough'].value_counts().index[0:5])
plt.show()
plt.title("Top 5 dog breeds")
sns.countplot(y=df['BreedName'],order=df['BreedName'].value_counts().index[0:10]) 
plt.show()
stopwords=set(['UNKNOWN','NAME NOT PROVIDED','NAME'])
stopwords.union(set(STOPWORDS))
wordcloud = WordCloud( background_color='white',
                        stopwords=stopwords,
                          max_words=300,
                         ).generate(str(df[df['AnimalGender']=='M']['AnimalName']))

fig = plt.figure(figsize=(15, 12))

ax = fig.add_subplot(1,2,1)
ax.imshow(wordcloud)
ax.set_title('Male',size=30)
ax.axis('off')


ax = fig.add_subplot(1,2,2)
wordcloud = WordCloud( background_color='white',
                        stopwords=stopwords,
                          max_words=300,
                         ).generate(str(df[df['AnimalGender']=='F']['AnimalName']))

ax.set_title('Female',size=30)
ax.imshow(wordcloud)
ax.axis('off')

plt.show()
