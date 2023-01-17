# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling
%matplotlib inline
df_rough= pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv")
df= df_rough.iloc[:,1:11]       #df.iloc[5:10,3:8] # rows 5-10 and columns 3-8
df.describe()
df.info()
df.isnull().sum()
df['Review Text']=df['Review Text'].astype(str)
df['Review Length']=df['Review Text'].apply(len)
g = sns.FacetGrid(data=df, col='Rating')
g.map(plt.hist, 'Review Length', bins=50)
df.corr()
sns.heatmap(data=df.corr(), annot= True, cbar=False, center= .8)
AgeCat= pd.cut(df['Age'], np.arange(start=0, stop=100, step=10))
print(AgeCat.unique()) 
plt.figure(figsize=(2,10))
df.groupby(['Rating', AgeCat]).size().unstack(0).plot.bar(stacked=True)
df.groupby(['Department Name', pd.cut(df['Age'], np.arange(0,100,10))]).size().unstack(0).plot.bar(stacked=True)
z=df.groupby(by=['Department Name'],as_index=False).count().sort_values(by='Class Name',ascending=False)
print(z.head())
sns.set_style("whitegrid")
ax = sns.barplot(x=z['Department Name'],y=z['Class Name'], data=z)
plt.ylabel("Count")
plt.title("Counts Vs Department Name")
a= df.groupby('Department Name')
a.describe().head()
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
text = " ".join(review for review in df['Review Text'])
print ("There are {} words in the combination of all review.".format(len(text)))
text= re.sub('(\s+)(a|an|and|the|on|in|of|if|is|i|)(\s+)', '', text)  
print ("There are {} words in the combination of all review.".format(len(text)))
# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.add("fabric")

# Generate a word cloud image
wordcloud = WordCloud(background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
import collections
words = text.split()
print(collections.Counter(words))
