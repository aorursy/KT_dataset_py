# Start with loading all necessary libraries

import numpy as np

import pandas as pd

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import matplotlib.pyplot as plt

%matplotlib inline
#import warnings

#warnings.filterwarnings("ignore")
# Load in the dataframe

df = pd.read_csv("/kaggle/input/trainparadatos/train.csv", index_col=0)
df.head(2).T

#df.info()
df.describe()
import seaborn as sns

plt.figure(figsize=(15,10))

sns.heatmap(df.corr(),vmin=-0.5,cmap='coolwarm',annot=True);
df2=df.dropna(subset=['lat','lng'])

plt.scatter(df2.lat,df2.lng)

plt.title('Scatter plot Lat-Long');

#ax1=df2.plot(kind='scatter',x='lat',y='lng');
from stop_words import get_stop_words

stop_words = get_stop_words('es')



text = " ".join(titulo for titulo in df.titulo.apply(str))

print ("There are {} words in the combination of all review.".format(len(text)))



# Generate a word cloud image

wordcloud = WordCloud(stopwords=stop_words, max_words=500, background_color="white").generate(text)



# Display the generated image:

# the matplotlib way:

plt.figure(figsize=[20,10])

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
#from stop_words import get_stop_words

#stop_words = get_stop_words('es')



text = " ".join(titulo for titulo in df.tipodepropiedad.apply(str))

print ("There are {} words in the combination of all review.".format(len(text)))



# Generate a word cloud image

wordcloud = WordCloud(stopwords=stop_words, max_words=500, background_color="white").generate(text)



# Display the generated image:

# the matplotlib way:

plt.figure(figsize=[20,10])

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
#from stop_words import get_stop_words

#stop_words = get_stop_words('es')



text = " ".join(titulo for titulo in df.descripcion.apply(str))

print ("There are {} words in the combination of all review.".format(len(text)))



# Generate a word cloud image

wordcloud = WordCloud(stopwords=stop_words, max_words=500, background_color="white").generate(text)



# Display the generated image:

# the matplotlib way:

plt.figure(figsize=[20,10])

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()