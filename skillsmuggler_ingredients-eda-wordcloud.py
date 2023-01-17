# Load libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
df = pd.read_json('../input/train.json')
df.head()
'''
Input: dataframe and cuisine
Output: WordCloud image
Processing: Create a list of all ingredients in given cuisine
            Create wordcluod based on the count of each type of ingredient
'''

def word_cloud(df, cuisine):
    
    # Read the whole text.
    # For selected cuisine
    lst = []
    for each in df[df['cuisine'] == cuisine]['ingredients']:
        lst = lst + each
        
    text = ' '.join(lst)
    
    # Create word cloud
    wordcloud = WordCloud(background_color='white', max_words=100, width=2000, height=1000).generate(text)

    # Display the generated image:
    plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
# greek
word_cloud(df, 'greek')
# southern_us
word_cloud(df, 'southern_us')
# indian
word_cloud(df, 'indian')
# jamaican
word_cloud(df, 'jamaican')
# filipino
word_cloud(df, 'filipino')
# spanish
word_cloud(df, 'spanish')
# italian
word_cloud(df, 'italian')
# mexican
word_cloud(df, 'mexican')
# chinese
word_cloud(df, 'chinese')
# british
word_cloud(df, 'british')
# thai
word_cloud(df, 'thai')
# vietnamese
word_cloud(df, 'vietnamese')
# cajun_creole
word_cloud(df, 'cajun_creole')
# brazilian
word_cloud(df, 'brazilian')
# japanese
word_cloud(df, 'japanese')
# irish
word_cloud(df, 'irish')
# korean
word_cloud(df, 'korean')
# moroccan
word_cloud(df, 'moroccan')
# russian
word_cloud(df, 'russian')
