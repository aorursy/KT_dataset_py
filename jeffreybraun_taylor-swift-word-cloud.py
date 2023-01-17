import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image

STOPWORDS.add('verse')
STOPWORDS.add('chorus')
STOPWORDS.add('Verse')
STOPWORDS.add('chorus')
STOPWORDS.add('bridge')

def clean_lyric(lyric):
    if '[' in lyric:
        return ''
    else:
        return lyric
    
df = pd.read_csv('/kaggle/input/taylorswiftlyrics/final_taylor_swift_lyrics.tsv', sep= '\t')
df['lyric_clean'] = df['lyric'].apply(lambda x: clean_lyric(x))

mask = np.array(Image.open('/kaggle/input/taylor-swift-mask-2/tsiwft.png'))

wordcloud = WordCloud( background_color='white',stopwords = STOPWORDS,
                       mask = mask,
                        width=1000,
                        height=4000).generate(" ".join(df['lyric_clean']))
plt.figure(figsize=(20,20))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Taylor Swift Songs Word Cloud',fontsize=10)
plt.show()
