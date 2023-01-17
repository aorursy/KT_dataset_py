import pandas as pd

import nltk

sw = pd.read_csv('../input/Complete Steven Wilson.csv')



# cleaning

sw.loc[sw['lyrics'].isnull(), 'lyrics'] = ''



sw_lyrics = sw['lyrics']
# Word cloud configuration

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



plt.figure(figsize=(20,20))

wc = WordCloud(height=300, width=500, background_color="white", max_words=10000, stopwords=STOPWORDS, max_font_size=50)
# To The Bone

bone = sw[sw['album']=='To The Bone']['lyrics']

wc.generate(' '.join(bone))



plt.imshow(wc.recolor(colormap= 'RdBu'), interpolation='bilinear')

plt.title('To The Bone')

plt.axis('off')
# 4 1/2

four = sw[sw['album']=='4 1/2']['lyrics']

wc.generate(' '.join(four))



plt.imshow(wc.recolor(colormap= 'YlGn'), interpolation='bilinear')

plt.title('4 1/2')

plt.axis('off')
# Hand Cannot Erase

hand = sw[sw['album']=='Hand Cannot Erase']['lyrics']

wc.generate(' '.join(hand))



plt.imshow(wc.recolor(colormap= 'cool'), interpolation='bilinear')

plt.title('Hand Cannot Erase')

plt.axis('off')
# The Raven That Refused to Sing (And Other Stories)

raven = sw[sw['album']=='The Raven That Refused to Sing (And Other Stories)']['lyrics']

wc.generate(' '.join(raven))



plt.imshow(wc.recolor(colormap= 'pink'), interpolation='bilinear')

plt.title('The Raven That Refused to Sing (And Other Stories)')

plt.axis('off')
# Grace for Drowning

grace = sw[sw['album']=='Grace for Drowning']['lyrics']

wc.generate(' '.join(grace))



plt.imshow(wc.recolor(colormap= 'copper'), interpolation='bilinear')

plt.title('Grace for Drowning')

plt.axis('off')
# Insurgentes 

ins = sw[sw['album']=='Insurgentes (2016 Remaster)']['lyrics']

wc.generate(' '.join(ins))



plt.imshow(wc.recolor(colormap= 'autumn'), interpolation='bilinear')

plt.title('Insurgentes')

plt.axis('off')