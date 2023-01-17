import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

stopwords = set(STOPWORDS)

# Filtering out some words that are too common, and we want to focus on the
# words that are diffrent between varieties
stopwords.add("wine")
stopwords.add("flavor")
stopwords.add("flavors")
stopwords.add("finish")
stopwords.add("drink")
stopwords.add("now")
stopwords.add("note")
stopwords.add("notes")

df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
df = df[['variety', 'points', 'description']]
df = df[~df.variety.str.contains('Blend|,')]
top_reviewed = df[['variety']].groupby('variety').size().reset_index(name='count')

top_scored = df[['variety', 'points']].groupby('variety').mean()
top_reviewed = top_reviewed.join(top_scored, on='variety')

top_reviewed['points'] = top_reviewed['count'] * (top_reviewed['points'] / 100)

top_reviewed.sort_values('count', ascending=False)[:20].sort_values('count').set_index("variety", drop=True).plot(kind='barh', title='Top 20 Varieties by Review Count')
plt.show()
for number, variety_name in enumerate(top_reviewed.sort_values('count', ascending=False)[:10]['variety']):
    variety = df[df['variety'] == variety_name]
    reviews = ' '.join(' '.join(variety['description']).split())

    image_path = os.path.join('../input/wine-bottle-images/', variety_name + '.jpg')
    bottle_coloring = np.array(Image.open(image_path))
    image_colors = ImageColorGenerator(bottle_coloring)
    
    wc = WordCloud(background_color="white", max_words=2000, stopwords=stopwords, mask=bottle_coloring)
    wc.generate(text=reviews)

    try:
        plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
        plt.axis("off")
        plt.title(str(number + 1) + '. ' + variety_name)
        plt.show()
    except:
        pass