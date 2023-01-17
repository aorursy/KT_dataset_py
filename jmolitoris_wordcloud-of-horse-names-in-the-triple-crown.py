import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from wordcloud import WordCloud, STOPWORDS

from PIL import Image

import seaborn as sns

from os import path, getcwd

import os

from matplotlib import cm

from matplotlib.colors import ListedColormap
df = pd.read_csv("../input/triple-crown-of-horse-races-2005-2019/TripleCrownRaces_2005-2019.csv")
horse_words = " "

stopwords = set(STOPWORDS)

for val in df['Horse']:

    val = str(val)

    tokens = val.split()

    

    for i in range(len(tokens)):

        tokens[i] = tokens[i].lower()

    for words in tokens:

        horse_words = horse_words + words + ' '
brg = cm.get_cmap('gist_heat', 256)

new_colors = brg(np.linspace(0, 1, 256))

new_colors = new_colors[:-50]

newcmp = ListedColormap(new_colors)
print(os.listdir("../input"))
mask = np.array(Image.open('../input/horsejpg/horse.jpg'))



wordcloud = WordCloud(width = 2400, height = 1800, mask = mask, background_color = 'white', colormap = newcmp,

                      contour_width = 0.5, contour_color = 'black')

wordcloud.generate(horse_words)

plt.figure(figsize = (10, 8))

plt.imshow(wordcloud, interpolation = "quadric", aspect = 'auto', origin = 'upper')

plt.axis("off")

plt.tight_layout(pad = 0)

plt.savefig('wordcloud.png', facecolor='w', bbox_inches='tight')