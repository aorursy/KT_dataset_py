import pandas as pd

from wordcloud import WordCloud

import matplotlib.pyplot as plt

%matplotlib inline

pop = pd.read_excel('../input/population.xlsx')

pop.head()
pop['percent'] = [(x*100)/sum(pop['2015_total']) for x in pop['2015_total']]
d = dict(zip(pop.name, pop.percent))
plt.figure(figsize = (15,10))

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black')

wordcloud.fit_words(d)

plt.imshow(wordcloud,interpolation = 'bilinear')