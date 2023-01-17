import pandas as pd

data=pd.read_csv("../input/youtube-new/USvideos.csv")

data.head()
from matplotlib import pyplot as plt

from wordcloud import WordCloud, STOPWORDS

string=str(data.title)

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=1000,

                      height=1000).generate(string)

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis("off")

plt.show()