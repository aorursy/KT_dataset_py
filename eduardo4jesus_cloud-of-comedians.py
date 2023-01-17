import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
quotes = pd.read_csv('../input/scomedy-quotes.csv')
quotes.head()
comedians = quotes.groupby('comedian').count()
comedians['freq'] = comedians['text']/comedians['text'].sum()
comedians = comedians.sort_values(by=['freq'], ascending=False)
comedians = comedians.drop(columns=['tags', 'text'])
comedians.head()
comedians.head(50).plot.bar(figsize=(15,10))
plt.xticks(rotation=90);
plt.xlabel("Comedian");
plt.ylabel("Percentage of Quotes");
wordcloud = WordCloud(width=800, height=600, max_font_size=80, background_color="white", colormap='Set1')
wordcloud = wordcloud.generate_from_frequencies(comedians.to_dict()['freq'])
wordcloud.to_file("comedians_cloud.png");
plt.figure(figsize=[14,21])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()