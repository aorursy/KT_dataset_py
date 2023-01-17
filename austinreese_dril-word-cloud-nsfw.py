import pandas as pd

import matplotlib.pyplot as plt



from wordcloud import WordCloud, STOPWORDS



tweets = pd.read_csv("../input/dril-tweets/driltweets.csv")
tweetText = tweets.content.tolist()

words = []

STOPWORDS.add("im")

STOPWORDS.add("will")

for t in tweetText:

    for w in t.split():

        if w.strip() not in STOPWORDS:

            words.append(w.strip())

            

unique_string = (" ").join(words)

wordcloud = WordCloud(width = 1500, height = 750).generate(unique_string)

plt.figure(figsize=(24,12))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()

plt.close()