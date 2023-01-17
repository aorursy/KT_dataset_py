import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import plotly.graph_objects as go



from wordcloud import WordCloud, STOPWORDS



DIR_INPUT = '/kaggle/input/tweet-sentiment-extraction'
train_df = pd.read_csv(DIR_INPUT + '/train.csv')

train_df.head()
train_df['sentiment'].value_counts(normalize=True)
dist = train_df['sentiment'].value_counts()



fig = go.Figure([go.Bar(x=dist.index, y=dist.values)])

fig.update_layout(

    title='Sentiment distribution in train dataset'

)

fig.show()
text = 'TEXT: \n{}\n\nSELECTED_TEXT: \n{}\n\nSENTIMENT: \n{}'



for i in range(3):

    print("============")

    print(text.format(train_df.iloc[i, 1],

                      train_df.iloc[i, 2],

                      train_df.iloc[i, 3]))

    print("============\n\n")

rnd_comments = train_df[train_df['sentiment'] == 'neutral'].sample(n=2000)['text'].values

wc = WordCloud(background_color="black", max_words=2000, stopwords=STOPWORDS)

wc.generate(" ".join(rnd_comments))



plt.figure(figsize=(20,10))

plt.axis("off")

plt.title("Frequent words in neutral comments", fontsize=20)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)

plt.show()
rnd_comments = train_df[train_df['sentiment'] == 'negative'].sample(n=2000)['text'].values

wc = WordCloud(background_color="black", max_words=2000, stopwords=STOPWORDS)

wc.generate(" ".join(rnd_comments))



plt.figure(figsize=(20,10))

plt.axis("off")

plt.title("Frequent words in negative comments", fontsize=20)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)

plt.show()
rnd_comments = train_df[train_df['sentiment'] == 'positive'].sample(n=2000)['text'].values

wc = WordCloud(background_color="black", max_words=2000, stopwords=STOPWORDS)

wc.generate(" ".join(rnd_comments))



plt.figure(figsize=(20,10))

plt.axis("off")

plt.title("Frequent words in positive comments", fontsize=20)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)

plt.show()
test_df = pd.read_csv(DIR_INPUT + '/test.csv')

test_df.head()
test_df['sentiment'].value_counts(normalize=True)
dist = test_df['sentiment'].value_counts()



fig = go.Figure([go.Bar(x=dist.index, y=dist.values)])

fig.update_layout(

    title='Sentiment distribution in test dataset'

)

fig.show()
text = 'TEXT: \n{}\n\nSENTIMENT: \n{}'



for i in range(3):

    print("============")

    print(text.format(test_df.iloc[i, 1],

                      test_df.iloc[i, 2]))

    print("============\n\n")

rnd_comments = test_df[test_df['sentiment'] == 'neutral'].sample(n=1000)['text'].values

wc = WordCloud(background_color="black", max_words=2000, stopwords=STOPWORDS)

wc.generate(" ".join(rnd_comments))



plt.figure(figsize=(20,10))

plt.axis("off")

plt.title("Frequent words in neutral comments", fontsize=20)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)

plt.show()
rnd_comments = test_df[test_df['sentiment'] == 'negative'].sample(n=1000)['text'].values

wc = WordCloud(background_color="black", max_words=2000, stopwords=STOPWORDS)

wc.generate(" ".join(rnd_comments))



plt.figure(figsize=(20,10))

plt.axis("off")

plt.title("Frequent words in negative comments", fontsize=20)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)

plt.show()
rnd_comments = test_df[test_df['sentiment'] == 'positive'].sample(n=1000)['text'].values

wc = WordCloud(background_color="black", max_words=2000, stopwords=STOPWORDS)

wc.generate(" ".join(rnd_comments))



plt.figure(figsize=(20,10))

plt.axis("off")

plt.title("Frequent words in positive comments", fontsize=20)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)

plt.show()