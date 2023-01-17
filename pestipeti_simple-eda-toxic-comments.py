import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff



from wordcloud import WordCloud, STOPWORDS
DIR_INPUT = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification'
train_df1 = pd.read_csv(DIR_INPUT + '/jigsaw-toxic-comment-train.csv')

train_df1['src'] = 0

train_df1.head()
train_df2 = pd.read_csv(DIR_INPUT + '/jigsaw-unintended-bias-train.csv')

train_df2['src'] = 1

train_df2.head()
keep_cols = ['id', 'comment_text', 'toxic', 'src']

train_df = train_df1[keep_cols].append(train_df2[keep_cols])

train_df.head()
del train_df1, train_df2
train_df['toxic'] = (train_df['toxic'] > 0.5).astype(np.uint)
print("We have {} english comments in the train datasets.".format(train_df.shape[0]))
train_df['toxic'].value_counts(normalize=True)
train_df.groupby(by=['toxic', 'src']).count()[['id']]
fig = go.Figure([go.Bar(x=['Not-toxic', 'Toxic'], y=train_df.toxic.value_counts())])

fig.update_layout(

    title='Toxic/non-toxic comments distribution in the train dataset'

)

fig.show()
train_df['comment_text_len'] = train_df['comment_text'].apply(lambda x : len(x))

train_df['comment_text_word_cnt'] = train_df['comment_text'].apply(lambda x : len(x.split(' ')))
fig = px.histogram(train_df, x='comment_text_len', color='toxic', nbins=200)

fig.show(renderer="kaggle")
fig = px.histogram(train_df[train_df['src'] == 0],

                   x='comment_text_len',

                   color='toxic',

                   nbins=200,

                   title='Text length - Source: Jigsaw toxic comment (train)')

fig.show(renderer="kaggle")
fig = px.histogram(train_df[train_df['src'] == 1],

                   x='comment_text_len',

                   color='toxic',

                   nbins=200,

                   title='Text length - Source: Jigsaw unintended bias (train)')

fig.show(renderer="kaggle")
fig = px.histogram(train_df[train_df['src'] == 0],

                   x='comment_text_word_cnt',

                   color='toxic',

                   nbins=200,

                   title='Word count - Source: Jigsaw toxic comment (train)')

fig.show(renderer="kaggle")
fig = px.histogram(train_df[train_df['src'] == 1],

                   x='comment_text_word_cnt',

                   color='toxic',

                   nbins=200,

                   title='Word count - Source: Jigsaw toxic comment (train)')

fig.show(renderer="kaggle")
valid_df = pd.read_csv(DIR_INPUT + '/validation.csv')

valid_df.head()
per_lang = valid_df['lang'].value_counts()

fig = go.Figure([go.Bar(x=per_lang.index, y=per_lang.values)])

fig.update_layout(

    title='Language distribution in the validation dataset'

)

fig.show()
valid_df.toxic.value_counts(normalize=True)
fig = go.Figure([go.Bar(x=['Not-toxic', 'Toxic'], y=valid_df.toxic.value_counts())])

fig.update_layout(

    title='Language distribution in the validation dataset'

)

fig.show()
per_lang = valid_df.groupby(by=['lang', 'toxic']).count()[['id']]

per_lang
data = []



for lang in valid_df['lang'].unique():

    y = per_lang[per_lang.index.get_level_values('lang') == lang].values.flatten()

    data.append(go.Bar(name=lang, x=['Non-toxic', 'Toxic'], y=y))



fig = go.Figure(data=data)

fig.update_layout(

    title='Language distribution in the validation dataset',

    barmode='group'

)

fig.show()
test_df = pd.read_csv(DIR_INPUT + '/test.csv')

test_df.head()
test_df['lang'].value_counts()
per_lang = test_df['lang'].value_counts()

fig = go.Figure([go.Bar(x=per_lang.index, y=per_lang.values)])

fig.update_layout(

    title='Language distribution in the test dataset',

)

fig.show()
toxic_samples = train_df[train_df['toxic'] == 1].sample(n=5)['comment_text']



for toxic in toxic_samples.values:

    print("")

    print("==============================")

    print(toxic)

    print("==============================")

    print("")
rnd_comments = train_df.sample(n=2500)['comment_text'].values

wc = WordCloud(background_color="black", max_words=2000, stopwords=STOPWORDS.update(['Trump', 'people', 'one', 'will']))

wc.generate(" ".join(rnd_comments))



plt.figure(figsize=(20,10))

plt.axis("off")

plt.title("Random words", fontsize=20)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)

plt.show()
rnd_comments = train_df[train_df['toxic'] == 0].sample(n=10000)['comment_text'].values

wc = WordCloud(background_color="black", max_words=2000, stopwords=STOPWORDS.update(['Trump', 'people', 'one', 'will']))

wc.generate(" ".join(rnd_comments))



plt.figure(figsize=(20,10))

plt.axis("off")

plt.title("Frequent words in non-toxic comments", fontsize=20)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)

plt.show()
rnd_comments = train_df[train_df['toxic'] == 1].sample(n=10000)['comment_text'].values

wc = WordCloud(background_color="black", max_words=2000, stopwords=STOPWORDS.update(['Trump', 'people', 'one', 'will']))

wc.generate(" ".join(rnd_comments))



plt.figure(figsize=(20,10))

plt.axis("off")

plt.title("Frequent words in toxic comments", fontsize=20)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)

plt.show()