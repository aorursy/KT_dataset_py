import pandas as pd



from bs4 import BeautifulSoup

import re



import spacy



from sklearn.manifold import TSNE

from sklearn.cluster import KMeans



import seaborn as sns



from collections import Counter



from bokeh.plotting import output_notebook, figure, show

from bokeh.models import ColumnDataSource, Select, CustomJS

from bokeh.layouts import column

from bokeh.transform import factor_cmap, linear_cmap



output_notebook(hide_banner=True)
!pip install pscript
messages = pd.read_csv('../input/ForumMessages.csv')

messages = messages[messages.Message.notna()]

messages['PostDate'] = pd.to_datetime(messages['PostDate'], infer_datetime_format=True)

messages = messages.sort_values('PostDate')

messages.tail()
messages_str = ' |sep| '.join(messages.Message.tolist())



messages_str = re.sub(r'<code>.*?</code>', '', messages_str, flags=re.DOTALL)

messages_str = re.sub('<-', '', messages_str)



messages_str = BeautifulSoup(messages_str, 'lxml').get_text()



messages_str = re.sub(r'http\S+', '', messages_str)

messages_str = re.sub(r'@\S+', '', messages_str)



messages['Message'] = messages_str.split(' |sep| ')



messages.tail()
corpus = messages[messages.Medal == 3].Message.tolist()[-1000:]

corpus[-5:]
nlp = spacy.load('en_core_web_lg', disable=['ner'])
batch = nlp.pipe(corpus)

corpus_tok = []

for doc in batch:

    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and token.has_vector and not token.is_stop]

    tokens_str = ' '.join(tokens)

    if tokens_str != '':

        corpus_tok.append(tokens_str)



corpus_tok[-5:]
batch_tok = nlp.pipe(corpus_tok)

X = []

for doc in batch_tok:

    X.append(doc.vector)
X_emb = TSNE(random_state=0).fit_transform(X)

df = pd.DataFrame(X_emb, columns=['x', 'y'])

df.tail()
sns.scatterplot('x', 'y', data=df, edgecolor='none', alpha=0.5)
model = KMeans(n_clusters=3)

df['Label'] = model.fit_predict(X_emb)

df['Tokens'] = corpus_tok

df.tail()
palette = sns.color_palette(n_colors=3)

sns.scatterplot('x', 'y', data=df, edgecolor='none', alpha=0.5, hue='Label', palette=palette)
cluster0 = ' '.join(df[df.Label == 0].Tokens.tolist())

words0 = Counter(cluster0.split())

words0.most_common(10)
cluster1 = ' '.join(df[df.Label == 1].Tokens.tolist())

words1 = Counter(cluster1.split())

words1.most_common(10)
cluster2 = ' '.join(df[df.Label == 2].Tokens.tolist())

words2 = Counter(cluster2.split())

words2.most_common(10)
s = ColumnDataSource(df)



p = figure(plot_width=600, plot_height=400, toolbar_location=None, tools=['hover'], tooltips='@Tokens')



cmap = linear_cmap('Label', palette=palette.as_hex(), low=df.Label.min(), high=df.Label.max())

p.circle('x', 'y', source=s, color=cmap)



tokens_all = ' '.join(df.Tokens.tolist()).split()

options = sorted(set(tokens_all))

options.insert(0, 'Please choose...')

select = Select(value='Please choose...', options=options)



def callback(s=s, window=None):

    indices = [i for i, x in enumerate(s.data['Tokens']) if cb_obj.value in x]

    s.selected.indices = indices

    s.change.emit()

    

select.js_on_change('value', CustomJS.from_py_func(callback))

    

show(column(select, p))
corpus = messages[(messages.Medal == 1) | (messages.Medal == 2)].Message.tolist()[-1000:]



batch = nlp.pipe(corpus)

corpus_tok = []

for doc in batch:

    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and token.has_vector and not token.is_stop]

    tokens_str = ' '.join(tokens)

    if tokens_str != '':

        corpus_tok.append(tokens_str)



batch_tok = nlp.pipe(corpus_tok)

X = []

for doc in batch_tok:

    X.append(doc.vector)

    

X_emb = TSNE(random_state=0).fit_transform(X)



df = pd.DataFrame(X_emb, columns=['x', 'y'])

df['Tokens'] = corpus_tok



sns.scatterplot('x', 'y', data=df, edgecolor='none', alpha=0.5)
model = KMeans(n_clusters=2)

df['Label'] = model.fit_predict(X_emb)



palette = sns.color_palette(n_colors=2)

sns.scatterplot('x', 'y', data=df, edgecolor='none', alpha=0.5, hue='Label', palette=palette)
cluster0 = ' '.join(df[df.Label == 0].Tokens.tolist())

words0 = Counter(cluster0.split())

words0.most_common(10)
cluster1 = ' '.join(df[df.Label == 1].Tokens.tolist())

words1 = Counter(cluster1.split())

words1.most_common(10)
s = ColumnDataSource(df)



p = figure(plot_width=600, plot_height=400, toolbar_location=None, tools=['hover'], tooltips='@Tokens')



cmap = linear_cmap('Label', palette=palette.as_hex(), low=df.Label.min(), high=df.Label.max())

p.circle('x', 'y', source=s, color=cmap)



tokens_all = ' '.join(df.Tokens.tolist()).split()

options = sorted(set(tokens_all))

options.insert(0, 'Please choose...')

select = Select(value='Please choose...', options=options)



def callback(s=s, window=None):

    indices = [i for i, x in enumerate(s.data['Tokens']) if cb_obj.value in x]

    s.selected.indices = indices

    s.change.emit()

    

select.js_on_change('value', CustomJS.from_py_func(callback))

    

show(column(select, p))
topics = pd.read_csv('../input/ForumTopics.csv').rename(columns={'Title': 'TopicTitle'})

topics.head()
forums = pd.read_csv('../input/Forums.csv').rename(columns={'Title': 'ForumTitle'})

forums.head()
df1 = pd.merge(messages[['ForumTopicId', 'PostDate', 'Medal']], topics[['Id', 'ForumId', 'TopicTitle']], left_on='ForumTopicId', right_on='Id')

df1 = df1.drop(['ForumTopicId', 'Id'], axis=1)

df1.head()
df2 = pd.merge(df1, forums[['Id', 'ForumTitle']], left_on='ForumId', right_on='Id')

df2 = df2.drop(['ForumId', 'Id'], axis=1)

df2.head()
bronze = df2[(df2.Medal == 3) & (df2.PostDate > '2019-01-01 00:00:00')]

bronze_gr = bronze.groupby('ForumTitle').count()

bronze_ind = bronze_gr.sort_values('Medal')[-10:].index.values

bronze = bronze[bronze.ForumTitle.isin(bronze_ind)]



silver_gold = df2[((df2.Medal == 1) | (df2.Medal == 2)) & (df2.PostDate > '2019-01-01 00:00:00')]

silver_gold_gr = silver_gold.groupby('ForumTitle').count()

silver_gold_ind = silver_gold_gr.sort_values('Medal')[-10:].index.values

silver_gold = silver_gold[silver_gold.ForumTitle.isin(silver_gold_ind)]
sns.countplot(y='ForumTitle', data=bronze)
sns.countplot(y='ForumTitle', data=silver_gold)