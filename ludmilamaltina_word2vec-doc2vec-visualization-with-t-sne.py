import numpy as np

import pandas as pd

import nltk

import re

import gensim

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.manifold import TSNE

from bokeh.models import ColumnDataSource, LabelSet, HoverTool

from bokeh.plotting import figure

from bokeh.io import show, output_notebook
train = pd.read_csv('..//input/train.csv')

train.shape
stopWords = set(nltk.corpus.stopwords.words('english'))  # стоп-слова для английского языка из nltk

tokenized = []



for i, line in enumerate(train['comment_text']):

    

    if i % 10000 == 0:  # отслеживаем прогресс

        print(i)

        

    add = gensim.utils.simple_preprocess(line)



    if len(add)!=0:

        comment = []

        for w in add:

            if w not in stopWords:

                comment.append(w)



    if len(comment)!=0:

        tokenized.append(comment)

    else:

        tokenized.append([])
print('comment_0:', train['comment_text'][0])

print('preprocessed_0:', tokenized[0])
train['comments_words'] = tokenized  # записываем предобработанные комментарии в столбец 'comments_words'

train['comments'] = train['comments_words'].map(lambda x: ' '.join(x))  # записываем предобработанные комментарии без деления на слова в столбец 'comments'

train['comments'][0]  # посмотрим на предобработанный 0-ой комментарий без деления на слова
model_w2v = gensim.models.Word2Vec(tokenized, window=5, min_count=5, size=50, seed=0)

model_w2v.train(sentences=tokenized, total_examples=len(tokenized), epochs=30)
def show_similar(w):

    print('Слова, близкие к слову {}:'.format(w))

    similar_words = model_w2v.wv.most_similar(positive=[w])

    for w in similar_words:

        print(w)

    print('\n')
for w in ['person', 'opinion', 'source', 'wiki', 'search']:

    show_similar(w)
def words_for_tsne(tokenized):

    '''функция, выбирающая 1000 наиболее частых слов'''

    count_vectorizer = CountVectorizer(max_features=1000)

    count_vec = count_vectorizer.fit_transform(tokenized)

    words = count_vectorizer.vocabulary_.keys()

    return words
ws = words_for_tsne(train['comments'])

# ws  # можно посмотреть выбранные слова
output_notebook()

words_top_vec = model_w2v.wv[ws]

tsne = TSNE(n_components=2, random_state=0)

words_top_tsne = tsne.fit_transform(words_top_vec)

p = figure(tools="pan,wheel_zoom,reset,save",

           toolbar_location="above",

           title="Word2Vec t-SNE for most common words")



source = ColumnDataSource(data=dict(x1=words_top_tsne[:,0],

                                    x2=words_top_tsne[:,1],

                                    names=list(ws)))



p.scatter(x="x1", y="x2", size=8, source=source)



labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,

                  text_font_size="8pt", text_color="#555555",

                  source=source, text_align='center')

p.add_layout(labels)

show(p)
from gensim.models.doc2vec import Doc2Vec, TaggedDocument





documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenized)]

model_d2v = Doc2Vec(vector_size=50, min_count=5, seed=0)
model_d2v.build_vocab(documents)
model_d2v.train(documents, total_examples=len(documents), epochs=50)
documents[0]
similar_doc = model_d2v.docvecs.most_similar([0])

similar_doc
for ind, similarity in similar_doc:

    print(similarity)  # насколько этот документ похож на 0-ой документ

    print(train['comment_text'][ind])  # текст документа

    print(train['toxic'][ind])  # токсичность документа

    print('-' * 150)
train['toxic_score'] = (train['toxic'] + train['severe_toxic'] + train['obscene'] + train['threat'] + train['insult'] + train['identity_hate'] > 0).astype(int)

train['toxic_score'].value_counts()
nrows = 10000

tsne = TSNE(n_components=2, random_state=0)

tsne_d2v = tsne.fit_transform(model_d2v.docvecs.vectors_docs[:nrows])

tsne_d2v_df = pd.DataFrame(data=tsne_d2v, columns=["x", "y"])  # запишем результаты TSNE преобразования в датафрейм

tsne_d2v_df["toxic_score"] = train["toxic_score"].values[:nrows]  # добавим столбец, отвечающий за наличие хотя бы одного типа токсичности
tsne_d2v_df.head()  # посмотрим первые строки полученного датафрейма
tsne_d2v_df.shape  # размерность полученного датафрейма
output_notebook()



docs_top_tsne = tsne.fit_transform(tsne_d2v_df)

p = figure(tools="pan,wheel_zoom,reset,save",

           toolbar_location="above",

           title="Doc2Vec t-SNE for first 10000 documents")



colormap = np.array(["green", "red"])



source = ColumnDataSource(data=dict(x1=tsne_d2v_df["x"],

                                    x2=tsne_d2v_df["y"],

                                    color=colormap[tsne_d2v_df["toxic_score"]],

                                    toxic_score=tsne_d2v_df["toxic_score"]

                                   ))



p.scatter(x="x1", y="x2", color='color', legend='toxic_score', alpha=0.5, size=8, source=source)

hover = p.select(dict(type = HoverTool))

hover.tooltips = {"toxic_score":"@toxic_score"}



show(p)