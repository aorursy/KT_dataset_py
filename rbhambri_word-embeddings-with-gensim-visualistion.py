import pandas as pd

df = pd.read_csv('../input/train.csv')

corpus_text = '\n'.join(df[:50000]['comment_text'])

sentences = corpus_text.split('\n')

sentences = [line.lower().split(' ') for line in sentences]
def clean(s):

    return [w.strip(',."!?:;()\'') for w in s]

sentences = [clean(s) for s in sentences if len(s) > 0]
from gensim.models import Word2Vec



model = Word2Vec(sentences, size=100, window=5, min_count=3, workers=4)
vectors = model.wv

# del model
vectors['good']
print(vectors.similarity('you', 'your'))

print(vectors.similarity('you', 'internet'))
vectors.most_similar('kill')
len(model.wv.vocab)
# build a list of the terms, integer indices,

# and term counts from the food2vec model vocabulary

ordered_vocab = [(term, voc.index, voc.count) for term, voc in model.wv.vocab.items()]



# sort by the term counts, so the most common terms appear first

ordered_vocab = sorted(ordered_vocab, key=lambda k: -k[2])



# unzip the terms, integer indices, and counts into separate lists

ordered_terms, term_indices, term_counts = zip(*ordered_vocab)

# print(ordered_terms)

# create a DataFrame with the food2vec vectors as data,

# and the terms as row labels

word_vectors = pd.DataFrame(model.wv.syn0norm[term_indices, :], index=ordered_terms)



word_vectors
def get_related_terms(token, topn=10):

    """

    look up the topn most similar terms to token

    and print them as a formatted list

    """



    for word, similarity in model.most_similar(positive=[token], topn=topn):

        print (word, round(similarity, 3))
get_related_terms(u'killed')
get_related_terms(u'japanese')
get_related_terms(u'asshole')
get_related_terms(u'discussion')
get_related_terms(u'wikipedia')
get_related_terms(u'please')
get_related_terms(u'vandalism')
get_related_terms(u'media')
get_related_terms(u'language')
get_related_terms(u'perhaps')
get_related_terms(u'sex')
get_related_terms(u'conflict')
get_related_terms(u'bastard')
get_related_terms(u'jewish')
get_related_terms(u'introduction')
def word_algebra(add=[], subtract=[], topn=1):

    """

    combine the vectors associated with the words provided

    in add= and subtract=, look up the topn most similar

    terms to the combined vector, and print the result(s)

    """

    answers = model.most_similar(positive=add, negative=subtract, topn=topn)

    

    for term, similarity in answers:

        print(term)
word_algebra(add=[u'i', u'will'])
word_algebra(add=[u'you', u'will'])
word_algebra(add=[u'i', u'am'])
word_algebra(add=[u'mother', u'fuck'])
word_algebra(add=[ u'fuck', 'you'])
from sklearn.manifold import TSNE
tsne_input = word_vectors

tsne_input = tsne_input.head(5000)
tsne_input
tsne = TSNE()

tsne_vectors = tsne.fit_transform(tsne_input.values)
tsne_vectors = pd.DataFrame(tsne_vectors,

                            index=pd.Index(tsne_input.index),

                            columns=[u'x_coord', u'y_coord'])



tsne_vectors.head()
tsne_vectors[u'word'] = tsne_vectors.index
tsne_vectors.head()
from bokeh.plotting import figure, show, output_notebook

from bokeh.models import HoverTool, ColumnDataSource, value



output_notebook()
# add our DataFrame as a ColumnDataSource for Bokeh

plot_data = ColumnDataSource(tsne_vectors)



# create the plot and configure the

# title, dimensions, and tools

tsne_plot = figure(title=u't-SNE Word Embeddings',

                   plot_width = 800,

                   plot_height = 800,

                   tools= (u'pan, wheel_zoom, box_zoom,'

                           u'box_select, reset'),

                   active_scroll=u'wheel_zoom')



# add a hover tool to display words on roll-over

tsne_plot.add_tools( HoverTool(tooltips = u'@word') )



# draw the words as circles on the plot

tsne_plot.circle(u'x_coord', u'y_coord', source=plot_data,

                 color=u'blue', line_alpha=0.2, fill_alpha=0.1,

                 size=10, hover_line_color=u'black')



# configure visual elements of the plot

tsne_plot.title.text_font_size = value(u'16pt')

tsne_plot.xaxis.visible = False

tsne_plot.yaxis.visible = False

tsne_plot.grid.grid_line_color = None

tsne_plot.outline_line_color = None



# engage!

show(tsne_plot);