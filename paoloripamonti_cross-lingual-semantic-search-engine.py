!pip uninstall --quiet --yes tensorflow

!pip install --quiet tensorflow-gpu==1.13.1

!pip install --quiet tf-sentencepiece

!pip install --quiet simpleneighbors

!pip install --quiet tensorflow-hub
# tensorflow

import tensorflow as tf

import tensorflow_hub as hub

import tf_sentencepiece  # Not used directly but needed to import TF ops.



# Utility

import os

import pandas as pd

from tqdm import tqdm

from tqdm import trange

import sklearn.metrics.pairwise

from simpleneighbors import SimpleNeighbors
# The 16-language multilingual module is the default but feel free

# to pick others from the list and compare the results.

module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/1'  #@param ['https://tfhub.dev/google/universal-sentence-encoder-multilingual/1', 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/1', 'https://tfhub.dev/google/universal-sentence-encoder-xling-many/1']



# Set up graph.

g = tf.Graph()

with g.as_default():

    text_input = tf.placeholder(dtype=tf.string, shape=[None])

    multiling_embed = hub.Module(module_url)

    embedded_text = multiling_embed(text_input)

    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])

g.finalize()



# Initialize session.

session = tf.Session(graph=g)

session.run(init_op)
corpus_metadata = [

    ('en', 'en-it.txt.zip', 'News-Commentary.en-it.en', 'English'),

    ('it', 'en-it.txt.zip', 'News-Commentary.en-it.it', 'Italian'),

]



language_to_sentences = {}

language_to_news_path = {}

for language_code, zip_file, news_file, language_name in corpus_metadata:

    zip_path = tf.keras.utils.get_file(

        fname=zip_file,

        origin='http://opus.nlpl.eu/download.php?f=News-Commentary/v11/moses/' + zip_file,

        extract=True)

    news_path = os.path.join(os.path.dirname(zip_path), news_file)

    language_to_sentences[language_code] = pd.read_csv(news_path, sep='\t', header=None)[0]

    language_to_news_path[language_code] = news_path



    print('{:,} {} sentences'.format(len(language_to_sentences[language_code]), language_name))
%%time

batch_size = 2048

language_to_embeddings = {}

for language_code, zip_file, news_file, language_name in corpus_metadata:

    print('\nComputing {} embeddings'.format(language_name))

    with tqdm(total=len(language_to_sentences[language_code])) as pbar:

        for batch in pd.read_csv(language_to_news_path[language_code], sep='\t',header=None, chunksize=batch_size):

            language_to_embeddings.setdefault(language_code, []).extend(session.run(embedded_text, feed_dict={text_input: batch[0]}))

            pbar.update(len(batch))
%%time



# Takes about 8 minutes



num_index_trees = 40

language_name_to_index = {}

embedding_dimensions = len(list(language_to_embeddings.values())[0][0])

for language_code, zip_file, news_file, language_name in corpus_metadata:

    print('\nAdding {} embeddings to index'.format(language_name))

    index = SimpleNeighbors(embedding_dimensions, metric='dot')



    for i in trange(len(language_to_sentences[language_code])):

        index.add_one(language_to_sentences[language_code][i], language_to_embeddings[language_code][i])



    print('Building {} index with {} trees...'.format(language_name, num_index_trees))

    index.build(n=num_index_trees)

    language_name_to_index[language_name] = index
%%time



# Takes about 13 minutes



num_index_trees = 60

print('Computing mixed-language index')

combined_index = SimpleNeighbors(embedding_dimensions, metric='dot')

for language_code, zip_file, news_file, language_name in corpus_metadata:

    print('Adding {} embeddings to mixed-language index'.format(language_name))

    for i in trange(len(language_to_sentences[language_code])):

        annotated_sentence = '({}) {}'.format(language_name, language_to_sentences[language_code][i])

        combined_index.add_one(annotated_sentence, language_to_embeddings[language_code][i])



print('Building mixed-language index with {} trees...'.format(num_index_trees))

combined_index.build(n=num_index_trees)
sample_query = 'Il mercato azionario è sceso di quattro punti.'  #@param ["Global warming", "Researchers made a surprising new discovery last week.", "The stock market fell four points.", "Lawmakers will vote on the proposal tomorrow."] {allow-input: true}

index_language = 'Italian'  #@param ["Arabic", "Chinese", "English", "French", "German", "Russian", "Spanish"]

num_results = 10  #@param {type:"slider", min:0, max:100, step:10}



query_embedding = session.run(embedded_text, feed_dict={text_input: [sample_query]})[0]



search_results = language_name_to_index[index_language].nearest(query_embedding, n=num_results)



print('{} sentences similar to: "{}"\n'.format(index_language, sample_query))

search_results