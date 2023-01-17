PREPROCESS_AND_TRAIN = True
import numpy as np 

import pandas as pd 

import os

from glob import glob

from tqdm.auto import tqdm

import json

import re

from unidecode import unidecode

from collections import Counter

import pickle

import gensim 

import logging

import sys



# We create a logger handler so that gensim outputs its logging to stdout, which will then appear in the notebook cells.

# This is nice to have to be able to see the progress of the training process

logger = logging.getLogger()



if logger.hasHandlers():

        logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)

console_handler.setLevel(logging.INFO)

console_handler.setFormatter(logging.Formatter('%(asctime)s : %(levelname)s : %(message)s'))

logger.addHandler(console_handler)

logger.info('hi')
# Regex used for cleaning and tokenisation

space = re.compile('\s+')

reference = re.compile(r'[\(\[]\d+(, ?\d+)?[\)\]]')

links = re.compile(r'https?://\S+')

sentence  = re.compile(r'(\S{3,})[.!?]\s')

hyphen_1 = re.compile(r'([A-Z0-9])\-(.)')

hyphen_2 = re.compile(r'(.)\-([A-Z0-9])')



license_phrases = [r'\(which was not peer-reviewed\)',

                    'The copyright holder for this preprint',

                    'It is made available under a is the author/funder',

                    'who has granted medRxiv a license to display the preprint in perpetuity',

                    'medRxiv preprint',

                    r'(CC(?:-BY)?(?:-NC)?(?:-ND)? \d\.\d (?:International license)?)',

                    'Submitted manuscript.', 

                   'Not peer reviewed.']



license_phrases = re.compile('|'.join(license_phrases), re.I)

PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE) # from gensim - removes digits - keeps only other alpha numeric and tokenises on everything else

PAT_ALL = re.compile(r'((\d+(,\d+)*(\.\d+)?)+|([\w_])+)', re.UNICODE) # Includes digits - tokenises on space and non alphanumeric characters (except _)



def clean_text(text):

    text = text.replace('\t', ' ').replace('\n', ' ')

    text = sentence.sub(r'\1 _SENT_ ', text)

    text = text.replace('doi:', ' http://a.website.com/')

    text = unidecode(text) # converts any non-unicode characters to a unicode equivalent

    text = hyphen_1.sub(r'\1\2', text)

    text = hyphen_2.sub(r'\1\2', text)

    text = links.sub(' ', text)

    text = license_phrases.sub(' ', text)

    text = reference.sub(' ', text)

    text = space.sub(' ', text)



    return text.strip()



def fetch_tokens(text, reg_pattern):

    for match in reg_pattern.finditer(text):

        yield match.group()



def tokenise(text, remove_stop=False, lowercase=False, include_digits=True):

    text = clean_text(text)

    

    if lowercase:

        text = text.lower()

    

    if include_digits:

        tokens = list(fetch_tokens(text, reg_pattern=PAT_ALL))

    else:

        tokens = list(fetch_tokens(text, reg_pattern=PAT_ALPHABETIC))

            

    if remove_stop:

        return ' '.join([x for x in tokens if x.lower() not in stopWords])

    else:

        return ' '.join(tokens)
test_text = """A few test sentences for the tokeniser and cleaner.

With a web link http://something.co.uk/ and including a large number 100,000.50 and small ones 5.5.

But not including footnote references (10, 34) and [23].

Remove hyphens from Covid-19 and MERS-SARS to keep these as one token;

But add spaces for hyphenated-lowercase-words to keep the individual words as tokens."""
print(tokenise(test_text))
LICENSE_TYPES = ['comm_use_subset',

                 'noncomm_use_subset',

                 'pmc_custom_license',

                 'biorxiv_medrxiv']



# DATA_PATH = '/kaggle/input/CORD-19-research-challenge/2020-03-13'

DATA_PATH = '/kaggle/input/CORD-19-research-challenge'

PRE_PROCESSED_PATH = '/kaggle/input/paragraph-search-using-word2vec'
def iter_info():

    """

    Custom iterator to extract paragraphs from body text and reference text, alongside some of the document's meta data 

    """

    for licence in LICENSE_TYPES:

        for root, dirs, files in os.walk(os.path.join(DATA_PATH, licence)):

            for name in files: 

                fname = os.path.join(root, name)

                document = json.load(open(fname))

            

                refs = {}

                for para_num, para in enumerate(document['body_text']):

                    paragraph = {'title': document['metadata']['title'],

                                'id': document['paper_id'],

                                'contains_refs': False,

                                'section': para['section'],

                                'para_num': para_num,

                                'para_text': para['text'],

                                'licence': licence}



                    # check for references within this paragraph, but only add text once per paragraph

                    for ref in para['ref_spans']:

                        if ref['ref_id'] in document['ref_entries'].keys() and ref['ref_id'] not in refs.keys():

                            refs[ref['ref_id']] = True

                            paragraph['contains_refs'] = True 

                            paragraph['para_text'] += ' _STARTREF_ ' + document['ref_entries'][ref['ref_id']]['text'] + ' _ENDREF_ '

                    yield paragraph



                # Add any references text that has been missed

                for ref_id, ref_element in document['ref_entries'].items():

                    if ref_id not in refs.keys():

                        paragraph = {'title': document['metadata']['title'],

                                    'id': document['paper_id'],

                                    'contains_refs': False,

                                    'para_num': ref_id,

                                    'para_text': ref_element['text'],

                                    'section': 'References',

                                    'licence': licence}

                        yield paragraph
if PREPROCESS_AND_TRAIN:

    all_data = []

    texts = iter_info()

    for cc, t in tqdm(enumerate(texts)):

        t['tokenised'] = tokenise(t['para_text'])

        all_data.append(t)

#         if cc == 50000:

#             break

    # Convert to dataframe and save

    all_data = pd.DataFrame(all_data)

    all_data.to_pickle('CORD_19_all_papers.pkl')

else:

    all_data = pd.read_pickle(os.path.join(PRE_PROCESSED_PATH, 'CORD_19_all_papers.pkl'))
all_data.head()
class Paragraphs(object):

    def __init__(self, corpus_df, min_words_in_sentence=10):

        self.corpus_df = corpus_df

        self.min_words = min_words_in_sentence



    def __iter__(self): 

        for tot_paras, para in enumerate(self.corpus_df.tokenised, 1):

            if (tot_paras % 100000 == 0):

                logger.info("Read {0} paras".format(tot_paras))

            word_list = para.replace('\n', ' ').split(' ')

            if len(word_list) >= self.min_words:

                yield word_list
if PREPROCESS_AND_TRAIN:

    documents = Paragraphs(all_data, min_words_in_sentence=10)



    vocab = Counter()

    for line in tqdm(documents):

        vocab.update(line)



    # Save these word frequencies

    vocab = dict(vocab)

    pickle.dump(vocab, open('covid_vocab_frequencies.pkl', 'wb'))

else:

    vocab = pickle.load(open(os.path.join(PRE_PROCESSED_PATH, 'covid_vocab_frequencies.pkl'), 'rb'))
vocab['COVID19']
print(f'{len(vocab)} unique tokens in total.')

print(f'Most common words are: {Counter(vocab).most_common(10)}')

print(f'COVID19 mentioned {vocab["COVID19"]} times.')
EMBEDDING_DIMS = 100
if PREPROCESS_AND_TRAIN:

    # build vocabulary within Gensim

    model = gensim.models.Word2Vec(

        size=EMBEDDING_DIMS,

        window=6,

        min_count=5)



    model.build_vocab(documents)

    print(f'Number of paragraphs in corpus: {model.corpus_count}')
if PREPROCESS_AND_TRAIN:

    model.train(sentences=documents, 

            epochs=5, 

            total_examples=model.corpus_count, 

            total_words=model.corpus_total_words)

    model.save('covid_w2v')    

else:

    model = gensim.models.Word2Vec.load(os.path.join(PRE_PROCESSED_PATH, 'covid_w2v'))
def fetch_similar(positive=[], negative=[], k=10):

    label = ';'.join(positive)

    if negative:

        label += ' - ' + ';'.join(negative)

    return pd.DataFrame([(x, y, vocab[x]) for x, y in model.wv.most_similar(positive, negative, topn=k)], 

                        columns=[label, 'relevance', 'frequency'])
model.wv.most_similar(['COVID19'])
fetch_similar(['coronavirus'], )
from ipywidgets import widgets, interact, Layout

def view_similar(positive='Beijing,Paris', negative='China', top_k=10):

    if negative=='':

        negative = []

    else:

        negative = negative.split(',')

    return fetch_similar(positive=positive.split(','),

                        negative=negative,

                        k=top_k)

interact(view_similar);
interact(view_similar, positive='COVID19,runny,nose,sore,throat,sneezing,rhinorrhea', negative='common,cold');
# interact(view_similar, positive='flu,runny,nose,sore,throat,sneezing,rhinorrhea', negative='common,cold');
def fetch_vector(words):

    tokens = tokenise(words).split()



    if len(tokens) > 0:

        myvectors = np.zeros((1, EMBEDDING_DIMS), dtype=np.float32)

        for word in tokens:

            found = False

            if word not in model.wv.vocab:

                if word.title() in model.wv.vocab:

                    word = word.title()

                    found = True

                elif word.lower() in model.wv.vocab:

                    word = word.lower()

                    found = True

                elif word.upper() in model.wv.vocab:

                    word = word.upper()

                    found = True

            else:

                found = True



            if found:

                wordvec = model.wv.get_vector(word).astype(np.float32) / np.log(5 + vocab[word])



        

        myvectors[0, :] = wordvec



        # normalize

        myvectors = myvectors / np.linalg.norm(myvectors, axis=1, keepdims=True)

        return myvectors

    

def normalise(vec):

    return vec / np.linalg.norm(vec, axis=0, keepdims=True)



def calc_vector(tokens):

    # calculate vectors for document

    # 0 = sentences

    # 1 = paragraphs

    # 2 = document

    sentence_vecs = []

    para_vecs = []

    para_sents = []

    vec = np.zeros((3, EMBEDDING_DIMS), dtype=np.float32)

    word_count = 0

    

    for word in tokens:

        found = False

        if word == '_SENT_':

            sentence_vecs.append(normalise(vec[0]))

            vec[0] = np.zeros((EMBEDDING_DIMS,), dtype=np.float32)

        elif word == '_PARA_':

            sentence_vecs.append(normalise(vec[0]))

            vec[0] = np.zeros((EMBEDDING_DIMS,), dtype=np.float32)

            para_vecs.append(normalise(vec[1]))

            para_sents.append(sentence_vecs)

            sentence_vecs = []

            vec[1] = np.zeros((EMBEDDING_DIMS,), dtype=np.float32)

    

        elif word not in model.wv.vocab:

            if word.title() in model.wv.vocab:

                word = word.title()

                found = True

            elif word.lower() in model.wv.vocab:

                word = word.lower()

                found = True

            elif word.upper() in model.wv.vocab:

                word = word.upper()

                found = True

        else:

            found = True



        if found:

            wordvec = model.wv.get_vector(word).astype(np.float32) / np.log(5 + vocab[word])

#             word_count += 1

            vec[0] += wordvec

            vec[1] += wordvec

            vec[2] += wordvec

            

    # add last sentece / para

#     if word_count > 0:

    sentence_vecs.append(normalise(vec[0]))

    para_vecs.append(normalise(vec[1]))

    para_sents.append(sentence_vecs)

    return {'sentences': para_sents, 

            'paragraphs': para_vecs,

            'document': normalise(vec[2])}
def fetch_vectors_from_dataframe(df):

    results = []

    doc_id = df.iloc[0].id

    para_ids = []

    doc_tokens = []

    

    for row in tqdm(df.itertuples()):

        if row.id != doc_id and len(doc_tokens) > 0:

            # new document

            # fetch vectors for previous document

            tokens = ' _PARA_ '.join(doc_tokens).split()

            results.append({**calc_vector(tokens), 'id': doc_id, 'para_ids': para_ids})

            

            # reset

            doc_id = row.id

            doc_tokens = []

            para_ids = []

            

        # append paragraphs

        doc_tokens.append(row.tokenised)

        para_ids.append(row[0])

        

    tokens = ' _PARA_ '.join(doc_tokens).split()

    results.append({**calc_vector(tokens), 'id': doc_id, 'para_ids': para_ids})

    return results
# paragraph_vectors = fetch_vectors_from_dataframe(all_data.iloc[:30000][['id', 'tokenised']])
!pip install faiss-cpu
import faiss

from faiss import write_index
index_cosine = faiss.IndexFlatIP(EMBEDDING_DIMS*3)              # Create an inner product index to allow cosine distance searches, and specify that the input vectors will have 100 dimensions
# Vector for each sentence is going to consist of concatenation of 

# sentence vec, para vec, document vec
def add_vectors_to_faiss_from_dataframe(df, index_cosine):

    results = []

    doc_id = df.iloc[0].id

    para_ids = []

    doc_tokens = []

    master_index = []

    sentence_index = []

    

    for row in tqdm(df.itertuples()):

        if row.id != doc_id and len(doc_tokens) > 0:

            # new document

            # fetch vectors for previous document

            tokens = ' _PARA_ '.join(doc_tokens).split()

            x = {**calc_vector(tokens), 'id': doc_id, 'para_ids': para_ids}

            sentence_lengths = [len(y) for y in x['sentences']]

            tmp = [[p]*s for p, s in zip(x['para_ids'], sentence_lengths)]

            tmp_2 = [list(range(s)) for s in sentence_lengths]

            master_index += [x for y in tmp for x in y]

            sentence_index += [x for y in tmp_2 for x in y]



            para = np.vstack(np.repeat(np.expand_dims(y, 0), s, axis=0) for y, s in zip (x['paragraphs'], sentence_lengths))

            doc = np.repeat(np.expand_dims(x['document'], 0), sum(sentence_lengths), axis=0)



            sent = np.vstack(x['sentences'])

            assert sent.shape == doc.shape

            assert doc.shape == para.shape

            index_cosine.add(np.concatenate([sent, para, doc], axis=1))             # Add our document vectors to the index

            

            

            # reset

            doc_id = row.id

            doc_tokens = []

            para_ids = []

            

        # append paragraphs

        doc_tokens.append(row.tokenised)

        para_ids.append(row[0])

        

    tokens = ' _PARA_ '.join(doc_tokens).split()

    x = {**calc_vector(tokens), 'id': doc_id, 'para_ids': para_ids}

    sentence_lengths = [len(y) for y in x['sentences']]

    tmp = [[p]*s for p, s in zip(x['para_ids'], sentence_lengths)]

    tmp_2 = [list(range(s)) for s in sentence_lengths]

    master_index += [x for y in tmp for x in y]

    sentence_index += [x for y in tmp_2 for x in y]



    para = np.vstack(np.repeat(np.expand_dims(y, 0), s, axis=0) for y, s in zip (x['paragraphs'], sentence_lengths))

    doc = np.repeat(np.expand_dims(x['document'], 0), sum(sentence_lengths), axis=0)



    sent = np.vstack(x['sentences'])

    assert sent.shape == doc.shape

    assert doc.shape == para.shape

    index_cosine.add(np.concatenate([sent, para, doc], axis=1))             # Add our document vectors to the index

    return master_index, sentence_index
# if PREPROCESS_AND_TRAIN:

#     master_index, sentence_index = add_vectors_to_faiss_from_dataframe(all_data.loc[:, ['id', 'tokenised']], index_cosine)

#     pickle.dump(master_index, open('master_index.pkl', 'wb'))

#     pickle.dump(sentence_index, open('sentence_index.pkl', 'wb'))

#     write_index(index_cosine, 'index_cosine.faiss')
# pd.options.display.max_colwidth=1000



# def search_db(words, k=10, min_length=70):

#     vec = fetch_vector(words)

#     D, I = index_cosine.search(np.expand_dims(np.repeat(vec, 3), 0), k+200)

#     tmp = all_data.loc[[master_index[t] for t in I[0]], ['title', 'para_num', 'para_text']].copy()

#     tmp['relevance'] = D[0]

#     tmp = tmp.loc[tmp.para_text.str.len()>min_length, ['relevance', 'para_text', 'title', 'para_num', ]].iloc[:k]

#     return tmp



# interact(search_db, words=widgets.Text(

#     value='COVID19 symptoms',

#     placeholder='Enter search phrase',

#     description='Search phrase:',

#     disabled=False,

#     layout=Layout(width='80%', height='80px')

# ));
# We can use the pandas shift function to find the preceding and following paragraphs

# all_data['preceding_para'] = np.where((all_data.id.shift(1, fill_value='') == all_data.id), all_data.para_text.shift(1, fill_value=''), '')

# all_data['following_para'] = np.where((all_data.id.shift(-1, fill_value='') == all_data.id), all_data.para_text.shift(-1, fill_value=''), '')



# # Note that we add some html formatting to the cell to make the indexed paragraph stand out

# all_data['para_context'] = [f'<p>{x}</p><p style="color:red;">{y}</p><p>{z}</p>'.replace('\n', '</br>') for x, y, z in zip(all_data.preceding_para, 

#                                                                 all_data.para_text.fillna(''),

#                                                                 all_data.following_para)]

# # now delete the other preceding and following para fields

# del all_data['preceding_para']

# del all_data['following_para']
# all_sources = pd.read_csv(os.path.join(DATA_PATH, 'metadata.csv'))
# Drop the entries that don't have full text

# all_sources = all_sources.groupby('sha').first()

# all_sources.head()
# all_data.set_index('id', inplace=True)



# all_data['doi'] = all_sources['doi']

# all_data['publish_year'] = all_sources['publish_time'].fillna('0').apply(lambda x: str(x)[:4])



# all_data.reset_index(inplace=True)
# from IPython.display import display, HTML
# def search_db(words, k=10, min_length=70, showmore=1000, covid_only=True, date_range=[1957,2020]):

#     vec = fetch_vector(words)

#     D, I = index_cosine.search(np.expand_dims(np.repeat(vec,3), axis=0), k+5000)

#     tmp = all_data.loc[[master_index[t] for t in I[0]], ['title', 'para_context', 'para_text', 'tokenised', 'publish_year', 'doi']].copy()

#     tmp['relevance'] = np.round(D[0], 2)

#     tmp['sentence_number'] = [sentence_index[t] for t in I[0]] 

#     # identify the most relevant sentences

#     tmp['relevant_sentence'] = tmp.apply(lambda x: x.tokenised.split('_SENT_')[x.sentence_number], axis=1)

    

#     if covid_only:

#         # We will inlcude documents that have a blank title

#         tmp = tmp.loc[tmp.title.fillna('covid').str.lower().str.contains('covid')]

        

#     if date_range[0]>1957 or date_range[1]<2020:

#         # Also include documents missing the published date

#         tmp = tmp.loc[(tmp.publish_year=='0') | ((tmp.publish_year>=str(date_range[0])) & (tmp.publish_year<=str(date_range[1])))]

#     if len(tmp) > 0:

#         tmp = tmp.loc[tmp.para_text.str.len()>min_length, :].iloc[:k].copy()

#         tmp['title'] = tmp.apply(lambda x: f'<a href="http://doi.org/{x.doi}" target="_blank">{x.title}</a>', axis=1)



#         tmp['para_context'] = tmp['para_context'].apply(lambda x: x.replace('_STARTREF_', '<p><i>').replace('_ENDREF_', '</i></p>'))



#         pd.options.display.max_colwidth = showmore

#         return HTML(tmp[['relevance', 'para_context', 'relevant_sentence', 'title', 'publish_year']].rename(columns={'relevance': 'Relevance', 'para_context': 'Body text', 'title': 'Paper title', 'publish_year': 'Published'}).set_index('Relevance').to_html().replace('&lt;', '<').replace('&gt;', '>'))

#     else:

#         print('No results found')
# interact(search_db, words=widgets.Text(

#     value='COVID19 symptoms',

#     placeholder='Enter search phrase',

#     description='Search phrase:',

#     disabled=False,

#     layout=Layout(width='80%', height='80px')

# ),

#         date_range=widgets.IntRangeSlider(

#     value=[1957, 2020],

#     min=1957,

#     max=2020,

#     step=1,

#     description='Publish date:',

#     disabled=False,

#     continuous_update=False,

#     orientation='horizontal',

#     readout=True,

#     readout_format='d',

# ));
if not PREPROCESS_AND_TRAIN:

    all_data.to_pickle('CORD_19_all_papers.pkl')

    pickle.dump(vocab, open('covid_vocab_frequencies.pkl', 'wb'))

    pickle.dump(paragraph_vectors, open('all_para_vectors.pkl', 'wb'))

    model.save('covid_w2v')    

    write_index(index_cosine, 'index_cosine.faiss')