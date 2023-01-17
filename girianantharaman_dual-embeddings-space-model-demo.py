import sys, os, csv, glob, json, uuid, pickle, math

import nltk 

import gensim, logging

import numpy as np, scipy, pandas as pd

from operator import itemgetter

from IPython.display import HTML, display

import tabulate
CONTENT_INDEX = 9

csv.field_size_limit(sys.maxsize)

CONTENT_PATH = './inputs/contents/'

TOKENS_PATH = './inputs/tokens/'

CENTROIDS_PATH = './inputs/centroids/'

BM25_PATH = './inputs/bm25/'



if not os.path.exists(CONTENT_PATH):

    os.makedirs(CONTENT_PATH)

    

if not os.path.exists(TOKENS_PATH):

    os.makedirs(TOKENS_PATH)

    

if not os.path.exists(CENTROIDS_PATH):

    os.makedirs(CENTROIDS_PATH)



if not os.path.exists(BM25_PATH):

    os.makedirs(BM25_PATH)
count = 0



for fname in glob.iglob('./inputs/*.csv', recursive=False):

    f = open(fname)

    reader = csv.reader(f)

    for line in reader:

        count = count + 1

        content = line[CONTENT_INDEX]

        cname = CONTENT_PATH + str(count) + '.txt'

        tname = TOKENS_PATH + str(count) + '.tokens'

        cf = open(cname, 'w')

        cf.write(content)

        cf.close()

        tf = open(tname, 'w')

        for sentence in nltk.sent_tokenize(content):

            tf.write("%s\n" % sentence.lower())

        tf.close()
class MySentences(object):

    def __init__(self, dirname):

        self.dirname = dirname

 

    def __iter__(self):

        for fname in glob.iglob(self.dirname +'*.tokens', recursive=True):

            for line in open(fname):

                yield nltk.word_tokenize(line)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = MySentences('./inputs/tokens/') 

model1 = gensim.models.Word2Vec(sentences, min_count=1)
model1.save('./model/w2v-lc.model')

model1.wv.save_word2vec_format('./model/w2v-lc.model.bin', binary=True)

vocab = dict([(k, v.index) for k, v in model1.wv.vocab.items()])

with open('./model/w2v-lc-vocab.json', 'w') as f:

    f.write(json.dumps(vocab))
model1.wv.most_similar(positive=['texas', 'senate'], negative=['alabama'])
for fname in glob.iglob('./inputs/contents/*.txt', recursive=False):

    for line in open(fname):

        centroid_in = (np.mean(np.array([get_embedding(x) for x in nltk.word_tokenize(line.lower())]), axis=0))

        centroid_out = (np.mean(np.array([get_embedding(x, out=True) for x in nltk.word_tokenize(line.lower())]), axis=0))

        out_dict = { fname : (centroid_in, centroid_out) }

        pickle_file = './inputs/centroids/' + os.path.basename(fname).replace('.txt', '.p')

        pickle.dump(out_dict, open(pickle_file, "wb"))
class BM25Sentences(object):

    def __init__(self, pattern):

        self.pattern = pattern

 

    def __iter__(self):

        for fname in glob.iglob(self.pattern, recursive=True):

            for line in open(fname):

                yield nltk.word_tokenize(line)
sentences = BM25Sentences('./inputs/tokens/*.tokens')

dictionary = gensim.corpora.Dictionary(line for line in sentences)

dictionary.compactify()

print(dictionary)
dictionary.save('./inputs/bm25/allnews.dict')
bm25dict = dictionary.load('./inputs/bm25/allnews.dict') 



class MyCorpus(object):

    def __init__(self, dirname):

        self.dirname = dirname

        self.count = 142573

 

    def __iter__(self):

        for x in range(self.count):

            fname = self.dirname + str(x+1) + '.tokens'

            doc = open(fname).read().replace('\n', '')

            yield bm25dict.doc2bow(nltk.word_tokenize(doc))
citer = MyCorpus(TOKENS_PATH)

corpus = [x for x in citer]

print (len(corpus))
gensim.corpora.MmCorpus.serialize('./inputs/bm25/allnewscorpus.mm', corpus)
bm25corpus = gensim.corpora.MmCorpus('./inputs/bm25/allnewscorpus.mm')
model = gensim.models.Word2Vec.load('./model/w2v-lc.model')
centroid_dict = {}

for fname in glob.iglob('./inputs/centroids/*.p', recursive=False):

    centroid_dict.update(pickle.load(open(fname, "rb")))
clean_centroid_dict = {k: centroid_dict[k] for k in centroid_dict if not np.isnan(centroid_dict[k][0]).any()}
def get_embedding(x, out=False):

    if x in model.wv.vocab:

        if out == True:

            return model.syn1neg[model.wv.vocab[x].index]

        else:

            return model[x]

    else:

        return np.zeros(100)
def score_document(q_embeddings, d_centroid):

    individual_csims = [(1 - scipy.spatial.distance.cosine(qin, d_centroid)) for qin in q_embeddings]

    return (sum(individual_csims)/len(q_embeddings))
bm25dict = gensim.corpora.Dictionary().load('./inputs/bm25/allnews.dict') 

bm25corpus = gensim.corpora.MmCorpus('./inputs/bm25/allnewscorpus.mm')

bm25 = gensim.summarization.bm25.BM25(bm25corpus)

average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)
query = 'political stability and economic health'

query_words = nltk.word_tokenize(query.lower())
scores = bm25.get_scores(bm25dict.doc2bow(query_words), average_idf)
best_result = ['./inputs/contents/'+str(x+1)+'.txt' for x in (sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5])]

for fname in best_result:

    print(fname)
query_ins = [get_embedding(x) for x in query_words]

q_len = len(query_ins)

print('Num words in query: ', len(query_words), 'Num query word in vectors: ', q_len)
scores_in_in = []

scores_in_out = []

for k,v in clean_centroid_dict.items():

    scores_in_in.append((k, score_document(query_ins, v[0])))

    scores_in_out.append((k, score_document(query_ins, v[1])))



scores_in_in = sorted(scores_in_in, key=itemgetter(1), reverse=True)

scores_in_out = sorted(scores_in_out, key=itemgetter(1), reverse=True)
print('TOP 5 IN-IN:')

top_5_in_in = [x[0] for x in scores_in_in[:5]]



for fname in top_5_in_in:

    print(fname)
print('TOP 5 IN-OUT:')

top_5_in_out = [x[0] for x in scores_in_out[:5]]



for fname in top_5_in_out:

    print(fname)
table = [["BM25",30931, 40023, 71852, 133532, 1620],

         ["DESM-IN-IN", 140797, 32221, 31472, 39594, 135444],

         ["DESM-IN-OUT", 73280, 140797, 32221, 42404, 42105]]

display(HTML(tabulate.tabulate(table, tablefmt='html')))