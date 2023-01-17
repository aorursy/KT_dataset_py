import numpy as np
import pandas as pd
import re
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from fastcache import clru_cache as lru_cache

print( 'read data' )
train_df = pd.read_csv( '../input/train.csv' )
train_size = len( train_df.index )
test_df = pd.read_csv( '../input/test.csv' )
ids_test = test_df[ 'qid' ].values
clazz_train = train_df[ 'target' ].values

print( 'training set' )
train_df.head()
print( 'test set' )
test_df.head()
df = pd.concat( [ train_df, test_df ], axis=0, sort=False, ignore_index=True )

print( 'training set size:', train_size )
print( 'test set size:', len(ids_test) )
print( 'vectorize' )
# stemmer
stemmer = PorterStemmer()
@lru_cache(2048)
def stem(s):
	return stemmer.stem(s)
whitespace = re.compile(r'\s+')
non_letter = re.compile(r'\W+')
def tokenize(text):
	text = text.lower()
	text = non_letter.sub(' ', text)
	tokens = []
	for t in text.split():
		t = stem(t)
		tokens.append(t)
	return tokens

vectorizer = TfidfVectorizer( use_idf=True, lowercase=True, tokenizer=tokenize )
vecs = vectorizer.fit_transform( df[ 'question_text' ].values )
vecs_train = vecs[ :train_size ]
vocabulary = {v:k for k,v in vectorizer.vocabulary_.items()}
for i,k in zip(range(20),vocabulary.keys()):
    print( k,vocabulary[k] )
# make weights for class
cnum = np.sum(clazz_train == 1)
weight = cnum / ( len(clazz_train) - cnum )
# sum of each class
index = np.arange( train_size )
index_posi = index[ clazz_train == 0 ]
index_nega = index[ clazz_train == 1 ]
vecs_posi = vecs_train[ index_posi ].sum( axis=0 ) * ( weight )
vecs_nega = vecs_train[ index_nega ].sum( axis=0 ) * ( 1.0 - weight)
print(vecs_nega[:10])
print(vecs_posi[:10])
# class nega is class value 1
vecs_score = np.array(vecs_nega - vecs_posi).reshape((-1,))
print(vecs_score[:10])
print( 'positive/negative words in trainig set:' )
rank = np.argsort( vecs_score )
# 'positive word' is class 0
print( '  positive 20:' )
for r in rank[:20]:
	print( vecs_score[ r ], vocabulary[ r ] )
print( '  negative 20:' )
for r in rank[-20:]:
	print( vecs_score[ r ], vocabulary[ r ] )
print( 'find words only in test set' )
vecs_test = vecs[ train_size: ]
train_words = np.nonzero( vecs_train.sum( axis=0 ) )[1]
test_words = np.nonzero( vecs_test.sum( axis=0 ) )[1]
onlytest_words = [ t for t in test_words if t not in train_words ]
print( '  words only in test:' )
for o in onlytest_words[:20]:
	print( o, vocabulary[ o ] )
print( 'find nearest word' )
# load word vector
def load_vectors(fname):
	fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
	n, d = map(int, fin.readline().split())
	data = {}
	for line in fin:
		tokens = line.rstrip().split(' ')
		if tokens[0] in vectorizer.vocabulary_:
			data[tokens[0]] = np.array( list( map(float, tokens[1:]) ) )
	return data

word_vec = load_vectors( '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec' )
# find nearest word
def get_onlytest_words( o ):
	result = 0.0
	if vocabulary[ o ] in word_vec:
		nmin = np.inf
		nidx = None
		# use 2000 negative words
		for k in rank[-2000:]:
			if k not in onlytest_words and vocabulary[ k ] in word_vec:
				n = np.linalg.norm( word_vec[ vocabulary[ o ] ] - word_vec[ vocabulary[ k ] ] )
				if nmin > n:
					nmin = n
					nidx = k
		if nidx is not None:
			# use nearest word
			result = vecs_score[ nidx ]
	return o, result
# use multiprocessing
from multiprocessing import Pool
with Pool(4) as p:
	for o, val in p.map(get_onlytest_words, onlytest_words):
		vecs_score[ o ] = val
print('predict for trainig data set')
vecs_train_posi = vecs_train[ index_posi ]
score_posi = vecs_train_posi.multiply( vecs_score ).sum( axis=1 )
score_posi = np.array(score_posi).reshape((-1,))
print('posi:',score_posi.min(),score_posi.max(),score_posi.mean(),np.median(score_posi))

vecs_train_nega = vecs_train[ index_nega ]
score_nega = vecs_train_nega.multiply( vecs_score ).sum( axis=1 )
score_nega = np.array(score_nega).reshape((-1,))
print('nega:',score_nega.min(),score_nega.max(),score_nega.mean(),np.median(score_nega))
print('make split point:')
score_min = min(score_posi.min(),score_nega.max())
score_max = max(score_posi.max(),score_nega.max())
score_div = score_min
score_div_p = score_div
score_gain = score_max - score_min
count_true = -np.inf
for _ in range(10):
	c = len( score_posi[ score_posi<=score_div ] ) + len( score_nega[ score_nega>score_div ] )
	if c > count_true:
		count_true = c
		score_div_p = score_div
		score_gain /= 2.0
		score_div += score_gain
	else:
		score_gain /= 2.0
		score_div -= score_gain
print('split point is',score_div_p)
print('scoreing for training data set')
TP = len( score_posi[ score_posi<=score_div_p ] )
FP = len( score_nega[ score_nega<=score_div_p ] )
FN = len( score_posi[ score_posi>score_div_p ] )
TN = len( score_nega[ score_nega>score_div_p ] )
PR = TP / (TP+FP)
RC = TP / (TP+FN)
print('F1 score of training data set is',2*RC*PR / (RC+PR))
score = vecs_test.multiply( vecs_score ).sum( axis=1 )
score = np.array(score).reshape((-1,))
print(score[:20])
score[ score <= 0 ] = 0
score[ score > 0 ] = 1
score = score.astype( np.int )
print(score[:20])
print('class0 count:',len(score[ score == 0 ]))
print('class1 count:',len(score[ score == 1 ]))
test_df[ 'prediction' ] = score
test_df[ 'prediction' ] = test_df[ 'prediction' ].astype( np.int )
test_df[ [ 'qid','prediction'] ].to_csv( 'submission.csv', index=None )