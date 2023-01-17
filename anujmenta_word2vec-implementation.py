import math, csv

import heapq

from nltk.tokenize import word_tokenize

from nltk.corpus.reader.wordnet import ADJ, ADJ_SAT, ADV, NOUN, VERB

from nltk.stem import WordNetLemmatizer

import time  # For computing running time

import gensim

import numpy as np



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
stopwords = frozenset(

        	[u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours',

             u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it',

             u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who',

             u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been',

             u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the',

             u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with',

             u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below',

             u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further',

             u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each',

             u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not', u'only', u'own', u'same',

             u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u'should', u'now', u"'s", u'?', u'50/50'])

def cleaner(question_string):

	"""

	Given question string, returns word2vec vector of the questions tring

	:param question_string : The given question as a string.

	"""

	words = word_tokenize(question_string)[:-1]

	non_stop_words = []

	for w in words:

		if w.lower().strip('-') not in stopwords and w.lower() in model.vocab:

			w = WordNetLemmatizer().lemmatize(w, NOUN)

			non_stop_words.append(w.lower().strip('-'))

	#print non_stop_words

	vectors = [model[word] for word in non_stop_words]

	vector = sum(vectors)/float(len(non_stop_words))

	return vector



def numpy_cosine(q1_vec, q2_vec):

	"""

	Cosine similarity between q1 and q2 question instances using their vectors

	:param q1_vec: cleaner(question1)

	:param q2_vec: cleaner(question2)

	:return: similarity between q1 and q2

	"""

	#	print q1_vec

	cosine_similarity = np.dot(q1_vec, q2_vec)/(np.linalg.norm(q1_vec)*np.linalg.norm(q2_vec))

	#	print np.dot(q1_vec, q2_vec)

	#	print type(np.dot(q1_vec, q2_vec))

	return cosine_similarity

MODEL_Googlenews_DIR = 'GoogleNews-vectors-negative300.bin'

model = gensim.models.Word2Vec.load_word2vec_format(MODEL_Googlenews_DIR, binary=True)

opener = open('test.csv', 'r')

reader = csv.reader(opener)



with open('output.csv', 'wb') as resultFile:

	wr = csv.writer(resultFile, dialect='excel')

	wr.writerow(['test_id', 'is_duplicate'])

	header = opener.readline()

	for line in reader:

		qid = line[0]

		question1 = line[1]

		question2 = line[2]

		try:

			wr.writerow([int(qid), numpy_cosine(cleaner(question1), cleaner(question2))])

		except:

			if qid=='test_id':

				pass

			wr.writerow([ int(qid), 1 ])