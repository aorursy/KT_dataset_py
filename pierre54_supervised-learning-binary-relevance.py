import requests

import re

import csv

from lxml import html

from lxml import etree



import pandas as pd

from collections import defaultdict

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

import nltk

import pickle



import copy

import math



from sklearn import linear_model

import os.path



import csv
# This function uses the Springer API to get the articles related to physics and their keywords

def get_training_set():

	# To create a free account : https://dev.springer.com/signup

	api_key = "put your API key here"

	host = "http://api.springer.com"

	path = "/metadata/pam"



	query = "physics"



	articles = open('./physics_articles.csv', 'w')

	csvwriter = csv.writer(articles)

	csvwriter.writerow(['id', 'title', 'content', 'tags'])



	i = 0

	s = 1

	step = 100

	

	# The maximum number of articles to download

	maxi = 1000



	while True:

		get_params = {

		'q': query,

		's': s,

		'p': step,

		'api_key': api_key

		}



		r = requests.get(host + path, params=get_params)

		response = html.fromstring(r.content)



		# The total number of articles got from the query

		total = int(response.xpath('//result/total/text()')[0])



		doil = response.xpath('//doi/text()')



		# We have to make one request per articles

		for doi in doil:	

			i += 1

			

			get_params = {

				'q': doi,

				'api_key': api_key

			}

	

			r = requests.get(host + path, params=get_params)

			response = html.fromstring(r.content)

	

			keywords = response.xpath('//facet[@name="keyword"]/facet-value/text()')

	

			if(len(keywords) == 0):

				continue

	

			title = response.xpath('//article/title/text()')

	

			if(len(title) == 0):

				continue

	

			abstract = response.xpath('//body/p')

	

			if(len(abstract) == 0):

				continue

	

			title = title[0]

			abstract = re.sub('<.*?>', '', str(etree.tostring(abstract[0]).decode('utf-8')))

	

			tag_list = []



			for k in keywords:

				if(re.search('[0-9:\\(\\)]', k)):

					continue



				k = re.sub(' ', '-', k.strip()).lower()

				tag_list.append(k)

		

			tags = ' '.join(tag_list)

	

			csvwriter.writerow([i, title, abstract, tags])



		if(s + step >= total or s + step >= maxi):

			break

		else:

			s += step



	articles.close()

# This function creates a list of dictionaries (one per document/articles), a dictionary of all the words in the corpus (dico_global) and one dictionary that counts the number of documents who contains a word (dico_unique).

def create_dictionary():

	train = pd.read_csv('./physics_articles.csv')



	dico_list = []

	dico_global = defaultdict(int)

	dico_unique = defaultdict(int)

	stemmer = SnowballStemmer("english")

	stop = stopwords.words('english')



	for i in range(0, train.count()['id']):

		row = train.loc[i]

		dico = defaultdict(int)

	

		for word, token in nltk.pos_tag(nltk.word_tokenize(row['title'].lower())):

			if(word not in stop and (token.startswith("NN") or token.startswith("VB") or token.startswith("JJ"))):

				word = re.sub('[^a-z \\-]', '', word)

				word = stemmer.stem(word)

				if(dico[word] == 0):

					dico_unique[word] += 1

				dico[word] += 1

				dico_global[word] += 1

	

		for word, token in nltk.pos_tag(nltk.word_tokenize(re.sub('<.*?>', '', row['content'].lower()))):

			if(word not in stop and (token.startswith("NN") or token.startswith("VB") or token.startswith("JJ"))):

				word = re.sub('[^a-z \\-]', '', word)

				word = stemmer.stem(word)

				if(dico[word] == 0):

					dico_unique[word] += 1

				dico[word] += 1

				dico_global[word] += 1



		dico_list.append(dico)

	

	pickle.dump(dico_list, open('./dico_list.bin', 'wb'))

	pickle.dump(dico_unique, open('./dico_unique.bin', 'wb'))

	pickle.dump(dico_global, open('./dico_global.bin', 'wb'))

# Application of TD-IDF on dico_list

# I also chose to only keep the ten most important words in each document in order to reduce the dictionaries

# and I removed the words present in more than 10% of the documents and present in only one document

def tfidf():

	train = pd.read_csv('./physics_articles.csv')

	dico_list = pickle.load(open('./dico_list.bin', 'rb'))

	dico_global = pickle.load(open('./dico_global.bin', 'rb'))

	dico_unique = pickle.load(open('./dico_unique.bin', 'rb'))

	

	dico_list_tfidf = defaultdict(float)

	word_to_keep = set()



	for i in range(0, train.count()['id']):

		for word, occur in dico_list[i].items():

			row = train.loc[i]

			text = row['title'] + ' ' + re.sub('<.*?>', '', row['content'])

			tf = occur / len(text.split())

			idf = math.log(train.count()['id'] / dico_unique[word], 10)

			dico_list_tfidf[i][word] = tf * idf

	

		word_to_keep.update(set(sorted(dico_list[i], key=dico_list[i].get, reverse=True)[:10]))

	

	# word_to_keep has to be an ordered list, not a set

	word_to_keep = list(word_to_keep)

	

	# It's not possible to edit the dictionary in a for loop

	copy_dico_global = copy.copy(dico_global)



	for word in copy_dico_global:

		if(word not in word_to_keep or dico_global[word] == 1 or dico_global[word] / train.count()['id'] > 0.1):

			dico_global.pop(word)

		

	del copy_dico_global



	pickle.dump(dico_list_tfidf, open('./dico_list_tfidf.bin', 'wb'))

	pickle.dump(word_to_keep, open('./word_to_keep.bin', 'wb'))

# This function create the X matrix who will be send to the sklearn algorithm

def create_x():

	dico_list_tfidf = pickle.load(open('./dico_list_tfidf.bin', 'rb'))

	word_to_keep = pickle.load(open('./word_to_keep.bin', 'rb'))

	

	X = []

	

	for i in range(0, train.count()['id']):

		x = []

		for word in word_to_keep:

			x.append(dico_list_tfidf[i][word])

		X.append(x)



	pickle.dump(X, open('./X.bin', 'wb'))
# This function creates the keywords array who contains all the different tags in the training set

def get_keywords():

	train = pd.read_csv('./physics_articles.csv')

	keywords = []



	for i in range(0, train.count()['id']):

		row = train.loc[i]

		for word in row['tags'].split():

			keywords.append(word)



	keywords = list(set(keywords))

	

	pickle.dump(keywords, open('./keywords.bin', 'wb'))



# This function create the Y matrix who will be send to the sklearn algorithm

# It's a matrix because we will use binary relevance: each column of the matrix correspond to one tag/keyword

def create_y():

	train = pd.read_csv('./physics_articles.csv')

	keywords = pickle.load(open('./keywords.bin', 'rb'))

	

	Y = []



	for keyword in keywords:

		print(keyword)

		y = []

		for i in range(0, train.count()['id']):

			row = train.loc[i]

			if(keyword in row['tags'].split()):

				y.append(1)

			else:

				y.append(0)

		Y.append(y)



	pickle.dump(Y, open('./Y.bin', 'wb'))

# This function create one model per keyword

def create_models():

	X = pickle.load(open('./X.bin', 'rb'))

	Y = pickle.load(open('./Y.bin', 'rb'))

	

	for i in range(0, len(Y)):

		# I added this condition because I got a problem during the creation of the models

		if(not os.path.isfile('./clf/' + str(i) + '.bin')):

			# You can adjust the C parameter but, with smaller value on smaller training set, I got a very bad in-sample error (almost 100% with C=1)

			clf = linear_model.LogisticRegression(C=100)

			clf.fit(X, Y[i])

		

			pickle.dump(clf, open('./clf/' + str(i) + '.bin', 'wb'))

# This function predicts tags according to our models

def predict():

	test = pd.read_csv('./test.csv')

	dico_unique = pickle.load(open('./dico_unique.bin', 'rb'))

	

	predict_csv = open('./predict.csv', 'w')

	csvwriter = csv.writer(predict_csv)

	csvwriter.writerow(['id', 'tags'])



	stemmer = SnowballStemmer("english")



	for i in range(0, int(test.count()['id'])):

		row = test.loc[i]

		dico = defaultdict(float)

		predicted_keywords = ""

	

		for word in nltk.word_tokenize(row['title'].lower()):

			word = re.sub('[^a-z \\-]', '', word)

			word = stemmer.stem(word)

			if(word in word_to_keep):

				dico[word] += 1

	

		for word in nltk.word_tokenize(row['content'].lower()):

			word = re.sub('[^a-z \\-]', '', word)

			word = stemmer.stem(word)

			if(word in word_to_keep):

				dico[word] += 1

	

		x = []

		length = len(row['content'].split()) + len(row['title'].split())

	

		for word in word_to_keep:

			tf = dico[word] / length

			idf = math.log(test.count()['id'] / dico_unique[word], 10)

			x.append(tf * idf)

	

		for j in range(0, len(keywords)):

			clf = pickle.load(open("./clf/" + str(j) + ".bin", 'rb'))

			res = clf.predict([x])[0]

		

			if(res == 1):

				predicted_keywords += (" " + keywords[j])

	

		csvwriter.writerow([row['id'], predicted_keywords])



	predict_csv.close()