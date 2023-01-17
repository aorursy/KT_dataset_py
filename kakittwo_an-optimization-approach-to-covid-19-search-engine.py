import nltk

nltk.download('stopwords')

nltk.download('punkt')



IS_FAST = True
# Data cleaning

import json

from nltk.tokenize import word_tokenize

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords

from collections import defaultdict

from gensim import corpora

from gensim.models.phrases import Phrases, Phraser

import glob



class dataCleaningRobot():

	def __init__(self, path, max_count_arg = 10000000):

		self.max_count = max_count_arg

		self.paths_to_files =  glob.glob(path + "/*.json")

		self.path = path



	def getText(self):

		self.getDocDicFromPath()

		self.filterTokens()



	def getDocDicFromPath(self):

		self.dicOfTexts = {}

		mycount = 0

		for filename in self.paths_to_files:

			with open(filename) as f:

				data = json.load(f)



			wordList = []

			paperID = data['paper_id']



			for eachTextBlk in data['abstract'] + data['body_text']:

				wordList += word_tokenize(eachTextBlk['text'])



			self.dicOfTexts[paperID] = wordList

			mycount +=1 

			if mycount > self.max_count:

				break



	def filterTokens(self):

		self.dicOfFilterTexts = {}



		### Token-based filtering   

		newStopWords = set(['preprint', 'copyright', 'doi', 'http', 'licens', 'biorxiv', 'medrxiv'])

		stopWords = set( stopwords.words('english'))



		porter = PorterStemmer()

		self.wordCtFreq = defaultdict(int)



		for eachText in self.dicOfTexts:

			filtered = []

			for word in self.dicOfTexts[eachText] :

				if word.isalpha() and len(word) > 2 :

					token = word.lower()

					if token in stopWords:

						continue

					token = porter.stem(token)

					if token in newStopWords:

						continue



					filtered.append(token)

					self.wordCtFreq[token] += 1



			self.dicOfFilterTexts[eachText] = filtered

		

		### Count-based filtering

		for eachText in self.dicOfFilterTexts:

			filtered = []



			for word in self.dicOfFilterTexts[eachText] :

				if self.wordCtFreq[word] > 10 :

					filtered.append(word)



			self.dicOfFilterTexts[eachText] = filtered



	def getDicCorpus(self):

		texts = [ self.dicOfFilterTexts[eachitem] for eachitem in self.dicOfFilterTexts ] 

		self.dictionary = corpora.Dictionary(texts)

		self.corpus = [self.dictionary.doc2bow(text) for text in texts]



	def getSingleWordCount(self):

		return self.getCountInfo(self.wordCtFreq)



	def getBigramCount(self):

		return self.getCountInfo(self.bigramCtFreq)



	def getCountInfo(self, ctFreqDic):



		word_freq_list = []

		for each_token in ctFreqDic: 

			word_freq_list.append([ctFreqDic[each_token], each_token])

		word_freq_list.sort(reverse= True)



		# print(word_freq_list[0:5])

		wordList = []

		countList = []



		for eachitem in word_freq_list:

			wordList.append(eachitem[1])

			countList.append(eachitem[0])



		return wordList, countList, word_freq_list





	def getBigramData(self):

		texts = [ self.dicOfFilterTexts[eachitem] for eachitem in self.dicOfFilterTexts ] 

		self.bigram = Phrases(texts)

		self.bigram_model = Phraser(self.bigram)

		bigram_texts = [self.bigram_model[self.dicOfFilterTexts[eachitem]] for eachitem in self.dicOfFilterTexts]



		self.bigramCtFreq = defaultdict(int)

		for eachText in bigram_texts:

			for word in eachText :

				tmp_array = word.split('_') 



				### Only extract two words

				if len(tmp_array) > 1 :

					self.bigramCtFreq[word] += 1





		self.bigram_dictionary = corpora.Dictionary(bigram_texts)

		self.bigram_corpus = [self.bigram_dictionary.doc2bow(text) for text in bigram_texts]





# Feature generation

from gensim import models

import numpy as np

from gensim.test.utils import datapath

from nltk.stem.porter import PorterStemmer

from gensim.corpora import Dictionary



class featureGenRobot():

	def __init__(self):

		assert(True)



	def getModels(self):

		self.num_lsi_topics = 2

		self.num_lda_topics = 10

		self.tfidf_model = models.TfidfModel(self.corpus)

		self.corpus_tfidf = self.tfidf_model[self.corpus]

		self.lsi_model = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_lsi_topics)  

		self.lda_model = models.LdaModel(self.corpus,id2word=self.dictionary, num_topics=self.num_lda_topics, iterations=1500, passes=20, minimum_probability=0.0)



	def saveModelsDict(self):

		prefix = '/kaggle/working/'

		self.lda_model.save(datapath(prefix + "lda_debug.model"))

		self.lsi_model.save(datapath(prefix + "lsi_debug.model"))

		self.tfidf_model.save(datapath(prefix + "tfidf_model_debug.model"))

		self.dictionary.save_as_text(prefix + "dictionary_debug.txt")



	def loadModelsDict(self):

		prefix = '/kaggle/working/'



		self.num_lsi_topics = 2 

		self.num_lda_topics = 10



		self.tfidf_model = models.TfidfModel.load(prefix + "tfidf_model_debug.model")

		self.lsi_model = models.LsiModel.load(prefix + "lsi_debug.model")

		self.lda_model = models.LdaModel.load(prefix + "lda_debug.model")



		self.dictionary = Dictionary.load_from_text(prefix+ "dictionary_debug.txt")



	def getFeaVec(self, text):

		porter = PorterStemmer()

		myList = text.lower().split()

		myList2 = [ porter.stem(word.lower()) for word in myList ]

		bow_vec = self.dictionary.doc2bow(myList2)

		return self.getFeaVecFromBow(bow_vec)



	def getFeaVecFromBow(self, bow_vec):

		### Computer BOW, TFIDF , LSI, LDA values . models topic division as dense features. 

		vector_tfidf = self.tfidf_model[bow_vec]

		vector_lsi = self.lsi_model[bow_vec]

		vector_lda = self.lda_model[bow_vec]

		

		### Convert LSI, LDA values as dense vectors. 

		lsi_topic = self.num_lsi_topics 

		lda_topic = self.num_lda_topics



		N = lsi_topic + lda_topic

		denseVector = np.zeros(N)

		

		base = 0 

		for i in range(len(vector_lsi)):

			idx = vector_lsi[i][0]

			denseVector[base + idx] = vector_lsi[i][1]



		base = len(vector_lsi)

		for i in range(len(vector_lda)):

			idx = vector_lda[i][0]

			denseVector[base + idx] = vector_lda[i][1]



		### Convert BOW, IFIDF as sparse vectors.

		m1 = len(self.dictionary)

		m2 = len(self.dictionary)

		sparseVec = np.zeros(m1 + m2)

		base = 0 

		for eachitem in bow_vec:

			idx = eachitem[0]

			val = eachitem[1]

			sparseVec[base + idx] = val



		base = m1

		for eachitem in vector_tfidf:

			idx = eachitem[0]

			val = eachitem[1]

			sparseVec[base + idx] = val



		### Convert dense and sparse feature vectors.

		combined = np.concatenate((sparseVec, denseVector))

		#combined = denseVector

		return denseVector, sparseVec, combined



	def genFeaVecMap(self, dictionary_arg, corpus_arg):

		self.dictionary = dictionary_arg

		self.corpus = corpus_arg

		self.getModels()



# Example generation

import random

import numpy as np 





class exampleGenRobot():

	def __init__(self, dicOfFilterTexts, dictionary, featureRobot):

		assert(True)

		self.dicOfFilterTexts = dicOfFilterTexts

		self.dictionary = dictionary

		self.featureRobot= featureRobot



	def getIJPair(self):	

		self.ijPairs = []

		N = len(self.dicOfFilterTexts)

		posNum = 100

		negNum = 100

		docRatio = 0.01



		keyList = list(self.dicOfFilterTexts)



		for id_text in self.dicOfFilterTexts:

			wholeText = self.dicOfFilterTexts[id_text]

			sentArr = [ [] for i in range(posNum)  ]

			### Pos eg

			for word in wholeText:

				for i in range(posNum):

					if random.random() < docRatio:

						sentArr[i].append(word)

			

			for i in range(int(posNum/2)):

				bow_vec1 = self.dictionary.doc2bow(sentArr[2*i])

				bow_vec2 = self.dictionary.doc2bow(sentArr[2*i+1])

				denseVector1, sparseVec1, combined1 = self.featureRobot.getFeaVecFromBow(bow_vec1)

				denseVector2, sparseVec2, combined2 = self.featureRobot.getFeaVecFromBow(bow_vec2)

				self.ijPairs.append([combined1, combined2, 1])



			### Neg eg

			for j in range(negNum):



				k = random.choice(keyList)

				while ( k == id_text):

					k = random.choice(keyList)



				negSent = []

				for word in self.dicOfFilterTexts[k]:

					if random.random() < docRatio:

						negSent.append(word)



				bow_vecneg = self.dictionary.doc2bow(negSent)

				bow_vec1 = self.dictionary.doc2bow(sentArr[j])



				denseVector1, sparseVec1, combined1 = self.featureRobot.getFeaVecFromBow(bow_vec1)

				denseVectorneg, sparseVecneg, combinedneg = self.featureRobot.getFeaVecFromBow(bow_vecneg)

				self.ijPairs.append([combined1, combinedneg, 0])

				

		train_X = np.zeros((len(self.ijPairs),  self.ijPairs[0][0].shape[0] + self.ijPairs[0][1].shape[0] ))

		

		y = np.zeros(len(self.ijPairs))



		for i in range(len(self.ijPairs)) :

			train_X[i] = np.concatenate((self.ijPairs[i][0], self.ijPairs[i][1]))

			y[i] = self.ijPairs[i][2]



		return train_X, y 





# Model training

from sklearn.model_selection import train_test_split

import tensorflow as tf



class simGenerator():

	def __init__(self,train_eg, train_labels):

		self.train_eg = train_eg

		self.train_labels = train_labels



	def trainModel(self):

		train_examples, test_examples, train_labels, test_labels = train_test_split(self.train_eg, self.train_labels, test_size=0.33, random_state=42)

		train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))

		test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

		

		BATCH_SIZE = 4

		SHUFFLE_BUFFER_SIZE = 16



		train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

		test_dataset = test_dataset.batch(BATCH_SIZE)



		print(train_examples.shape[1])



		model = tf.keras.Sequential()

		model.add(tf.keras.layers.Dense(10, input_dim=train_examples.shape[1], activation='relu'))

		model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



		model.compile(optimizer=tf.keras.optimizers.RMSprop(),

			loss=tf.keras.losses.BinaryCrossentropy(),

			metrics=['binary_accuracy'])



		model.fit(train_dataset, epochs=10)

		print(model.evaluate(test_dataset))





		model_json = model.to_json()

		with open("/kaggle/working/kmodel.js", "w") as json_file:

			json_file.write(model_json)



		model.save_weights("/kaggle/working/kmodel.h5")

		print("Saved model to disk")



		self.trained_model = model



		
# Retrieval 



import numpy as np



class retrievalMethod():

	def __init__(self, sim_model,dicOfFilterTexts, corpus, featureBot):

		self.dicOfFilterTexts = dicOfFilterTexts

		self.sim_model = sim_model

		self.corpus = corpus

		self.featureBot= featureBot



	def findClosest(self, searchtext, limit=10, offset=0):

		denseVector, sparseVec, combined = self.featureBot.getFeaVec(searchtext)

		texts = [(eachitem, self.dicOfFilterTexts[eachitem]) for eachitem in self.dicOfFilterTexts] 



		relevantList = []

		for i in range(len(self.corpus)):

			each_bow = self.corpus[i]

			denseVector1, sparseVec1, combined1 = self.featureBot.getFeaVecFromBow(each_bow)

			modelVec = np.concatenate((combined1, combined))

			modelVec = modelVec.reshape((1, modelVec.shape[0]))

			prob = self.sim_model.predict(modelVec)

			relevantList.append((prob, i))



		relevantList.sort(reverse=True)



		return [{

		"prob": r[0],

		"paper_id": texts[r[1]][0]

		} for r in relevantList[offset:offset + limit]]





# Example usage 



print("Data cleaning")

fileList = "/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json"

if IS_FAST:

    cleaner_bot = dataCleaningRobot(fileList, 70)

else:

    cleaner_bot = dataCleaningRobot(fileList)



cleaner_bot.getText()

cleaner_bot.getDicCorpus()



print("Feature generation")

feat_bot = featureGenRobot()

feat_bot.genFeaVecMap(cleaner_bot.dictionary, cleaner_bot.corpus)

feat_bot.saveModelsDict()



print("Ground truth generation")

fileList = "/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json"

cleaner_bot_small = dataCleaningRobot(fileList, 70)

cleaner_bot_small.getText()

cleaner_bot_small.getDicCorpus()



eg_bot = exampleGenRobot(cleaner_bot_small.dicOfFilterTexts, cleaner_bot.dictionary, feat_bot)

train_eg, train_labels = eg_bot.getIJPair()



print("Model training")

sim_bot = simGenerator(train_eg , train_labels)

model = sim_bot.trainModel()



retrieve_bot = retrievalMethod(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)

retrieve_bot.findClosest("vaccine and drug") 
### Single word count visualization

import glob

import matplotlib.pyplot as plt

from wordcloud import WordCloud

import numpy as np





wordList, countList, word_freq_list = cleaner_bot.getSingleWordCount()



plt.plot(range(len(countList)), countList)  

plt.xlabel("Top i vacab")

plt.ylabel("Word count")

plt.show()





offset = 0 

numItems = 10

plt.barh(wordList[offset:offset+ numItems], countList[offset:offset+ numItems])

plt.show()



offset = 50 

numItems = 10

plt.barh(wordList[offset:offset+ numItems], countList[offset:offset+ numItems])

plt.show()
### Bigram word count visualization



cleaner_bot.getBigramData()

wordList_bi, countList_bi, word_freq_list_bi = cleaner_bot.getBigramCount()

plt.xlabel("Top i bigram")

plt.ylabel("Bigram count")

plt.plot(range(len(countList_bi)), countList_bi)  

plt.show()


offset = 0 

numItems = 10

plt.barh(wordList_bi[offset:offset+ numItems], countList_bi[offset:offset+ numItems])

plt.show()
offset = 20 

numItems = 10

plt.barh(wordList_bi[offset:offset+ numItems], countList_bi[offset:offset+ numItems])

plt.show()
### Word cloud visualization



texts = ""

for eachDoc in cleaner_bot_small.dicOfFilterTexts:

  for eachword in cleaner_bot_small.dicOfFilterTexts[eachDoc]:

    if cleaner_bot_small.wordCtFreq[eachword] > countList[-30] and cleaner_bot_small.wordCtFreq[eachword] < countList[30]:

      texts = texts + " " + eachword





wordcloud = WordCloud(width = 800, height = 800, 

      background_color ='white', 

      min_font_size = 10).generate(texts)



plt.imshow(wordcloud, interpolation='bilinear')

plt.show()
from IPython.core.display import display, HTML

from jinja2 import Template

import glob



class evaluatorRobot():

	def __init__(self):

		

		self.table_template = Template('''

		<table>

		<thead>

		<tr>

		<th>Title</th>

		<th>Authors</th>

		<th>Abstract</th>

		<th>Paper ID</th>

		</tr>

		</thead>

		<tbody>

		{% for paper in papers %}

		<tr>

		<td>{{ paper.title }}</td>

		<td>{{ paper.authors }}</td>

		<td>

		{% for paragraph in paper.abstract %}

		<p>{{ paragraph }}</p>

		{% endfor %}

		</td>

		<td>{{ paper.paper_id }}</td>

		</tr>

		{% endfor %}

		</tbody>

		</table>

		''')



	def loadPath(self, path):

		self.pathtofiles = path



	def load_paper(self, paper_id):

		matches = glob.glob(self.pathtofiles + "/" + f'{paper_id}.json', recursive=True)

		filename = matches[0]

		with open(filename) as f:

			data = json.load(f)

		return data



	def formatPaper(self, raw_paper):

		paper = self.load_paper(raw_paper['paper_id'])

		authors = [f'{author["first"]} {author["last"]}' for author in paper['metadata']['authors']]

		abstract_paragraphs = [paragraph['text'][:100] + '...' for paragraph in paper['abstract']]

	

		return {

			'title': paper['metadata']['title'],

			'authors': ', '.join(authors),

			'abstract': abstract_paragraphs,

			'paper_id': paper['paper_id'],

			"prob": raw_paper['prob']

		}



	def presentResults(self, results):

		papers = [self.formatPaper(r) for r in results]

		render = self.table_template.render(papers=papers)

		display(HTML(render))

retrieve_bot = retrievalMethod(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)

results = retrieve_bot.findClosest("drug and vaccines", limit=5, offset=0)



eval_bot = evaluatorRobot()

eval_bot.loadPath(cleaner_bot.path)

eval_bot.presentResults(results)



# Retrieval 



import numpy as np

from scipy import spatial

from nltk.stem.porter import PorterStemmer



class retrievalMethod2():

	def __init__(self, sim_model,dicOfFilterTexts, corpus, featureBot):

		self.dicOfFilterTexts = dicOfFilterTexts

		self.sim_model = sim_model

		self.corpus = corpus

		self.featureBot= featureBot

        

	def findClosestWithSeed(self,  searchtext, limit=10, offset=0):

		# = ["therapeutic"]

		#key_words_2  = ["animal" ,"model"]

		texts = [ [eachitem , self.dicOfFilterTexts[eachitem]] for eachitem in self.dicOfFilterTexts ] 

		relevantList = []

		alreadyFoundDic = {}



		### Key word match first : score range 0.7 to 1 

		key_words_1 = searchtext.split()

		porter = PorterStemmer()



		kw_stem = []

		for word in key_words_1:

			kw_stem.append(porter.stem(word))



		kw_matched_list = []



		index = 0

		for doc_id in self.dicOfFilterTexts:

			N = len(self.dicOfFilterTexts[doc_id])

			count  = 0 

			total_count = 0

			for i in range(N - len(kw_stem)):

				total_count += 1

				if self.dicOfFilterTexts[doc_id][i:i+ len(kw_stem)] == kw_stem:

					count += 1 



			if count > 0:

				kw_matched_list.append([ count*1.0/total_count, count , doc_id, index])



			index += 1 



		kw_matched_list.sort(reverse=True)

		kw_matched_score_list= []

		for j in range(len(kw_matched_list)):

			index = kw_matched_list[j][-1]

			#print("d1", j)

			relevantList.append([1 - (1-0.7)*j/len(kw_matched_list), index])

			kw_matched_score_list.append([1 - (1-0.7)*j/len(kw_matched_list), index])

			alreadyFoundDic[index] = True



		### Cosine distance match : score range 0.3 to 0.7 

		cos_sim_matched_list = []



		for each_already_rel in kw_matched_score_list:

			score, kk = each_already_rel[0],  each_already_rel[1]

			denseVector, sparseVec, combined = self.featureBot.getFeaVecFromBow(self.corpus[kk])



			for i  in range(len(self.corpus )):

				if  i in alreadyFoundDic:

					continue

				each_bow = self.corpus[i]

				denseVector1, sparseVec1, combined1 = self.featureBot.getFeaVecFromBow(each_bow)

				similarity = 1 - spatial.distance.cosine(denseVector, denseVector1)

				if similarity > 0.9 :

					cos_sim_matched_list.append([score*similarity, i ] )





		cos_sim_matched_list.sort(reverse=True)

		for j in range(len(cos_sim_matched_list)):

			index = cos_sim_matched_list[j][-1]

			#print("d2", j)

			relevantList.append([0.7 - (0.7-0.3)*j/len(cos_sim_matched_list), index])

			alreadyFoundDic[index] = True





		### s distance match using doc  : score range 0.1 to 0.3 

		### Cosine distance match : score range 0.3 to 0.7 

		doc_sim_matched_list = []



		for each_already_rel in kw_matched_score_list:

			score, kk = each_already_rel[0],  each_already_rel[1]

			denseVector, sparseVec, combined = self.featureBot.getFeaVecFromBow(self.corpus[kk])

			for i  in range(len(self.corpus )):

				if  i in alreadyFoundDic:

					continue

				each_bow = self.corpus[i]

				denseVector1, sparseVec1, combined1 = self.featureBot.getFeaVecFromBow(each_bow)



				modelVec = np.concatenate((combined1, combined))

				modelVec= modelVec.reshape((1, modelVec.shape[0]))

				prob = self.sim_model.predict(modelVec)

				

				if prob > 0.9:

					doc_sim_matched_list.append([score*prob, i ] )





		doc_sim_matched_list.sort(reverse=True)

		for j in range(len(doc_sim_matched_list)):

			index = doc_sim_matched_list[j][-1]

			#print("d3", j)

			relevantList.append([0.3 - (0.3-0.1)*j/len(doc_sim_matched_list), index])

			alreadyFoundDic[index] = True



		relevantList.sort(reverse = True)



		return [{

			"prob": r[0],

			"paper_id": texts[r[1]][0]

		} for r in relevantList[offset:offset + limit]]







retrieve_bot = retrievalMethod2(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)

results = retrieve_bot.findClosestWithSeed("vaccine") 



eval_bot = evaluatorRobot()

eval_bot.loadPath(cleaner_bot.path)

eval_bot.presentResults(results)



retrieve_bot = retrievalMethod2(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)

results = retrieve_bot.findClosestWithSeed("drug") 



eval_bot = evaluatorRobot()

eval_bot.loadPath(cleaner_bot.path)

eval_bot.presentResults(results)



retrieve_bot = retrievalMethod2(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)

results = retrieve_bot.findClosestWithSeed("therapeutic") 



eval_bot = evaluatorRobot()

eval_bot.loadPath(cleaner_bot.path)

eval_bot.presentResults(results)



retrieve_bot = retrievalMethod2(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)

results = retrieve_bot.findClosestWithSeed("universal vaccine") 



eval_bot = evaluatorRobot()

eval_bot.loadPath(cleaner_bot.path)

eval_bot.presentResults(results)



retrieve_bot = retrievalMethod2(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)

results = retrieve_bot.findClosestWithSeed("prophylaxis") 



eval_bot = evaluatorRobot()

eval_bot.loadPath(cleaner_bot.path)

eval_bot.presentResults(results)



retrieve_bot = retrievalMethod2(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)

results = retrieve_bot.findClosestWithSeed("animal model") 



eval_bot = evaluatorRobot()

eval_bot.loadPath(cleaner_bot.path)

eval_bot.presentResults(results)



retrieve_bot = retrievalMethod2(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)

results = retrieve_bot.findClosestWithSeed("enhanced disease") 



eval_bot = evaluatorRobot()

eval_bot.loadPath(cleaner_bot.path)

eval_bot.presentResults(results)





retrieve_bot = retrievalMethod2(sim_bot.trained_model, cleaner_bot.dicOfFilterTexts, cleaner_bot.corpus, feat_bot)

results = retrieve_bot.findClosestWithSeed("prioritize") 



eval_bot = evaluatorRobot()

eval_bot.loadPath(cleaner_bot.path)

eval_bot.presentResults(results)








