#coding:utf-8

import pandas as pd

import numpy as np

import re

from sklearn.naive_bayes import BernoulliNB
def readData():

	SMS_df = pd.read_csv('../input/spam.csv',usecols=[0,1],encoding='latin-1')

	SMS_df.columns=['label','content']

	n = int(SMS_df.shape[0])

    # split into training data and test data

	return SMS_df.iloc[:int(n/2)], SMS_df.iloc[int(n/2):]
def generate_tabu_list(path, tabu_size=200,ignore=3):

	train_df,_ = readData()

	spam_TF_dict = dict()

	ham_TF_dict = dict()

	IDF_dict = dict()



	# ignore all other than letters.

	for i in range(train_df.shape[0]):

		finds = re.findall('[A-Za-z]+', train_df.iloc[i].content)

		if train_df.iloc[i].label == 'spam':

			for find in finds:

				if len(find)<ignore: continue

				find = find.lower()

				try:

					spam_TF_dict[find] = spam_TF_dict[find] + 1

				except:	

					spam_TF_dict[find] = spam_TF_dict.get(find,1)

					ham_TF_dict[find] = ham_TF_dict.get(find,0)

		else:

			for find in finds:

				if len(find)<ignore: continue

				find = find.lower()

				try:

					ham_TF_dict[find] = ham_TF_dict[find] + 1

				except:	

					spam_TF_dict[find] = spam_TF_dict.get(find,0)

					ham_TF_dict[find] = ham_TF_dict.get(find,1)

		

		word_set = set()

		for find in finds:

			if len(find)<ignore: continue

			find = find.lower()

			if not(find in word_set):

				try:

					IDF_dict[find] = IDF_dict[find] + 1

				except:	

					IDF_dict[find] = IDF_dict.get(find,1)

			word_set.add(find)



	word_df = pd.DataFrame(list(zip(ham_TF_dict.keys(),ham_TF_dict.values(),spam_TF_dict.values(),IDF_dict.values())))

	word_df.columns = ['keyword','ham_TF','spam_TF','IDF']

	word_df['ham_TF'] = word_df['ham_TF'].astype('float')/train_df[train_df['label']=='ham'].shape[0]

	word_df['spam_TF'] = word_df['spam_TF'].astype('float')/train_df[train_df['label']=='spam'].shape[0]

	word_df['IDF'] = np.log10(train_df.shape[0]/word_df['IDF'].astype('float'))

	word_df['ham_TFIDF'] = word_df['ham_TF']*word_df['IDF']

	word_df['spam_TFIDF'] = word_df['spam_TF']*word_df['IDF']

	word_df['diff']=word_df['spam_TFIDF']-word_df['ham_TFIDF']



	selected_spam_key = word_df.sort_values('diff',ascending=False)



	print('>>>Generating Tabu List...\n  Tabu List Size: {}\n  File Name: {}\n  The words shorter than {} are ignored by model\n'.format(tabu_size, path, ignore))

	file = open(path,'w')

	for word in selected_spam_key.head(tabu_size).keyword:

		file.write(word+'\n')

	file.close()
def read_tabu_list():

	file = open('tabu.txt','r')

	keyword_dict = dict()

	i = 0

	for line in file:

		keyword_dict.update({line.strip():i})

		i+=1

	return keyword_dict



def convert_Content(content, tabu):

	m = len(tabu)

	res = np.int_(np.zeros(m))

	finds = re.findall('[A-Za-z]+', content)

	for find in finds:

		find=find.lower()

		try:

			i = tabu[find]

			res[i]=1

		except:

			continue

	return res
def learn():

	global tabu, m

	train,_ = readData()

	n = train.shape[0]

	X = np.zeros((n,m)); Y=np.int_(train.label=='spam')

	for i in range(n):

		X[i,:] = convert_Content(train.iloc[i].content, tabu)



	NaiveBayes = BernoulliNB()

	NaiveBayes.fit(X, Y)



	Y_hat = NaiveBayes.predict(X)

	print('>>>Learning...\n  Learning Sample Size: {}\n  Accuarcy (Training sample): {:.2f}％\n'.format(n,sum(np.int_(Y_hat==Y))*100./n))

	return NaiveBayes
def test(NaiveBayes):

	global tabu, m

	_,test = readData()

	n = test.shape[0]

	X = np.zeros((n,m)); Y=np.int_(test.label=='spam')

	for i in range(n):

		X[i,:] = convert_Content(test.iloc[i].content, tabu)

	Y_hat = NaiveBayes.predict(X)

	print ('>>>Cross Validation...\n  Testing Sample Size: {}\n  Accuarcy (Testing sample): {:.2f}％\n'.format(n,sum(np.int_(Y_hat==Y))*100./n))

	return



def predictSMS(SMS):

	global NaiveBayes, tabu, m

	X = convert_Content(SMS, tabu)

	Y_hat = NaiveBayes.predict(X.reshape(1,-1))

	if int(Y_hat) == 1:

		print ('SPAM: {}'.format(SMS))

	else:

		print ('HAM: {}'.format(SMS))
print('UCI SMS SPAM CLASSIFICATION PROBLEM SET\n  -- implemented by Bernoulli Naive Bayes Model\n')

tabu_file = 'tabu.txt'          # user defined tabu file

tabu_size = 300                 # how many features are used to classify spam

word_len_ignored = 3            # ignore those words shorter than this variable

# build a tabu list based on the training data

generate_tabu_list(tabu_file,tabu_size, word_len_ignored)



tabu = read_tabu_list()

m = len(tabu)

# train the Naive Bayes Model using training data

NaiveBayes=learn()

# Test Model using testing data

test(NaiveBayes)

print('>>>Testing')

# I select two messages from the test data here.

predictSMS('Ya very nice. . .be ready on thursday')

predictSMS('Had your mobile 10 mths? Update to the latest Camera/Video phones for FREE. KEEP UR SAME NUMBER, Get extra free mins/texts. Text YES for a call')