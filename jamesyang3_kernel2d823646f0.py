import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from nltk.corpus import stopwords

import numpy as np

import sys

import string

import re

from nltk.stem import WordNetLemmatizer 



#global adjustable variables:

inputs = 0



# train_size = 0.90 #percentage of training data utilized

# random_state = 0

# max_features = 10000

# max_df = 0.95

# min_df = 2

#from the slides, J(A,B) = |A ∩ B| / |A ∪ B| = |A ∩ B| / (|A| + |B| - |A ∩ B|)

def jaccard(A, B): 

    a = set(A.lower().split()) 

    b = set(B.lower().split())

    c = a.intersection(b)



    j_value = float(len(c)) / (len(a) + len(b) - len(c))

    return j_value



#This function simply returns the number of sentiments for the entire training dataset.

def total_number_of_sentiments():

	#load in the data

	df_reviews = pd.read_csv("train.csv")

	#seperate positives, neutrals, negatives into an array call sentiments

	sentiments = df_reviews.groupby('sentiment')['textID'].nunique()

	

	#put them into variables

	total_pos = sentiments[0]

	total_neutral = sentiments[1]

	total_neg = sentiments[2]

	return total_pos, total_neutral, total_neg



# For program train_data, I have referenced Nick Koprowicz from his kaggle noteboook 

# (https://www.kaggle.com/nkoprowicz/a-simple-solution-using-only-word-counts)

# on his section for displaying the count vectorization vocabulary in an effort to make the results cleaner.

# His setup method is straight forward and very easy to adjust with my added global variable adjustments.



def load_and_train(train_size=None, random_state=None, max_feat=None, max_d=None, min_d=None):

	if train_size is None:

		train_size = 0.90

	if random_state is None:

		random_state = 0

	if max_feat is None:

		max_feat = 10000

	if max_d is None:

		max_d= 0.95

	if min_d is None:

		min_d = 2

	#Loads the values into variables

	train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

	test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

	sample = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

	#drops the NaN

	train[train['text'].isna()]

	train = train.dropna()

	train[train['text'].isna()]



	#################################################################

	# (These attempts did not assist in prediction value)

	# Attempt on filtering out junk characters as done in Implementation Assignment 2

	################################################################

	# filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

	# translate_dict = dict((c, " ") for c in filters)

	# translate_map = str.maketrans(translate_dict)

	# train['text'] = train['text'].str.translate(translate_map)

	################################################################

	#Filtering out http://

	# filters = 'http://'

	# translate_dict = dict((c, " ") for c in filters)

	# translate_map = str.maketrans(translate_dict)

	# train['text'] = train['text'].str.translate(translate_map)



	# Referencing Rajaram's Kaggle (https://www.kaggle.com/rajaram1988/ignored-stop-words-using-only-word-counts) to 

	# implement filtering out web links

	###############################################################





	train['text'] = train['text'].map(lambda x: re.sub('\\n',' ',str(x)))

    

	# remove any text starting with User... 

	train['text'] = train['text'].map(lambda x: re.sub("\[\[User.*",'',str(x)))

	    

	# remove IP addresses or user IDs

	train['text'] = train['text'].map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))

	    

	#remove http links in the text

	train['text'] = train['text'].map(lambda x: re.sub("(http://.*?\s)|(http://.*)",'',str(x)))

	###############################################################





	#Gets rid of case sensitivity so words like hi vs HI are the same.

	train['text'] = train['text'].str.lower()

	test['text'] = test['text'].str.lower()

	

	#splits the training data into what the values were above

	X_train, X_val = train_test_split(train, train_size = train_size, random_state = random_state)



	#simply organizes the sentiments into categories of positive, neutral, negative

	pos_train = X_train[X_train['sentiment'] == 'positive']

	neutral_train = X_train[X_train['sentiment'] == 'neutral']

	neg_train = X_train[X_train['sentiment'] == 'negative']



	cv = CountVectorizer(max_df=max_d, min_df=min_d, max_features=max_feat, stop_words='english')



	X_train_cv = cv.fit_transform(X_train['text'].values.astype('U'))



	X_pos = cv.transform(pos_train['text'])

	X_neutral = cv.transform(neutral_train['text'])

	X_neg = cv.transform(neg_train['text'])



	pos_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())

	neutral_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())

	neg_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())



	pos_words = {}

	neutral_words = {}

	neg_words = {}



	for k in cv.get_feature_names():

	    pos = pos_count_df[k].sum()

	    neutral = neutral_count_df[k].sum()

	    neg = neg_count_df[k].sum()

	    

	    pos_words[k] = pos/pos_train.shape[0]

	    neutral_words[k] = neutral/neutral_train.shape[0]

	    neg_words[k] = neg/neg_train.shape[0]

	    

	# We need to account for the fact that there will be a lot of words used in tweets of every sentiment.  

	# Therefore, we reassign the values in the dictionary by subtracting the proportion of tweets in the other 

	# sentiments that use that word.



	neg_words_adj = {}

	pos_words_adj = {}

	neutral_words_adj = {}



	for key, value in neg_words.items():

	    neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])



	for key, value in pos_words.items():

	    pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])

	    

	for key, value in neutral_words.items():

	    neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])



	return X_val, pos_words_adj, neg_words_adj, neutral_words_adj, train, test, sample

# For program calculation, I have referenced Nick Koprowicz from his kaggle noteboook 

# (https://www.kaggle.com/nkoprowicz/a-simple-solution-using-only-word-counts)

# on his section for calculating the text. The way of calculating the text cannot vary and his method was clean

# in implementing the positive and negative word weights. 

def calculate_selected_text(df_row, pos_words_adj, neg_words_adj, neutral_words_adj, tol):

    

    tweet = df_row['text']

    sentiment = df_row['sentiment']

    

    if(sentiment == 'neutral'):

        return tweet

    

    elif(sentiment == 'positive'):

        dict_to_use = pos_words_adj # Calculate word weights using the pos_words dictionary

    elif(sentiment == 'negative'):

        dict_to_use = neg_words_adj # Calculate word weights using the neg_words dictionary

        

    words = tweet.split()

    words_len = len(words)

    subsets = [words[i:j+1] for i in range(words_len) for j in range(i,words_len)]

    

    score = 0

    selection_str = '' # This will be our choice

    lst = sorted(subsets, key = len) # Sort candidates by length

    

    #Calculating the 

    for i in range(len(subsets)):

        

        new_sum = 0 # Sum for the current substring

        

        # Calculate the sum of weights for each word in the substring

        for p in range(len(lst[i])):

            if(lst[i][p].translate(str.maketrans('','',string.punctuation)) in dict_to_use.keys()):

            	new_sum += dict_to_use[lst[i][p].translate(str.maketrans('','',string.punctuation))]

            	after_character = 0

            	before_character = 0

            if (inputs == 1):  # If there are inputs.

                    # Example of the for loop: "This guy cool"



                    # Calculate the second word to impact the first word:

                if (p + 1 < len(lst[i]) and p - 1 > 0):



                    # Ex: Word selected at lst[i][p] = "guy"

                    # after_character = "cool"

                    # before_character = "This"



                    after_character = lst[i][p + 1].translate(

                        str.maketrans('', ''.string.punctuation)) in dict_to_use.keys()

                    before_character = lst[i][p - 1].translate(

                        str.maketrans('', ''.string.punctuation)) in dict_to_use.keys()



                    # Calculate the weight of each of the words individually and determine which sum is greater.

                    if (abs(after_character) > abs(before_character)):  # If the weight of the word is more

                        new_sum += dict_to_use[lst[i][p].translate(

                            str.maketrans('', '', string.punctuation))] + after_character



                    elif (abs(after_characvter) < abs(before_character)):  # If the weight of the other word is more

                        new_sum += dict_to_use[lst[i][p].translate(

                            str.maketrans('', '', string.punctuation))] + before_character



                    else:  # if the weight is 0 (edge words in the tweet) #otherwise its on the edge

                        new_sum += dict_to_use[lst[i][p].translate(str.maketrans('', '', string.punctuation))]

        # If the sum is greater than the score, update our current selection

        if(new_sum > score + tol):

            score = new_sum

            selection_str = lst[i]

            tol = tol*5 # Increase the tolerance a bit each time we choose a selection



    # If we didn't find good substrings, return the whole text

    if(len(selection_str) == 0):

        selection_str = words

        

    return ' '.join(selection_str)



# For program calculation, I have referenced Nick Koprowicz from his kaggle noteboook 

# (https://www.kaggle.com/nkoprowicz/a-simple-solution-using-only-word-counts)

# on his section for calculating the jaccard. The way of calculating the text cannot vary and his method was clean

# in implementing the positive and negative word weights. 



def calculation(train_size=None, random_state=None, max_features=None, max_df=None, min_df=None):

	if train_size is None:

		train_size = 0.90

	if random_state is None:

		random_state = 0

	if max_features is None:

		max_feat = 10000

	if max_df is None:

		max_df= 0.95

	if min_df is None:

		min_df = 2

	X_val, pos_words_adj, neg_words_adj, neutral_words_adj = load_and_train(train_size, random_state, max_features, max_df, min_df)



	pd.options.mode.chained_assignment = None

	tol = 0.001



	X_val['predicted_selection'] = ''



	for index, row in X_val.iterrows():

	    

	    selected_text = calculate_selected_text(row, pos_words_adj, neg_words_adj, neutral_words_adj, 3)

	    

	    X_val.loc[X_val['textID'] == row['textID'], ['predicted_selection']] = selected_text



	X_val['jaccard'] = X_val.apply(lambda x: jaccard(x['selected_text'], x['predicted_selection']), axis = 1)

	print("For train_size: ", train_size)

	print('The jaccard score for the validation set is:', np.mean(X_val['jaccard']))

	return np.mean(X_val['jaccard'])



def run_output(train, test, sample):

	pos_tr = train[train['sentiment'] == 'positive']

	neutral_tr = train[train['sentiment'] == 'neutral']

	neg_tr = train[train['sentiment'] == 'negative']

	cv = CountVectorizer(max_df=0.95, min_df=2,

                                     max_features=10000,

                                     stop_words='english')

	final_cv = cv.fit_transform(train['text'])



	X_pos = cv.transform(pos_tr['text'])

	X_neutral = cv.transform(neutral_tr['text'])

	X_neg = cv.transform(neg_tr['text'])



	pos_final_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())

	neutral_final_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())

	neg_final_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())



	pos_words = {}

	neutral_words = {}

	neg_words = {}



	for k in cv.get_feature_names():

	    pos = pos_final_count_df[k].sum()

	    neutral = neutral_final_count_df[k].sum()

	    neg = neg_final_count_df[k].sum()

	    

	    pos_words[k] = pos/(pos_tr.shape[0])

	    neutral_words[k] = neutral/(neutral_tr.shape[0])

	    neg_words[k] = neg/(neg_tr.shape[0])



	neg_words_adj = {}

	pos_words_adj = {}

	neutral_words_adj = {}



	for key, value in neg_words.items():

	    neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])

	    

	for key, value in pos_words.items():

	    pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])

	    

	for key, value in neutral_words.items():

	    neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])



	tol = 0.001



	for index, row in test.iterrows():

	    

	    selected_text = calculate_selected_text(row, pos_words_adj, neg_words_adj, neutral_words_adj, tol)

	    

	    sample.loc[sample['textID'] == row['textID'], ['selected_text']] = selected_text





	sample.to_csv('submission.csv', index = False)



def run_multiple_times():

	best = 0

	# for train_size in np.arange(0.1, 0.9, 0.1): #doesn't do much

	# 	temp = calculation(train_size)

	# 	if best < temp:

	# 		best = temp

	# print("best train size is: ", best)

	# for max_features in np.arange(10000,20000,2500):

	# 	temp = calculation(None, None, max_features)

	# 	if best < temp:

	# 		best = temp

	# print("best max_feature size is: ", best)

	# calculation()

	X_val, pos_words_adj, neg_words_adj, neutral_words_adj, train, test, sample = load_and_train()

	run_output(train, test, sample)

run_multiple_times()

# load_and_split_data()