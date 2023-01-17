import pandas as pd

# timing function

import numpy as np

import time   

start = time.clock() #_________________ measure efficiency timing



# read data

#test = pd.read_csv('../input/test.csv',encoding='utf8')[:100]

#test.fillna(value='leeg',inplace=True)

train = pd.read_csv('../input/train.csv',encoding='utf8')[:100]

print(train.head(2))

train.fillna(value='leeg',inplace=True)



import codecs, difflib, Levenshtein

from sklearn.metrics import jaccard_similarity_score

print('leven', Levenshtein.ratio(train.iloc[0]['question1'], train.iloc[0]['question2']) )

print('diffl', difflib.SequenceMatcher(None, train.iloc[0]['question1'], train.iloc[0]['question2']).ratio() )

print('jac', jaccard_similarity_score([10,20,10],[10,20,20],normalize=False) )





def euclidean_distance(x,y):

    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

def manhattan_distance(x,y):

    return sum(abs(a-b) for a,b in zip(x,y))



print ( manhattan_distance([10,20,10],[10,20,20]) )



from decimal import Decimal



def nth_root(value, n_root):

    root_value = 1/float(n_root)

    return round (Decimal(value) ** Decimal(root_value),3)



def minkowski_distance(x,y,p_value):

    return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),p_value)



print ( minkowski_distance([0,3,4,5],[7,6,3,-1],3) )

def square_rooted(x):

    return round(np.sqrt(sum([a*a for a in x])),3)



def cosine_similarity(x,y):

    numerator = sum(a*b for a,b in zip(x,y))

    denominator = square_rooted(x)*square_rooted(y)

    

    return round(numerator/float(denominator),3)



print ( cosine_similarity([3, 45, 7, 2], [2, 54, 13, 15]) )



def jaccard_similarity(x,y):

    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))

    union_cardinality = len(set.union(*[set(x), set(y)]))

    return intersection_cardinality/float(union_cardinality)



print ( jaccard_similarity([0,1,2,5,6],[0,2,3,5,7,9]) )



end = time.clock()

print('open:',end-start)
# -*- coding: utf-8 -*-

import numpy as np

import pandas as pd

import xgboost as xgb

import datetime

import operator

from sklearn.cross_validation import train_test_split

from collections import Counter

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

from pylab import plot, show, subplot, specgram, imshow, savefig

from nltk import word_tokenize

from scipy.spatial import distance



RS = 12357

ROUNDS = 315



print("Started")

np.random.seed(RS)

input_folder = '../input/'



def find_bigrams(input_list):

    lijst=list(zip(input_list, input_list[1:]))

    #remove stopwords on second position, do not remove starting stopwords

    lijst2=[tup for tup in lijst if tup[1] not in stopwords.words('english')]

    return lijst2



def BiJaccard(str1, str2):

    bigram1=set(find_bigrams(str1))

    bigram2=set(find_bigrams(str2))

    #print(bigram1,bigram2)

    return float( len( set(bigram1).intersection(bigram2) ) / len( set(bigram1).union(bigram2)) ) 



def jaro_winkler(ying, yang): #, long_tolerance, winklerize):

    winklerize=True

    long_tolerance=True

    ying_len = len(ying)

    yang_len = len(yang)



    if not ying_len or not yang_len:

        return 0.0



    min_len = max(ying_len, yang_len)

    search_range = (min_len // 2) - 1

    if search_range < 0:

        search_range = 0



    ying_flags = [False]*ying_len

    yang_flags = [False]*yang_len



    # looking only within search range, count & flag matched pairs

    common_chars = 0

    for i, ying_ch in enumerate(ying):

        low = i - search_range if i > search_range else 0

        hi = i + search_range if i + search_range < yang_len else yang_len - 1

        for j in range(low, hi+1):

            if not yang_flags[j] and yang[j] == ying_ch:

                ying_flags[i] = yang_flags[j] = True

                common_chars += 1

                break



    # short circuit if no characters match

    if not common_chars:

        return 0.0



    # count transpositions

    k = trans_count = 0

    for i, ying_f in enumerate(ying_flags):

        if ying_f:

            for j in range(k, yang_len):

                if yang_flags[j]:

                    k = j + 1

                    break

            if ying[i] != yang[j]:

                trans_count += 1

    trans_count /= 2



    # adjust for similarities in nonmatched characters

    common_chars = float(common_chars)

    weight = ((common_chars/ying_len + common_chars/yang_len +

              (common_chars-trans_count) / common_chars)) / 3



    # winkler modification: continue to boost if strings are similar

    if winklerize and weight > 0.7 and ying_len > 3 and yang_len > 3:

        # adjust for up to first 4 chars in common

        j = min(min_len, 4)

        i = 0

        while i < j and ying[i] == yang[i] and ying[i]:

            i += 1

        if i:

            weight += i * 0.1 * (1.0 - weight)



        # optionally adjust for long strings

        # after agreeing beginning chars, at least two or more must agree and

        # agreed characters must be > half of remaining characters

        if (long_tolerance and min_len > 4 and common_chars > i+1 and

                2 * common_chars >= min_len + i):

            weight += ((1.0 - weight) * (float(common_chars-i-1) / float(ying_len+yang_len-i*2+2)))



    return weight



def train_xgb(X, y, params):

	print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, RS))

	x, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RS)



	xg_train = xgb.DMatrix(x, label=y_train)

	xg_val = xgb.DMatrix(X_val, label=y_val)



	watchlist  = [(xg_train,'train'), (xg_val,'eval')]

	return xgb.train(params, xg_train, ROUNDS, watchlist)



def predict_xgb(clr, X_test):

	return clr.predict(xgb.DMatrix(X_test))



def create_feature_map(features):

	outfile = open('xgb.fmap', 'w')

	i = 0

	for feat in features:

		outfile.write('{0}\t{1}\tq\n'.format(i, feat))

		i = i + 1

	outfile.close()



def add_word_count(x, df, word):

	x['q1_' + word] = df['question1'].apply(lambda x: (word in str(x).lower())*1)

	x['q2_' + word] = df['question2'].apply(lambda x: (word in str(x).lower())*1)

	x[word + '_both'] = x['q1_' + word] * x['q2_' + word]



#starting parameters

params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'logloss'

params['eta'] = 0.11

params['max_depth'] = 5

params['silent'] = 1

params['seed'] = RS



df_train = pd.read_csv(input_folder + 'train.csv',encoding='utf8') [:50000]

df_test  = pd.read_csv(input_folder + 'test.csv',encoding='utf8')  [:50000]

df_train.fillna(value='leeg',inplace=True)

df_test.fillna(value='leeg',inplace=True)

print("Original data: X_train: {}, X_test: {}".format(df_train.shape, df_test.shape))


print("Features processing, be patient...")



	# If a word appears only once, we ignore it completely (likely a typo)

	# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller

def get_weight(count, eps=10000, min_count=2):

	return 0 if count < min_count else 1 / (count + eps)

    

    

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)

words = (" ".join(train_qs)).lower().split()

counts = Counter(words)

weights = {word: get_weight(count) for word, count in counts.items()}



stops = set(stopwords.words("english"))

def word_shares(row):



    q1_list = word_tokenize(row['question1'])

    q1 = set(q1_list)

    q1words = q1.difference(stops)

    if len(q1words) == 0:

        return '0:0:0:0:0:0:0:0:0:0:0'

        print(row['id'])

        

    q2_list = word_tokenize(row['question2']) #.lower().split()

    q2 = set(q2_list)

    q2words = q2.difference(stops)

    if len(q2words) == 0:

        return '0:0:0:0:0:0:0:0:0:0:0'

    

    BiJac=BiJaccard(list(q1),list(q2))

    JaroWink=jaro_winkler(row['question1'],row['question2'])

    words_hamming = sum(1 for i in zip(q1_list, q2_list) if i[0]==i[1])/max(len(q1_list), len(q2_list))



    #q1stops = q1.intersection(stops)

    #q2stops = q2.intersection(stops)



    q1_2gram = set([i for i in zip(q1_list, q1_list[1:])])

    q2_2gram = set([i for i in zip(q2_list, q2_list[1:])])

    shared_2gram = q1_2gram.intersection(q2_2gram)



		

    shared_words = q1words.intersection(q2words)

    differ_words = q1words.symmetric_difference(q2words)

	#total_words  = shared_words+differwords

    

    shared_weights = [weights.get(w, 0) for w in shared_words]

    differ_weights = [weights.get(w, 0) for w in differ_words]

    q1_weights = [weights.get(w, 0) for w in q1words]

    q2_weights = [weights.get(w, 0) for w in q2words]

		

    total_weights = q1_weights + q1_weights

		

    R1 = np.sum(shared_weights) / np.sum(total_weights) #tfidf share

    R2 = len(shared_words) / (len(q1words) + len(q2words) - len(shared_words)) #count share

    R31 = len(differ_words) / len(q1words) #stops in q1

    R32 = len(differ_words) / len(q2words) #stops in q2

    Rcosine_denominator = (np.sqrt(np.dot(q1_weights,q1_weights))*np.sqrt(np.dot(q2_weights,q2_weights)))

    Rcosine = np.dot(shared_weights, shared_weights)/Rcosine_denominator

    Rcosdif = np.dot(differ_weights, differ_weights)/Rcosine_denominator    

    if len(q1_2gram) + len(q2_2gram) == 0:

        R2gram = 0

    else:

        R2gram = len(shared_2gram) / (len(q1_2gram) + len(q2_2gram))

    return '{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R31, R32, R2gram, Rcosine, words_hamming,BiJac,JaroWink,Rcosdif)



df = pd.concat([df_train, df_test])

df['word_shares'] = df.apply(word_shares, axis=1, raw=True)

x = pd.DataFrame()



x['word_match']       = df['word_shares'].apply(lambda x: float(x.split(':')[0]))

x['word_match_2root'] = np.sqrt(x['word_match'])

x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))

x['shared_count']     = df['word_shares'].apply(lambda x: float(x.split(':')[2]))



x['stops1_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[3]))

x['stops2_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[4]))

x['shared_2gram']     = df['word_shares'].apply(lambda x: float(x.split(':')[5]))

x['cosine']           = df['word_shares'].apply(lambda x: float(x.split(':')[6]))

x['words_hamming']    = df['word_shares'].apply(lambda x: float(x.split(':')[7]))

x['bi_jacq']    = df['word_shares'].apply(lambda x: float(x.split(':')[8]))

x['Jaro_Wink']    = df['word_shares'].apply(lambda x: float(x.split(':')[9]))

x['Cosdiff']    = df['word_shares'].apply(lambda x: float(x.split(':')[10]))

x['diff_stops_r']     = x['stops1_ratio'] - x['stops2_ratio']



x['len_q1'] = df['question1'].apply(lambda x: len(str(x)))

x['len_q2'] = df['question2'].apply(lambda x: len(str(x)))

x['diff_len'] = x['len_q1'] - x['len_q2']

	

x['caps_count_q1'] = df['question1'].apply(lambda x:sum(1 for i in str(x) if i.isupper()))

x['caps_count_q2'] = df['question2'].apply(lambda x:sum(1 for i in str(x) if i.isupper()))

x['diff_caps'] = x['caps_count_q1'] - x['caps_count_q2']



x['len_char_q1'] = df['question1'].apply(lambda x: len(str(x).replace(' ', '')))

x['len_char_q2'] = df['question2'].apply(lambda x: len(str(x).replace(' ', '')))

x['diff_len_char'] = x['len_char_q1'] - x['len_char_q2']



x['len_word_q1'] = df['question1'].apply(lambda x: len(str(x).split()))

x['len_word_q2'] = df['question2'].apply(lambda x: len(str(x).split()))

x['diff_len_word'] = x['len_word_q1'] - x['len_word_q2']



x['avg_world_len1'] = x['len_char_q1'] / x['len_word_q1']

x['avg_world_len2'] = x['len_char_q2'] / x['len_word_q2']

x['diff_avg_word'] = x['avg_world_len1'] - x['avg_world_len2']



x['exactly_same'] = (df['question1'] == df['question2']).astype(int)

x['duplicated'] = df.duplicated(['question1','question2']).astype(int)

add_word_count(x, df,'how')

add_word_count(x, df,'what')

add_word_count(x, df,'which')

add_word_count(x, df,'who')

add_word_count(x, df,'where')

add_word_count(x, df,'when')

add_word_count(x, df,'why')



print(x.columns)

print(x.describe())



feature_names = list(x.columns.values)

create_feature_map(feature_names)

print("Features: {}".format(feature_names))



x_train = x[:df_train.shape[0]]

x_test  = x[df_train.shape[0]:]

y_train = df_train['is_duplicate'].values

del x, df_train



if 1: # Now we oversample the negative class - on your own risk of overfitting!

	pos_train = x_train[y_train == 1]

	neg_train = x_train[y_train == 0]

	print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))

	p = 0.165

	scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1

	while scale > 1:

		neg_train = pd.concat([neg_train, neg_train])

		scale -=1

	neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])

	print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))

	x_train = pd.concat([pos_train, neg_train])

	y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()

	del pos_train, neg_train

	
	print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(x_train.shape, len(y_train), x_test.shape))

	clr = train_xgb(x_train, y_train, params)

	preds = predict_xgb(clr, x_test)



	print("Writing output...")

	sub = pd.DataFrame()

	sub['test_id'] = df_test['test_id']

	sub['is_duplicate'] = preds *.75

	sub.to_csv("xgb_seed{}_n{}.csv".format(RS, ROUNDS), index=False)



	print("Features importances...")

	importance = clr.get_fscore(fmap='xgb.fmap')

	importance = sorted(importance.items(), key=operator.itemgetter(1))

	ft = pd.DataFrame(importance, columns=['feature', 'fscore'])



	ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))

	plt.gcf().savefig('features_importance.png')