from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# package for regular expressions

import re
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

# identify files

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# view a sample

nRowsRead = None # 1000 # specify 'None' if want to read whole file

# fake_or_real_news.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows



df = pd.read_csv('/kaggle/input/fake_or_real_news.csv', delimiter=',', nrows = nRowsRead)

df.dataframeName = 'fake_or_real_news.csv'

df.columns = ['rec', 'title', 'text', 'label']

df = df.set_index('rec')



# shuffle

r,c = df.shape

df1 = df.sample( r )



# view 10 first listed

df1.head(10)
# visualize data set label values

plotPerColumnDistribution(df1, 10, 5)
sample_size = 5000
def clean_special_char(text) : 

    cleaned = text

    cleaned = re.sub(r'[\n]+', ' ', cleaned)

    cleaned = re.sub(r'\x91\x92\x96', '', cleaned)

    

    # substitute anything not ([^...]) alpha (A-Za-z) with '' 

    cleaned = re.sub('[^A-Za-z ]+', '', cleaned)

    # OR

    # substitute anything not ([^...]) alpha (A-Za-z) numeric (0-9) with ''

    # cleaned = re.sub('[^A-Za-z0-9 ]+', '', cleaned)

    

    return cleaned
# gets unique words from sampled text

def get_unique_words(df, field, num_words=1000, verbose=False) :



    # get texts

    data_word_lists = df[field].apply(lambda x : np.unique(x.split()[:num_words])).values



    # get cleaned words

    data_word_list = [  clean_special_char(y) for x in data_word_lists for y in x]

    # verify

    if verbose : print( 'total words found:', len(data_word_list) )



    # get unique words

    unique_words = np.unique(np.array(data_word_list))

    print('unique words found : ', len(unique_words))

    

    # verify

    if verbose : print( '100 of the unique words :', unique_words[:100] )

    

    return data_word_lists, data_word_list, unique_words



# bar graph of number of occurances of set of 'words' in list of list of title/text words

def bar_word_occ(text, words, occ_threshold) :

    

    # get number of occurances of unique words (not including spaces/blanks)

    wrd_cnt = [ [x, text.count(x)] for x in words if text.count(x) > occ_threshold if x != '']



    # extract words and counts

    wrd = np.array([ t[0] for t in wrd_cnt ])

    cnt = np.array([ t[1] for t in wrd_cnt ])



    # get sort order by count

    sort_order = np.argsort(cnt)



    # get ordered lists of words and counts

    w = wrd[sort_order]

    c = cnt[sort_order]



    # plot it

    plt.figure(figsize=(70,8))

    plt.bar(w, c)

    

    # labels

    plt.title('Real/Fake News Word Occurance (>' + str(occ_threshold) + ')' )

    

# histogram of the frequency of unique words in passed in list of lists

def hist_text_lengths(lolist, title_prefix='') :

    # get word counts

    word_lengths = np.array([ len(x) for x in lolist ] )



    # # verify

    # print( 'first 100\'s word count :', word_lengths[:100])

        

    twlhist = plt.hist(word_lengths)

    plt.title(title_prefix + 'Word Count Distribution')
# sample real news entries

reals = df1[df1.label=='REAL'][['title', 'text']].iloc[:sample_size]

reals
print('real news title - unique words')

real_titles, real_title_words, real_title_uniques = get_unique_words(reals, 'title')
bar_word_occ(real_title_words, real_title_uniques, 100)
hist_text_lengths(real_titles, 'Real Title ')
print('real news text - unique words')

real_texts, real_text_words, real_text_uniques = get_unique_words(reals, 'text')
# To reduce computational complexity : 

# sample onlyt 2000 texts from the original sample

bar_word_occ(real_text_words[:2000], real_text_uniques, 5)



# get actual word count for sample (requires lots of computation)

# bar_word_occ(real_text_words, real_text_uniques, 50)
hist_text_lengths(real_texts, 'Real Texts ')
fakes = df1[df1.label=='FAKE'][['title', 'text']]

fakes
print('fake news title - unique words')

fake_titles, fake_title_words, fake_title_uniques = get_unique_words(fakes, 'title')
# get number of occurances of unique words

bar_word_occ(fake_title_words, fake_title_uniques, 100)
hist_text_lengths(fake_titles, 'Fake Titles ')
print('fake news text - unique words')

fake_texts, fake_text_words, fake_text_uniques = get_unique_words(fakes, 'text')
hist_text_lengths(fake_texts, 'Fake Texts ')
# real vs fake title uniques

fakes_title_exclusives = [ x for x in fake_title_uniques if x not in real_title_uniques ]

print( 'fake title exclusive unique words found: {}'.format(len(fakes_title_exclusives)) )

print( '100 of the fake title exclusive unique words :', fakes_title_exclusives[:100])
# real vs fake text uniques

fakes_text_exclusives = [ x for x in fake_text_uniques if x not in real_text_uniques ]

print( 'fake text exclusive words found:', len(fakes_text_exclusives))

print( '100 of the fake text exclusive words :', fakes_text_exclusives[:100])
import time



# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split
# make dictionary of unique words and [0,1] or [0, len(unique words)] index words 

def make_uniques_dict(word_dictionary, normalized=False) :

    

    wdic = {}

    i = 0

    d = len(word_dictionary)



    for w in word_dictionary :

        

        value = i/d if normalized else d

        

        wdic.update({w: i})

        i += 1

        

    return wdic



def encode_one_hot(news_feat, dic, verbose=False) :

    

    x_matrix = np.zeros((len(news_feat), len(dic)))



    for t in range(len(news_feat)) :

        for w in news_feat[t] :

    

            w = clean_special_char(w)

            

            try :

                x_matrix[t, dic[w] ] = 1

            except:

                x_matrix[t, 0 ] = 1

                    

        if verbose : print(t)

    

    return x_matrix
# get target values

y_values = np.array(df.label.apply(lambda x : 1 if x=="REAL" else 0).values)



# split

x_train, x_test, y_train, y_test = train_test_split(df[['title', 'text']], y_values, test_size=0.2, random_state=3611)
# get title words and uniques

titles, title_words, title_uniques = get_unique_words(x_train, 'title')



# build title unique word dictionary

tw_dic = make_uniques_dict(title_uniques)



# encode

x_train_titles = encode_one_hot(titles, tw_dic)
# # verify

# print('num records"', len(x_titles) )

# print('first ten titles and [01] encoded titles:')

# # verify

# for i in x_titles[:10] :

# #     print(titles[00], end=' : ')

#     for j in i :

#         if j > 0 : 

#             print(np.round(j, 3), end=', ')

#     print('')
# make classifier

# X, y = make_classification(n_samples=100, random_state=1)

# x_train, x_test, y_train, y_test = train_test_split(x_titles, y_, test_size=0.2, random_state=3611)



stime = time.process_time()

clf = MLPClassifier(random_state=1, batch_size=(2110), max_iter=1000, verbose=True ).fit(x_train_titles, y_train)

# clf = MLPClassifier(random_state=1, hidden_layer_sizes=(1000), batch_size=(2110), max_iter=1000, verbose=True ).fit(x_train, y_train)

print('proc time:', time.process_time() - stime)
#test



# assume new titles are unknown

titles, title_words, title_uniques = get_unique_words(x_test, 'title')



# do not build text unique word dictionary. use training set dictionary.



# encode

x_test_titles = encode_one_hot(titles, tw_dic)



# test (with unknowns)

clf.score(x_test_titles, y_test)
# To reduce computational complexity : 

# use (maximum) only the first 100 words of a news text

x_train['text_cropped'] = x_train['text'].apply(lambda x : x[:300])



# build text word dictionary

texts, texts_words, texts_uniques = get_unique_words(x_train, 'text_cropped')



# build text unique word dictionary

xw_dic = make_uniques_dict(texts_uniques)



# encode

x_train_text = encode_one_hot(texts, xw_dic)
stime = time.process_time()

clf = MLPClassifier(random_state=1, batch_size=(100), max_iter=1000, verbose=True ).fit(x_train_text, y_train)

print('proc time:', time.process_time() - stime)
# To reduce computational complexity : 

# use (maximum) only the first 100 words of a news text

x_test['text_cropped'] = x_test['text'].apply(lambda x : x[:300])



# assume new titles are unknown

texts, texts_words, texts_uniques = get_unique_words(x_test, 'text_cropped')



# do not build text unique word dictionary. use training set dictionary.



# encode

x_test_texts = encode_one_hot(texts, xw_dic)



# test

clf.score(x_test_texts, y_test)
# concatonate news title and text to new column 'titletext'

x_train['titletext'] = x_train.title + ' ' + x_train.text

# df1['titletext'] = df1.title.apply(lambda x : x.split()) + df1.text.apply(lambda x : x.split()[:100])



# build title+text word dictionary

titlestexts, titlestexts_words, titlestexts_uniques = get_unique_words(x_train, 'titletext', 100)



# build title+text unique word dictionary

titlestextsw_dic = make_uniques_dict(titlestexts_uniques)



# encode

x_train_titlestexts = encode_one_hot(titlestexts, titlestextsw_dic)
stime = time.process_time()

clf = MLPClassifier(random_state=1, batch_size=(2110), max_iter=1000, verbose=True ).fit(x_train_titlestexts, y_train)

# clf = MLPClassifier(random_state=1, hidden_layer_sizes=(1000), batch_size=(2110), max_iter=1000, verbose=True ).fit(x_train, y_train)

print('proc time:', time.process_time() - stime)
# concatonate news title and text to new column 'titletext'

x_test['titletext'] = x_test.title + ' ' + x_test.text



# assume new titles are unknown

titlestexts, titlestexts_words, titlestexts_uniques = get_unique_words(x_test, 'titletext')



# do not build text unique word dictionary. use training set dictionary.



# encode

x_test_titlestexts = encode_one_hot(titlestexts, titlestextsw_dic)



clf.score(x_test_titlestexts, y_test)