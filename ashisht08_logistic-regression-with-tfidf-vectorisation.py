# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import model_selection

import nltk

from nltk.corpus import stopwords

import string

from collections import defaultdict

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score

from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import csr_matrix, hstack

import math

from matplotlib import colors



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

sub_df = pd.read_csv('../input/sample_submission.csv')

print("Train shape : ", train_df.shape)

print("Test shape : ", test_df.shape)

print("Submission shape : ", sub_df.shape)

# train_df.head()

# test_df.head()



# Copy train_df

train_df_copy = train_df.copy()
# Drop qid

train_df.drop(columns="qid", inplace=True)

# test_df.drop(columns="qid", inplace=True)
def ngram_extractor(text, n_gram):

    token = [token for token in text.lower().strip(string.punctuation).split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



# Function to generate a dataframe with n_gram and top max_row frequencies

def generate_ngrams(df, col, n_gram, max_row):

    temp_dict = defaultdict(int)

    for question in df[col]:

        for word in ngram_extractor(question, n_gram):

            temp_dict[word] += 1

    temp_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda x: x[1])[::-1]).head(max_row)

    temp_df.columns = ["word", "wordcount"]

    return temp_df



def comparison_plot(df_1,df_2,col_1,col_2, space, ngram):

    plt.rcParams.update({'font.size': 14})  

    fig, ax = plt.subplots(1, 2, figsize=(20,5))

#     ax[0].set(xscale="log")

#     ax[1].set(xscale="log")

    sns.barplot(x=col_2, y=col_1, data=df_1, ax=ax[0], color=colors.CSS4_COLORS.get('lightblue'))

    sns.barplot(x=col_2, y=col_1, data=df_2, ax=ax[1], color=colors.CSS4_COLORS.get('lightsalmon'))



    ax[0].set_xlabel('Word count', size=14, color="black")

    ax[0].set_ylabel(ngram.capitalize(), size=14, color="black")

    ax[0].set_title('Top '+ ngram + ' in sincere questions', size=15)



    ax[1].set_xlabel('Word count', size=14, color="black")

    ax[1].set_ylabel(ngram.capitalize(), size=14, color="black")

    ax[1].set_title('Top '+ ngram + ' in insincere questions', size=15)



    #fig.subplots_adjust(wspace=space)

    fig.tight_layout()

    

    plt.show()

    fig.savefig(ngram + '.pdf', format='pdf')



sincere_1gram = generate_ngrams(train_df[train_df["target"]==0], 'question_text', 1, 20)

insincere_1gram = generate_ngrams(train_df[train_df["target"]==1], 'question_text', 1, 20)



comparison_plot(sincere_1gram,insincere_1gram,'word','wordcount', 0.25, "unigrams")



sincere_2gram = generate_ngrams(train_df[train_df["target"]==0], 'question_text', 2, 20)

insincere_2gram = generate_ngrams(train_df[train_df["target"]==1], 'question_text', 2, 20)



comparison_plot(sincere_2gram,insincere_2gram,'word','wordcount', 0.25, "bigrams")



sincere_3gram = generate_ngrams(train_df[train_df["target"]==0], 'question_text', 3, 20)

insincere_3gram = generate_ngrams(train_df[train_df["target"]==1], 'question_text', 3, 20)



comparison_plot(sincere_3gram,insincere_3gram,'word','wordcount', 0.25, "trigrams")
misspells = ["ain't", "aren't", "can't", "'cause", "could've", "couldn't", "didn't", 

    "doesn't", "don't", "hadn't", "hasn't", "haven't", "he'd", "he'll", "he's", "how'd", 

    "how'd'y", "how'll", "how's", "i'd", "i'd've", "i'll", "i'll've", "i'm", "i've", "isn't", 

    "it'd", "it'd've", "it'll", "it'll've", "it's", "let's", "ma'am", "mayn't", "might've", 

    "mightn't", "mightn't've", "must've", "mustn't", "mustn't've", "needn't", "needn't've", 

    "o'clock", "oughtn't", "oughtn't've", "shan't", "sha'n't", "shan't've", "she'd", "she'd've", 

    "she'll", "she'll've", "she's", "should've", "shouldn't", "shouldn't've", "so've", "so's", 

    "this's", "that'd", "that'd've", "that's", "there'd", "there'd've", "there's", "here's", 

    "they'd", "they'd've", "they'll", "they'll've", "they're", "they've", "to've", "wasn't", 

    "we'd", "we'd've", "we'll", "we'll've", "we're", "we've", "weren't", "what'll", "what'll've", 

    "what're", "what's", "what've", "when's", "when've", "where'd", "where's", "where've", "who'll", 

    "who'll've", "who's", "who've", "why's", "why've", "will've", "won't", "won't've", "would've", 

    "wouldn't", "wouldn't've", "y'all", "y'all'd", "y'all'd've", "y'all're", "y'all've", "you'd", 

    "you'd've", "you'll", "you'll've", "you're", "you've", "colour", "centre", "favourite", 

    "travelling", "counselling", "theatre", "cancelled", "labour", "organisation", "wwii", 

    "citicise", "youtu ", "qoura", "sallary", "whta", "narcisist", "howdo", "whatare", "howcan", 

    "howmuch", "howmany", "whydo", "doi", "thebest", "howdoes", "mastrubation", "mastrubate", 

    "mastrubating", "pennis", "etherium", "narcissit", "bigdata", "2k17", "2k18", "qouta", 

    "exboyfriend", "airhostess", "whst", "watsapp", "demonitisation", "demonitization", "demonetisation",

    'demonitization']



# List Of Bad Words by Google-Profanity Words 

bad_words = ['cockknocker', 'n1gger', 'ing', 'fukker', 'nympho', 'fcuking', 'gook', 'freex', 

             'arschloch', 'fistfucked', 'chinc', 'raunch', 'fellatio', 'splooge',

             'nutsack', 'lmfao', 'wigger', 'bastard', 'asses', 'fistfuckings', 'blue', 'waffle', 

             'beeyotch', 'pissin', 'dominatrix', 'fisting', 'vullva', 'paki', 'cyberfucker', 'chuj',

             'penuus', 'masturbate', 'b00b*', 'fuks', 'sucked', 'fuckingshitmotherfucker', 'feces', 'panty', 

             'coital', 'wh00r.', 'whore', 'condom', 'hells', 'foreskin', 'wanker', 'hoer', 'sh1tz', 'shittings', 

             'wtf', 'recktum', 'dick*', 'pr0n', 'pasty', 'spik', 'phukked', 'assfuck', 'xxx', 'nigger*', 'ugly',

             's_h_i_t', 'mamhoon', 'pornos', 'masterbates', 'mothafucks', 'Mother', 'Fukkah', 'chink', 'pussy', 

             'palace', 'azazel', 'fistfucking', 'ass-fucker', 'shag', 'chincs', 'duche', 'orgies', 'vag1na', 'molest', 

             'bollock', 'a-hole', 'seduce', 'Cock*', 'dog-fucker', 'shitz', 'Mother', 'Fucker', 'penial', 'biatch',

             'junky', 'orifice', '5hit', 'kunilingus', 'cuntbag', 'hump', 'butt', 'fuck', 'titwank', 'schaffer', 

             'cracker', 'f.u.c.k', 'breasts', 'd1ld0', 'polac', 'boobs', 'ritard', 'fuckup', 'rape', 'hard', 'on', 

             'skanks', 'coksucka', 'cl1t', 'herpy', 's.o.b.', 'Motha', 'Fucker', 'penus', 'Fukker', 'p.u.s.s.y.', 

             'faggitt', 'b!tch', 'doosh', 'titty', 'pr1k', 'r-tard', 'gigolo', 'perse', 'lezzies', 'bollock*', 

             'pedophiliac', 'Ass', 'Monkey', 'mothafucker', 'amcik', 'b*tch', 'beaner', 'masterbat*', 'fucka', 

             'phuk', 'menses', 'pedophile', 'climax', 'cocksucking', 'fingerfucked', 'asswhole', 'basterdz',

             'cahone', 'ahole', 'dickflipper', 'diligaf', 'Lesbian', 'sperm', 'pisser', 'dykes', 'Skanky',

             'puuker', 'gtfo', 'orgasim', 'd0ng', 'testicle*', 'pen1s', 'piss-off', '@$$', 'fuck', 'trophy', 

             'arse*', 'fag', 'organ', 'potty', 'queerz', 'fannybandit', 'muthafuckaz', 'booger', 'pussypounder',

             'titt', 'fuckoff', 'bootee', 'schlong', 'spunk', 'rumprammer', 'weed', 'bi7ch', 'pusse', 'blow', 'job', 

             'kusi*', 'assbanged', 'dumbass', 'kunts', 'chraa', 'cock', 'sucker', 'l3i+ch', 'cabron', 'arrse', 'cnut', 

             'murdep', 'fcuk', 'phuked', 'gang-bang', 'kuksuger', 'mothafuckers', 'ghey', 'clit', 'licker', 'feg', 

             'ma5terbate', 'd0uche', 'pcp', 'ejaculate', 'nigur', 'clits', 'd0uch3', 'b00bs', 'fucked', 'assbang', 

             'mutha', 'goddamned', 'cazzo', 'lmao', 'godamn', 'kill', 'coon', 'penis-breath', 'kyke', 'heshe', 'homo',

             'tawdry', 'pissing', 'cumshot', 'motherfucker', 'menstruation', 'n1gr', 'rectus', 'oral', 'twats', 

             'scrot', 'God', 'damn', 'jerk', 'nigga', 'motherfuckin', 'kawk', 'homey', 'hooters', 'rump', 

             'dickheads', 'scrud', 'fist', 'fuck', 'carpet', 'muncher', 'cipa', 'cocaine', 'fanyy', 'frigga', 

             'massa', '5h1t', 'brassiere', 'inbred', 'spooge', 'shitface', 'tush', 'Fuken', 'boiolas', 'fuckass', 'wop*',

             'cuntlick', 'fucker', 'bodily', 'bullshits', 'hom0', 'sumofabiatch', 'jackass', 'dilld0', 'puuke', 'cums', 

             'pakie', 'cock-sucker', 'pubic', 'pron', 'puta', 'penas', 'weiner', 'vaj1na', 'mthrfucker', 'souse', 'loin',

             'clitoris', 'f.ck', 'dickface', 'rectal', 'whored', 'bookie', 'chota', 'bags', 'sh!t', 'pornography', 'spick', 'seamen',

             'Phukker', 'beef', 'curtain', 'eat', 'hair', 'pie', 'mother', 'fucker', 'faigt', 'yeasty', 'Clit', 'kraut', 'CockSucker', 

             'Ekrem*', 'screwing', 'scrote', 'fubar', 'knob', 'end', 'sleazy', 'dickwhipper', 'ass', 'fuck', 'fellate', 'lesbos', 

             'nobjokey', 'dogging', 'fuck', 'hole', 'hymen', 'damn', 'dego', 'sphencter', 'queef*', 'gaylord', 'va1jina', 'a55', 

             'fuck', 'douchebag', 'blowjob', 'mibun', 'fucking', 'dago', 'heroin', 'tw4t', 'raper', 'muff', 'fitt*', 'wetback*',

             'mo-fo', 'fuk*', 'klootzak', 'sux', 'damnit', 'pimmel', 'assh0lez', 'cntz', 'fux', 'gonads', 'bullshit', 'nigg3r', 

             'fack', 'weewee', 'shi+', 'shithead', 'pecker', 'Shytty', 'wh0re', 'a2m', 'kkk', 'penetration', 'kike', 'naked', 

             'kooch', 'ejaculation', 'bang', 'hoare', 'jap', 'foad', 'queef', 'buttwipe', 'Shity', 'dildo', 'dickripper', 

             'crackwhore', 'beaver', 'kum', 'sh!+', 'qweers', 'cocksuka', 'sexy', 'masterbating', 'peeenus', 'gays', 

             'cocksucks', 'b17ch', 'nad', 'j3rk0ff', 'fannyflaps',

             'God-damned', 'masterbate', 'erotic', 'sadism', 'turd', 'flipping', 'the', 'bird',

             'schizo', 'whiz', 'fagg1t', 'cop', 'some', 

             'wood', 'banger', 'Shyty', 'f', 'you', 'scag', 'soused', 'scank',

             'clitorus', 'kumming', 'quim', 'penis', 'bestial', 'bimbo', 'gfy',

             'spiks', 'shitings', 'phuking', 'paddy', 'mulkku', 'anal', 

             'leakage', 'bestiality', 'smegma', 'bull', 'shit', 'pillu*', 'schmuck',

             'cuntsicle', 'fistfucker', 'shitdick', 'dirsa', 'm0f0']



stopwords = set(stopwords.words('english'))
def get_features(dataframe):

    dataframe["text_size"] = dataframe["question_text"].apply(len).astype('uint16')

    dataframe["capital_size"] = dataframe["question_text"].apply(lambda x: sum(1 for c in x if c.isupper())).astype('uint16')

    dataframe["capital_rate"] = dataframe.apply(lambda x: float(x["capital_size"]) / float(x["text_size"]), axis=1).astype('float16')

    dataframe["exc_count"] = dataframe["question_text"].apply(lambda x: x.count("!")).astype('uint16')

    dataframe["question_count"] = dataframe["question_text"].apply(lambda x: x.count("?")).astype('uint16')

    dataframe["unq_punctuation_count"] = dataframe["question_text"].apply(lambda x: sum(x.count(p) for p in '∞θ÷α•à−β∅³π‘₹´°£€\×™√²')).astype('uint16')

    dataframe["symbol_count"] = dataframe["question_text"].apply(lambda x: sum(x.count(p) for p in '*&$%')).astype('uint16')

    dataframe["words_count"] = dataframe["question_text"].apply(lambda x: len(x.split())).astype('uint16')

    dataframe["unique_words"] = dataframe["question_text"].apply(lambda x: (len(set(x.split())))).astype('uint16')

    dataframe["unique_rate"] = dataframe["unique_words"] / dataframe["words_count"]

    dataframe["word_max_length"] = dataframe["question_text"].apply(lambda x: max([len(word) for word in x.split()]) ).astype('uint16')  # useless

    dataframe["mistake_count"] = dataframe["question_text"].apply(lambda x: sum(x.lower().count(w) for w in misspells)).astype('uint16')

    dataframe['num_stopwords'] = dataframe["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords])).astype('uint16')

    dataframe['num_punctuation_chars'] = dataframe["question_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation])).astype('uint16')

    dataframe['num_word_tokens_Title'] = dataframe["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()])).astype('uint16')

    dataframe['avg_word_token_length'] = dataframe["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()])).astype('uint16')

    dataframe['num_smilies'] = dataframe['question_text'].apply(lambda x: sum(x.count(w) for w in (':-)', ':)', ';-)', ';)')))

    dataframe['num_sad'] = dataframe['question_text'].apply(lambda x: sum(x.count(w) for w in (':-<', ':()', ';-()', ';(')))

    dataframe["badwordcount"] = dataframe['question_text'].apply(lambda comment: sum(comment.count(w) for w in bad_words))

    dataframe['num_chars'] =    dataframe['question_text'].apply(len)

    dataframe["normchar_badwords"] = dataframe["badwordcount"]/dataframe['num_chars']

    dataframe["normword_badwords"] = dataframe["badwordcount"]/dataframe['text_size']

    return dataframe
# # Get features

# get_features(train_df)



# # Split training data into train and validaiton sets (70/20/10)

# train_x, temp_x, train_y, temp_y = model_selection.train_test_split(train_df.drop(columns='target'), train_df['target'], test_size=0.3)

# valid_x, test_x, valid_y, test_y = model_selection.train_test_split(temp_x, temp_y, test_size=1/3)

# print("No samples in training set: {}".format(len(train_x)))

# print("No samples in validation set: {}".format(len(valid_x)))

# print("No samples in test set: {}".format(len(test_x)))
# Simple logistic regression model fitting/training

# model = LogisticRegression(solver='lbfgs', dual=False, class_weight='balanced', C=0.5, max_iter=10000)

# model.fit(train_x.drop(columns='question_text'), train_y)
# # Logistic regression prediction

# # Probability output

# model.predict_proba(valid_x.drop(columns='question_text'))

# # Model score

# model.score(valid_x.drop(columns='question_text'), valid_y)

# # Class output

# pred_valid_y = model.predict(valid_x.drop(columns='question_text'))

# pred_test_y = model.predict(test_x.drop(columns='question_text'))

# print("F1-score (Valid): {}".format(f1_score(valid_y, pred_valid_y)))

# print("F1-score (Test): {}".format(f1_score(test_y, pred_test_y)))
# Submission

# get_features(test_df)

# pred_sub_test_y = model.predict(test_df.drop(columns='question_text'))

# sub_df.prediction = pred_sub_test_y

# sub_df.to_csv("submission.csv", index=False)

#TFIDF Vectorizer

# TODO Try with different tokenizer

# tfidf_vectorizer = TfidfVectorizer(

#         #ngram_range=(1,2),

#         min_df=3,

#         max_df=0.9,

#         strip_accents='unicode',

#         use_idf=True,

#         smooth_idf=True,

#         sublinear_tf=True,

#         max_features=9000

#     ).fit(pd.concat([train_df['question_text'], test_df['question_text']]))



tfidf_vectorizer = TfidfVectorizer(

    ngram_range=(1,1),

    max_features=9000,

    sublinear_tf=True, 

    strip_accents='unicode', 

    analyzer='word', 

    token_pattern="\w{1,}", 

    stop_words="english",

    max_df=0.95,

    min_df=2

).fit(pd.concat([train_df['question_text'], test_df['question_text']]))
# Get features

get_features(train_df)



# Split training data into train and validaiton sets (70/20/10)

train_x, temp_x, train_y, temp_y = model_selection.train_test_split(train_df.drop(columns='target'), train_df['target'], test_size=0.3)

valid_x, test_x, valid_y, test_y = model_selection.train_test_split(temp_x, temp_y, test_size=1/3)

print("No samples in training set, shape, type: {}, {}, {}".format(len(train_x), train_x.shape, type(train_x)))

print("No samples in validation set, shape, type: {}, {}, {}".format(len(valid_x), valid_x.shape, type(valid_x)))

print("No samples in test set, shape, type: {}, {}, {}".format(len(test_x), test_x.shape, type(test_x)))
# Transform data

train_x_tfidf = tfidf_vectorizer.transform(train_x['question_text'].fillna("na_").values).tocsr()

valid_x_tfidf = tfidf_vectorizer.transform(valid_x['question_text'].fillna("na_").values).tocsr()

test_x_tfidf = tfidf_vectorizer.transform(test_x['question_text'].fillna("na_").values).tocsr()



# Stack features

train_x_tfidf_f = hstack([csr_matrix(train_x.drop(columns='question_text')), train_x_tfidf])

valid_x_tfidf_f = hstack([csr_matrix(valid_x.drop(columns='question_text')), valid_x_tfidf])

test_x_tfidf_f = hstack([csr_matrix(test_x.drop(columns='question_text')), test_x_tfidf])

print(type(train_x), type(train_x_tfidf), type(train_x_tfidf_f))

print(train_x.shape, train_x_tfidf.shape, train_x_tfidf_f.shape)
# Fit model on transformed data

model = LogisticRegression(solver='lbfgs', dual=False, class_weight='balanced', C=0.5, max_iter=10000)

model.fit(train_x_tfidf_f, train_y)

#model.fit(train_x_tfidf, train_y)
# Prediction

# pred_valid_y = model.predict(valid_x_tfidf)

# pred_test_y = model.predict(test_x_tfidf)

pred_valid_y = model.predict_proba(valid_x_tfidf_f)

pred_test_y = model.predict_proba(test_x_tfidf_f)

# print("F1-score (Valid): {}".format(f1_score(valid_y, pred_valid_y)))

# print("F1-score (Test): {}".format(f1_score(test_y, pred_test_y)))



def threshold_search(y_true, y_proba):

    best_threshold = 0

    best_score = 0

    for threshold in [i * 0.01 for i in range(100)]:

        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)

        if score > best_score:

            best_threshold = threshold

            best_score = score

    search_result = {'threshold': best_threshold, 'f1': best_score}

    return search_result



t_result_0 = threshold_search(valid_y, pred_valid_y[:,0])

t_result_1 = threshold_search(valid_y, pred_valid_y[:,1])

                 

# Find optimal threshold

#print("Threshold (0) and F1-score: {} {}".format(t_result_0['threshold'], t_result_0['f1']))

print("Threshold (1) and F1-score: {} {}".format(t_result_1['threshold'], t_result_1['f1']))



pred_test_y_thresh = np.zeros(pred_test_y.shape[0])

pred_test_y_thresh[pred_test_y[:,1] > t_result_1['threshold']] = 1



print("F1-score (Test): {}".format(f1_score(test_y, pred_test_y_thresh)))