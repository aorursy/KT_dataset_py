# First my program will install all the required libraries and will read the csv file and will convert it to dataframe df.
# It will give column names to the datafram df and then it will drop some columns except text and sentiment.
# It will count the tweets group by sentiment.
# It will add new column pre_clean_len to dataframe which is length of each tweet.
# plot pre_clean_len column.
# check for any tweets greater than 140 characters.
# for each text i am calling tweet_cleaner function which will remove convert words to lower case, remove URL, remove hashtag, remove @mentions, HTML decoding, UTF-8 BOM decoding and converting words like isn't to is not.
# And all this it will store in list called clean_tweet_texts.
# Again it will tokenize the tweets in clean_tweet_texts and will do lemmatizing for every word in  list and after lemmatization it will join all the wirds again and will store it in new list called clean_df1.
# This clean_df1 is then converted to dataframe and a sentiment column is added to it which is from old dataframe df.
# Again it will add new column pre_clean_len to dataframe which is length of each tweet.
# Again check for any tweets greater than 140 characters.
# All the tweets is given to new variable x.
# All the tweets sentiments is given to new variable y and plot the see shaoe of both x and y variable.
# Now split the dataset in ratio 80:20 whereas 80% is for training and 20% is for testing.
# Split both the x and y variables.
# make a new instance vect of Tf-idf vectorizer and pass parameter as analyzer = "word" and ngrams_range = (1,1).
# this ngrams_range is for feature selection is given (1,1) it will only select unigrams, (2,2) only bigrams, (3,3) only trigrams, (1,2) unigrams and bigrams, (1,3) unigrams, bigrams and trigrams.
# we can also remove stop words over here by simply add new parameter stop_words = 'english'.
# fit or traing data tweets to vect.
# transform our training data tweets.
# transform our testing data tweets.
# import naive bayes and make object of it. Fit our traing tweets data and training tweets sentiment to the model.
# do 10- fold cross validation on the training data and  calculate the mean accuracy of it.
# predict the sentiments of testing tweets data.
# calculate the accuracy of predicted sentiments with the original tweets sentiment of testing data.
# plot the confusion matrix between original testing sentiment data and predicted sentiment.
# import logistic regression and make object of it. Fit our traing tweets data and training tweets sentiment to the model.
# do 10- fold cross validation on the training data and  calculate the mean accuracy of it.
# predict the sentiments of testing tweets data.
# calculate the accuracy of predicted sentiments with the original tweets sentiment of testing data.
# plot the confusion matrix between original testing sentiment data and predicted sentiment.
# import SVM and make object of it. Fit our traing tweets data and training tweets sentiment to the model.
# do 10- fold cross validation on the training data and  calculate the mean accuracy of it.
# predict the sentiments of testing tweets data.
# calculate the accuracy of predicted sentiments with the original tweets sentiment of testing data.
# plot the confusion matrix between original testing sentiment data and predicted sentiment.

import pandas as pd #import pandas
import numpy as numpy #import numpy
from sklearn.utils import shuffle # to shuffle the data 
import random # import random
import sklearn # import sklearn
import nltk # import nltk
from nltk.corpus import stopwords #import stop words
import re # import regular expression
from nltk.tokenize import word_tokenize # import word_tokenize
import matplotlib
import gensim
import random
import re
from collections import Counter
import unicodedata as udata
import string
import matplotlib.pyplot as plt #import matplotlib.pyplot 
df = pd.read_csv("../input/training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None) #read csv file without header as dataframe
from sklearn.feature_extraction.text import TfidfVectorizer #  import TF-idf vectorizer
df = shuffle(df) # shuffle csv file
#tweets1 = df.iloc[0:9999,]
#tweets1.to_csv('tweets1.csv', sep=',')
%matplotlib inline
#data
print(sklearn.__version__)
print(matplotlib.__version__)
print(numpy.__version__)
print(pd.__version__)
print(nltk.__version__)
# some useful functions:

class Voc:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = ["PAD", "UNK"] # might be changed
        self.n_words = 10000 + 2 # might be changed

    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    def remove_punctuation(self, sentence):
        sentence = self.unicodeToAscii(sentence)
        sentence = re.sub(r"([.!?])", r" \1", sentence)
        sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)
        sentence = re.sub(r"\s+", r" ", sentence).strip()
        return sentence

    def fit(self, train_df, train_df_no_label, USE_Word2Vec=True):
        print("Voc fitting...")
        
        # tokenize
        tokens = []
        sentences = []
        
        for sequence in train_df["seq"]:
            token = sequence.strip(" ").split(" ")
            tokens += token
            sentences.append(token)


        for sequence in train_df_no_label["seq"]:
            token = sequence.strip(" ").split(" ")
            tokens += token
            sentences.append(token)

        # Using Word2Vec
        if USE_Word2Vec:
            dim = 100
            print("Word2Vec fitting")
            model = Word2Vec(sentences, size=dim, window=5, min_count=20, workers=20, iter=20)
            print("Word2Vec fitting finished....")
            # gensim index2word 
            self.index2word += model.wv.index2word
            self.n_words = len(self.index2word)
            # build up numpy embedding matrix
            embedding_matrix = [None] * len(self.index2word) # init to vocab length
            embedding_matrix[0] = np.random.normal(size=(dim,))
            embedding_matrix[1] = np.random.normal(size=(dim,))
            # plug in embedding
            for i in range(2, len(self.index2word)):
                embedding_matrix[i] = model.wv[self.index2word[i]]
                self.word2index[self.index2word[i]] = i
            
            # 
            self.embedding_matrix = np.array(embedding_matrix)
            return
        else:
            # Counter
            counter = Counter(tokens)
            voc_list = counter.most_common(10000)

            for i, (voc, freq) in enumerate(voc_list):
                self.word2index[voc] = i+2
                self.index2word[i+2] = voc
                self.word2count[voc] = freq

def print_to_csv(y_, filename):
    d = {"id":[i for i in range(len(y_))],"label":list(map(lambda x: str(x), y_))}
    df = pd.DataFrame(data=d)
    df.to_csv(filename, index=False)


class BOW():
    def __init__(self):
        self.vectorizer = CountVectorizer(max_features=10000)

    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    def remove_punctuation(self, sentence):
        sentence = self.unicodeToAscii(sentence)
        sentence = re.sub(r"([.!?])", r" \1", sentence)
        sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)
        sentence = re.sub(r"\s+", r" ", sentence).strip()
        return sentence

    def fit(self, train_df, train_df_no_label):
        # prepare copus
    
        corpus = list(map(lambda x: self.remove_punctuation(x), train_df['seq']))
        corpus += list(map(lambda x: self.remove_punctuation(x), train_df_no_label['seq']))
        print("BOW fitting")
        self.vectorizer.fit(corpus)
        self.dim = len(self.vectorizer.get_feature_names())
        print("BOW fitting done")
        return self

    def batch_generator(self, df, batch_size, shuffle=True, training=True):
         # (B, Dimension)
        N = df.shape[0]
        df_matrix = df.as_matrix()
        
        if shuffle == True:
            random_permutation = np.random.permutation(N)
            
            # shuffle
            X = df_matrix[random_permutation, 1]
            y = df_matrix[random_permutation, 0].astype(int) # 0 is label's index
        else:
            X = df_matrix[:, 1]
            y = df_matrix[:, 0].astype(int)
        #
        quotient = X.shape[0] // batch_size
        remainder = X.shape[0] - batch_size * quotient

        for i in range(quotient):
            batch = {}
            batch_X = self.vectorizer.transform(X[i*batch_size:(i+1)*batch_size]).toarray()
            batch['X'] = Variable(torch.from_numpy(batch_X)).float()
            if training:
                batch_y = y[i*batch_size:(i+1)*batch_size]
                batch['y'] = Variable(torch.from_numpy(batch_y))
            else:
                batch['y'] = None
            batch['lengths'] = None
            yield batch
            
        if remainder > 0: 
            batch = {}
            batch_X = self.vectorizer.transform(X[-remainder:]).toarray()
            batch['X'] = Variable(torch.from_numpy(batch_X)).float()
            if training:
                batch_y = y[-remainder:]
                batch['y'] = Variable(torch.from_numpy(batch_y))
            else:
                batch['y'] = None
            batch['lengths'] = None
            yield batch

class Preprocess:
    '''
        Preprocess raw data
    '''
    def __init__(self):
        self.regex_remove_punc = re.compile('[%s]' % re.escape(string.punctuation))
        pass
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalizeString(self, sentence):
        sentence = self.unicodeToAscii(sentence.strip())
        #sentence = self.unicodeToAscii(sentence.lower().strip())
        # remove punctuation
        if False:
            sentence = self.regex_remove_punc.sub('', sentence)
        sentence = re.sub(r"([.!?])", r" \1", sentence)
        sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)
        sentence = re.sub(r"\s+", r" ", sentence).strip()
        return sentence

    def remove_punctuation(self, sentence):
        sentence = self.regex_remove_punc.sub('', sentence)
        return sentence

    def read_txt(self, train_filename, test_filename, train_filename_no_label):
        train_df = None
        test_df = None
        train_df_no_label = None
        
        if train_filename is not None:
            train_df = pd.read_csv(train_filename, header=None, names=["label", "seq"], sep="\+\+\+\$\+\+\+",
                                  engine="python")
            # remove puncuation
            #train_df["seq"] = train_df["seq"].apply(lambda seq: self.normalizeString(seq))
            
        
        if test_filename is not None:
            with open(test_filename, "r") as f:
                reader = csv.reader(f, delimiter=",")
                rows = [[row[0], ",".join(row[1:])] for row in reader]
                test_df = pd.DataFrame(rows[1:], columns=rows[0]) # first row is column name
            # remove puncuation
            #test_df["text"] = test_df["text"].apply(lambda seq: self.normalizeString(seq))
        if train_filename_no_label is not None:
            train_df_no_label = pd.read_csv(train_filename_no_label, sep="\n", header=None, names=["seq"])
            train_df_no_label.insert(loc=0, column="nan", value=0)
            # remove puncuation
            #train_df_no_label["seq"] = train_df_no_label["seq"].apply(lambda seq: self.normalizeString(seq))
        
        return train_df, test_df, train_df_no_label

class Sample_Encode:
    '''
        Transform 
    '''
    def __init__(self, voc):
        self.voc = voc

    def sentence_to_index(self, sentence):
        encoded = list(map(lambda token: self.voc.word2index[token] if token in self.voc.word2index \
            else UNK_token, sentence))
        return encoded

    def pad_batch(self, index_batch):
        '''
            Return padded list with size (B, Max_length)
        '''
        return list(itertools.zip_longest(*index_batch, fillvalue=PAD_token))

    def batch_to_Variable(self, sentence_batch, training=True):
        '''
            Input: a numpy of sentence
            ex. ["i am a", "jim l o "]

            Output: a torch Variable and sentence lengths
        '''
        # split sentence
        sentence_batch = sentence_batch.tolist()
        
        # apply
        for training_sample in sentence_batch:
            # split training sentence
            training_sample[1] = training_sample[1].strip(" ").split(" ")

        # encode batch
        index_label_batch = [(training_sample[0], self.sentence_to_index(training_sample[1])) \
            for training_sample in sentence_batch]

        # sort sentence batch (in order to fit torch pack_pad_sequence)
        #index_label_batch.sort(key=lambda x: len(x[1]), reverse=True) 
        
        # index batch
        index_batch = [training_sample[1] for training_sample in index_label_batch]
        label_batch = [training_sample[0] for training_sample in index_label_batch]

        # record batch's length
        lengths = [len(indexes) for indexes in index_batch]

        # padded batch
        padded_batch = self.pad_batch(index_batch)

        # transform to Variable
        if training:
            pad_var = Variable(torch.LongTensor(padded_batch), volatile=False)
        else:
            pad_var = Variable(torch.LongTensor(padded_batch), volatile=True)

        # label
        if training:
            label_var = Variable(torch.LongTensor(label_batch), volatile=False)
        else:
            label_var = None

        
        return pad_var, label_var, lengths
    
    def generator(self, df, batch_size, shuffle=False, training=True):
        '''
        Return sample batch Variable
            batch['X'] is (T, B)
        '''
        df_matrix = df.as_matrix()
        if shuffle == True:
            random_permutation = np.random.permutation(len(df['seq']))
            
            # shuffle
            df_matrix = df_matrix[random_permutation]
        #
        quotient = df.shape[0] // batch_size
        remainder = df.shape[0] - batch_size * quotient

        for i in range(quotient):
            batch = {}
            X, y, lengths = self.batch_to_Variable(df_matrix[i*batch_size:(i+1)*batch_size], training)
            batch['X'] = X
            batch['y'] = y
            batch['lengths'] = lengths
            yield batch
            
        if remainder > 0: 
            batch = {}
            X, y, lengths = self.batch_to_Variable(df_matrix[-remainder:],training)
            batch['X'] = X
            batch['y'] = y
            batch['lengths'] = lengths
            yield batch

def trim(text_list, threshold=2):
    result = []
    for _, text in enumerate(text_list):
        grouping = []
        for _, g in itertools.groupby(text):
            grouping.append(list(g))
        r = ''.join([g[0] if len(g)<threshold else g[0]*threshold for g in grouping])
        result.append(r)
    return result

def token_counter(corpus):
    tokenizer = Tokenizer(num_words=None,filters="\n")
    tokenizer.fit_on_texts(corpus)
    sequences = tokenizer.texts_to_sequences(corpus)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
 
stemmer = gensim.parsing.porter.PorterStemmer()
def preprocess(string, use_stem = True):
    string = string.replace("i ' m", "im").replace("you ' re","youre").replace("didn ' t","didnt")    .replace("can ' t","cant").replace("haven ' t", "havent").replace("won ' t", "wont")    .replace("isn ' t","isnt").replace("don ' t", "dont").replace("doesn ' t", "doesnt")    .replace("aren ' t", "arent").replace("weren ' t", "werent").replace("wouldn ' t","wouldnt")    .replace("ain ' t","aint").replace("shouldn ' t","shouldnt").replace("wasn ' t","wasnt")    .replace(" ' s","s").replace("wudn ' t","wouldnt").replace(" .. "," ... ")    .replace("couldn ' t","couldnt")
    for same_char in re.findall(r'((\w)\2{2,})', string):
        string = string.replace(same_char[0], same_char[1])
    for digit in re.findall(r'\d+', string):
        string = string.replace(digit, "1")
    for punct in re.findall(r'([-/\\\\()!"+,&?\'.]{2,})',string):
        if punct[0:2] =="..":
            string = string.replace(punct, "...")
        else:
            string = string.replace(punct, punct[0])
    return string

def getFrequencyDict(lines):
    freq = {}
    for s in lines:
        for w in s:
            if w in freq: freq[w] += 1
            else:         freq[w] = 1
    return freq

def initializeCmap(lines):
    print('  Initializing conversion map...')
    cmap = {}
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            cmap[w] = w
    print('    Conversion map size:', len(cmap))
    return cmap

def convertAccents(lines, cmap):
    print('  Converting accents...')
    for i, s in enumerate(lines):
        s = [(''.join(c for c in udata.normalize('NFD', w) if udata.category(c) != 'Mn')) for w in s]
        for j, w in enumerate(s):
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s
    clist = 'abcdefghijklmnopqrstuvwxyz0123456789.!?'
    for i, s in enumerate(lines):
        s = [''.join([c for c in w if c in clist]) for w in s]
        for j, w in enumerate(s):
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertPunctuations(lines, cmap):
    print('  Converting punctuations...')
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            excCnt, queCnt, dotCnt = w.count('!'), w.count('?'), w.count('.')
            if queCnt:        s[j] = '_?'
            elif excCnt >= 5: s[j] = '_!!!'
            elif excCnt >= 3: s[j] = '_!!'
            elif excCnt >= 1: s[j] = '_!'
            elif dotCnt >= 2: s[j] = '_...'
            elif dotCnt >= 1: s[j] = '_.'
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertNotWords(lines, cmap):
    print('  Converting not words...')
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            if w[0] == '_': continue
            if w == '2':        s[j] = 'to'
            elif w.isnumeric(): s[j] = '_n'
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertTailDuplicates(lines, cmap):
    print('  Converting tail duplicates...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            w = re.sub(r'(([a-z])\2{2,})$', r'\g<2>\g<2>', w)
            s[j] = re.sub(r'(([a-cg-kmnp-ru-z])\2+)$', r'\g<2>', w)
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertHeadDuplicates(lines, cmap):
    print('  Converting head duplicates...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            s[j] = re.sub(r'^(([a-km-z])\2+)', r'\g<2>', w)
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertInlineDuplicates(lines, cmap, minfreq=64):
    print('  Converting inline duplicates...')
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            w = re.sub(r'(([a-z])\2{2,})', r'\g<2>\g<2>', w)
            s[j] = re.sub(r'(([ahjkquvwxyz])\2+)', r'\g<2>', w)  # repeated 2+ times, impossible
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            if freq[w] > minfreq: continue
            if w == 'too': continue
            w1 = re.sub(r'(([a-z])\2+)', r'\g<2>', w) # repeated 2+ times, replace by 1
            f0, f1 = freq.get(w,0), freq.get(w1,0)
            fm = max(f0, f1)
            if fm == f0:   pass
            else:          s[j] = w1;
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertSlang(lines, cmap):
    print('  Converting slang...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            if w == 'u': lines[i][j] = 'you'
            if w == 'dis': lines[i][j] = 'this'
            if w == 'dat': lines[i][j] = 'that'
            if w == 'luv': lines[i][j] = 'love'
            w1 = re.sub(r'in$', r'ing', w)
            w2 = re.sub(r'n$', r'ing', w)
            f0, f1, f2 = freq.get(w,0), freq.get(w1,0), freq.get(w2,0)
            fm = max(f0, f1, f2)
            if fm == f0:   pass
            elif fm == f1: s[j] = w1;
            else:          s[j] = w2;
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertSingular(lines, cmap, minfreq=512):
    print('  Converting singular form...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            if freq[w] > minfreq: continue
            w1 = re.sub(r's$', r'', w)
            w2 = re.sub(r'es$', r'', w)
            w3 = re.sub(r'ies$', r'y', w)
            f0, f1, f2, f3 = freq.get(w,0), freq.get(w1,0), freq.get(w2,0), freq.get(w3,0)
            fm = max(f0, f1, f2, f3)
            if fm == f0:   pass
            elif fm == f1: s[j] = w1;
            elif fm == f2: s[j] = w2;
            else:          s[j] = w3;
            cmap[original_lines[i][j]] = s[j]
    lines[i] = s

def convertRareWords(lines, cmap, min_count=16):
    print('  Converting rare words...')
    freq = getFrequencyDict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            if freq[w] < min_count: s[j] = '_r'
            cmap[original_lines[i][j]] = s[j]
        lines[i] = s

def convertCommonWords(lines, cmap):
    print('  Converting common words...')
    #beverbs = set('is was are were am s'.split())
    #articles = set('a an the'.split())
    #preps = set('to for of in at on by'.split())

    for i, s in enumerate(lines):
        #s = [word if word not in beverbs else '_b' for word in s]
        #s = [word if word not in articles else '_a' for word in s]
        #s = [word if word not in preps else '_p' for word in s]
        lines[i] = s

def convertPadding(lines, maxlen=38):
    print('  Padding...')
    for i, s in enumerate(lines):
        lines[i] = [w for w in s if w]
    for i, s in enumerate(lines):
        lines[i] = s[:maxlen]

def preprocessLines(lines):
    global original_lines
    original_lines = lines[:]
    cmap = initializeCmap(original_lines)
    convertAccents(lines, cmap)
    convertPunctuations(lines, cmap)
    convertNotWords(lines, cmap)
    convertTailDuplicates(lines, cmap)
    convertHeadDuplicates(lines, cmap)
    convertInlineDuplicates(lines, cmap)
    convertSlang(lines, cmap)
    convertSingular(lines, cmap)
    convertRareWords(lines, cmap)
    convertCommonWords(lines, cmap)
    convertPadding(lines)
    return lines, cmap

def readData(path, label=True):
    print('  Loading', path+'...')
    _lines, _labels = [], []
    with open(path, 'r', encoding='utf_8') as f:
        for line in f:
            if label:
                _labels.append(int(line[0]))
                line = line[10:-1]
            else:
                line = line[:-1]
            _lines.append(line.split())
    if label: return _lines, _labels
    else:     return _lines

def padLines(lines, value, maxlen):
    maxlinelen = 0
    for i, s in enumerate(lines):
        maxlinelen = max(len(s), maxlinelen)
    maxlinelen = max(maxlinelen, maxlen)
    for i, s in enumerate(lines):
        lines[i] = (['_r'] * max(0, maxlinelen - len(s)) + s)[-maxlen:]
    return lines

def getDictionary(lines):
    _dict = {}
    for s in lines:
        for w in s:
            if w not in _dict:
                _dict[w] = len(_dict) + 1
    return _dict

def transformByDictionary(lines, dictionary):
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w in dictionary: lines[i][j] = dictionary[w]
            else:               lines[i][j] = dictionary['']

def transformByConversionMap(lines, cmap, iter=2):
    cmapRefine(cmap)
    for it in range(iter):
        for i, s in enumerate(lines):
            s0 = []
            for j, w in enumerate(s):
                if w in cmap and w[0] != '_':
                    s0 = s0 + cmap[w].split()
                elif w[0] == '_':
                    s0 = s0 + [w]
            lines[i] = [w for w in s0 if w]

def transformByWord2Vec(lines, w2v):
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w in w2v.wv:
                lines[i][j] = w2v.wv[w]
            else:
                lines[i][j] = w2v.wv['_r']

def readTestData(path):
    print('  Loading', path + '...')
    _lines = []
    with open(path, 'r', encoding='utf_8') as f:
        for i, line in enumerate(f):
            if i:
                start = int(np.log10(max(1, i-1))) + 2
                _lines.append(line[start:].split())
    return _lines

def savePrediction(y, path, id_start=0):
    pd.DataFrame([[i+id_start, int(y[i])] for i in range(y.shape[0])],
                 columns=['id', 'label']).to_csv(path, index=False)

def savePreprocessCorpus(lines, path):
    with open(path, 'w', encoding='utf_8') as f:
        for line in lines:
            f.write(' '.join(line) + '\n')

def savePreprocessCmap(cmap, path):
    with open(path, 'wb') as f:
        pickle.dump(cmap, f)

def loadPreprocessCmap(path):
    print('  Loading', path + '...')
    with open(path, 'rb') as f:
        cmap = pickle.load(f)
    return cmap

def loadPreprocessCorpus(path):
    print('  Loading', path + '...')
    lines = []
    with open(path, 'r', encoding='utf_8') as f:
        for line in f:
            lines.append(line.split())
    return lines

def removePunctuations(lines):
    rs = {'_!', '_!!', '_!!!', '_.', '_...', '_?'}
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w in rs:
                s[j] = ''
        lines[i] = [w for w in x if w]

def removeDuplicatedLines(lines):
    lineset = set({})
    for line in lines:
        lineset.add(' '.join(line))
    for i, line in enumerate(lineset):
        lines[i] = line.split()
    del lines[-(len(lines)-len(lineset)):]
    return lineset

def shuffleData(lines, labels):
    for i, s in enumerate(lines):
        lines[i] = (s, labels[i])
    np.random.shuffle(lines)
    for i, s in enumerate(lines):
        labels[i] = s[1]
        lines[i] = s[0]

def cmapRefine(cmap):
    cmap['teh'] = cmap['da'] = cmap['tha'] = 'the'
    cmap['evar'] = 'ever'
    cmap['likes'] = cmap['liked'] = cmap['lk'] = 'like'
    cmap['wierd'] = 'weird'
    cmap['kool'] = 'cool'
    cmap['yess'] = 'yes'
    cmap['pleasee'] = 'please'
    cmap['soo'] = 'so'
    cmap['noo'] = 'no'
    cmap['lovee'] = cmap['loove'] = cmap['looove'] = cmap['loooove'] = cmap['looooove'] \
        = cmap['loooooove'] = cmap['loves'] = cmap['loved'] = cmap['wuv'] \
        = cmap['loovee'] = cmap['lurve'] = cmap['lov'] = cmap['luvs'] = 'love'
    cmap['lovelove'] = 'love love'
    cmap['lovelovelove'] = 'love love love'
    cmap['ilove'] = 'i love'
    cmap['liek'] = cmap['lyk'] = cmap['lik'] = cmap['lke'] = cmap['likee'] = 'like'
    cmap['mee'] = 'me'
    cmap['hooo'] = 'hoo'
    cmap['sooon'] = cmap['soooon'] = 'soon'
    cmap['goodd'] = cmap['gud'] = 'good'
    cmap['bedd'] = 'bed'
    cmap['badd'] = 'bad'
    cmap['sadd'] = 'sad'
    cmap['madd'] = 'mad'
    cmap['redd'] = 'red'
    cmap['tiredd'] = 'tired'
    cmap['boredd'] = 'bored'
    cmap['godd'] = 'god'
    cmap['xdd'] = 'xd'
    cmap['itt'] = 'it'
    cmap['lul'] = cmap['lool'] = 'lol'
    cmap['sista'] = 'sister'
    cmap['w00t'] = 'woot'
    cmap['srsly'] = 'seriously'
    cmap['4ever'] = cmap['4eva'] = 'forever'
    cmap['neva'] = 'never'
    cmap['2day'] = 'today'
    cmap['homee'] = 'home'
    cmap['hatee'] = 'hate'
    cmap['heree'] = 'here'
    cmap['cutee'] = 'cute'
    cmap['lemme'] = 'let me'
    cmap['mrng'] = 'morning'
    cmap['gd'] = 'good'
    cmap['thx'] = cmap['thnx'] = cmap['thanx'] = cmap['thankx'] = cmap['thnk'] = 'thanks'
    cmap['jaja'] = cmap['jajaja'] = cmap['jajajaja'] = 'haha'
    cmap['eff'] = cmap['fk'] = cmap['fuk'] = cmap['fuc'] = 'fuck'
    cmap['2moro'] = cmap['2mrow'] = cmap['2morow'] = cmap['2morrow'] \
        = cmap['2morro'] = cmap['2mrw'] = cmap['2moz'] = 'tomorrow'
    cmap['babee'] = 'babe'
    cmap['theree'] = 'there'
    cmap['thee'] = 'the'
    cmap['woho'] = cmap['wohoo'] = 'woo hoo'
    cmap['2gether'] = 'together'
    cmap['2nite'] = cmap['2night'] = 'tonight'
    cmap['nite'] = 'night'
    cmap['dnt'] = 'dont'
    cmap['rly'] = 'really'
    cmap['gt'] = 'get'
    cmap['lat'] = 'late'
    cmap['dam'] = 'damn'
    cmap['4ward'] = 'forward'
    cmap['4give'] = 'forgive'
    cmap['b4'] = 'before'
    cmap['tho'] = 'though'
    cmap['kno'] = 'know'
    cmap['grl'] = 'girl'
    cmap['boi'] = 'boy'
    cmap['wrk'] = 'work'
    cmap['jst'] = 'just'
    cmap['geting'] = 'getting'
    cmap['4get'] = 'forget'
    cmap['4got'] = 'forgot'
    cmap['4real'] = 'for real'
    cmap['2go'] = 'to go'
    cmap['2b'] = 'to be'
    cmap['gr8'] = cmap['gr8t'] = cmap['gr88'] = 'great'
    cmap['str8'] = 'straight'
    cmap['twiter'] = 'twitter'
    cmap['iloveyou'] = 'i love you'
    cmap['loveyou'] = cmap['loveya'] = cmap['loveu'] = 'love you'
    cmap['xoxox'] = cmap['xox'] = cmap['xoxoxo'] = cmap['xoxoxox'] \
        = cmap['xoxoxoxo'] = cmap['xoxoxoxoxo'] = 'xoxo'
    cmap['cuz'] = cmap['bcuz'] = cmap['becuz'] = 'because'
    cmap['iz'] = 'is'
    cmap['aint'] = 'am not'
    cmap['fav'] = 'favorite'
    cmap['ppl'] = 'people'
    cmap['mah'] = 'my'
    cmap['r8'] = 'rate'
    cmap['l8'] = 'late'
    cmap['w8'] = 'wait'
    cmap['m8'] = 'mate'
    cmap['h8'] = 'hate'
    cmap['l8ter'] = cmap['l8tr'] = cmap['l8r'] = 'later'
    cmap['cnt'] = 'cant'
    cmap['fone'] = cmap['phonee'] = 'phone'
    cmap['f1'] = 'fONE'
    cmap['xboxe3'] = 'eTHREE'
    cmap['jammin'] = 'jamming'
    cmap['onee'] = 'one'
    cmap['1st'] = 'first'
    cmap['2nd'] = 'second'
    cmap['3rd'] = 'third'
    cmap['inet'] = 'internet'
    cmap['recomend'] = 'recommend'
    cmap['ah1n1'] = cmap['h1n1'] = 'hONEnONE'
    cmap['any1'] = 'anyone'
    cmap['every1'] = cmap['evry1'] = 'everyone'
    cmap['some1'] = cmap['sum1'] = 'someone'
    cmap['no1'] = 'no one'
    cmap['4u'] = 'for you'
    cmap['4me'] = 'for me'
    cmap['2u'] = 'to you'
    cmap['yu'] = 'you'
    cmap['yr'] = cmap['yrs'] = cmap['years'] = 'year'
    cmap['hr'] = cmap['hrs'] = cmap['hours'] = 'hour'
    cmap['min'] = cmap['mins'] = cmap['minutes'] = 'minute'
    cmap['go2'] = cmap['goto'] = 'go to'
    for key, value in cmap.items():
        if not key.isalpha():
            if key[-1:] == 'k':
                cmap[key] = '_n'
            if key[-2:]=='st' or key[-2:]=='nd' or key[-2:]=='rd' or key[-2:]=='th':
                cmap[key] = '_ord'
            if key[-2:]=='am' or key[-2:]=='pm' or key[-3:]=='min' or key[-4:]=='mins' \
                    or key[-2:]=='hr' or key[-3:]=='hrs' or key[-1:]=='h' \
                    or key[-4:]=='hour' or key[-5:]=='hours'\
                    or key[-2:]=='yr' or key[-3:]=='yrs'\
                    or key[-3:]=='day' or key[-4:]=='days'\
                    or key[-3:]=='wks':
                cmap[key] = '_time'
def preprocessTestingData(path):
    print('Loading testing data...')
    lines = readTestData(path)

    cmap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/cmap.pkl')
    w2v_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/word2vec.pkl')
    cmap = loadPreprocessCmap(cmap_path)
    transformByConversionMap(lines, cmap)
    
    lines = padLines(lines, '_', maxlen)
    w2v = Word2Vec.load(w2v_path)
    transformByWord2Vec(lines, w2v)
    return lines

def preprocessTrainingData(label_path, nolabel_path, retrain=False, punctuation=True):
    print('Loading training data...')
    if retrain:
        preprocess(label_path, nolabel_path)

    lines, labels = readData(label_path)
    corpus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/corpus.txt')
    cmap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/cmap.pkl')
    w2v_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/word2vec.pkl')
    lines = readData(corpus_path, label=False)[:len(lines)]
    shuffleData(lines, labels)
    labels = np.array(labels)

    cmap = loadPreprocessCmap(cmap_path)
    transformByConversionMap(lines, cmap)
    if not punctuation:
        removePunctuations(lines)

    lines = padLines(lines, '_', maxlen)
    w2v = Word2Vec.load(w2v_path)
    transformByWord2Vec(lines, w2v)
    return lines, labels

def preprocess(label_path, nolabel_path):
    print('Preprocessing...')
    labeled_lines, labels = readData(label_path)
    nolabel_lines = readData(nolabel_path, label=False)
    lines = labeled_lines + nolabel_lines

    lines, cmap = preprocessLines(lines)
    corpus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/corpus.txt')
    cmap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/cmap.pkl')
    w2v_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/word2vec.pkl')
    savePreprocessCorpus(lines, corpus_path)
    savePreprocessCmap(cmap, cmap_path)

    transformByConversionMap(lines, cmap)
    removeDuplicatedLines(lines)

    print('Training word2vec...')
    model = Word2Vec(lines, size=256, min_count=16, iter=16, workers=16)
    model.save(w2v_path)
# search from github
df.columns = ["sentiment", "id", "date", "query", "user", "text"] # give column names
#data
df = df.drop(["id", "date", "query", "user"], axis = 1) #drop some column from the dataframe 
#data
df.head() # get the first 5 rows from the dataframe
df.sentiment.value_counts() # count the number of sentiments with respect to their tweet(4 stands for positive tweet and 0 stands for negative tweet)
df['pre_clean_len'] = [len(t) for t in df.text] # add new column pre_clean_len to dataframe which is length of each tweet
plt.boxplot(df.pre_clean_len) # plot pre_clean_len column
plt.show()
df[df.pre_clean_len > 140].head(10)  # check for any tweets greater than 140 characters
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()

pat1 = r'@[A-Za-z0-9_]+'        # remove @ mentions fron tweets
pat2 = r'https?://[^ ]+'        # remove URL's from tweets
combined_pat = r'|'.join((pat1, pat2)) #addition of pat1 and pat2
www_pat = r'www.[^ ]+'         # remove URL's from tweets
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",   # converting words like isn't to is not
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def tweet_cleaner(text):  # define tweet_cleaner function to clean the tweets
    soup = BeautifulSoup(text, 'lxml')    # call beautiful object
    souped = soup.get_text()   # get only text from the tweets 
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")    # remove utf-8-sig codeing
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed) # calling combined_pat
    stripped = re.sub(www_pat, '', stripped) #remove URL's
    lower_case = stripped.lower()      # converting all into lower case
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case) # converting word's like isn't to is not
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)       # will replace # by space
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1] # Word Punct Tokenize and only consider words whose length is greater than 1
    return (" ".join(words)).strip() # join the words
nums = [0,400000,800000,1200000,1600000] # used for batch processing tweets
#nums = [0, 9999]
clean_tweet_texts = [] # initialize list
for i in range(nums[0],nums[4]): # batch process 1.6 million tweets                                                               
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))  # call tweet_cleaner function and pass parameter as all the tweets to clean the tweets and append cleaned tweets into clean_tweet_texts list
#clean_tweet_texts
word_tokens = [] # initialize list for tokens
for word in clean_tweet_texts:  # for each word in clean_tweet_texts
    word_tokens.append(word_tokenize(word)) #tokenize word in clean_tweet_texts and append it to word_tokens list
# word_tokens
# stop = set(stopwords.words('english'))
# clean_df =[]
# for m in word_tokens:
#     a = [w for w in m if not w in stop]
#     clean_df.append(a)
# Lemmatizing
df1 = [] # initialize list df1 to store words after lemmatization
from nltk.stem import WordNetLemmatizer # import WordNetLemmatizer from nltk.stem
lemmatizer = WordNetLemmatizer() # create an object of WordNetLemmatizer
for l in word_tokens: # for loop for every tokens in word_token
    b = [lemmatizer.lemmatize(q) for q in l] #for every tokens in word_token lemmatize word and giev it to b
    df1.append(b) #append b to list df1
# Stemming
# df1 = [] 
# from nltk.stem import PorterStemmer
# ps = PorterStemmer()
# for l in word_tokens:
#     b = [ps.stem(q) for q in l]
#     df1.append(b)
#df
clean_df1 =[] # initialize list clean_df1 to join word tokens after lemmatization
for c in df1:  # for loop for each list in df1
    a = " ".join(c) # join words in list with space in between and giev it to a
    clean_df1.append(a) # append a to clean_df1
#clean_df1
clean_df = pd.DataFrame(clean_df1,columns=['text']) # convert clean_tweet_texts into dataframe and name it as clean_df
clean_df['target'] = df.sentiment # from earlier dataframe get the sentiments of each tweet and make a new column in clean_df as target and give it all the sentiment score
#clean_df
clean_df['clean_len'] = [len(t) for t in clean_df.text] # Again make a new coloumn in the dataframe and name it as clean_len which will store thw number of words in the tweet
clean_df[clean_df.clean_len > 140].head(10) # agin check id any tweet is more than 140 characters
X = clean_df.text # get all the text in x variable
y = clean_df.target # get all the sentiments into y variable
print(X.shape) #print shape of x
print(y.shape) # print shape of y
from sklearn.cross_validation import train_test_split #from sklearn.cross_validation import train_test_split to split the data into training and tesing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 1) # split the data into traing and testing set where ratio is 80:20


# X_train is the tweets of training data, X_test is the testing tweets which we have to predict, y_train is the sentiments of tweets in the traing data and y_test is the sentiments of the tweets  which we will use to measure the accuracy of the model
vect = TfidfVectorizer(analyzer = "word", ngram_range=(1,3)) # Get Tf-idf object and save it as vect. We can select features from here we just have simply change 
                                                                                     #the ngram range to change the features also we can remove stop words over here with the help of stop parameter
vect.fit(X_train) # fit or traing data tweets to vect
X_train_dtm = vect.transform(X_train) # transform our training data tweets
X_test_dtm = vect.transform(X_test)# transform our testing data tweets
from sklearn.naive_bayes import MultinomialNB # import Multinomial Naive Bayes model from sklearn.naive_bayes
nb = MultinomialNB(alpha = 10) # get object of Multinomial naive bayes model with alpha parameter = 10
nb.fit(X_train_dtm, y_train)# fit our both traing data tweets as well as its sentiments to the multinomial naive bayes model
from sklearn.model_selection import cross_val_score  # import cross_val_score from sklear.model_selection
accuracies = cross_val_score(estimator = nb, X = X_train_dtm, y = y_train, cv = 10) # do K- fold cross validation on our traing data and its sentimenst with 10 fold cross validation
accuracies.mean() # measure the mean accuray of 10 fold cross validation
y_pred_nb = nb.predict(X_test_dtm) # predict the sentiments of testing data tweets
from sklearn import metrics # import metrics from sklearn
metrics.accuracy_score(y_test, y_pred_nb) # measure the accuracy of our model on the testing data
from sklearn.metrics import confusion_matrix # import confusion matrix from the sklearn.metrics
confusion_matrix(y_test, y_pred_nb) # plot the confusion matrix between our predicted sentiments and the original testing data sentiments
from sklearn.linear_model import LogisticRegression # import Logistic Regression model from sklearn.linear_model
logisticRegr = LogisticRegression(C = 1.1) # get object of logistic regression model with cost parameter = 1.1
logisticRegr.fit(X_train_dtm, y_train)# fit our both traing data tweets as well as its sentiments to the logistic regression model
from sklearn.model_selection import cross_val_score # import cross_val_score from sklear.model_selection
accuracies = cross_val_score(estimator = logisticRegr, X = X_train_dtm, y = y_train, cv = 10) # do K- fold cross validation on our traing data and its sentimenst with 10 fold cross validation
accuracies.mean() # measure the mean accuray of 10 fold cross validation
y_pred_lg = logisticRegr.predict(X_test_dtm)  # predict the sentiments of testing data tweets
from sklearn import metrics # import metrics from sklearn
metrics.accuracy_score(y_test, y_pred_lg) # measure the accuracy of our model on the testing data
from sklearn.metrics import confusion_matrix # import confusion matrix from the sklearn.metrics
confusion_matrix(y_test, y_pred_lg) # plot the confusion matrix between our predicted sentiments and the original testing data sentiments
from sklearn.svm import LinearSVC # import SVC model from sklearn.svm
svm_clf = LinearSVC(random_state=0) # get object of SVC model with random_state parameter = 0
svm_clf.fit(X_train_dtm, y_train)# fit our both traing data tweets as well as its sentiments to the SVC model
from sklearn.model_selection import cross_val_score  # import cross_val_score from sklear.model_selection
accuracies = cross_val_score(estimator = svm_clf, X = X_train_dtm, y = y_train, cv = 10)# do K- fold cross validation on our traing data and its sentimenst with 10 fold cross validation
accuracies.mean() # measure the mean accuray of 10 fold cross validation
y_pred_svm = svm_clf.predict(X_test_dtm)  # predict the sentiments of testing data tweets
from sklearn import metrics  # import metrics from sklearn
metrics.accuracy_score(y_test, y_pred_svm)  # measure the accuracy of our model on the testing data
from sklearn.metrics import confusion_matrix # import confusion matrix from the sklearn.metrics
confusion_matrix(y_test, y_pred_svm)# plot the confusion matrix between our predicted sentiments and the original testing data sentiments
