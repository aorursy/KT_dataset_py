import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from wordcloud import WordCloud



import nltk.data

from nltk.corpus import subjectivity, stopwords, wordnet, sentiwordnet

from nltk import word_tokenize, pos_tag

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.stem.wordnet import WordNetLemmatizer



import matplotlib.pyplot as plt

# set plot size

plt.rcParams['figure.figsize'] = (8.0, 8.0)
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sid = SentimentIntensityAnalyzer()

lemmatizer = WordNetLemmatizer()

STOP_WORDS = set(stopwords.words('english'))



# a little memoization for synset word scores

WORD_SCORES = {}



# for replacing contractions post-tokenization

CONTRACTION_MAP = {"'s": "is",

                   "'ll": 'will',

                   "n't": "not",

                   "'ve": "have",

                   "'re": "are",

                   "'m": "am",

                   "ca": "can",

                   "'d": "would"}



# this maps nltk 'universal' tags to wordnet tags

POS_TAG_MAP = {'NOUN': 'n', 'ADJ': 'a', 'VERB': 'v', 'ADV': 'r'}
def normalize_arr(arr, mn= None, mx= None):

    if not mn:

        mn, mx = min(arr), max(arr)

    return list(map(lambda x : (x - mn)/ (mx - mn), arr))



def replace_contractions(token):

    if token in CONTRACTION_MAP:

        return CONTRACTION_MAP[token]

    return token
#break down lines into sentences & score sentences

def get_sentences(lines):

    """break down lines into sentences

    returns a list of [(sentence, polarity score)] 

    tuples

    """

    sentences = []

    for line in lines:

        these_sents = sentence_tokenizer.tokenize(line)

        for sent in these_sents:

            sentences.append((sent, sid.polarity_scores(sent)))

    return sentences

  

    

def word_senti_score(word, POS):

    """returns nltk sentiwordnet...

    Args:

        word (str): Description

        pos (str): part of speech should be 

                   gotta be in NLTK wordnet

    Returns:

        TYPE: pos & neg values... skips neu

    """

    p, n = 0., 0.

    try:

        p, n =  WORD_SCORES[(word, POS)]

    except KeyError:

        scores = list(sentiwordnet.senti_synsets(word, POS))

        if scores: # this will average all synset words for given POS

            p = sum([s.pos_score() for s in scores])/ len(scores)

            n = sum([s.neg_score() for s in scores])/len(scores)

        WORD_SCORES[(word, POS)] = (p, n)

    return p, n



# workhorse for breaking down sentences, pos_tagging, lemmatization, returns tagged

#lemmatized words with their initial scores

def get_words(sent, sent_score, word_clean, stopwords=[], exceptions=[]):

    """tag and tokenize sentance, do cleanup on words

        and return list of word, POS pairs with their synset

        scores combined with the score of their context sentence



    Args:

        sent (str): sentence, not tokenized

        sent_score (tuple) : pos and neg scores for sentence

        word_clean (function): cleaning function to be run on

                               words after tagging

        stopwords (List): list of stopwords

        exceptions (list, optional): these words will escape the

                                     lemmatizer.

    Returns:

        List of tuples: [(word, POS, positive score, negative score)]

    """

    tagged = pos_tag(word_tokenize(sent), tagset='universal')

    words = [(word_clean(x), y) for (x,y) in tagged]

    res = []

    s_pos, s_neg = sent_score

    for (w, t) in words:

        #basically throwing everything not in i'm ok with that since

        #i've already used full sentences for polarity scores

        if t in POS_TAG_MAP:

            POS = POS_TAG_MAP[t]

            if w in exceptions: # don't lemmatize words like 'ISIS'

                word, POS = w,POS

            else:

                 word = lemmatizer.lemmatize(w, POS)

            if word not in stopwords: 

                p, n = word_senti_score(word, POS)

                # this is a little arbitrary but it seems to work well...

                # I add the (often 0. in the case of neutral words) senti_synset

                # score to the vader score for that sentence....

                w_pos = 1. * (p + s_pos )

                w_neg = 1. * (n + s_neg)

                res.append((word, POS, w_pos, w_neg))

    return res

    



def get_vocab(sentences, word_getter):

    words = []

    for sentence, score in sentences:

        s_pos, s_neg = score['pos'] , score['neg']

        words += word_getter(sentence, (s_pos, s_neg))

    unique_words = set([e[0] for e in words])

    vocab = [list(unique_words), [], [], []] # word, count, pos, neg ... because pandas joins make everyting slower

    for u_word in unique_words:

        w_dat = [e for e in words if  e[0] == u_word]

        count = len(w_dat)

        vocab[1].append(count)

        # ... then i get the mean for all uses of that word (within a single individuals

        # vocabulary)

        p, n = sum([e[-2] for e in w_dat])/ float(count), sum([e[-1] for e in w_dat])/ float(count)

        vocab[2].append(p)

        vocab[3].append(n)

                        

    #then i scale scores for entire vocab between 0 & 1

    vocab[2] = normalize_arr(vocab[2])

    vocab[3] = normalize_arr(vocab[3])

    return vocab





def get_data(lines, additional_stopwords=[], exceptions=[]):

    sentences = get_sentences(lines)

    (words, counts, pos_vals, neg_vals) = get_vocab(sentences, 

                                                    word_getter= lambda s, sc: get_words(s, sc,

                                                                                   word_clean=lambda x: replace_contractions(x.lower()),

                                                                                    stopwords=additional_stopwords | STOP_WORDS,

                                                                                    exceptions=exceptions)                                                                             )

    return pd.DataFrame({'word': words, 

                    'count': counts, 

                     'pos': pos_vals, 

                     'neg': neg_vals}, 

                       columns = ['word', 'count', 'pos', 'neg'])
def gen_cloud(data):

    counts = dict([(w, data[w]['count']) for w in data])

    def sent_color_function(word=None, font_size=None, position=None,

                            orientation=None, font_path=None, random_state=None):

        """sentiment color generator for WordCloud 

        """

        r, g, b = 126 + int(255 * data[word]['neg']), 126, 126 + int(255 * data[word]['pos'])

        if r > 255:

            v = r - 255

            b = max(b - v, 0)

            g = max(g - v, 0)

            r = 255

        if b > 255:

            v = b - 255

            r = max(r - v, 0)

            g = max(g - v, 0) 

            b = 255

        return "rgb({}, {}, {})".format(r, g, b)



    wordcloud = WordCloud(  max_font_size = 100,

                            width= 800, 

                            height = 400,

                            color_func=sent_color_function).generate_from_frequencies(counts)

    return wordcloud





def show_cloud(cloud, label):

    plt.title(label)

    plt.imshow(cloud)

    plt.axis("off")

    plt.show()

    

def show_basics(vocab):

    print('unique word count : {}\n'.format(vocab.shape[0]))

    print( 'top 10 most used frequent words:')

    print( vocab.nlargest(10, 'count'), '\n')

    print( 'most positive words:')

    print( vocab.nlargest(10, 'pos'), '\n')

    print( 'most negative words:')

    print( vocab.nlargest(10, 'neg'), '\n')

    
junk = set([ 'say', 'get', 'think', 'go', 'people', 'well', 'come', 'would', 'could',

             'would', 'want', 'become', 'donald', 'hillary', 'lester', 'make', 'chris', 'know', 

             'take', 'lot', 'tell', 'way', 'need', 'give', 'see', 'year', 'many', 'talk', 'clinton', 

             'trump', 'really', 'look', 'let', 'much', 'thing', 'country', 'president', 'also'])



exceptions = ['isis', 'isil', 'sanders']
# read in data

df = pd.read_csv('D:/debate.csv', encoding= "latin1")


#get vocab as pandas for canidates

clinton_vocab = get_data(  list(df['Text'][df['Speaker'] == 'Clinton'].values), 

                                additional_stopwords=junk, 

                                exceptions=exceptions)



trump_vocab = get_data( list(df['Text'][df['Speaker'] == 'Trump'].values), 

                           additional_stopwords=junk, 

                           exceptions=exceptions)
# build clinton's cloud

clinton_cloud = gen_cloud(dict(clinton_vocab.set_index('word').to_dict('index')))

show_cloud(clinton_cloud, label = 'CLINTON')
#show some basic data

show_basics(clinton_vocab)
# ditto trump...

trump_cloud = gen_cloud(dict(trump_vocab.set_index('word').to_dict('index')))

show_cloud(trump_cloud, label = 'TRUMP')
#show some basic data

show_basics(trump_vocab)