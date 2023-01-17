import numpy as np

import pandas as pd

from tqdm import tqdm_notebook

tqdm_notebook().pandas()

from keras.preprocessing.text import Tokenizer

from nltk.corpus import stopwords

import pickle as pkl

import re
MAX_NUM_WORDS = 10000

TOXICITY_COLUMN = 'target'

TEXT_COLUMN = 'comment_text'

EMBEDDINGS_FILES = [

    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',

    '../input/glove840b300dtxt/glove.840B.300d.txt',

    '../input/paragram-300-sl999/paragram_300_sl999.txt'

]

EMBEDDINGS_DIMENSION = 300

# All comments must be truncated or padded to be the same length.

MAX_SEQUENCE_LENGTH = 250
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
# Make sure all comment_text values are strings

train[TEXT_COLUMN] = train[TEXT_COLUMN].astype(str) 
def remove_stop_words(text, stopword_list):

    return ' '.join([word for word in text.split() if word not in stopword_list])
english_stopwords = set(stopwords.words('english'))

train[TEXT_COLUMN] = train[TEXT_COLUMN].progress_apply(lambda x: remove_stop_words(x, english_stopwords))
def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have",

 "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
train[TEXT_COLUMN] = train[TEXT_COLUMN].progress_apply(lambda x: clean_contractions(x, contraction_mapping))
def fixing_with_regex(text) -> str:

    """

    Additional fixing of words.



    :param text: text to clean

    :return: cleaned text

    """



    mis_connect_list = ['\b(W|w)hat\b', '\b(W|w)hy\b', '(H|h)ow\b', '(W|w)hich\b', '(W|w)here\b', '(W|w)ill\b']

    mis_connect_re = re.compile('(%s)' % '|'.join(mis_connect_list))



    text = re.sub(r" (W|w)hat+(s)*[A|a]*(p)+ ", " WhatsApp ", text)

    text = re.sub(r" (W|w)hat\S ", " What ", text)

    text = re.sub(r" \S(W|w)hat ", " What ", text)

    text = re.sub(r" (W|w)hy\S ", " Why ", text)

    text = re.sub(r" \S(W|w)hy ", " Why ", text)

    text = re.sub(r" (H|h)ow\S ", " How ", text)

    text = re.sub(r" \S(H|h)ow ", " How ", text)

    text = re.sub(r" (W|w)hich\S ", " Which ", text)

    text = re.sub(r" \S(W|w)hich ", " Which ", text)

    text = re.sub(r" (W|w)here\S ", " Where ", text)

    text = re.sub(r" \S(W|w)here ", " Where ", text)

    text = mis_connect_re.sub(r" \1 ", text)

    text = text.replace("What sApp", ' WhatsApp ')



    # Clean repeated letters.

    text = re.sub(r"(I|i)(I|i)+ng", "ing", text)

    text = re.sub(r"(-+|\.+)", " ", text)



    text = re.sub(r'[\x00-\x1f\x7f-\x9f\xad]', '', text)

    text = re.sub(r'(\d+)(e)(\d+)', r'\g<1> \g<3>', text)  # is a dup from above cell...

    text = re.sub(r"(-+|\.+)\s?", "  ", text)

    text = re.sub("\s\s+", " ", text)

    text = re.sub(r'ᴵ+', '', text)



    text = re.sub(r"(W|w)on(\'|\’)t ", "will not ", text)

    text = re.sub(r"(C|c)an(\'|\’)t ", "can not ", text)

    text = re.sub(r"(Y|y)(\'|\’)all ", "you all ", text)

    text = re.sub(r"(Y|y)a(\'|\’)ll ", "you all ", text)



    text = re.sub(r"(I|i)(\'|\’)m ", "i am ", text)

    text = re.sub(r"(A|a)in(\'|\’)t ", "is not ", text)

    text = re.sub(r"n(\'|\’)t ", " not ", text)

    text = re.sub(r"(\'|\’)re ", " are ", text)

    text = re.sub(r"(\'|\’)s ", " is ", text)

    text = re.sub(r"(\'|\’)d ", " would ", text)

    text = re.sub(r"(\'|\’)ll ", " will ", text)

    text = re.sub(r"(\'|\’)t ", " not ", text)

    text = re.sub(r"(\'|\’)ve ", " have ", text)



    text = re.sub(

        r'(by|been|and|are|for|it|TV|already|justhow|some|had|is|will|would|should|shall|must|can|his|here|there|them|these|their|has|have|the|be|that|not|was|he|just|they|who)(how)',

        '\g<1> \g<2>', text)



    return text
train[TEXT_COLUMN] = train[TEXT_COLUMN].progress_apply(lambda x: fixing_with_regex(x))
def clean_number(text: str) -> str:

    """

    Cleans numbers.



    :param text: text to clean

    :return: cleaned text

    """

    text = re.sub(r'(\d+)([a-zA-Z])', '\g<1> \g<2>', text)

    text = re.sub(r'(\d+) (th|st|nd|rd) ', '\g<1>\g<2> ', text)

    text = re.sub(r'(\d+),(\d+)', '\g<1>\g<2>', text)

    text = re.sub(r'(\d+),', '\g<1>', text)

    text = re.sub(r'(\d+)(e)(\d+)', '\g<1> \g<3>', text)



    return text
train[TEXT_COLUMN] = train[TEXT_COLUMN].progress_apply(lambda x: clean_number(x))
tokenizer_filter = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'



# Create a text tokenizer.

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters=tokenizer_filter)

tokenizer.fit_on_texts(train[TEXT_COLUMN])
def load_embeddings(path):

    embeddings_dict = {}

    with open(path) as f:

        for line in f:

            values = line.strip().split(' ')

            word = values[0]

            coef = np.asarray(values[1:], dtype='float32')

            if len(coef) == 300:

                embeddings_dict[word] = coef

    return embeddings_dict
%%time

embeddings_dict = {

    **load_embeddings(EMBEDDINGS_FILES[0]),

    **load_embeddings(EMBEDDINGS_FILES[1]),

    **load_embeddings(EMBEDDINGS_FILES[2])

}
def build_matrix(word_index, embeddings_dict):

    embeddings_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDINGS_DIMENSION))



    for word, i in word_index.items():

        embedding_vector = embeddings_dict.get(word)

        if embedding_vector is not None:

            embeddings_matrix[i] = embedding_vector

    

    return embeddings_matrix    
%%time

embeddings_matrix = build_matrix(tokenizer.word_index, embeddings_dict)
def check_coverage(vocab, embeddings_index):

    known_words = {}

    unknown_words = {}

    nb_known_words = 0

    nb_unknown_words = 0

    for word in vocab.keys():

        try:

            known_words[word] = embeddings_index[word]

            nb_known_words += vocab[word]

        except:

            unknown_words[word] = vocab[word]

            nb_unknown_words += vocab[word]

            pass



    print('Found embeddings for {:.3%} of vocab'.format(len(known_words) / len(vocab)))

    print('Found embeddings for  {:.3%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))

    unknown_words = sorted(unknown_words.items(),  key=lambda kv: kv[1])[::-1]



    return unknown_words
check_coverage(tokenizer.word_counts, embeddings_dict)
with open('embedding_matrix.pickle', 'wb') as f:

    pkl.dump(embeddings_matrix, f)
with open('tokenizer.pickle', 'wb') as f:

    pkl.dump(tokenizer, f)
with open('preprocessed_train_data.pickle', 'wb') as f:

    pkl.dump(train, f)