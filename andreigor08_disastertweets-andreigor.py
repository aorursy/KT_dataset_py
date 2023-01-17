import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string

import re # Regular expressions



# Sklearn

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix, classification_report





# Visualization

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import seaborn as sns



# NLTK

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk import pos_tag

from nltk.stem import PorterStemmer



# Keras

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.preprocessing import text

from keras.preprocessing import sequence

from keras.models import Sequential, load_model

from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import Adadelta, Adam, RMSprop

from keras.utils import np_utils

from keras.layers.embeddings import Embedding

from keras.layers import Flatten

from keras.layers.recurrent import LSTM, GRU

from keras.layers import GlobalAveragePooling1D,Lambda,Input,GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D,TimeDistributed
X_train = pd.read_csv('../input/nlp-getting-started/train.csv')



X_test = pd.read_csv('../input/nlp-getting-started/test.csv')
X_train.head()
X_test.head()
print('Train dataset shape: {}'.format(X_train.shape))

print('Test dataset shape: {}'.format(X_test.shape))
print('Disaster tweet:')

print(X_train.text[0])

print('---------------------------')

print('Normal tweet:')

print(X_train.text[24])
print('Amount of missing data by column in the training dataset')

print(X_train.isnull().sum())

print('------------------------------------')

print('Porcentagem de keywords faltantes no train dataset: {:.3f}%'.format(X_train.keyword.isnull().sum()/X_train.shape[0]*100))

print('Porcentagem de location faltantes no train dataset: {:.3f}%'.format(X_train.location.isnull().sum()/X_train.shape[0]*100))
print('Amount of missing data by column in the training dataset:')

print(X_test.isnull().sum())

print('------------------------------------')

print('Porcentagem de keywords faltantes no test dataset: {:.3f}%'.format(X_test.keyword.isnull().sum()/X_test.shape[0]*100))

print('Porcentagem de location faltantes no test dataset: {:.3f}%'.format(X_test.location.isnull().sum()/X_test.shape[0]*100))
# Criando a representação, área de plot

fig1, ax1 = plt.subplots(figsize = (4,4))



# Conjunto de dados a ser representado

sns.set(style="darkgrid")

targets = X_train.target.value_counts()

labels = ['Not Disaster', 'Disaster']



# Criando o gŕafico

ax1.pie(targets, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 90, colors = ['skyblue',(0.8,0.4,0.4)])



# Opções Adicionais

plt.title('Target class distribution')

ax1.axis('equal')



# Mostrando o gŕafico

plt.show()
def plot_distplot(disaster_data, non_disater_data, feature):

    fig, axes = plt.subplots(figsize = [5,5], nrows = 1, ncols = 1, dpi = 100)



    sns.set(style = 'darkgrid')

    sns.distplot(non_disater_data, label = 'Not Disaster', ax = axes, color = 'blue')

    sns.distplot(disaster_data, label = 'Disaster', ax = axes, color = 'red')

    axes.set_xlabel('')

    axes.tick_params(axis='x', labelsize=12)

    axes.tick_params(axis='y', labelsize=12)

    axes.legend()

    axes.set_title(f'{feature} Distribution in Training set')



    plt.show()

# Number of characteres in tweets

disaster_char_len = X_train[X_train['target']==1]['text'].map(lambda x: len(str(x)))

non_disaste_char_len = X_train[X_train['target']==0]['text'].map(lambda x: len(str(x)))



plot_distplot(disaster_char_len,non_disaste_char_len, 'Number of Characters' )
# Number of words

total_words_disaster = X_train[X_train.target == 1]['text'].map(lambda x: len(list(x.split())))



total_words_non_disaster = X_train[X_train.target == 0]['text'].map(lambda x: len(list(x.split())))



plot_distplot(total_words_disaster,total_words_non_disaster, feature = 'Total Words')
# Mean word lenght



mean_word_disaster = X_train[X_train.target == 1]['text'].map(lambda x: np.mean([len(item) for item in list(x.split())]))

mean_word_non_disaster = X_train[X_train.target == 0]['text'].map(lambda x: np.mean([len(item) for item in list(x.split())]))



plot_distplot(mean_word_disaster,mean_word_non_disaster, feature = 'Mean Word Lenght' )
# Amount of hashtags



hashtags_disaster = X_train[X_train.target == 1]['text'].map(lambda x: str(x).count('#'))

hashtags_not_disaster = X_train[X_train.target == 0]['text'].map(lambda x: str(x).count('#'))



fig, axes = plt.subplots(figsize = [5,5], nrows = 1, ncols = 1, dpi = 100)



sns.set(style = 'darkgrid')

sns.distplot(hashtags_not_disaster, label = 'Not Disaster', ax = axes, color = 'blue', kde_kws = {'bw':0.1})

sns.distplot(hashtags_disaster, label = 'Disaster', ax = axes, color = 'red',kde_kws = {'bw':'scott'})

axes.set_xlabel('')

axes.tick_params(axis='x', labelsize=12)

axes.tick_params(axis='y', labelsize=12)

axes.legend()

axes.set_title('Amount of Hashtags Distribution in Training set')



plt.show()
data = {'I':[1,1], 'like': [1,0], 'hate': [0,1], 'databases': [1,1]}

DTM_Example = pd.DataFrame(data, index = ['D1', 'D2'])



DTM_Example
def document_term_matrix(dataframe, ngrams = (1,1)):

    """

    Returns a DataFrame that is a DTM with the specified N-Grams and the text data as input

    """

    cv = CountVectorizer(stop_words = "english", ngram_range=ngrams, min_df = 5)

    data_cv = cv.fit_transform(dataframe.text) #Learn the vocabulary dictionary and return document-term matrix.

    

    # data_cv.toarray() returns an array representation of the bag of words

    # get_feature_names returns all the words in the corpus in array-like form

    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names()) # create the dtm_dataframe

    

    # reset the index to match de input dataframe index

    data_dtm.index = dataframe.index

    data_dtm['target_label'] = X_train.target

    return data_dtm





# Top words for each category



def top_ngrams(k, df):

    '''

    Returns the k top commom ngrams for disaster and for not disaster tweets, given a DTM.

    '''

    

    # Not Disaster

    

    # Get only the normal tweets

    target_label_df_ndisaster = pd.DataFrame(df[df['target_label'] == 0])

    

    # Aggregate the top ngrams

    word_counts_ndisaster = [target_label_df_ndisaster[column].sum() for column in target_label_df_ndisaster.columns]

    target_label_df_ndisaster = pd.DataFrame(word_counts_ndisaster, target_label_df_ndisaster.columns)

    

    # Sort from descending order

    top_words_ndisaster = target_label_df_ndisaster[0].sort_values(ascending = False)[0:k]

    

    

    # Disaster

    

    # Get only the disaster tweets

    target_label_df_disaster = pd.DataFrame(df[df['target_label'] == 1]).drop('target_label', axis = 1)

    

    # Aggregate the top ngrams

    word_counts_disaster = [target_label_df_disaster[column].sum() for column in target_label_df_disaster.columns]

    target_label_df_disaster = pd.DataFrame(word_counts_disaster, target_label_df_disaster.columns)

    

    # Sort from descending order

    top_words_disaster = target_label_df_disaster[0].sort_values(ascending = False)[0:k]

    

    return top_words_ndisaster, top_words_disaster

    

    

def plot_top_ngrams(not_disaster_ngrams, disaster_ngrams):

    """

    Plots a bar graph showing the top n-grams for disaster and non disaster tweets

    """

    # Gets the N of N-gram

    n_gram = len((not_disaster_ngrams.index[0]).split())

    n_gram_dict = {1: 'Words', 2: 'Bigrams', 3: 'Trigrams'}

    

    

    fig, axes = plt.subplots(figsize = [24,12], nrows = 1, ncols = 2)

    sns.set(style="darkgrid")

    sns.barplot(y = not_disaster_ngrams.index, x = not_disaster_ngrams.values, ax = axes[0])

    sns.barplot(y = disaster_ngrams.index, x = disaster_ngrams.values, ax = axes[1])



    axes[0].set_title(f'Top Non Disaster {n_gram_dict[n_gram]}')

    axes[0].set_xlabel(f'{n_gram_dict[n_gram]} Count')

    axes[0].set_ylabel(f'{n_gram_dict[n_gram]}')



    axes[1].set_title(f'Top Disaster {n_gram_dict[n_gram]}')

    axes[1].set_xlabel(f'{n_gram_dict[n_gram]} Count')

    axes[1].set_ylabel(f'{n_gram_dict[n_gram]}')



    plt.tight_layout()

    plt.show()
def generate_display_wordclouds(dataframe):

    full_text_non_disaster = ''.join(dataframe[dataframe['target'] == 0].text)

    full_text_disaster = ''.join(dataframe[dataframe['target'] == 1].text)



        

    wc_not_disaster = WordCloud(background_color="white", colormap="Dark2",

                   max_font_size=150, random_state=42)

    

    wc_disaster = WordCloud(background_color="white", colormap="Dark2",

                   max_font_size=150, random_state=42)

    

    wc_not_disaster.generate(full_text_non_disaster)

    wc_disaster.generate(full_text_disaster)

    

    

    fig, axes = plt.subplots(figsize = [20,8], nrows = 1, ncols = 2)

    

    axes[0].imshow(wc_not_disaster, interpolation = 'bilinear')

    axes[1].imshow(wc_disaster, interpolation = 'bilinear')

    axes[0].axis("off")

    axes[1].axis("off")

    

    

    axes[0].set_title('Not Disaster')

    axes[1].set_title('Disaster')

    

    

    

    

    



    
# Document Term Matrix from the original input

raw_data_dtm = document_term_matrix(X_train)



# Get the top disaster and not disaster words

top_ndisaster_words, top_disaster_words = top_ngrams(20, raw_data_dtm)



# Plot the top words

plot_top_ngrams(top_ndisaster_words,top_disaster_words)

# Generate the raw data word clouds

generate_display_wordclouds(X_train)
abbreviations = {

    "$" : " dollar ",

    "€" : " euro ",

    "4ao" : "for adults only",

    "a.m" : "before midday",

    "a3" : "anytime anywhere anyplace",

    "aamof" : "as a matter of fact",

    "acct" : "account",

    "adih" : "another day in hell",

    "afaic" : "as far as i am concerned",

    "afaict" : "as far as i can tell",

    "afaik" : "as far as i know",

    "afair" : "as far as i remember",

    "afk" : "away from keyboard",

    "app" : "application",

    "approx" : "approximately",

    "apps" : "applications",

    "asap" : "as soon as possible",

    "asl" : "age, sex, location",

    "atk" : "at the keyboard",

    "ave." : "avenue",

    "aymm" : "are you my mother",

    "ayor" : "at your own risk", 

    "b&b" : "bed and breakfast",

    "b+b" : "bed and breakfast",

    "b.c" : "before christ",

    "b2b" : "business to business",

    "b2c" : "business to customer",

    "b4" : "before",

    "b4n" : "bye for now",

    "b@u" : "back at you",

    "bae" : "before anyone else",

    "bak" : "back at keyboard",

    "bbbg" : "bye bye be good",

    "bbc" : "british broadcasting corporation",

    "bbias" : "be back in a second",

    "bbl" : "be back later",

    "bbs" : "be back soon",

    "be4" : "before",

    "bfn" : "bye for now",

    "blvd" : "boulevard",

    "bout" : "about",

    "brb" : "be right back",

    "bros" : "brothers",

    "brt" : "be right there",

    "bsaaw" : "big smile and a wink",

    "btw" : "by the way",

    "bwl" : "bursting with laughter",

    "c/o" : "care of",

    "cet" : "central european time",

    "cf" : "compare",

    "cia" : "central intelligence agency",

    "csl" : "can not stop laughing",

    "cu" : "see you",

    "cul8r" : "see you later",

    "cv" : "curriculum vitae",

    "cwot" : "complete waste of time",

    "cya" : "see you",

    "cyt" : "see you tomorrow",

    "dae" : "does anyone else",

    "dbmib" : "do not bother me i am busy",

    "diy" : "do it yourself",

    "dm" : "direct message",

    "dwh" : "during work hours",

    "e123" : "easy as one two three",

    "eet" : "eastern european time",

    "eg" : "example",

    "embm" : "early morning business meeting",

    "encl" : "enclosed",

    "encl." : "enclosed",

    "etc" : "and so on",

    "faq" : "frequently asked questions",

    "fawc" : "for anyone who cares",

    "fb" : "facebook",

    "fc" : "fingers crossed",

    "fig" : "figure",

    "fimh" : "forever in my heart", 

    "ft." : "feet",

    "ft" : "featuring",

    "ftl" : "for the loss",

    "ftw" : "for the win",

    "fwiw" : "for what it is worth",

    "fyi" : "for your information",

    "g9" : "genius",

    "gahoy" : "get a hold of yourself",

    "gal" : "get a life",

    "gcse" : "general certificate of secondary education",

    "gfn" : "gone for now",

    "gg" : "good game",

    "gl" : "good luck",

    "glhf" : "good luck have fun",

    "gmt" : "greenwich mean time",

    "gmta" : "great minds think alike",

    "gn" : "good night",

    "g.o.a.t" : "greatest of all time",

    "goat" : "greatest of all time",

    "goi" : "get over it",

    "gps" : "global positioning system",

    "gr8" : "great",

    "gratz" : "congratulations",

    "gyal" : "girl",

    "h&c" : "hot and cold",

    "hp" : "horsepower",

    "hr" : "hour",

    "hrh" : "his royal highness",

    "ht" : "height",

    "ibrb" : "i will be right back",

    "ic" : "i see",

    "icq" : "i seek you",

    "icymi" : "in case you missed it",

    "idc" : "i do not care",

    "idgadf" : "i do not give a damn fuck",

    "idgaf" : "i do not give a fuck",

    "idk" : "i do not know",

    "ie" : "that is",

    "i.e" : "that is",

    "ifyp" : "i feel your pain",

    "IG" : "instagram",

    "iirc" : "if i remember correctly",

    "ilu" : "i love you",

    "ily" : "i love you",

    "imho" : "in my humble opinion",

    "imo" : "in my opinion",

    "imu" : "i miss you",

    "iow" : "in other words",

    "irl" : "in real life",

    "j4f" : "just for fun",

    "jic" : "just in case",

    "jk" : "just kidding",

    "jsyk" : "just so you know",

    "l8r" : "later",

    "lb" : "pound",

    "lbs" : "pounds",

    "ldr" : "long distance relationship",

    "lmao" : "laugh my ass off",

    "lmfao" : "laugh my fucking ass off",

    "lol" : "laughing out loud",

    "ltd" : "limited",

    "ltns" : "long time no see",

    "m8" : "mate",

    "mf" : "motherfucker",

    "mfs" : "motherfuckers",

    "mfw" : "my face when",

    "mofo" : "motherfucker",

    "mph" : "miles per hour",

    "mr" : "mister",

    "mrw" : "my reaction when",

    "ms" : "miss",

    "mte" : "my thoughts exactly",

    "nagi" : "not a good idea",

    "nbc" : "national broadcasting company",

    "nbd" : "not big deal",

    "nfs" : "not for sale",

    "ngl" : "not going to lie",

    "nhs" : "national health service",

    "nrn" : "no reply necessary",

    "nsfl" : "not safe for life",

    "nsfw" : "not safe for work",

    "nth" : "nice to have",

    "nvr" : "never",

    "nyc" : "new york city",

    "oc" : "original content",

    "og" : "original",

    "ohp" : "overhead projector",

    "oic" : "oh i see",

    "omdb" : "over my dead body",

    "omg" : "oh my god",

    "omw" : "on my way",

    "p.a" : "per annum",

    "p.m" : "after midday",

    "pm" : "prime minister",

    "poc" : "people of color",

    "pov" : "point of view",

    "pp" : "pages",

    "ppl" : "people",

    "prw" : "parents are watching",

    "ps" : "postscript",

    "pt" : "point",

    "ptb" : "please text back",

    "pto" : "please turn over",

    "qpsa" : "what happens", #"que pasa",

    "ratchet" : "rude",

    "rbtl" : "read between the lines",

    "rlrt" : "real life retweet", 

    "rofl" : "rolling on the floor laughing",

    "roflol" : "rolling on the floor laughing out loud",

    "rotflmao" : "rolling on the floor laughing my ass off",

    "rt" : "retweet",

    "ruok" : "are you ok",

    "sfw" : "safe for work",

    "sk8" : "skate",

    "smh" : "shake my head",

    "sq" : "square",

    "srsly" : "seriously", 

    "ssdd" : "same stuff different day",

    "tbh" : "to be honest",

    "tbs" : "tablespooful",

    "tbsp" : "tablespooful",

    "tfw" : "that feeling when",

    "thks" : "thank you",

    "tho" : "though",

    "thx" : "thank you",

    "tia" : "thanks in advance",

    "til" : "today i learned",

    "tl;dr" : "too long i did not read",

    "tldr" : "too long i did not read",

    "tmb" : "tweet me back",

    "tntl" : "trying not to laugh",

    "ttyl" : "talk to you later",

    "u" : "you",

    "u2" : "you too",

    "u4e" : "yours for ever",

    "utc" : "coordinated universal time",

    "w/" : "with",

    "w/o" : "without",

    "w8" : "wait",

    "wassup" : "what is up",

    "wb" : "welcome back",

    "wtf" : "what the fuck",

    "wtg" : "way to go",

    "wtpa" : "where the party at",

    "wuf" : "where are you from",

    "wuzup" : "what is up",

    "wywh" : "wish you were here",

    "yd" : "yard",

    "ygtr" : "you got that right",

    "ynk" : "you never know",

    "zzz" : "sleeping bored and tired"

}
contractions = { 

"ain't": "is not",

"aren't": "are not",

"can't": "cannot",

"can't've": "cannot have",

"'cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"hadn't": "had not",

"hadn't've": "had not have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he would",

"he'd've": "he would have",

"he'll": "he will",

"he'll've": "he will have",

"he's": "he is",

"how'd": "how did",

"how'd'y": "how do you",

"how'll": "how will",

"how's": "how has",

"i'd": "i had",

"i'd've": "i would have",

"i'll": "i shall",

"i'll've": "i will have",

"i'm": "i am",

"i've": "i have",

"isn't": "is not",

"it'd": "it had ",

"it'd've": "it would have",

"it'll": "it will",

"it'll've": "it will have",

"it's": "it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"might've": "might have",

"mightn't": "might not",

"mightn't've": "might not have",

"must've": "must have",

"mustn't": "must not",

"mustn't've": "must not have",

"needn't": "need not",

"needn't've": "need not have",

"o'clock": "of the clock",

"oughtn't": "ought not",

"oughtn't've": "ought not have",

"shan't": "shall not",

"sha'n't": "shall not",

"shan't've": "shall not have",

"she'd": "she would",

"she'd've": "she would have",

"she'll": "she will",

"she'll've": "she will have",

"she's": "she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"so've": "so have",

"so's": "so is",

"that'd": "that had",

"that'd've": "that would have",

"that's": "that is",

"there'd": "there would",

"there'd've": "there would have",

"there's": "there is",

"they'd": "they would",

"they'd've": "they would have",

"they'll": "they will",

"they'll've": "they will have",

"they're": "they are",

"they've": "they have",

"to've": "to have",

"wasn't": "was not",

"we'd": "we would",

"we'd've": "we would have",

"we'll": "we will",

"we'll've": "we will have",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'll": "what will",

"what'll've": "what will have",

"what're": "what are",

"what's": "what is",

"what've": "what have",

"when's": "when is",

"when've": "when have",

"where'd": "where did",

"where's": "where is",

"where've": "where have",

"who'll": "who will",

"who'll've": "who will have",

"who's": "who is",

"who've": "who have",

"why's": "why is",

"why've": "why have",

"will've": "will have",

"won't": "will not",

"won't've": "will not have",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

"y'all'd": "you all would",

"y'all'd've": "you all would have",

"y'all're": "you all are",

"y'all've": "you all have",

"you'd": "you would",

"you'd've": "you would have",

"you'll": "you will",

"you'll've": "you will have",

"you're": "you are",

"you've": "you have"

}
def fix_abreviations(text):

    """

    Expand te abreviations (slangs) into its formal word

    """

    words = text.split(' ')

    clean_text = []

    for word in words:

        clean_text.append(abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word)

    

    clean_text = ' '.join(clean_text)

    return clean_text



def fix_contractions(text):

    """

    Expands the english contractions

    """

    words = text.split(' ')

    clean_text = []

    for word in words:

        clean_text.append(contractions[word.lower()] if word.lower() in contractions.keys() else word)

    

    clean_text = ' '.join(clean_text)

    

    return clean_text



def clean_data(text):

    text = text.lower()

    

    # Fix contractions

    text = fix_contractions(text)

    

    # Fix abreviations

    

    text = fix_abreviations(text)

    

    

    text = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', text) # remove punctuations

    text = re.sub(r'\n', ' ', text)  # remove line breaks

    text = re.sub(r'https?://\S+', '', text) # remove links

    text = re.sub(r'\s+', ' ', text) # remove unnecessary spacings

    text = re.sub(r'\w*\d\w*', '', text) # remove words containing numbers



    # Remove Special characters -  adapted from https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert#3.-Target-and-N-grams

    text = re.sub(r"û_", "", text)

    text = re.sub(r"ûò", "", text)

    text = re.sub(r"ûòåê", "", text)

    text = re.sub(r"ûó", "", text)

    text = re.sub(r"ûó", "", text)

    text = re.sub(r"û÷", "", text)

    text = re.sub(r"ûª", "", text)

    text = re.sub(r"å_", "", text)

    text = re.sub(r"ûï", "", text)

    text = re.sub(r"åê", "", text)

    text = re.sub(r"åè", "", text)

    text = re.sub(r"å¨", "", text)

    text = re.sub(r"åÇ", "", text)

    text = re.sub(r"ìñ", "", text)

    text = re.sub(r"ìñ", "", text)

    text = re.sub(r"ìü", "", text)

    text = re.sub(r"ââ", "", text)

    text = re.sub(r"åç", "", text)



    

    

    return text

    

clean_df = pd.DataFrame(X_train['text'].apply(clean_data))

clean_df['target'] = X_train.target



clean_df.head()
print('Disaster tweets cleaned')

print('Before:\n {} \n After:\n {}'.format(X_train.text[5], clean_df.text[5]))

print('##############################################')

print('Before:\n {} \n After:\n {}'.format(X_train.text[8], clean_df.text[8]))

print('##############################################')

print('Before:\n {} \n After:\n {}'.format(X_train.text[12], clean_df.text[12]))

print('##############################################')

print('---------------------------------------------')

print('Normal tweets cleaned')

print('Before:\n {} \n After:\n {}'.format(X_train.text[30], clean_df.text[30]))

print('##############################################')

print('Before:\n {} \n After:\n {}'.format(X_train.text[34], clean_df.text[34]))

print('##############################################')

print('Before:\n {} \n After:\n {}'.format(X_train.text[42], clean_df.text[42]))

# Document Term Matrix cleaned

clean_data_dtm = document_term_matrix(clean_df)

dtm_bigrams = document_term_matrix(clean_df, ngrams = (2,2))

dtm_trigrams = document_term_matrix(clean_df, ngrams = (3,3))





# Get the top disaster and not disaster words

top_ndisaster_words, top_disaster_words = top_ngrams(20, clean_data_dtm)



top_ndisaster_bigrams, top_disaster_bigrams = top_ngrams(20, dtm_bigrams)



top_ndisaster_trigrams, top_disaster_trigrams = top_ngrams(20, dtm_trigrams)



plot_top_ngrams(top_ndisaster_words,top_disaster_words)
plot_top_ngrams(top_ndisaster_bigrams,top_disaster_bigrams)
plot_top_ngrams(top_ndisaster_trigrams,top_disaster_trigrams)
generate_display_wordclouds(clean_df)
def remove_words_in_commom(top_disaster_words, top_ndisaster_words):

    top_disaster_words = top_disaster_words.index

    top_ndisaster_words = top_ndisaster_words.index



    remove_word = [word for word in top_ndisaster_words if word in top_disaster_words]

    remove_word = remove_word[0:-1]

    remove_word.append('wa')

    return remove_word
def preprocessing(text):

    # clean the text using the clean_data function

    preprocessed_text = clean_data(text)

    

    #  tokenizes the sentences into words based on whitespaces

    tokens = [word for sent in nltk.sent_tokenize(preprocessed_text) for word in nltk.word_tokenize(sent)]

    stopwds = stopwords.words('english')

    tokens = [token for token in tokens if token not in stopwds]

    

    

    # Lemmatization

    tagged_corpus = pos_tag(tokens)

    Noun_tags = ['NN','NNP','NNPS','NNS']

    Verb_tags = ['VB','VBD','VBG','VBN','VBP','VBZ']

    lemmatizer = WordNetLemmatizer()

    

    # The following function, prat_lemmatize, has been created only for the reasons of mismatch 

    # between the pos_tag function and intake values of lemmatize function. 

    def prat_lemmatize(token, tag):

        if tag in Noun_tags:

            return lemmatizer.lemmatize(token,'n')

        elif tag in Verb_tags:

            return lemmatizer.lemmatize(token,'v')

        else:

            return lemmatizer.lemmatize(token,'n')

     

    pre_proc_text =   " ".join([prat_lemmatize(token,tag) for token,tag in tagged_corpus])

    return pre_proc_text

    
# Getting only the target column

y = X_train.target 



# Getting only the text column



X = X_train.text



# Splitting the data into training and validation

x_train, x_test, y_train, y_test = train_test_split(

                                    X, y, test_size=0.33, random_state=42)
# Preprocessing train and test sets

x_train = x_train.apply(preprocessing)

x_test = x_test.apply(preprocessing)



# Creating a TfidfVectorizer object.

# words and bigrams that appear less then 5 times in the vocabulary dont add a lot of information and are removed

tfidf = TfidfVectorizer(min_df = 5, ngram_range = (1,2), stop_words = 'english')



# Learning the vocabulary from the train set

text_vec = tfidf.fit_transform(x_train)



# Transforming the test set into a TF-IDF representation

test_vec_test = tfidf.transform(x_test)



# Getting the results dataframes

x_train = pd.DataFrame(text_vec.toarray(), columns = tfidf.get_feature_names())

x_test = pd.DataFrame(test_vec_test.toarray(), columns = tfidf.get_feature_names())
lr = LogisticRegression(solver='liblinear', random_state=777)



# This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.

scaler = MinMaxScaler()

pipeline = Pipeline([('scale',scaler), ('lr', lr),])



pipeline.fit(x_train, y_train)



print ('Training accuracy: %.4f' % pipeline.score(x_train, y_train))

print ('Test accuracy: %.4f' % pipeline.score(x_test, y_test))

# Splitting the data into training and validation

x_train, x_test, y_train, y_test = train_test_split(

                                    X, y, test_size=0.33, random_state=42)
# Cleaning the train and test sets

x_train = pd.DataFrame(x_train.apply(clean_data))

x_test = pd.DataFrame(x_test.apply(clean_data))

# Creating the tokenizer object

tokenizer = text.Tokenizer()



# Leaning the toknes

tokenizer.fit_on_texts(x_train.text)



# Size of the vocabulary

vocab_size = len(tokenizer.word_index) + 1



# Applying the tokenization 

encoded_docs = tokenizer.texts_to_sequences(x_train.text)

encoded_docs_test = tokenizer.texts_to_sequences(x_test.text)



max_lenght = 25



# Padding the documents

padded_docs = sequence.pad_sequences(encoded_docs, maxlen= max_lenght, padding = 'post')

padded_docs_test = sequence.pad_sequences(encoded_docs_test, maxlen= max_lenght, padding = 'post')
padded_docs.shape
model = Sequential()

model.add(Embedding(vocab_size, 100, input_length=max_lenght))

model.add(SpatialDropout1D(0.2))

model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))

# compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# summarize the model

print(model.summary())
# fit the model

model.fit(padded_docs, y_train, epochs=3, verbose=1, validation_data = (padded_docs_test, y_test))

# evaluate the model

loss, accuracy = model.evaluate(padded_docs, y_train, verbose=0)

loss, accuracy_val = model.evaluate(padded_docs_test, y_test, verbose=0)



print('Train Accuracy: %f' % (accuracy*100))

print('Test Accuracy: %f' % (accuracy_val*100))
# Splitting the data into training and validation

x_train, x_test, y_train, y_test = train_test_split(

                                    X, y, test_size=0.33, random_state=42)



# Cleaning the train and test sets

x_train = pd.DataFrame(x_train.apply(clean_data))

x_test = pd.DataFrame(x_test.apply(clean_data))



# Creating the tokenizer object

tokenizer = text.Tokenizer()



# Leaning the toknes

tokenizer.fit_on_texts(x_train.text)



# Size of the vocabulary

vocab_size = len(tokenizer.word_index) + 1



# Applying the tokenization 

encoded_docs = tokenizer.texts_to_sequences(x_train.text)

encoded_docs_test = tokenizer.texts_to_sequences(x_test.text)



max_lenght = 25



# Padding the documents

padded_docs = sequence.pad_sequences(encoded_docs, maxlen= max_lenght, padding = 'post')

padded_docs_test = sequence.pad_sequences(encoded_docs_test, maxlen= max_lenght, padding = 'post')
embeddings_index = dict()

with open('../input/glovetwitter/glove.twitter.27B.100d.txt','r') as f:

    for line in f:

        values = line.split()

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs

f.close()

print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs

embedding_matrix = np.zeros((vocab_size, 100))

for word, i in tokenizer.word_index.items():

	embedding_vector = embeddings_index.get(word)

	if embedding_vector is not None:

		embedding_matrix[i] = embedding_vector
model2 = Sequential()

e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=25, trainable=False)

model2.add(e)

model2.add(SpatialDropout1D(0.2))

model2.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))

model2.add(Dense(1, activation='sigmoid'))

model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model2.summary())



# CallBack Function



mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
# Training

history = model2.fit(padded_docs, y_train, epochs=5, verbose=1, validation_data = (padded_docs_test, y_test), callbacks=[mc])

# evaluate the model

loss, accuracy = model2.evaluate(padded_docs_test, y_test, verbose=0)

print('Validation Accuracy: %f' % (accuracy*100))
saved_model = load_model('best_model.h5')
y_pred = np.round(saved_model.predict(padded_docs_test))

cm = confusion_matrix(y_test, y_pred)/y_test.shape[0]*100


sns.set()

fig, ax0 = plt.subplots(figsize = (14,7))



ax = sns.heatmap(data = cm, annot=True, fmt = '.1f', square=1, linewidths=.5, cmap="YlGnBu")

ax0.set_title('Confusion Matrix')

for t in ax.texts: t.set_text(t.get_text() + " %")

plt.show()
# Applying the preprocessing steps to the X_test dataset

X_test = pd.DataFrame(X_test.text.apply(clean_data))

X_test = tokenizer.texts_to_sequences(X_test.text)

X_test = sequence.pad_sequences(X_test, maxlen= max_lenght, padding = 'post')
# Predicting the results



predictions = saved_model.predict(X_test)



# Making the submission dataframe

X_submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

X_submission['target'] = np.round(predictions)

X_submission['target'] = X_submission['target'].astype(int)
X_submission.to_csv('sub.csv', index = False)