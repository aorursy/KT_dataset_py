import numpy as np

import pandas as pd

import re

import string

import unicodedata

from collections import Counter

import os

import pickle



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



# Important! NLTK lemmatizer needs a POS (parts of speech) tags.

# https://www.kaggle.com/alvations/basic-nlp-with-nltk

from nltk.stem import WordNetLemmatizer, PorterStemmer

from nltk import pos_tag

from nltk import word_tokenize

import nltk
def load_data(file_prefix=''):

    """

    Load test and train data from csv.

    Parameters

    __________

    file_prefix: str

        Optional prefix to add to "train.csv" or "test.csv" file names

    

    Returns

    _______

    df_train: DataFrame

        Full raw training dataset

    df_test: DataFrame

        Full raw test dataset

    """

    

    # Select local path vs kaggle kernel

    path = os.getcwd()

    if 'data-projects/kaggle_quora/notebooks' in path:

        data_dir = '../data/raw/'

    else:

        data_dir = '../input/'



    df_train = pd.read_csv(data_dir + file_prefix +'train.csv')

    df_test = pd.read_csv(data_dir + file_prefix +'test.csv')

    return df_train, df_test



def normalize_unicode(text):

    """

    author: Kevin Liao

    unicode string normalization

    """

    return unicodedata.normalize('NFKD', text)





def remove_newline(text):

    """

    author: Kevin Liao

    remove \n and  \t

    """

    text = re.sub('\n', ' ', text)

    text = re.sub('\t', ' ', text)

    text = re.sub('\b', ' ', text)

    text = re.sub('\r', ' ', text)

    return text



def clean_latex(text):

    """

    author: Kevin Liao

    convert r"[math]\vec{x} + \vec{y}" to English

    """

    # edge case

    text = re.sub(r'\[math\]', ' LaTex math ', text)

    text = re.sub(r'\[\/math\]', ' LaTex math ', text)

    text = re.sub(r'\\', ' LaTex ', text)



    pattern_to_sub = {

        r'\\mathrm': ' LaTex math mode ',

        r'\\mathbb': ' LaTex math mode ',

        r'\\boxed': ' LaTex equation ',

        r'\\begin': ' LaTex equation ',

        r'\\end': ' LaTex equation ',

        r'\\left': ' LaTex equation ',

        r'\\right': ' LaTex equation ',

        r'\\(over|under)brace': ' LaTex equation ',

        r'\\text': ' LaTex equation ',

        r'\\vec': ' vector ',

        r'\\var': ' variable ',

        r'\\theta': ' theta ',

        r'\\mu': ' average ',

        r'\\min': ' minimum ',

        r'\\max': ' maximum ',

        r'\\sum': ' + ',

        r'\\times': ' * ',

        r'\\cdot': ' * ',

        r'\\hat': ' ^ ',

        r'\\frac': ' / ',

        r'\\div': ' / ',

        r'\\sin': ' Sine ',

        r'\\cos': ' Cosine ',

        r'\\tan': ' Tangent ',

        r'\\infty': ' infinity ',

        r'\\int': ' integer ',

        r'\\in': ' in ',

    }

    # post process for look up

    pattern_dict = {k.strip('\\'): v for k, v in pattern_to_sub.items()}

    # init re

    patterns = pattern_to_sub.keys()

    pattern_re = re.compile('(%s)' % '|'.join(patterns))



    def _replace(match):

        """

        reference: https://www.kaggle.com/hengzheng/attention-capsule-why-not-both-lb-0-694 # noqa

        """

        return pattern_dict.get(match.group(0).strip('\\'), match.group(0))

    return pattern_re.sub(_replace, text)



def decontracted(text):

    """

    author: Kevin Liao

    de-contract the contraction

    """

    try:

        # specific

        text = re.sub(r"(W|w)on(\'|\’)t", "will not", text)

        text = re.sub(r"(C|c)an(\'|\’)t", "can not", text)

        text = re.sub(r"(Y|y)(\'|\’)all", "you all", text)

        text = re.sub(r"(Y|y)a(\'|\’)ll", "you all", text)



        # general

        text = re.sub(r"(I|i)(\'|\’)m", "i am", text)

        text = re.sub(r"(A|a)in(\'|\’)t", "is not", text)

        text = re.sub(r"n(\'|\’)t", " not", text)

        text = re.sub(r"(\'|\’)re", " are", text)

        text = re.sub(r"(\'|\’)s", " is", text)

        text = re.sub(r"(\'|\’)d", " would", text)

        text = re.sub(r"(\'|\’)ll", " will", text)

        text = re.sub(r"(\'|\’)t(?!h)", " not", text)

        text = re.sub(r"(\'|\’)ve", " have", text)

    except:

        print('error processing text:{}'.format(text))

        

    return text



def remove_string(text, string_to_omit=['']):

    """

    author: Kevin Liao

    Substrings to delete if present.

    """    

    # light arg checking

    if type(string_to_omit) == str:

        string_to_omit = [string_to_omit]

    

    re_tok = re.compile(f'({string_to_omit})')

    return re_tok.sub(r'', text)    



def spacing_digit(text):

    """

    author: Kevin Liao

    add space before and after digits

    """

    re_tok = re.compile('([0-9])')

    return re_tok.sub(r' \1 ', text)





def spacing_number(text):

    """

    author: Kevin Liao

    add space before and after numbers

    """

    re_tok = re.compile('([0-9]{1,})')

    return re_tok.sub(r' \1 ', text)





def remove_number(text):

    """

    author: Kevin Liao

    numbers are not toxic

    """

    return re.sub('\d+', ' ', text)



def remove_space(text):

    """

    author: Kevin Liao

    remove extra spaces and ending space if any

    """

    text = re.sub('\s+', ' ', text)

    text = re.sub('\s+$', '', text)

    return text



def clean_misspell(text):

    """

    adapted from: Kevin Liao

    misspell list (quora vs. fasttext wiki-news-300d-1M)

    """

    misspell_to_sub = {

        '“': ' " ',

        '”': ' " ',

        '°C': 'degrees Celsius',

        '&amp;': ' & ',

        '2k17': '2017',

        '2k18': '2018',

        '9/11': 'terrorist attack',

        'Aadhar': 'Indian identification number',

        'aadhar': 'Indian identification number',

        ' adhar': 'Indian identification number',

        'Adityanath': 'Indian monk Yogi Adityanath',

        'AFCAT': 'Indian air force recruitment exam',

        'airhostess': 'air hostess',

        'Ambedkarite': 'Dalit Buddhist movement ',

        'AMCAT': 'Indian employment assessment examination',

        'and/or': 'and or',

        'antibrahmin': 'anti Brahminism',

        'articleship': 'chartered accountant internship',

        'Asifa': 'abduction rape murder case ',

        'AT&T': 'telecommunication company',

        'atrracted': 'attract',

        'Awadesh': 'Indian engineer Awdhesh Singh',

        'Awdhesh': 'Indian engineer Awdhesh Singh',

        'Babchenko': 'Arkady Arkadyevich Babchenko faked death',

        'Barracoon': 'Black slave',

        'Bathla': 'Namit Bathla',

        'bcom': 'bachelor of commerce',

        'beyon´çe': 'Beyoncé',

        'Bhakts': 'Bhakt',

        'bhakts': 'Bhakt',

        'bigdata': 'big data',

        'biharis': 'Biharis',

        'BIMARU': 'Bihar Madhya Pradesh Rajasthan Uttar Pradesh',

        'BITSAT': 'Birla Institute of Technology entrance examination',

        'BNBR': 'be nice be respectful',

        'bodycams': 'body cams',

        'bodyshame': 'body shaming',

        'bodyshoppers': 'body shopping',

        'Bolsonaro': 'Jair Bolsonaro',

        'Boshniak': 'Bosniaks ',

        'Boshniaks': 'Bosniaks',

        'bremainer': 'anti Brexit',

        'bremoaner': 'Brexit remainer',

        'Brexiteer': 'Brexit supporter',

        'Brexiteers': 'Brexit supporters',

        'Brexiter': 'Brexit supporter',

        'Brexiters': 'Brexit supporters',

        'brexiters': 'Brexit supporters',

        'Brexiting': 'Brexit',

        'Brexitosis': 'Brexit disorder',

        'Brexshit': 'Brexit bullshit',

        'C#': 'computer programming language',

        'c#': 'computer programming language',

        'C++': 'computer programming language',

        'c++': 'computer programming language',

        'Cananybody': 'Can any body',

        'cancelled': 'canceled',

        'Castrater': 'castration',

        'castrater': 'castration',

        'centre': 'center',

        'Chodu': 'fucker',

        'Chutiya': 'Tibet people ',

        'Chutiyas': 'Tibet people ',

        'cishet': 'cisgender and heterosexual person',

        'citicise': 'criticize',

        'cliché': 'cliche',

        'clichéd': 'cliche',

        'clichés': 'cliche',

        'Clickbait': 'click bait ',

        'clickbait': 'click bait ',

        'coinbase': 'bitcoin wallet',

        'Coinbase': 'bitcoin wallet',

        'colour': 'color',

        'COMEDK': 'medical engineering and dental colleges of Karnataka entrance examination',

        'counselling': 'counseling',

        'Crimean': 'Crimea people ',

        'currancies': 'currencies',

        'currancy': 'currency',

        'cybertrolling': 'cyber trolling',

        'D&D': 'dungeons & dragons game',

        'daesh': 'Islamic State of Iraq and the Levant',

        'deadbody': 'dead body',

        'deaddict': 'de addict',

        'demcoratic': 'Democratic',

        'demonetisation': 'demonetization',

        'demonetisation': 'demonetization',

        'Demonetization': 'demonetization',

        'demonitisation': 'demonetization',

        'demonitization': 'demonetization',

        'deplorables': 'deplorable',

        'doI': 'do I',

        'Doklam': 'disputed Indian Chinese border area',

        'Doklam': 'Tibet',

        'Dönmeh': 'Islam',

        'Dravidanadu': 'Dravida Nadu',

        'dropshipping': 'drop shipping',

        'Drumpf ': 'Donald Trump fool ',

        'Drumpfs': 'Donald Trump fools',

        'Dumbassistan': 'dumb ass Pakistan',

        'emiratis': 'Emiratis',

        'Eroupian': 'European',

        'Etherium': 'Ethereum',

        'Eurocentric': 'Eurocentrism ',

        'exboyfriend': 'ex boyfriend',

        'facetards': 'Facebook retards',

        'Fadnavis': 'Indian politician Devendra Fadnavis',

        'favourite': 'favorite',

        'Fck': 'Fuck',

        'fck': 'fuck',

        'Feku': 'The Man of India ',

        'feminazism': 'feminism nazi',

        'FIITJEE': 'Indian tutoring service',

        'fiitjee': 'Indian tutoring service',

        'fortnite': 'Fortnite ',

        'Fortnite': 'video game',

        'Gixxer': 'motorcycle',

        'Golang': 'computer programming language',

        'golang': 'computer programming language',

        'Gujratis': 'Gujarati',

        'Gurmehar': 'Gurmehar Kaur Indian student activist',

        'h1b': 'US work visa',

        'H1B': 'US work visa',

        'hairfall': 'hair loss',

        'harrase': 'harass',

        'he/she': 'he or she',

        'healhtcare': 'healthcare',

        'him/her': 'him or her',

        'Hindians': 'North Indian who hate British',

        'Hinduphobia': 'Hindu phobic',

        'hinduphobia': 'Hindu phobic',

        'Hinduphobic': 'Hindu phobic',

        'hinduphobic': 'Hindu phobic',

        'his/her': 'his or her',

        'Hongkongese': 'HongKong people',

        'hongkongese': 'HongKong people',

        'howcan': 'how can',

        'Howdo': 'How do',

        'howdo': 'how do',

        'howdoes': 'how does',

        'howmany': 'how many',

        'howmuch': 'how much',

        'HYPS': ' Harvard Yale Princeton Stanford',

        'HYPSM': ' Harvard Yale Princeton Stanford MIT',

        'ICOs': 'cryptocurrencies initial coin offering',

        'Idiotism': 'idiotism',

        'IITian': 'Indian Institutes of Technology student',

        'IITians': 'Indian Institutes of Technology students',

        'IITJEE': 'Indian Institutes of Technology entrance examination',

        ' incel': ' involuntary celibates',

        ' incels': ' involuntary celibates',

        'indans': 'Indian',

        'jallikattu': 'Jallikattu',

        'JEE MAINS': 'Indian university entrance examination',

        'Jewdar': 'Jew dar',

        'Jewism': 'Judaism',

        'jewplicate': 'jewish replicate',

        'JIIT': 'Jaypee Institute of Information Technology',

        'Kalergi': 'Coudenhove-Kalergi',

        'Kashmirians': 'Kashmirian',

        'Khalistanis': 'Sikh separatist movement',

        'Khazari': 'Khazars',

        'kompromat': 'compromising material',

        'koreaboo': 'Korea boo ',

        'KVPY': 'entrance examination',

        'labour': 'labor',

        'langague': 'language',

        'LGBTQ': 'lesbian  gay  bisexual  transgender queer',

        'LGBT': 'lesbian  gay  bisexual  transgender',

        'Machedo': 'Indian internet celebrity',

        'madheshi': 'Madheshi',

        'Madridiots': 'Real Madrid idiot supporters',

        'mailbait': 'mail bait',

        'MAINS': 'exam',

        'marathis': 'Marathi',

        'marksheet': 'university transcript',

        'mastrubate': 'masturbate',

        'mastrubating': 'masturbating',

        'mastrubation': 'masturbation',

        'mastuburate': 'masturbate',

        'meninism': 'male feminism',

        'MeToo': 'feminist activism campaign',

        'Mewani': 'Indian politician Jignesh Mevani',

        'MGTOWS': 'Men Going Their Own Way',

        'micropenis': 'tiny penis',

        'moeslim': 'Muslim',

        'mongloid': 'Mongoloid',

        'mtech': 'Master of Engineering',

        'muhajirs': 'Muslim immigrant',

        'Myeshia': 'widow of Green Beret killed in Niger',

        'mysoginists': 'misogynists',

        'naïve': 'naive',

        'narcisist': 'narcissist',

        'narcissit': 'narcissist',

        'narcissit': 'narcissist',

        'Naxali ': 'Naxalite ',

        'Naxalities': 'Naxalites',

        'NICMAR': 'Indian university',

        'Niggeriah': 'Nigger',

        'Niggerism': 'Nigger',

        'NMAT': 'Indian MBA exam',

        'Northindian': 'North Indian ',

        'northindian': 'north Indian ',

        'northkorea': 'North Korea',

        'Novichok': 'Soviet Union agents',

        'organisation': 'organization',

        'Padmavat': 'Indian Movie Padmaavat',

        'Pahul': 'Amrit Sanskar',

        'penish': 'penis',

        'pennis': 'penis',

        'Pizzagate': 'Pizzagate conspiracy theory',

        'Pribumi': 'Native Indonesian',

        'qouta': 'quota',

        'quorans': 'advice website user',

        'quoran': 'advice website user',

        'Quorans': 'advice website user',

        'Quoran': 'advice website user',

        'quoras': 'advice website',

        'Qoura ': 'advice website ',

        'Qoura': 'advice website',

        'Quora': 'advice website',

        'Quroa': 'advice website',

        'QUORA': 'advice website',

        'R&D': 'research and development',

        'r&d': 'research and development',

        'r-aping': 'raping',

        'raaping': 'rape',

        'rapefugees': 'rapist refugee',

        'Rapistan': 'Pakistan rapist',

        'rapistan': 'Pakistan rapist',

        'Rejuvalex': 'hair growth formula',

        'ReleaseTheMemo': 'cry for the right and Trump supporters',

        'Remainers': 'anti Brexit',

        'remainers': 'anti Brexit',

        'remoaner': 'remainer ',

        'rohingya': 'Rohingya ',

        'sallary': 'salary',

        'Sanghis': 'Sanghi',

        'sh*t': 'shit',

        'shithole': ' shithole ',

        'shitlords': 'shit lords',

        'shitpost': 'shit post',

        'shitslam': 'shit Islam',

        'sickular': 'India sick secular ',

        'signuficance': 'significance',

        'SJW': 'social justice warrior',

        'SJWs': 'social justice warrior',

        'Skripal': 'Sergei Skripal',

        'Strzok': 'Hillary Clinton scandal',

        'suckimg': 'sucking',

        'superficious': 'superficial',

        'Swachh': 'Swachh Bharat mission campaign ',

        'Tambrahms': 'Tamil Brahmin',

        'Tamilans': 'Tamils',

        'Terroristan': 'terrorist Pakistan',

        'terroristan': 'terrorist Pakistan',

        'Tharki': 'pervert',

        'tharki': 'pervert',

        'theatre': 'theater',

        'theBest': 'the best',

        'thighing': 'masturbate',

        'travelling': 'traveling',

        'trollbots': 'troll bots',

        'trollimg': 'trolling',

        'trollled': 'trolled',

        'Trumpers': 'Trump supporters',

        'Trumpanzees': 'Trump chimpanzee fool',

        'Turkified': 'Turkification',

        'turkified': 'Turkification',

        'UCEED': 'Indian Institute of Technology Bombay entrance examination',

        'unacadamy': 'Indian online classroom',

        'Unacadamy': 'Indian online classroom',

        'unoin': 'Union',

        'unsincere': 'insincere',

        'UPES': 'Indian university',

        'UPSEE': 'Indian university entrance examination',

        'vaxxer': 'vocal nationalist ',

        'VITEEE': 'Vellore institute of technology',

        'watsapp': 'Whatsapp',

        'whattsapp': 'Whatsapp',

        'WBJEE': 'West Bengal entrance examination',

        'weatern': 'western',

        'westernise': 'westernize',

        'Whatare': 'What are',

        'whatare': 'what are',

        'whst': 'what',

        'Whta': 'What',

        'whydo': 'why do',

        'Whykorean': 'Why Korean',

        'Wjy': 'Why',

        'WMAF': 'White male married Asian female',

        'wumao ': 'cheap Chinese stuff',

        'wumaos': 'cheap Chinese stuff',

        'wwii': 'world war 2',

        ' xender': ' gender',

        'XXXTentacion': 'Tentacion',

        'youtu ': 'youtube ',

        'Zerodha': 'online stock brokerage',

        'Žižek': 'Slovenian philosopher Slavoj Žižek',

        'Zoë': 'Zoe',

        '卐': 'Nazi Germany'

    }



    escape_cars = re.compile('(\+|\*)')

    misspell = '|'.join([escape_cars.sub(r"\\\1",i) for i in misspell_to_sub.keys()])

    misspell_re = re.compile(misspell)

    

    def _replace(match):

        return misspell_to_sub.get(match.group(0), match.group(0))

    

    return misspell_re.sub(_replace, text)



def space_chars(text, chars_to_space):

    """

    Takes a string and list of characters, insert space before and after 

    characters that appear in text.

    

    Parameters

    ----------

    text : str

        String to search

    chars_to_space : list

        list of characters to find and space

        

    Returns

    -------

    str

        modified text string    

    """

    

    # light arg checking

    if type(chars_to_space) == str:

        chars_to_space = [chars_to_space]

        

    chars_to_space = set(chars_to_space)

    chars_to_space = '|'.join(chars_to_space)

    re_tok = re.compile('({})'.format(chars_to_space))

    

    return re_tok.sub(r' \1 ', text)



def preprocess(text, remove_num=False):

    """

    Text preprocessing pipeline

    

    Parameters

    ----------

    text : str

        String to process

        

    Returns

    -------

    str

        modified string  

    """    

    # 1. Normalize 

    # normalize_unicode(text)

    

    # 2. Remove new-lines

    # text = remove_newline(text)

    

    # 3. replace contractions (e.g. won't -> will not)

    text = decontracted(text)



    # 4. replace LateX with English

    text = clean_latex(text)

    

    # 5. space characters

    text = space_chars(text, ['\?', ',', '"', '\(', '\)', '%', ':', '\$', 

                              '\.', '\+', '\^', '/', '\{', '\}', '\!', 

                              '#', '=', '-','\|', '\[', '\]','\.'])

    

    # 6. handle number

    if remove_num:

        text = remove_number(text)

    else:

        text = spacing_digit(text)

    

    # 7. fix typos and swap terms that are not recognized by embedding

    text = clean_misspell(text)



    # 8. remove space

    text = remove_space(text)

    

    # 9. remove strings

    text = remove_string(text, '_')

    

    return text
def count_uppercase_words(text):

    """

    Count SHOUTY all-caps words more than 1 character long (eg not "I")

    

    Parameters

    ----------

    text : str

        String to process

        

    Returns

    -------

    int

        count of all-caps words in text  

    """   

    tokens = text.split()

    upper = [1 if u.isupper() and len(u) > 1 else 0 for u in tokens]

    return(sum(upper))



def programming_related(text):

    """

    Identify references to common programming languages, frameworks, 

    tools or databases in question text

    

    Parameters

    ----------

    text : str

        String to process

        

    Returns

    -------

    bool

        True if programming reference identified 

    """

    programming = ['javascript', 'html', 'css', 'sql', 'java', 'bash', 'python',

                   'c#', 'c++', 'c language', 'c programming', 'c programing',

                   'typescript', 'ruby', 'matlab', 'f#', 'clojure', 'haskell', 

                   'erlang', 'coffeescript', 'cobol', 'fortran', 'vba', '.net',

                   'asp.net', 'scala', 'perl', 'php', 'kotlin', 'node.js', 

                   'react.js', 'angular', 'django', 'cordova', 'tensorflow', 'keras',

                   'xamarin', 'hadoop', 'pytorch', 'mongo', 'redis', 'elasticsearch', 

                   'mariadb', 'azure', 'dynamodb', ' rds', 'redshift', 'cassandra',

                   'apache hive', 'bigquery', 'hbase', 'linux', 'raspberry pi', 

                   'rpi ', 'arduino', 'heroku', 'drupal', 'visual studio', 

                   'sublime text', 'rstudio', 'jupyter', 'pycharm', 'netbeans',

                   'emacs', 'vim ', 'komodo', 'graphql', 'golang']

    

    for word in text.split():

        if word.lower() in programming:

            return True

    

    return False



class FeatureEngineering():

    def __init__(self, doc_column, max_words=10):

        """

        Create features from text



        Parameters

        ----------

        doc_column : str

            column name of text to process

        

        max_words: int

            maximum number of leading words (1st word in sentence) to count

        """

        self._most_common = None

        self._data = None

        self._doc_column = doc_column

        self._max_words = max_words

    

    def fit(self, df):

        # save leading tokens (first word in sentences) information from fit dataset

        # keep top max_word, convert to one-hot and append to dataframe

        leading_tokens = df[self._doc_column].apply(lambda x: re.match('\w+|\d+|.', x)[0].lower())

        leading_token_count = Counter(leading_tokens)

        max_count = min(self._max_words, len(leading_token_count)-1)

        self._most_common = [w for w,c in leading_token_count.most_common(max_count)]

        self._data = df

        

    def transform(self, df):

        # Leading tokens

        # keep top max_word, convert to one-hot and append to dataframe

        leading_tokens = df[self._doc_column].apply(lambda x: re.match('\w+|\d+|.', x)[0].lower())

        df_leading_tokens = leading_tokens.apply(lambda x: x.lower() if x.lower() in self._most_common else 'other')

        df_leading_tokens = pd.get_dummies(df_leading_tokens)

        for token in self._most_common:

            if token not in df_leading_tokens.columns:

                df_leading_tokens[token] = 0

        df_leading_tokens = df_leading_tokens.rename(columns = {c: 'leading_word_' + c for c in df_leading_tokens.columns})

        # Using 'other' as reference category

        if 'leading_word_other' in df_leading_tokens.columns:

            df_leading_tokens = df_leading_tokens.drop('leading_word_other', axis=1)

        df = pd.concat([df, df_leading_tokens], axis=1)

        

        # Word count

        df['word_count'] = df[self._doc_column].apply(lambda x: len(re.findall(r'\w+',x)))



        # Character count

        df['char_count'] = df[self._doc_column].apply(lambda x: len(x))



        # How many question marks

        df['question_mark_count'] = df[self._doc_column].apply(lambda x: len(re.findall(r'\?',x)))



        # LaTex or Math symbols

        # Programming questions

        df['programming'] = df[self._doc_column].apply(lambda x: programming_related(x))



        # ALL CAPS Words

        df['caps_count'] = df[self._doc_column].apply(lambda x: count_uppercase_words(x))

        

        return df
# Important! NLTK lemmatizer needs a POS (parts of speech) tags to work correctly

# https://www.kaggle.com/alvations/basic-nlp-with-nltk

# TL/DR otherwise all words are assumed to be nouns and

# do -> doe

def penn2morphy(penntag):

    """ 

    Author: Liling Tan https://www.kaggle.com/alvations/basic-nlp-with-nltk

    Converts Penn Treebank tags to WordNet.

    """

    morphy_tag = {'NN':'n', 'JJ':'a',

                  'VB':'v', 'RB':'r'}

    try:

        return morphy_tag[penntag[:2]]

    except:

        return 'n' # if mapping isn't found, fall back to Noun.



# Note: Lemmatizing rather than stemming takes much longer

def tokenize(text, stem = False, stop_words = None):

    """

    Text tokenization pipeline



    Parameters

    ----------

    test : str

        document to tokenize.



    stem: bool

        If true use porter stemmer, else use WordNet lemmatizer.

        Note that lemmatizing is a much more expensive operation than 

        stemming text.

        

    stop_words: list

        optional list of stop words.

    

    Returns

    -------

    list

        ordered list of tokens 

    """

    

    if stop_words == None:

        # Starting from a small stop-word list. TODO: build this out further.

        stop_words = ('it', 'its', 'this', 'that', 'these', 'those', 

                      'a', 'an', 'the', 'and', 'but', 'if', 'or', 

                      'as', 'of', 'at', 'by', 'to', 'in', 'so')



    if stem:

        stemmer = PorterStemmer()

    else:

        lemmer = WordNetLemmatizer()

    

    # tokenize text into words

    tokens = nltk.word_tokenize(text)

    

    # drop punctuation

    tokens = [t for t in tokens if t.isalpha()]



    # drop stop words

    tokens = [t for t in tokens if t not in stop_words]



    # lowercase

    tokens  = [t.lower() for t in tokens]

    

    if stem:

        tokens = [stemmer.stem(t) for t in tokens]

    else:

        # parts-of-speech tags

        # required for nltk WordNetLemmatizer - if not supplied default is "noun"

        tagged_tokens = [t for t in pos_tag(tokens)]



        # lemmetize

        tokens = [lemmer.lemmatize(t, pos=penn2morphy(tag)) for t,tag in tagged_tokens]

    

    return tokens
df_train, df_test = load_data()



df_train['question_text_pr'] = df_train['question_text'].apply(preprocess)

df_test['question_text_pr'] = df_test['question_text'].apply(preprocess)



ef = FeatureEngineering('question_text', 20)

ef.fit(df_train)

df_train_plus = ef.transform(df_train)

df_test_plus = ef.transform(df_test)



df_train_plus.to_csv('processed_train.csv')

df_test_plus.to_csv('processed_test.csv')



del df_train_plus

del df_test_plus

del ef
%%time

# Simple tokenizer without any pre-processing of text, using nltk built in tokenizer

# and english stop words for comparison sake, as this comes out of the box

tfidf = TfidfVectorizer(tokenizer=nltk.word_tokenize, 

                        ngram_range=(1,4),

                        min_df=5,

                        max_df=0.9,

                        strip_accents='unicode',

                        use_idf=True,

                        smooth_idf=True,

                        sublinear_tf=True)



all_text = np.concatenate([df_train['question_text'], df_train['question_text']])

tfidf.fit(all_text)

X_train = tfidf.transform(df_train['question_text'])

X_train_feats = tfidf.get_feature_names()



# save results for use in further kernels, as this takes a long time to complete.

pickle.dump(X_train, open("tfidf_train_base.pickle", "wb"))

pickle.dump(X_train_feats, open("tfidf_feats_base.pickle", "wb"))
%%time

# Simple tokenizer without any pre-processing of text, using nltk built in tokenizer

# and english stop words for comparison sake, as this comes out of the box

tfidf = TfidfVectorizer(tokenizer=nltk.word_tokenize, 

                        ngram_range=(1,4),

                        min_df=5,

                        max_df=0.9,

                        strip_accents='unicode',

                        use_idf=True,

                        smooth_idf=True,

                        sublinear_tf=True)



tfidf.fit(all_text)

X_train = tfidf.transform(df_train['question_text'])

X_train_feats = tfidf.get_feature_names()



# save results for use in further kernels, as this takes a long time to complete.

pickle.dump(X_train, open("tfidf_train_pr.pickle", "wb"))

pickle.dump(X_train_feats, open("tfidf_feats_pr.pickle", "wb"))
%%time

# Create TD-IDF vectorizer on whole corpus (train + test), n-gram

countv2 = CountVectorizer(tokenizer=tokenize,

                          ngram_range=(1,2),

                          min_df=5,

                          max_df=0.9,

                          strip_accents='unicode')



countv2.fit(all_text)

X_train = countv2.transform(df_train['question_text_pr'])

X_train_feats = countv2.get_feature_names()



# save results for use in further kernels, as this takes a long time to complete.

pickle.dump(X_train, open("count_train_lem_ng2.pickle", "wb"))

pickle.dump(X_train_feats, open("count_feats_lem_ng2.pickle", "wb"))



del countv2
%%time

# Create TD-IDF vectorizer on whole corpus (train + test), n-gram

tfidf = TfidfVectorizer(tokenizer=tokenize, 

                        ngram_range=(1,2),

                        min_df=5,

                        max_df=0.9,

                        strip_accents='unicode',

                        use_idf=True,

                        smooth_idf=True,

                        sublinear_tf=True)



all_text = np.concatenate([df_train['question_text_pr'], df_train['question_text_pr']])

tfidf.fit(all_text)

X_train = tfidf.transform(df_train['question_text_pr'])

X_train_feats = tfidf.get_feature_names()



# save results for use in further kernels, as this takes a long time to complete.

pickle.dump(X_train, open("tfidf_train_lem_ng2.pickle", "wb"))

pickle.dump(X_train_feats, open("tfidf_feats_lem_ng2.pickle", "wb"))
%%time

# Create TD-IDF vectorizer on whole corpus (train + test), n-gram

tfidf = TfidfVectorizer(tokenizer=tokenize, 

                        ngram_range=(1,4),

                        min_df=5,

                        max_df=0.9,

                        strip_accents='unicode',

                        use_idf=True,

                        smooth_idf=True,

                        sublinear_tf=True)



all_text = np.concatenate([df_train['question_text_pr'], df_train['question_text_pr']])

tfidf.fit(all_text)

X_train = tfidf.transform(df_train['question_text_pr'])

X_test = tfidf.transform(df_test['question_text_pr'])

X_train_feats = tfidf.get_feature_names()



# save results for use in further kernels, as this takes a long time to complete.

pickle.dump(X_train, open("tfidf_train_lem_ng4.pickle", "wb"))

pickle.dump(X_test, open("tfidf_test_lem_ng4.pickle", "wb"))

pickle.dump(X_train_feats, open("tfidf_feats_lem_ng4.pickle", "wb"))



del tfidf

del X_train

del X_test

del X_train_feats
def load_word_embedding(filepath, verbose=True):

    """

    author: Theo Viel

    given a filepath to embeddings file, return a word to vec

    dictionary, in other words, word_embedding

    E.g. {'word': array([0.1, 0.2, ...])}

    """

    def _get_vec(word, *arr):

        return word, np.asarray(arr, dtype='float32')



    if verbose:

        print('load word embedding ......')

        

    try:

        word_embedding = dict(_get_vec(*w.split(' ')) for w in open(filepath))

    except UnicodeDecodeError:

        word_embedding = dict(_get_vec(*w.split(' ')) for w in open(

            filepath, encoding="utf8", errors='ignore'))



    if verbose:

        print('finished load of word embedding.')



    return word_embedding



def build_vocab(docs):

    """

    author: Theo Viel

    given a list or np.array of strings create a dictionary of unique words with frequencies.

    

    Parameters

    __________

    docs: list or np.array

        iterable of text

    

    Returns

    _______

    dict

        unique words as keys, frequencies as values

    """

    vocab = {}

    

    for doc in docs:

        for word in doc.split():

            vocab[word] = vocab.get(word, 0) + 1

                

    return vocab



def vocab_embedding_coverage(vocab, embedding, verbose = False):

    """

    author: Theo Viel

    

    given a dict representing the word frequency of a corpus, 

    calculate the percentage of unique words and 

    the percentage of the corpus matched in the embedding dict.

    

    Parameters

    __________

    vocab: dict

        word frequency of corpus

    embedding: dict

        embedding vector converted to dict

    verbse: bool

        print summary statistics

    

    Returns

    _______

    

    perc_words : float

        percentage of unique words identified in corpus

    perc_corpus : float

        percentage of corpus identified in corpus

    words_in_embedding: dict

        dictionary of unique words, frequency and whether found in embedding (true / false)

    """

    

    words_in_embedding = {}

    word_found_count = 0

    corpus_found_count = 0

    corpus_count = 0 

    

    for word, freq in vocab.items():

        corpus_count += freq

        words_in_embedding[word] = {

            'frequency': freq,

            'embedding': (word in embedding)

        }

        if word in embedding:

            word_found_count += 1

            corpus_found_count += freq

    

    perc_words = word_found_count / len(vocab)

    perc_corpus = corpus_found_count / corpus_count

    

    print('{}% of vocabulary words found in embedding files'.format(round(100*perc_words,2)))

    print('{}% of corpus found in embedding files'.format(round(100*perc_corpus,2)))

    

    return perc_words, perc_corpus, words_in_embedding
fasttext = load_word_embedding('../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec')
vocab = build_vocab(np.concatenate((df_train.question_text, df_test.question_text)))

w,c,words_in_embedding = vocab_embedding_coverage(vocab, fasttext, True)

df_words = pd.DataFrame.from_dict(words_in_embedding, orient = 'index')

df_words = df_words.sort_values(by='frequency', ascending=False)

df_words[np.logical_not(df_words.embedding)].head(10)
vocab = build_vocab(np.concatenate((df_train.question_text_pr, df_test.question_text_pr)))

w,c,words_in_embedding = vocab_embedding_coverage(vocab, fasttext, True)

df_words = pd.DataFrame.from_dict(words_in_embedding, orient = 'index')

df_words = df_words.sort_values(by='frequency', ascending=False)

df_words[np.logical_not(df_words.embedding)].head(10)