# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
GOOGLE_PATH = "/kaggle/input/google-quest-challenge/"



train = pd.read_csv(GOOGLE_PATH+"train.csv")

test = pd.read_csv(GOOGLE_PATH+"test.csv")

sample_submission = pd.read_csv(GOOGLE_PATH+"sample_submission.csv")
train.shape, test.shape
def url_id_ex(url):

    try:

        ids = int(url.split("/")[-2])

    except:

        ids = int(url.split("/")[-1])

    return ids



# category_type

train["category_type"] = train["url"].apply(lambda x : x.split(".")[0].split("/")[-1])

test["category_type"] = test["url"].apply(lambda x : x.split(".")[0].split("/")[-1])



train["quser_id"] = train["question_user_page"].apply(lambda x: int(x.split("/")[-1]))

train["auser_id"] = train["answer_user_page"].apply(lambda x: int(x.split("/")[-1]))

train["url_id"] = train["url"].apply(url_id_ex)



test["quser_id"] = test["question_user_page"].apply(lambda x: int(x.split("/")[-1]))

test["auser_id"] = test["answer_user_page"].apply(lambda x: int(x.split("/")[-1]))

test["url_id"] = test["url"].apply(url_id_ex)
## bert preprocess
# General imports

import numpy as np

import pandas as pd

import os, sys, gc, re, warnings, pickle, itertools, emoji, psutil, random, unicodedata



# custom imports

from gensim.utils import deaccent

from collections import Counter

from bs4 import BeautifulSoup

from multiprocessing import Pool



warnings.filterwarnings('ignore')

pd.options.display.max_columns = 10

pd.options.display.max_colwidth = 200
#################################################################################

## Multiprocessing Run.

# :df - DataFrame to split                      # type: pandas DataFrame

# :func - Function to apply on each split       # type: python function

# This function is NOT 'bulletproof', be carefull and pass only correct types of variables.

def df_parallelize_run(df, func):

    num_partitions, num_cores = 16, psutil.cpu_count()  # number of partitions and cores

    df_split = np.array_split(df, num_partitions)

    pool = Pool(num_cores)

    df = pd.concat(pool.map(func, df_split))

    pool.close()

    pool.join()

    return df



## Build of vocabulary from file - reading data line by line

## Line splited by 'space' and we store just first argument - Word

# :path - txt/vec/csv absolute file path        # type: str

def get_vocabulary(path):

    with open(path) as f:

        return [line.strip().split()[0] for line in f][0:]



## Check how many words are in Vocabulary

# :c_list - 1d array with 'comment_text'        # type: pandas Series

# :vocabulary - words in vocabulary to check    # type: list of str

# :response - type of response                  # type: str

def check_vocab(c_list, vocabulary, response='default'):

    try:

        words = set([w for line in c_list for w in line.split()])

        u_list = words.difference(set(vocabulary))

        k_list = words.difference(u_list)

    

        if response=='default':

            print('Unknown words:', len(u_list), '| Known words:', len(k_list))

        elif response=='unknown_list':

            return list(u_list)

        elif response=='known_list':

            return list(k_list)

    except:

        return []

        

## Seeder

# :seed to make all processes deterministic     # type: int

def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)



    if 'torch' in sys.modules:

        torch.manual_seed(seed)

        torch.cuda.manual_seed(seed)

        torch.backends.cudnn.deterministic = True

 

## Simple "Memory profilers" to see memory usage

def get_memory_usage():

    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 

        

def sizeof_fmt(num, suffix='B'):

    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:

        if abs(num) < 1024.0:

            return "%3.1f%s%s" % (num, unit, suffix)

        num /= 1024.0

    return "%.1f%s%s" % (num, 'Yi', suffix)

    

## Export pickle

def make_export(tr, tt, file_name):

    train_export = train[['id']]

    test_export = test[['id']]



    try:

        cur_shape = tr.shape[1]>1

        train_export = pd.concat([train_export, tr], axis=1)

        test_export = pd.concat([test_export, tt], axis=1)        

    except:

        train_export['p_comment'] = tr

        test_export['p_comment'] = tt

    

    train_export.to_pickle(file_name + '_x_train.pkl')

    test_export.to_pickle(file_name + '_x_test.pkl')



## Domain Search

re_3986_enhanced = re.compile(r"""

        # Parse and capture RFC-3986 Generic URI components.

        ^                                    # anchor to beginning of string

        (?:  (?P<scheme>    [^:/?#\s]+):// )?  # capture optional scheme

        (?:(?P<authority>  [^/?#\s]*)  )?  # capture optional authority

             (?P<path>        [^?#\s]*)      # capture required path

        (?:\?(?P<query>        [^#\s]*)  )?  # capture optional query

        (?:\#(?P<fragment>      [^\s]*)  )?  # capture optional fragment

        $                                    # anchor to end of string

        """, re.MULTILINE | re.VERBOSE)



re_domain =  re.compile(r"""

        # Pick out top two levels of DNS domain from authority.

        (?P<domain>[^.]+\.[A-Za-z]{2,6})  # $domain: top two domain levels.

        (?::[0-9]*)?                      # Optional port number.

        $                                 # Anchor to end of string.

        """, 

        re.MULTILINE | re.VERBOSE)



def domain_search(text):

    try:

        return re_domain.search(re_3986_enhanced.match(text).group('authority')).group('domain')

    except:

        return 'url'



## Load helper helper))

def load_helper_file(filename):

    with open(HELPER_PATH+filename+'.pickle', 'rb') as f:

        temp_obj = pickle.load(f)

    return temp_obj

        

## Preprocess helpers

def place_hold(w):

    return WPLACEHOLDER + '['+re.sub(' ', '___', w)+']'



def check_replace(w):

    return not bool(re.search(WPLACEHOLDER, w))



def make_cleaning(s, c_dict):

    if check_replace(s):

        s = s.translate(c_dict)

    return s

  

def make_dict_cleaning(s, w_dict):

    if check_replace(s):

        s = w_dict.get(s, s)

    return s



def export_dict(temp_dict, serial_num):

    pd.DataFrame.from_dict(temp_dict, orient='index').to_csv('dict_'+str(serial_num)+'.csv')



def print_dict(temp_dict, n_items=10):

    run = 0

    for k,v in temp_dict.items():

        print(k,'---',v)

        run +=1

        if run==n_items:

            break    

## ----------------------------------------------------------------------------------------------------
#################################################################################

HELPER_PATH             = '../input/jigsaw-general-helper-public/'



LOCAL_TEST = True       ## Local test - for test performance on part of the train set only

SEED = 42               ## Seed for enviroment

seed_everything(SEED)   ## Seed everything



WPLACEHOLDER = 'word_placeholder'
total_text = train["question_title"] + " " + train["question_body"]+ " " + train["answer"]


########################### Get basic helpers

#################################################################################

print('1.2. Basic helpers')

bert_uncased_vocabulary = load_helper_file('helper_bert_uncased_vocabulary')

bert_cased_vocabulary   = load_helper_file('helper_bert_cased_vocabulary')

bert_char_list          = list(set([c for line in bert_uncased_vocabulary+bert_cased_vocabulary for c in line]))



url_extensions          = load_helper_file('helper_url_extensions')

html_tags               = load_helper_file('helper_html_tags')

good_chars_dieter       = load_helper_file('helper_good_chars_dieter')

bad_chars_dieter        = load_helper_file('helper_bad_chars_dieter')

helper_contractions     = load_helper_file('helper_contractions')

global_vocabulary       = load_helper_file('helper_global_vocabulary')

global_vocabulary_chars = load_helper_file('helper_global_vocabulary_chars')

normalized_chars        = load_helper_file('helper_normalized_chars')

white_list_chars        = load_helper_file('helper_white_list_chars')

white_list_punct        = " '*-.,?!/:;_()[]{}<>=" + '"'

pictograms_to_emoji     = load_helper_file('helper_pictograms_to_emoji')

toxic_misspell_dict     = load_helper_file('helper_toxic_misspell_dict')

data = total_text

local_vocab = bert_uncased_vocabulary

verbose = True

global_lower=True

data = data.astype(str)

if verbose: print('#' *20 ,'Initial State:'); check_vocab(data, local_vocab)
if global_lower:

    data = data.apply(lambda x: x.lower())

    if verbose: print('#'*10 ,'Step - Lowering everything:'); check_vocab(data, local_vocab)
# Normalize chars and dots - SEE HELPER FOR DETAILS

# Global

data = data.apply(lambda x: ' '.join([make_cleaning(i,normalized_chars) for i in x.split()]))

data = data.apply(lambda x: re.sub('\(dot\)', '.', x))

data = data.apply(lambda x: deaccent(x))

if verbose: print('#'*10 ,'Step - Normalize chars and dots:'); check_vocab(data, local_vocab)

# Remove 'control' chars

# Global    

global_chars_list = list(set([c for line in data for c in line]))

chars_dict = {c:'' for c in global_chars_list if unicodedata.category(c)[0]=='C'}

data = data.apply(lambda x: ' '.join([make_cleaning(i,chars_dict) for i in x.split()]))

if verbose: print('#'*10 ,'Step - Control Chars:'); check_vocab(data, local_vocab)
# Remove hrefs

# Global    

data = data.apply(lambda x: re.sub(re.findall(r'\<a(.*?)\>', x)[0], '', x) if (len(re.findall(r'\<a (.*?)\>', x))>0) and ('href' in re.findall(r'\<a (.*?)\>', x)[0]) else x)

if verbose: print('#'*10 ,'Step - Remove hrefs:'); check_vocab(data, local_vocab)
# Convert or remove Bad Symbols

# Global

global_chars_list = list(set([c for line in data for c in line]))

chars = ''.join([c for c in global_chars_list if (c not in bert_char_list) and (c not in emoji.UNICODE_EMOJI) and (c not in white_list_chars)])

chars_dict = {}

for char in chars:

    try:

        new_char = unicodedata.name(char).split()[-1:][0].lower()

        if len(new_char)==1:

            chars_dict[ord(char)] = new_char

        else:

            chars_dict[ord(char)] = ''

    except:

        chars_dict[ord(char)] = ''

data = data.apply(lambda x: ' '.join([make_cleaning(i,chars_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Remove Bad Symbols:'); check_vocab(data, local_vocab)

if verbose: print(chars)

if verbose: print_dict(chars_dict)
# Remove Bad Symbols PART 2

# Global

global_chars_list = list(set([c for line in data for c in line]))

chars = '·' + ''.join([c for c in global_chars_list if (c not in white_list_chars) and (c not in emoji.UNICODE_EMOJI) and (c not in white_list_punct) and (ord(c)>256)])

chars_dict = {}

for char in chars:

    try:

        new_char = unicodedata.name(char).split()[-1:][0].lower()

        if len(new_char)==1:

            chars_dict[ord(char)] = new_char

        else:

            chars_dict[ord(char)] = ''

    except:

        chars_dict[ord(char)] = ''

data = data.apply(lambda x: ' '.join([make_cleaning(i,chars_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Remove Bad Symbols PART 2:'); check_vocab(data, local_vocab)

if verbose: print(chars)

if verbose: print_dict(chars_dict)
# Global

temp_vocab = list(set([c for line in data for c in line.split()]))

temp_vocab = [k for k in temp_vocab if check_replace(k)]

temp_dict = {}

for word in temp_vocab:

    if ('<' in word) and ('>' in word):

        for tag in html_tags:

            if ('<'+tag+'>' in word) or ('</'+tag+'>' in word):

                temp_dict[word] = BeautifulSoup(word, 'html5lib').text  

data = data.apply(lambda x: ' '.join([temp_dict.get(i, i) for i in x.split()]))

if verbose: print('#' * 10, 'Step - HTML tags:'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)
# Remove links (There is valuable information in links (probably you will find a way to use it)) 

# Global

temp_vocab = list(set([c for line in data for c in line.split()]))

temp_vocab = [k for k in temp_vocab if check_replace(k)]

url_rule = r'(?P<url>https?://[^\s]+)'

temp_dict = {k:domain_search(k) for k in temp_vocab if k!= re.compile(url_rule).sub('url', k)}

    

for word in temp_dict:

    new_value = temp_dict[word]

    if word.find('http')>2:

        temp_dict[word] =  word[:word.find('http')] + ' ' + place_hold(new_value)

    else:

        temp_dict[word] = place_hold(new_value)

            

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Convert urls part 1:'); check_vocab(data, local_vocab); 

if verbose: print_dict(temp_dict)
# Convert urls part 2

# Global

temp_vocab = list(set([c for line in data for c in line.split()]))

temp_vocab = [k for k in temp_vocab if check_replace(k)]

temp_dict = {}



for word in temp_vocab:

    url_check = False

    if 'file:' in word:

        url_check = True

    elif ('http' in word) or ('ww.' in word) or ('.htm' in word) or ('ftp' in word) or ('.php' in word) or ('.aspx' in word):

        if 'Aww' not in word:

            for d_zone in url_extensions:

                if '.' + d_zone in word:

                    url_check = True

                    break            

    elif ('/' in word) and ('.' in word):

        for d_zone in url_extensions:

            if '.' + d_zone + '/' in word:

                url_check = True

                break



    if url_check:

        temp_dict[word] =  place_hold(domain_search(word))

        

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Convert urls part 2:'); check_vocab(data, local_vocab); 

if verbose: print_dict(temp_dict)
# Normalize pictograms

# Local (only unknown words)

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if check_replace(k)]

temp_dict = {}

for word in temp_vocab:

    if len(re.compile('[a-zA-Z0-9]').sub('', word))>2:

        for pict in pictograms_to_emoji:

            if (pict in word) and (len(pict)>2):

                temp_dict[word] = word.replace(pict, pictograms_to_emoji[pict])

            elif pict==word:  

                temp_dict[word] = pictograms_to_emoji[pict]



data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Normalize pictograms:'); check_vocab(data, local_vocab); 

if verbose: print_dict(temp_dict)
# Isolate emoji

# Global

global_chars_list = list(set([c for line in data for c in line]))

chars = ''.join([c for c in global_chars_list if c in emoji.UNICODE_EMOJI])

chars_dict = {ord(c):f' {c} ' for c in chars}

data = data.apply(lambda x: ' '.join([make_cleaning(i,chars_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Isolate emoji:'); check_vocab(data, local_vocab)

if verbose: print(chars)
# Duplicated dots, question marks and exclamations

# Local

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if check_replace(k)]

temp_dict = {}

for word in temp_vocab:

    new_word = word

    if (Counter(word)['.']>1) or (Counter(word)['!']>1) or (Counter(word)['?']>1) or (Counter(word)[',']>1):

        if (Counter(word)['.']>1):

            new_word = re.sub('\.\.+', ' . . . ', new_word)

        if (Counter(word)['!']>1):

            new_word = re.sub('\!\!+', ' ! ! ! ', new_word)

        if (Counter(word)['?']>1):

            new_word = re.sub('\?\?+', ' ? ? ? ', new_word)

        if (Counter(word)[',']>1):

            new_word = re.sub('\,\,+', ' , , , ', new_word)

        temp_dict[word] = new_word

temp_dict = {k: v for k, v in temp_dict.items() if k != v}

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Duplicated Chars:'); check_vocab(data, local_vocab);

# Remove underscore for spam words

# Local

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if check_replace(k)]

temp_dict = {}

for word in temp_vocab:

    if (len(re.compile('[a-zA-Z0-9\-\.\,\/\']').sub('', word))/len(word) > 0.6) and ('_' in word):

        temp_dict[word] = re.sub('_', '', word)       

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Remove underscore:'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)
# Isolate spam chars repetition

# Local

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if check_replace(k)]

temp_dict = {}

for word in temp_vocab:

    if (len(re.compile('[a-zA-Z0-9\-\.\,\/\']').sub('', word))/len(word) > 0.6) and (len(Counter(word))==1) and (len(word)>2):

        temp_dict[word] = ' '.join([' ' + next(iter(Counter(word).keys())) + ' ' for i in range(3)])

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Spam chars repetition:'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)

# Normalize pictograms part 2

# Local (only unknown words)

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if check_replace(k)]

temp_dict = {}

for word in temp_vocab:

    if len(re.compile('[a-zA-Z0-9]').sub('', word))>1:

        for pict in pictograms_to_emoji:

            if pict==word:  

                temp_dict[word] = pictograms_to_emoji[pict]

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Normalize pictograms part 2:'); check_vocab(data, local_vocab); 

if verbose: print_dict(temp_dict)        
# Isolate brakets and quotes

# Global

chars = '()[]{}<>"'

chars_dict = {ord(c):f' {c} ' for c in chars}

data = data.apply(lambda x: ' '.join([make_cleaning(i,chars_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Brackets and quotes:'); check_vocab(data, local_vocab)

# Break short words

# Global

temp_vocab = list(set([c for line in data for c in line.split()]))

temp_vocab = [k for k in temp_vocab if check_replace(k)]

temp_vocab = [k for k in temp_vocab if len(k)<=20]

    

temp_dict = {}

for word in temp_vocab:

    if '/' in word:

        temp_dict[word] = re.sub('/', ' / ', word)

    

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Break long words:'); check_vocab(data, local_vocab); 

if verbose: print_dict(temp_dict)  
# Break long words

# Global

temp_vocab = list(set([c for line in data for c in line.split()]))

temp_vocab = [k for k in temp_vocab if check_replace(k)]

temp_vocab = [k for k in temp_vocab if len(k)>20]

    

temp_dict = {}

for word in temp_vocab:

    if '_' in word:

        temp_dict[word] = re.sub('_', ' ', word)

    elif '/' in word:

        temp_dict[word] = re.sub('/', ' / ', word)

    elif len(' '.join(word.split('-')).split())>2:

        temp_dict[word] = re.sub('-', ' ', word)

    

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Break long words:'); check_vocab(data, local_vocab); 

if verbose: print_dict(temp_dict)
# Remove/Convert usernames and hashtags (add username/hashtag word?????)

# Global

temp_vocab = list(set([c for line in data for c in line.split()]))

temp_vocab = [k for k in temp_vocab if check_replace(k)]

temp_dict = {}

for word in temp_vocab:

    new_word = word

    if (len(word) > 3) and (word[1:len(word)-1].isalnum()) and (not re.compile('[#@,.:;]').sub('', word).isnumeric()):

        if word[len(word)-1].isalnum():

            if (word.startswith('@')) or (word.startswith('#')):

                new_word = place_hold(new_word[0] + ' ' + new_word[1:]) 

        else:

            if (word.startswith('@')) or (word.startswith('#')):

                new_word = place_hold(new_word[0] + ' ' + new_word[1:len(word)-1]) + ' ' + word[len(word)-1]



    temp_dict[word] = new_word

temp_dict = {k: v for k, v in temp_dict.items() if k != v}

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - UserName and Hashtag:'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)        
# Remove ending underscore (or add quotation marks???)

# Local

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if (check_replace(k)) and ('_' in k)]

temp_dict = {}

for word in temp_vocab:

    new_word = word

    if word[len(word)-1]=='_':

        for i in range(len(word),0,-1):

            if word[i-1]!='_':

                new_word = word[:i]

                temp_dict[word] = new_word   

                break

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Remove ending underscore:'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)
# Remove starting underscore 

# Local

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if (check_replace(k)) and ('_' in k)]

temp_dict = {}

for word in temp_vocab:

    new_word = word

    if word[0]=='_':

        for i in range(len(word)):

            if word[i]!='_':

                new_word = word[i:]

                temp_dict[word] = new_word   

                break

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Remove starting underscore:'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)      
# End word punctuations

# Global

temp_vocab = list(set([c for line in data for c in line.split()]))

temp_vocab = [k for k in temp_vocab if (check_replace(k)) and (not k[len(k)-1].isalnum())]

temp_dict = {}

for word in temp_vocab:

    new_word = word

    for i in range(len(word),0,-1):

        if word[i-1].isalnum():

            new_word = word[:i] + ' ' + word[i:]

            break

    temp_dict[word] = new_word     

temp_dict = {k: v for k, v in temp_dict.items() if k != v}

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - End word punctuations:'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)                           
# Start word punctuations

# Global

temp_vocab = list(set([c for line in data for c in line.split()]))

temp_vocab = [k for k in temp_vocab if (check_replace(k)) and (not k[0].isalnum())]

temp_dict = {}

for word in temp_vocab:

    new_word = word

    for i in range(len(word)):

        if word[i].isalnum():

            new_word = word[:i] + ' ' + word[i:]

            break

    temp_dict[word] = new_word     

temp_dict = {k: v for k, v in temp_dict.items() if k != v}

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Start word punctuations:'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)  
# Find and replace acronims

# Local (only unknown words)

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if check_replace(k)]

temp_dict = {}

for word in temp_vocab:

    if (Counter(word)['.']>1) and (check_replace(word)):

        if (domain_search(word)!='') and (('www' in word) or (Counter(word)['/']>3)):

            temp_dict[word] = place_hold('url ' + domain_search(word))

        else: 

            if (re.compile('[\.\,]').sub('', word) in local_vocab) and (len(re.compile('[0-9\.\,\-\/\:]').sub('', word))>0):

                temp_dict[word] =  place_hold(re.compile('[\.\,]').sub('', word))

temp_dict = {k: v for k, v in temp_dict.items() if k != v}

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Find and replace acronims:'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)   
# Apply spellchecker for contractions

# Local (only unknown words)

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if (check_replace(k)) and ("'" in k)]

temp_dict = {}

for word in temp_vocab:

    if word in helper_contractions:

        temp_dict[word] = place_hold(helper_contractions[word])

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Contractions:'); check_vocab(data, local_vocab)

if verbose: print_dict(temp_dict)
# Isolate obscene (or just keep the word????)

# Local

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if check_replace(k)]

temp_dict = {}

for word in temp_vocab:

    new_word = re.compile('[a-zA-Z0-9\-\.\,\/\']').sub('', word)

    if len(Counter(new_word))>2:

        temp_dict[word] = place_hold('fuck')

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Possible obscene:'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)
# Remove 's (DO WE NEED TO REMOVE IT???)

# Local

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if check_replace(k)]

temp_dict = {k:k[:-2] for k in temp_vocab if (check_replace(k)) and (k.lower()[-2:]=="'s")}

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Remove "s:'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)                                        


# Convert backslash

# Global

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if (check_replace(k)) and ('\\' in k)]    

temp_dict = {k:re.sub('\\\\+', ' / ', k) for k in temp_vocab}

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Convert backslash:'); check_vocab(data, local_vocab)

if verbose: print_dict(temp_dict)
# Try remove duplicated chars (not sure about this!!!!!)

# Local (only unknown words)

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if check_replace(k)]

    

temp_dict = {}

temp_vocab_dup = []

    

for word in temp_vocab:

    temp_vocab_dup.append(''.join(ch for ch, _ in itertools.groupby(word)))

temp_vocab_dup = set(temp_vocab_dup)

temp_vocab_dup = temp_vocab_dup.difference(temp_vocab_dup.difference(set(local_vocab)))

            

for word in temp_vocab:

    new_word = ''.join(ch for ch, _ in itertools.groupby(word))

    if new_word in temp_vocab_dup:

        temp_dict[word] = new_word

temp_dict = {k: v for k, v in temp_dict.items() if (k != v) and (v in local_vocab)}

    

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Dup chars (with vocab check):'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)
# Isolate numbers

# Local (only unknown words)

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if check_replace(k)]

temp_dict = {}

for word in temp_vocab:

    if re.compile('[a-zA-Z]').sub('', word) == word:

        if re.compile('[0-9]').sub('', word) != word:

            temp_dict[word] = word



global_chars_list = list(set([c for line in temp_dict for c in line]))

chars = ''.join([c for c in global_chars_list if not c.isdigit()])

chars_dict = {ord(c):f' {c} ' for c in chars}                

temp_dict = {k:place_hold(make_cleaning(k,chars_dict)) for k in temp_dict}

    

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Isolate numbers:'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)
# Join dashes

# Local (only unknown words)

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if check_replace(k)]

    

temp_dict = {}

for word in temp_vocab:

    temp_dict[word] = re.sub('\-\-+', '-', word)

temp_dict = {k: v for k, v in temp_dict.items() if k != v}

    

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Join dashes:'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)
# Try join word (Sloooow)

# Local (only unknown words)

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if (check_replace(k)) and (Counter(k)['-']>1)]

    

temp_dict = {}

for word in temp_vocab:

    new_word = ''.join(['' if c in '-' else c for c in word])

    if (new_word in local_vocab) and (len(new_word)>3):

        temp_dict[word] = new_word    

    

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Try Split word:'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)
# Try Split word

# Local (only unknown words)

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if check_replace(k)]

    

temp_dict = {}

for word in temp_vocab:

    if len(re.compile('[a-zA-Z0-9\*]').sub('', word))>0:

        chars = re.compile('[a-zA-Z0-9\*]').sub('', word)

        temp_dict[word] = ''.join([' ' + c + ' ' if c in chars else c for c in word])

    

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Try Split word:'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)
# L33T vocabulary (SLOW)

# https://simple.wikipedia.org/wiki/Leet

# Local (only unknown words)

def convert_leet(word):

    # basic conversion 

    word = re.sub('0', 'o', word)

    word = re.sub('1', 'i', word)

    word = re.sub('3', 'e', word)

    word = re.sub('\$', 's', word)

    word = re.sub('\@', 'a', word)

    return word

            

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if check_replace(k)]

    

temp_dict = {}

for word in temp_vocab:

    new_word = convert_leet(word)

    if (new_word!=word): 

        if (len(word)>2) and (new_word in local_vocab):

            temp_dict[word] = new_word

    

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - L33T (with vocab check):'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)
# Open Holded words

# Global

temp_vocab = list(set([c for line in data for c in line.split()]))

temp_vocab = [k for k in temp_vocab if (not check_replace(k))]

temp_dict = {}

for word in temp_vocab:

    temp_dict[word] = re.sub('___', ' ', word[17:-1])

data = data.apply(lambda x: ' '.join([temp_dict.get(i, i) for i in x.split()]))

data = data.apply(lambda x: ' '.join([i for i in x.split()]))

if verbose: print('#' * 10, 'Step - Open Holded words:'); check_vocab(data, local_vocab)
# Search multiple form

# Local | example -> flashlights / flashlight -> False / True

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if (k[-1:]=='s') and (len(k)>4)]

temp_dict = {k:k[:-1] for k in temp_vocab if (k[:-1] in local_vocab)}

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Multiple form:'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)                                                                      
# Convert emoji to text

# Local 

temp_vocab = check_vocab(data, local_vocab, response='unknown_list')

temp_vocab = [k for k in temp_vocab if (k in emoji.UNICODE_EMOJI)]

temp_dict = {}

for word in temp_vocab:

    temp_dict[word] = re.compile('[:_]').sub(' ', emoji.UNICODE_EMOJI.get(word)) 

data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

if verbose: print('#' * 10, 'Step - Convert emoji to text:'); check_vocab(data, local_vocab);

if verbose: print_dict(temp_dict)