# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install symspellpy
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from tqdm import tqdm

import pandas as pd

tqdm.pandas()

import re

import itertools
df = pd.read_csv('/kaggle/input/clean30reduced27k/Clean 3.0 Reduced complaints 27K.csv',encoding = 'latin-1')

df.head()
def remove_consecutive_duplicates(s1):

     return (''.join(i for i, _ in itertools.groupby(s1)))

    

    

def removal(text):

    text = re.sub('[^A-Za-z]',' ',text)

    text = re.sub('xxxx','',text)

    text = re.sub('xxx','',text)    

    text = re.sub('xx','',text)

    text = re.sub('xx\/xx\/\d+','',text)

    #text = re.sub('UNKNOWN   UNKNOWN','UNKNOWN',text)

    text = re.sub('\n',' ',text)

    text = re.sub(' +',' ',text)

    

    return text



stop_words = stopwords.words('english')

words = []

for i in tqdm(range(len(stop_words))):

    words.append(re.sub('[^A-Za-z]','',stop_words[i]))

    

stop_words = list(set(stop_words+words))
lem = WordNetLemmatizer()

pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*')



count = 0

def cleaning(text):

    #text = df['Consumer complaint narrative'].astype(str)

    text = text.lower()

    text = pattern.sub(' ', text)

    text = removal(text)

   



    #text = remove_consecutive_duplicates(text)



    #text = check(text)

    #print(text)



    #text = segment(text)

    #print(text)

    #text = ' '.join(text)

    #print(text)

    #text = pattern.sub(' ', text)

    #text = check(text)

    #text = pattern.sub(' ', text)

    #print(text)

    #text = wordninja.split(text)

    text = main(text)

    text = pattern.sub(' ', text)

    text = word_tokenize(text)

    text = [lem.lemmatize(w,'v') for w in text]

    text = ' '.join(text)

    text = re.sub(r'\b\w{1,2}\b','', text)

    text = re.sub(' +', ' ',text)

    global count

    count = count+1

    print(count)

    return text







import pkg_resources



from symspellpy.symspellpy import SymSpell, Verbosity  # import the module



# maximum edit distance per dictionary precalculation

max_edit_distance_dictionary = 2

prefix_length = 7

# create object

sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)

# load dictionary

dictionary_path = pkg_resources.resource_filename(

    "symspellpy", "frequency_dictionary_en_82_765.txt")

bigram_path = pkg_resources.resource_filename(

    "symspellpy", "frequency_bigramdictionary_en_243_342.txt")

# term_index is the column of the term and count_index is the

# column of the term frequency

if not sym_spell.load_dictionary(dictionary_path, term_index=0,

                                 count_index=1):

    print("Dictionary file not found")

    

if not sym_spell.load_bigram_dictionary(bigram_path, term_index=0,

                                        count_index=2):

    print("Bigram dictionary file not found")

    



def main(text):

    





    # lookup suggestions for multi-word input strings (supports compound

    # splitting & merging)

   # text = "Hello {$1.00} ea equifax pnc bankamerica exprss transunion acknowledgement aaaaaassssaaaapppp "

    #text = cleaning(text)

    #text = re.sub(r'\b\w{1,1}\b', '', text)

    #text = list(text.rstrip()) 

    #text = removeDuplicates(text) 

    #text = ''.join(text)

    #print(text)

    #text = removeDuplicates()

    input_term = (text)

    # max edit distance per lookup (per single word, not per whole input string)

    max_edit_distance_lookup = 2

    suggestions = sym_spell.lookup_compound(input_term,

                                            max_edit_distance_lookup)

    # display suggestion term, edit distance, and term frequency

    words = []

    for suggestion in suggestions:

#         print("{}, {}, {}".format(suggestion.term, suggestion.distance,

#                                   suggestion.count))

        words.append(suggestion.term)

    return ' '.join(words)
cleaning('hello my name is a aasap equifax')
df1 = df[df['Company response to consumer']=='Others']

df2 = df[df['Company response to consumer']=='Closed with monetary relief']

df3 = df[df['Company response to consumer']=='Closed with non-monetary relief']



df1 = df1[:7000]

df2 = df2[:7000]

df3 = df3[:7000]



df = pd.concat([df1,df2,df3])

df = df.sample(frac=1)

df.reset_index(drop=True,inplace = True)

df.head()
%%time

import multiprocessing as mp

from tqdm import tqdm



pool = mp.Pool(mp.cpu_count())



df['Total Clean Symspell'] = pool.map(cleaning, df['Consumer complaint narrative'])



pool.terminate()



pool.join()
df.head()
df.to_csv('Symspell cleaning 21K total 7k each Multithreading.csv',index = False)