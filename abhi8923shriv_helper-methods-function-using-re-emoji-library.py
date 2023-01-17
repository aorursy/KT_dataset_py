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
import numpy as np

import pandas as pd

import os

import re   # regular expression library

import emoji

import numpy as np



#Count vectorizer for N grams

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer



# Nltk for tekenize and stopwords

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 



#Loading data



df=pd.read_csv('/kaggle/input/tweetsentimentextraction/train.csv')

df.head

# Boundaries



# Picking up the words with boundaries



def boundary(text):

    line=re.findall(r'\bneutral\b', text)

    return " ".join(line)
sentence="Most tweets are neutral in twitter"

boundary(sentence)
# Search



# Is the key word present in the sentence?



def search_string(text,key):

    return bool(re.search(r''+key+'',text))
sentence="happy mothers day to all moms"

search_string(sentence,'day')
# OR



def or_cond(text,key1,key2):

    line=re.findall(r"{}|{}".format(key1,key2),text)

    return " ".join(line)

    
sentence="sad or sorrow displays emotions"

or_cond(sentence,'sad','sorrow')
# AND



def and_cond(text):

    line=re.findall(r'(?=.*do)(?=.*die).*', text) 

    return " ".join(line)



                    

print("Both string present:",and_cond("do or die is a motivating phrase"))

print("Only one string present :",and_cond('die word is other side of emotion'))
# Dates # # mm-dd-yyyy format 



def find_dates(text):

    line=re.findall(r'\b(1[0-2]|0[1-9])/(3[01]|[12][0-9]|0[1-9])/([0-9]{4})\b',text)

    return line

    

    
# Only Words



def only_words(text):

    line=re.findall(r'\b[^\d\W]+\b', text)

    return " ".join(line)



    
sentence="the world population has grown from 1650 million to 6000 million"

only_words(sentence)
# pick sentence

# If we want to get all sentence with particular keyword.We can use below function



def pick_only_key_sentence(text,keyword):

    line=re.findall(r'([^.]*'+keyword+'[^.]*)', text)

    return line
sentence="my job is covid work.i enjoy my work.covid is dangerous"

pick_only_key_sentence(sentence,"covid")
# Duplicate Sentence

# Most webscrapped data contains duplicated sentence.This function could retrieve unique ones.



def pick_unique_sentence(text):

    line=re.findall(r'(?sm)(^[^\r\n]+$)(?!.*^\1$)', text)

    return line
sentence="I thank doctors\nDoctors are working very hard in this pandemic situation\nI thank doctors"

pick_unique_sentence(sentence)
# dates



def find_dates(text):

    line=re.findall(r'\b(1[0-2]|0[1-9])/(3[01]|[12][0-9]|0[1-9])/([0-9]{4})\b',text)

    return line
sentence="Todays date is 04/28/2020 for format mm/dd/yyyy, not 28/04/2020"

find_dates(sentence)
# Caps Words



# Extract words starting with capital letter.Some words like names,place or universal object are usually mentioned in a text starting with CAPS.



def find_capital(text):

    line=re.findall(r'\b[A-Z]\w+', text)

    return line
sentence="World is affected by corona crisis.No one other than God can save us from it"

find_capital(sentence)
# Length of characters

# one liner to identify length of characters in a sentence



df['char_length']=df['text'].str.len()

df[['text','char_length']].sample(3)
# Get ID



# Most data has IDs in it with some prefix.So if we want to pick only numbers in ID leaving the prefix out,we can apply below function.



def find_ID(text):

    line=re.findall(r'\bIND(\d+)',text)

    return line
sentence= " my india IND2345 is my picture"

find_ID(sentence)
# Hex code to Color

# Converting hex color codes to color names.We will install and import webcolors. (only for CSS3 colors)



!pip install webcolors

import webcolors



def find_color(string): 

    text = re.findall('\#(?:[0-9a-fA-F]{3}){1,2}',string)

    conv_name=[]

    for i in text:

        conv_name.append(webcolors.hex_to_name(i))

    return conv_name

    



sentence="Find the color of #00FF00 and #FF4500"

find_color(sentence)
# Tags

# Most of web scrapped data contains html tags.It can be removed from below re script



def remove_tag(string):

    text=re.sub('<.*?>','',string)

    return text

    
sentence="Markdown sentences can use <br> for breaks and <i></i> for italics"

remove_tag(sentence)
# IP Address

# Extract IP address from text.



def ip_add(string):

    text=re.findall('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',string)

    return text
sentence="An example of ip address is 125.16.100.1"

ip_add(sentence)
# Mac Address



# Extract Mac address from text.



def mac_add(string):

    text=re.findall('(?:[0-9a-fA-F]:?){12}',string)

    return text
sentence="MAC ADDRESSES of this laptop - 00:24:17:b1:cc:cc .Other details will be mentioned"

mac_add(sentence)
df.tail
# stop stopwords



def stop_word_fn(text):

    stop_words = set(stopwords.words('english')) 

    word_tokens = word_tokenize(text) 

    non_stop_words = [w for w in word_tokens if not w in stop_words] 

    stop_words= [w for w in word_tokens if w in stop_words] 

    return stop_words
sentence="this is testing for stop words,which is going to proof our logic"

stop_word_fn(sentence)
# N-grams



def ngrams_top(corpus,ngram_range,n=None):

    """

    List the top n words in a vocabulary according to occurrence in a text corpus.

    """

    vec = CountVectorizer(stop_words = 'english',ngram_range=ngram_range).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    total_list=words_freq[:n]

    df=pd.DataFrame(total_list,columns=['text','count'])

    return df

# Repetitive Character



# If you want to change match repetitive characters to n numbers,chage the return line in the rep function to grp[0:n].



def rep(text):

    grp = text.group(0)

    if len(grp) > 1:

        return grp[0:1] # can change the value here on repetition

def unique_char(rep,sentence):

    convert = re.sub(r'(\w)\1+', rep, sentence) 

    return convert

sentence="heyyyyy godd pls helppp me"

unique_char(rep,sentence)
# Number-Greater than 930



#Number greater than 930

def num_great(text): 

    line=re.findall(r'9[3-9][0-9]|[1-9]\d{3,}',text)

    return " ".join(line)

sentence="It is expected to be more than 935 corona death and 29974 observation cases across 29 states in india"

num_great(sentence)
# Number Lesser



def num_less(text):

    only_num=[]

    for i in text.split():

        line=re.findall(r'^(9[0-2][0-0]|[1-8][0-9][0-9]|[1-9][0-9]|[0-9])$',i) # 5 500

        only_num.append(line)

        all_num=[",".join(x) for x in only_num if x != []]

    return " ".join(all_num)

sentence="There are some countries where less than 920 cases exist with 1100 observations"

num_less(sentence)
# find dollar value



def find_dollar(text):

    line=re.findall(r'\$\d+(?:\.\d+)?',text)

    return " ".join(line)



# \$ - dollar sign followed by

# \d+ one or more digits

# (?:\.\d+)? - decimal which is optional

sentence="this shirt costs $20.56"

find_dollar(sentence)
# Phone Number

# Indian Mobile numbers have ten digit.I will write that pattern below



def find_phone_number(text):

    line=re.findall(r"\b\d{10}\b",text)

    return "".join(line)

    
sentence="my phone no. is 8888890999"

find_phone_number(sentence)
# Year

# Extract year from 1940 till 2020



def find_year(text):

    line=re.findall(r"\b(19[40][0-9]|20[0-1][0-9]|2020)\b",text)

    return line
sentence="India got independence on 1947."

find_year(sentence)
# Non Alphanumeric

# Alphanumeric, also referred to as alphameric, is a term that encompasses all of the letters and numerals in a given language set.



def find_nonalp(text):

    line = re.findall("[^A-Za-z0-9 ]",text)

    return line

    
sentence="Twitter has 9 lots of @ and # in posts.(general tweet)"

find_nonalp(sentence)



# here it has excluded alphabet and numbers.
# Punctuations

# Retrieve punctuations from sentence.

# Punctuation are marks used in printing and writing to separate sentences and clauses and to help make the meaning of sentences more clear. Commas, periods and question marks are examples of punctuation.

# There are 14 punctuation marks that are commonly used in English grammar. They are the period, question mark, exclamation point, comma, semicolon, colon, dash, hyphen, parentheses, brackets, braces, apostrophe, quotation marks, and ellipsis.



def find_punct(text):

    line = re.findall(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*', text)

    string="".join(line)

    return list(string)

example="Corona virus have kiled #24506 confirmed cases now.#Corona is un(tolerable)"

print(find_punct(example))
# Missing values



def missing_value_of_data(data):

    total=data.isnull().sum().sort_values(ascending=False)

    percentage=round(total/data.shape[0]*100,2)

    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])

missing_value_of_data(df)
# We will drop NA values

df=df.dropna()
#count values



def count_values_in_column(data,feature):

    total=data.loc[:,feature].value_counts(dropna=False)

    percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)

    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])
count_values_in_column(df,'sentiment')
# Unique Values



def unique_values_in_column(data,feature):

    unique_val=pd.Series(data.loc[:,feature].unique())

    return pd.concat([unique_val],axis=1,keys=['Unique Values'])
unique_values_in_column(df,'sentiment')
# Duplicate Values



def duplicated_values_data(data):

    dup=[]

    columns=data.columns

    for i in data.columns:

        dup.append(sum(data[i].duplicated()))

    return pd.concat([pd.Series(columns),pd.Series(dup)],axis=1,keys=['Columns','Duplicate count'])
duplicated_values_data(df)
# Stat

df.describe()
# Regex Helpers
# find URL 

# # re means regular expression library 



def find_url(string): 

    text = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',string)

    return "".join(text) # converting return value from list to string

    
sentence="I love spending so much time at https://www.kaggle.com/"

find_url(sentence)
# Emoticons

# Find and convert emoji to text



def find_emoji(text):

    emo_text=emoji.demojize(text)

    line=re.findall(r'\:(.*?)\:',emo_text)

    return line



    
sentence="I love ⚽ very very much 😁"

find_emoji(sentence)
# Remove Emoji from text



def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)

    
sentence="Its all about \U0001F600 face"

print(sentence)

remove_emoji(sentence)
# find email

# Extract email from text



def find_email(text):

    line = re.findall(r'[\w\.-]+@[\w\.-]+',str(text))

    return ",".join(line)

    
sentence="My gmail is abcdef99@gmail.com"

find_email(sentence)
# Hash

def find_hash(text):

    line=re.findall(r'(?<=#)\w+',text)

    return " ".join(line)
sentence="#Corona is latest trending now in the world" 

find_hash(sentence)
# Mention

# @ - Used to mention someone in tweets



def find_at(text):

    line=re.findall(r'(?<=@)\w+',text)

    

    return " ".join(line)
sentence="@David ji,can you help me out"

find_at(sentence)
# Number

# Pick only number from sentence



def find_number(text):

    line=re.findall(r'[0-9]+',text)

    return " ".join(line)
sentence="2833047567 people are affected by corona now"

find_number(sentence)