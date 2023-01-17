import pandas as pd

import numpy as np
data = pd.read_csv("../input/bbc-text.csv")
data.head(3)
def pri(x):

    for y in x:

        print(y)
def count_word(x):

#     print(type(x))

    word_list= x.split()

    return len(word_list)
print(count_word("  "))
data['word_count']= data['text'].apply(count_word)
data.head()
# smarter way

data['word_count1']= data['text'].apply(lambda x: len( x.split(" ")))

# show with and without split(" ")
# assignment find what is happening with and without split()

data.head()
def char_count(x):

    return len(x)
data['char_count']= data['text'].apply(char_count)
data['char_count1']= data['text'].apply(lambda x: len(x))
data[['char_count', 'char_count1']].head()
def avg_len(x):

    # use sun on list

    words= x.split()

    avg= sum(len(word) for word in words) / len(words)

    return avg

    

    
data['avg_len']= data['text'].apply(lambda x: sum(len(wrd) for wrd in x.split())/ len(x.split()))


from nltk.corpus import stopwords

stop = stopwords.words('english')

# we can make any dictionary or set of stop words
data['count_stp_wrd']= data['text'].apply(lambda x: len([x for x in x.split() if x in stop]))
data['count_stp_wrd'].head()


lambda x: len([x for x in x.split() if x.isdigit()])

#Number of Uppercase words

lambda x: len([x for x in x.split() if x.isupper()])
# special character count using starts with and endswith

hastags = data['text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
#Sample code to remove noisy words from a text



noise_list = ["is", "a", "this", "that"] 

def _remove_noise(input_text):

    words = input_text.split() 

    noise_free_words = [word for word in words if word not in noise_list] 

    noise_free_text = " ".join(noise_free_words) 

    return noise_free_text
# lowercase



lambda x: " ".join(x.lower() for x in x.split())



# https://www.geeksforgeeks.org/join-function-python/

# Syntax:

# string_name.join(iterable) 

# removal of puntucation

data['x']=data['text'].str.replace('[^\w\s]' , '')



# assignment how it works.
# remove stopwords from nltk stopwords

# many availbale library to many langugae

from nltk.corpus import stopwords

stop = stopwords.words('english')



lambda x: " ".join(x for x in x.split() if x not in stop)
tweet = "I luv &lt;3 cyberlab's class &amp; you're awsm people. ClassIsAwesome"


#Escaping HTML characters

from html.parser import HTMLParser

html_parser = HTMLParser()

tweet = html_parser.unescape(tweet)

print(tweet)
# Apostrophe removal

appo_dict = {"'s" : " is",  "'re" :  "are"}
def remove_appos(x):

    words= x.split()

    new_words=[]

    for word in words:

        flg= False;

        for suf in appo_dict:

            if(word.endswith(suf)):

                new_words.append(word[0:-len(suf)]+" "+appo_dict[suf])

                flg=True

                break

        if(flg==False):

            new_words.append(word)

    new_tweet= " ".join(new_words)

    return new_tweet

        

    # assignment do it with regex and find some smaller function like lambda and pythonistic way
print(remove_appos(tweet))

#https://www.tutorialspoint.com/python/string_endswith.htm
# common word counting

# .value_counts()

freq=pd.Series(' '.join(data['text']).split()).value_counts()

#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html
# rare word counting

rareword= freq[-10:]
# rare and frquent word removal

freq_word= list(freq.index[0:10])
print(freq_word)
rare_word= list(rareword.index)

print(rare_word)
# function to remove

lambda x: " ".join(x for x in x.split() if x not in freq)
# assignment split attached word and if this works or not

# cleaned = “ ”.join(re.findall(‘[A-Z][^A-Z]*’, tweet))
#spelling correction

from textblob import TextBlob

print(str(TextBlob("handsm").correct()))
from nltk.stem.wordnet import WordNetLemmatizer 

lem = WordNetLemmatizer()



from nltk.stem.porter import PorterStemmer 

stem = PorterStemmer()
word = "multiplying" 

print(lem.lemmatize(word, "v"))

print(stem.stem(word))
lookup_dict = {'rt':'Retweet', 'dm':'direct message', "awsm" : "awesome", "luv" :"love","plz": "please"}

def lookup_words(input_text):

    words = input_text.split() 

    new_words = [] 

    for word in words:

        if word.lower() in lookup_dict:

            word = lookup_dict[word.lower()]

        new_words.append(word) 

    new_text = " ".join(new_words) 

    return new_text
print(lookup_words("i dm you plz check. with luv"))