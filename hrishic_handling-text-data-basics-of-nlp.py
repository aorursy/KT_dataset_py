text_data = [" Interrobang. By HK Henriette ",
                "Parking And Going. By Karl Gautier",
                " Today Is The night. By Jarek Prakash "]
string_without_whitespace = [string.strip() for string in text_data]
string_without_whitespace
string_without_period = [string.replace(".", "") for string in string_without_whitespace]
string_without_period
capital_text = [string.upper() for string in string_without_period]
capital_text
from bs4 import BeautifulSoup
import nltk
# Create sample HTML content
html = """
<div class='full_name'><span style='font-weight:bold'>
Masego</span> Azra</div>"
"""

# Parse html
soup = BeautifulSoup(html, "lxml")

#Find div with class name 'full_name'
soup.find('div', {'class':'full_name'}).text
import string
text_data = ['Hi!!!! I. Love. This. Song....',
            '10000% Agree!!!! #LoveIT',
            'Right?!?!']
res_list = []
for string_tmp in text_data:
    res = ''
    for c in string_tmp:
        if c not in string.punctuation:
            res = res+c
    res_list.append(res)
res_list
from nltk.tokenize import word_tokenize
string = "The science of today is the technology of tomorrow. Thus the end"

word_tokenize(string)
from nltk.corpus import stopwords
text = "I am going to market today I had enjoyed your ride"
text = text.lower()
list_of_words = word_tokenize(text)

tmp = stopwords.words('english')
text_without_stopwords = [word for word in list_of_words if word not in tmp]

text_without_stopwords
tmp[:10]    #Here tmp is list of stopwords as assigned in above cell
from nltk.stem.porter import PorterStemmer

# Here we already have tokenized words from above output, i.e. text_without_stopwords

#Create porter
porter = PorterStemmer()

root_words = [porter.stem(word) for word in text_without_stopwords]
root_words
from nltk import pos_tag
from nltk import word_tokenize
text_data = "Chris loved outdoor running"
# Use pre-trained part of speech tagger
text_tagged = pos_tag(word_tokenize(text_data))
text_tagged
'''
NNP: Proper noun, singular
NN: Noun, singular or mass
RB: Adverb
VBD: Verb, past tense
VBG: Verb, gerund or present participle
JJ: Adjective
PRP: Personal pronoun
'''
print()
# Example to get all nouns
nouns = [word for word, tag in text_tagged if tag in {'NN','NNS','NNP','NNPS'}]
nouns
tweets = ["I am eating a burrito for breakfast",
        "Political science is an amazing field",
        "San Francisco is an awesome city"]

# Create list
tagged_tweets = []
# Tag each word and each tweet
for tweet in tweets:
    tweet_tag = nltk.pos_tag(word_tokenize(tweet))
    tagged_tweets.append([tag for word, tag in tweet_tag])
    
tagged_tweets
# Using one hot encoder
from sklearn.preprocessing import MultiLabelBinarizer

# Use one-hot encoding to convert the tags into features
one_hot_multi = MultiLabelBinarizer()
one_hot_multi.fit_transform(tagged_tweets)
# Using classes_ we can see that each feature is part-of-speech-tag
one_hot_multi.classes_
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Create text
text_data = np.array(['I love India. India!',
                        'Pune is the best',
                        'xyz beats both'])

# Create the bag of words feature matrix
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# Show feature matrix
bag_of_words
# Create feature matrix with arguments
count_2gram = CountVectorizer(ngram_range=(1,2),
    stop_words="english")

bag = count_2gram.fit_transform(text_data)
# View feature matrix
bag.toarray()
count_2gram.vocabulary_
# import requirements:
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# Create sample data
text_data = np.array(['I love India. India!',
                        'Japan is the best',
                        'NY beats both'])
# Create the tf-idf feature matrix
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)

feature_matrix.toarray()
