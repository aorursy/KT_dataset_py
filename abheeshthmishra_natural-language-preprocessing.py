import nltk
from nltk.corpus import brown
brown.words()
brown.categories()

print(len(brown.categories()))
data = brown.sents(categories = 'adventure')
len(data)
data = brown.sents(categories = 'fiction')
data
" ".join(data[1])
document = '''it was a very pleasent day. the weather was cool and there were light showers.

i went to the market to buty some fruits'''



sentence = "Send all the 50 document related to chapters 1,2,3,4 at prateek@cb.com"
from nltk.tokenize import sent_tokenize, word_tokenize
sents = sent_tokenize(document)

print(sents)
print(len(sents))

sentence.split()
word = word_tokenize(sentence)
word
from nltk.corpus import stopwords

sw = set(stopwords.words('english'))

print(sw)
def remove_stopwords(text,stopwords):

    useful_words = [w for w in text if w not in stopwords]

    return useful_words
text = "i am not bothered about her very much".split()

useful_text = remove_stopwords(text,sw)

print(useful_text)
'not' in sw
sentence = "Send all the 50 document related to chapters 1,2,3,4 at prateek@cb.com"

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer('[a-zA-Z]')

useful_text = tokenizer.tokenize(sentence)
useful_text
tokenizer = RegexpTokenizer('[a-zA-Z@.]+')

useful_text = tokenizer.tokenize(sentence)
useful_text
text   = """Foxes love to make jumpes. the quick brown fox was seen jumping over the

lovely dog from a 6th feet high wall"""
from nltk.stem.snowball import SnowballStemmer,PorterStemmer

from nltk.stem.lancaster import LancasterStemmer

#Snowball Stemmer ,Porter ,Lancaster Stemmer
ps = PorterStemmer()
ps.stem('jumping')
ps.stem('lovely')
ps.stem('loving')
ps.stem('jumped')
# let's work with snowball stemmer

ss = SnowballStemmer('english')
ss.stem('jumping')
'''# Lemitization

from nltk.stem import WordNetLemmatizer

wn = WordNetLemmatizer()

wn.lemmatize('jumping')'''
# Sample Corpus - Contains 4 Documents, each document can have 1 or more sentences

corpus = [

        'Indian cricket team will wins World Cup, says Capt. Virat Kohli. World cup will be held at Sri Lanka.',

        'We will win next Lok Sabha Elections, says confident Indian PM',

        'The nobel laurate won the hearts of the people.',

        'The movie Raazi is an exciting Indian Spy thriller based upon a real story.'

]

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
vectorized_corpus = cv.fit_transform(corpus)
vectorized_corpus = vectorized_corpus.toarray()
vectorized_corpus[0]
print(cv.vocabulary_)
len(cv.vocabulary_.keys())
#reverse maping

numbers = vectorized_corpus[2]

numbers
s = cv.inverse_transform(numbers)

print(s)
def myTokenizer(document):

    words = tokenizer.tokenize(document.lower())

    # Remove Stopwords

    words = remove_stopwords(words,sw)

    return words

    
#myTokenizer(sentence)

#print(sentence)
cv = CountVectorizer(tokenizer = myTokenizer)
vectorized_corpus = cv.fit_transform(corpus).toarray()
print(vectorized_corpus)

print(len(vectorized_corpus[0]))
cv.inverse_transform(vectorized_corpus)


# For Test Data

test_corpus = [

        'Indian cricket rock !',        

]
cv.transform(test_corpus).toarray()
sent_1  = ["this is good movie"]

sent_2 = ["this is good movie but actor is not present"]

sent_3 = ["this is not good movie"]
cv = CountVectorizer(ngram_range=(1,3))
docs = [sent_1[0],sent_2[0]]

cv.fit_transform(docs).toarray()

cv.vocabulary_
sent_1  = "this is good movie"

sent_2 = "this was good movie"

sent_3 = "this is not good movie"



corpus = [sent_1,sent_2,sent_3]
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
vc = tfidf.fit_transform(corpus).toarray()


print(vc)
tfidf.vocabulary_