import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

trainDataSet = pd.read_csv('../input/train.csv')
testDataSet = pd.read_csv('../input/test.csv')
trainDataSet.head
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
print (trainDataSet.shape)
print (testDataSet.shape)
print (trainDataSet.columns)
def cleanText(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub('\d+', ' ', text)
    text = text.strip(' ')
    return text
for className in class_names:
    print (trainDataSet[className].value_counts())
print('Percentage of comments that are not labelled:')
print(len(trainDataSet[(trainDataSet['toxic']==0) & 
             (trainDataSet['severe_toxic']==0) & 
             (trainDataSet['obscene']==0) & 
             (trainDataSet['threat']== 0) & 
             (trainDataSet['insult']==0) &
             (trainDataSet['identity_hate']==0)]) / len(trainDataSet))
tempToxicDataSet = trainDataSet[trainDataSet['toxic'] == 0][0:1]
tempInsultDataSet = trainDataSet[trainDataSet['toxic'] == 1][0:1]
frames = [tempToxicDataSet, tempInsultDataSet]
tempTrainDataSet = pd.concat(frames)
print (tempTrainDataSet.shape)

tempTestDataSet = testDataSet[0:1]
print (tempTestDataSet.shape)
train_text = tempTrainDataSet['comment_text']
train_target = tempTrainDataSet.loc[:, class_names]
test_text = tempTestDataSet['comment_text']
all_text = pd.concat([train_text, test_text])
StringData = ""
for i in all_text:
    StringData += i
StringData
print ("Original Text - ",StringData,"\n")
sentences = nltk.sent_tokenize(StringData)
print (len(sentences))
sentences
words = nltk.word_tokenize(StringData)
print (len(words))
words
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
for i in range(len(sentences)):
    print ("Actual Sentence - ",sentences[i],"\n")
    stemmingWords = nltk.word_tokenize(sentences[i])
    lemmatizingWords = nltk.word_tokenize(sentences[i])
    stemmedWords = [stemmer.stem(word) for word in stemmingWords]
    lemmatizedWords = [lemmatizer.lemmatize(word) for word in lemmatizingWords]
    print ("Stemmed Words - \n")
    print (stemmedWords)
    print ("Lemmatized Words - \n")
    print (lemmatizedWords)
    print ("____________________________________________")
from nltk.corpus import stopwords

for i in range(len(sentences)):
    print ("Actual Sentence - ",sentences[i],"\n")
    words = nltk.word_tokenize(sentences[i])
    newwords = [word for word in words if word not in stopwords.words('english')]
    print (newwords)
    print ("____________________________________________")
words = nltk.word_tokenize(StringData)
tagged_words = nltk.pos_tag(words)
words_tags = []
for tw in tagged_words:
    words_tags.append(tw[0]+"_"+tw[1])
print (words_tags)
import matplotlib

words = nltk.word_tokenize(StringData)
tagged_words = nltk.pos_tag(words)
namedEntity = nltk.ne_chunk(tagged_words)
print (namedEntity)
import re

sentences = nltk.sent_tokenize(StringData)
for i in range(len(sentences)):
    sentences[i] = sentences[i].lower()
    sentences[i] = re.sub(r'\W',' ',sentences[i])
    sentences[i] = re.sub(r'\s+',' ',sentences[i])
sentences
word2count = {}
for data in sentences:
    words = nltk.word_tokenize(data)
    for word in words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
import heapq
freq_words = heapq.nlargest(30, word2count, key=word2count.get)
freq_words
bagOfWords = []
for data in sentences:
    vector = []
    for word in freq_words:
        if word in nltk.word_tokenize(data):
            vector.append(1)
        else:
            vector.append(0)
    bagOfWords.append(vector)
bagOfWords = np.asarray(bagOfWords)
bagOfWords
# IDF Matrix
word_idfs = {}
for word in freq_words:
    document_count = 0
    for data in sentences:
        if word in nltk.word_tokenize(data):
            document_count += 1
    word_idfs[word] = np.log((len(sentences)/document_count)+1) # +1 is the bias and standard way of calulating TF-IDF
word_idfs
# TF Matrix
tf_matrix = {}
for word in freq_words:
    doc_tf = []
    for data in sentences:
        frequency = 0
        for w in nltk.word_tokenize(data):
            if w == word:
                frequency += 1
        tf_word = frequency/len(nltk.word_tokenize(data))
        doc_tf.append(tf_word)
    tf_matrix[word] = doc_tf
# tf_matrix
# TF_IDF MATRIX
tfidf_matrix = []
for word in tf_matrix.keys():
    tfidf = []
    for value in tf_matrix[word]:
        score = value * word_idfs[word]
        tfidf.append(score)
    tfidf_matrix.append(tfidf)
tfidf_matrix
X = np.asarray(tfidf_matrix)
X = np.transpose(X)
X.shape
import random

text = train_text[0]
words = nltk.word_tokenize(text)
print (text)
print ("\n")
ngrams = {}
n = 3
for i in range(len(words) - n):
    gram = ' '.join(words[i:i+n])
    if gram not in ngrams.keys():
        ngrams[gram] = []
    ngrams[gram].append(words[i+n])

currentgram = ' '.join(words[0:n])
result = currentgram
for i in range(10):
    if currentgram not in ngrams.keys():
        break
    possibilities = ngrams[currentgram]
    nextItem = possibilities[random.randrange(len(possibilities))]
    result  += ' '+nextItem
    rwords = nltk.word_tokenize(result)
    currentGram = ' '.join(rwords[len(rwords)-n:len(rwords)])
    
print (ngrams)
from sklearn.decomposition import TruncatedSVD

dataset = ["The amount of population is increasing day by day",
           "The concert was great",
           "I love to see Gordan Ramsay cook",
           "Google is introducing a new technology",
           "AI Robots are the example of great technology present today",
           "All of us were singing in the concert",
           "We have launch campaigns to stop pollution and global warming"
          ]
dataset = [line.lower() for line in dataset]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset)
#print (X[0]) # (documentNumber , position) tfidf value

lsa = TruncatedSVD(n_components = 4, n_iter = 100) # n_components are the number of concepts that you want to find from the data
lsa.fit(X)
#print (lsa.components_[3])
terms = vectorizer.get_feature_names()
concept_words = {}
for i,comp in enumerate(lsa.components_):
    componentTerms = zip(terms,comp)
    sortedTerms = sorted(componentTerms, key=lambda x:x[1], reverse=True)
    sortedTerms =  sortedTerms[:10]
    concept_words["Concept "+str(i)] = sortedTerms

for key in concept_words.keys():
    sentence_scores = []
    for sentence in dataset:
        words = nltk.word_tokenize(sentence)
        score = 0
        for word in words:
            for word_with_score in concept_words[key]:
                if word == word_with_score[0]:
                    score += word_with_score[1]
        sentence_scores.append(score)
    print("\n"+key+"\n")
    for sent_score in sentence_scores:
        print (sent_score)
from nltk.corpus import wordnet

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for s in syn.lemmas():
        synonyms.append(s.name())
        for a in s.antonyms():
            antonyms.append(a.name())
print (set(antonyms))
print (set(synonyms))
import matplotlib.pyplot as plt
from wordcloud import WordCloud

wc = WordCloud().generate(StringData)
plt.figure(figsize=(15,15))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

from nltk.corpus import stopwords
import string

oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
totalWords =[]
Sentences = trainDataSet['comment_text'].values
cleanedSentences = ""
for i in range(0,5000):
    cleanedText = cleanText(Sentences[i])
    cleanedSentences += cleanedText
    requiredWords = nltk.word_tokenize(cleanedText)
    for word in requiredWords:
        if word not in oneSetOfStopWords and word not in string.punctuation:
            totalWords.append(word)
    
wordfreqdist = nltk.FreqDist(totalWords)
mostcommon = wordfreqdist.most_common(50)
print(mostcommon)

wc = WordCloud().generate(cleanedSentences)
plt.figure(figsize=(15,15))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
import string
print (string.punctuation)
