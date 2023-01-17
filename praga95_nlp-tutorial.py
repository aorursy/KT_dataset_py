import sys
reload(sys)
sys.setdefaultencoding('utf8')
import nltk
nltk.download()
paragraph = """Thank you all so very much. Thank you to the Academy. 
               Thank you to all of you in this room. I have to congratulate 
               the other incredible nominees this year. The Revenant was 
               the product of the tireless efforts of an unbelievable cast
               and crew. First off, to my brother in this endeavor, Mr. Tom 
               Hardy. Tom, your talent on screen can only be surpassed by 
               your friendship off screen … thank you for creating a t
               ranscendent cinematic experience. Thank you to everybody at 
               Fox and New Regency … my entire team. I have to thank 
               everyone from the very onset of my career … To my parents; 
               none of this would be possible without you. And to my 
               friends, I love you dearly; you know who you are. And lastly,
               I just want to say this: Making The Revenant was about
               man's relationship to the natural world. A world that we
               collectively felt in 2015 as the hottest year in recorded
               history. Our production needed to move to the southern
               tip of this planet just to be able to find snow. Climate
               change is real, it is happening right now. It is the most
               urgent threat facing our entire species, and we need to work
               collectively together and stop procrastinating. We need to
               support leaders around the world who do not speak for the 
               big polluters, but who speak for all of humanity, for the
               indigenous people of the world, for the billions and 
               billions of underprivileged people out there who would be
               most affected by this. For our children’s children, and 
               for those people out there whose voices have been drowned
               out by the politics of greed. I thank you all for this 
               amazing award tonight. Let us not take this planet for 
               granted. I do not take tonight for granted. Thank you so very much."""
               
sentences = nltk.sent_tokenize(paragraph)
print(len(sentences))
sentences
words=[nltk.word_tokenize(sent) for sent in sentences]
words
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
words=[nltk.word_tokenize(sent) for sent in sentences]

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words]
    sentences[i] = ' '.join(words)      
sentences
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
newsentences = nltk.sent_tokenize(paragraph)
for i in range(len(newsentences)):
    words = nltk.word_tokenize(newsentences[i])
    words = [lemmatizer.lemmatize(word) for word in words]
    newsentences[i] = ' '.join(words)      
newsentences
nltk.download('stopwords')
from nltk.corpus import stopwords
for i in range(len(newsentences)):
    words = nltk.word_tokenize(newsentences[i])
    words = [word for word in words if word not in stopwords.words('english')]
    newsentences[i] = ' '.join(words)            
newsentences
words = nltk.word_tokenize(paragraph)
tagged_words=nltk.pos_tag(words)
tagged_words
word_tags = []
for tw in tagged_words:
    word_tags.append(tw[0]+"-"+tw[1])
word_tags
tagged_paragraph = ' '.join(word_tags)
tagged_paragraph
words = nltk.word_tokenize(paragraph)
words
tagged_words = nltk.pos_tag(words)
tagged_words
namedEnt = nltk.ne_chunk(tagged_words)

# Tokenize the article into sentences: sentences
sentences = nltk.sent_tokenize(paragraph)
# Tokenize each sentence into words: token_sentences
token_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Tag each tokenized sentence into parts of speech: pos_sentences
pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences] 

# Create the named entity chunks: chunked_sentences
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)

# Test for stems of the tree with 'NE' tags
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, "label") and chunk.label() == "NE":
            print(chunk)

article= """The Taj Mahal was built by Emperor Shah Jahan"""
words=nltk.word_tokenize(article)
tagged_words=nltk.pos_tag(words)
tagged_words
chunked_words=nltk.ne_chunk(tagged_words)
#chunked_words.draw()
import re
dataset = nltk.sent_tokenize(paragraph)
for i in range(len(dataset)):
    dataset[i] = dataset[i].lower()
    dataset[i] = re.sub(r'\W',' ',dataset[i])
    dataset[i] = re.sub(r'\s+',' ',dataset[i])
dataset[0:6]
word2count = {}
for data in dataset:
    words = nltk.word_tokenize(data)
    for word in words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
word2count
import heapq
freq_words = heapq.nlargest(25,word2count,key=word2count.get)
freq_words
import numpy as np
X = []
for data in dataset:
    vector = []
    for word in freq_words:
        if word in nltk.word_tokenize(data):
            vector.append(1)
        else:
            vector.append(0)
    X.append(vector)
        
X = np.asarray(X)
X
from collections import Counter
# Tokenize the article: tokens
tokens = nltk.word_tokenize(paragraph)
tokens = [word for word in tokens if word not in stopwords.words('english')]
# Convert the tokens into lowercase: lower_tokens
lower_tokens = [t.lower() for t in tokens ]
lower_tokens_1 = [re.sub(r'\W',' ',t) for t in lower_tokens]
lower_tokens_2 = [re.sub(r'\s+',' ',t) for t in lower_tokens_1]
alpha_only = [t for t in lower_tokens_2 if t.isalpha()]
# Create a Counter with the lowercase tokens: bow_simple
bow_simple =Counter(alpha_only)

# Print the 10 most common tokens
print(bow_simple.most_common(11))
paragraph = """Thank you all so very much. Thank you to the Academy. 
               Thank you to all of you in this room. I have to congratulate 
               the other incredible nominees this year. The Revenant was 
               the product of the tireless efforts of an unbelievable cast
               and crew. First off, to my brother in this endeavor, Mr. Tom 
               Hardy. Tom, your talent on screen can only be surpassed by 
               your friendship off screen … thank you for creating a t
               ranscendent cinematic experience. Thank you to everybody at 
               Fox and New Regency … my entire team. I have to thank 
               everyone from the very onset of my career … To my parents; 
               none of this would be possible without you. And to my 
               friends, I love you dearly; you know who you are. And lastly,
               I just want to say this: Making The Revenant was about
               man's relationship to the natural world. A world that we
               collectively felt in 2015 as the hottest year in recorded
               history. Our production needed to move to the southern
               tip of this planet just to be able to find snow. Climate
               change is real, it is happening right now. It is the most
               urgent threat facing our entire species, and we need to work
               collectively together and stop procrastinating. We need to
               support leaders around the world who do not speak for the 
               big polluters, but who speak for all of humanity, for the
               indigenous people of the world, for the billions and 
               billions of underprivileged people out there who would be
               most affected by this. For our children’s children, and 
               for those people out there whose voices have been drowned
               out by the politics of greed. I thank you all for this 
               amazing award tonight. Let us not take this planet for 
               granted. I do not take tonight for granted. Thank you so very much."""
               
               
# Tokenize sentences
dataset = nltk.sent_tokenize(paragraph)
for i in range(len(dataset)):
    dataset[i] = dataset[i].lower()
    dataset[i] = re.sub(r'\W',' ',dataset[i])
    dataset[i] = re.sub(r'\s+',' ',dataset[i])


# Creating word histogram
word2count = {}
for data in dataset:
    words = nltk.word_tokenize(data)
    for word in words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
            
# Selecting best 100 features
freq_words = heapq.nlargest(100,word2count,key=word2count.get)


# IDF Dictionary
word_idfs = {}
for word in freq_words:
    doc_count = 0
    for data in dataset:
        if word in nltk.word_tokenize(data):
            doc_count += 1
    word_idfs[word] = np.log(len(dataset)/(1+doc_count))
    
# TF Matrix
tf_matrix = {}
for word in freq_words:
    doc_tf = []
    for data in dataset:
        frequency = 0
        for w in nltk.word_tokenize(data):
            if word == w:
                frequency += 1
        tf_word = frequency/len(nltk.word_tokenize(data))
        doc_tf.append(tf_word)
    tf_matrix[word] = doc_tf
    
# Creating the Tf-Idf Model
tfidf_matrix = []
for word in tf_matrix.keys():
    tfidf = []
    for value in tf_matrix[word]:
        score = value * word_idfs[word]
        tfidf.append(score)
    tfidf_matrix.append(tfidf)   
    
# Finishing the Tf-Tdf model
X = np.asarray(tfidf_matrix)

X = np.transpose(X)
X
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
response = tfidf.fit_transform([paragraph])
feature_names = tfidf.get_feature_names()
feature_names
for col in response.nonzero()[1]:
    print (feature_names[col], ' - ', response[0, col])
import pandas as pd
tfidf_df= pd.DataFrame()
tfidf_df['Feature names']=feature_names
tfidf_df['weights']=[response[0,col] for col in response.nonzero()[1]]

tfidf_df=tfidf_df.sort_values(by=['weights'],ascending=False)
tfidf_df.head(10)
import random

# Sample data
text = """Global warming or climate change has become a worldwide concern. It is gradually developing into an unprecedented environmental crisis evident in melting glaciers, changing weather patterns, rising sea levels, floods, cyclones and droughts. Global warming implies an increase in the average temperature of the Earth due to entrapment of greenhouse gases in the earth’s atmosphere."""

# Order of the grams
n = 3

# Our N-Grams
ngrams = {}

# Creating the model
for i in range(len(text)-n):
    gram = text[i:i+n]
    if gram not in ngrams.keys():
        ngrams[gram] = []
    ngrams[gram].append(text[i+n])
ngrams
currentGram = text[0:n]
result = currentGram
for i in range(300):
    if currentGram not in ngrams.keys():
        break
    possibilities = ngrams[currentGram]
    nextItem = possibilities[random.randrange(len(possibilities))]
    result += nextItem
    currentGram = result[len(result)-n:len(result)]
  
print(result)
n = 3

# Our N-Grams
ngrams = {}

# Building the model
words = nltk.word_tokenize(text)
for i in range(len(words)-n):
    gram = ' '.join(words[i:i+n])
    if gram not in ngrams.keys():
        ngrams[gram] = []
    ngrams[gram].append(words[i+n])
    
# Testing the model
currentGram = ' '.join(words[0:n])
result = currentGram
for i in range(30):
    if currentGram not in ngrams.keys():
        break
    possibilities = ngrams[currentGram]
    nextItem = possibilities[random.randrange(len(possibilities))]
    result += ' '+nextItem
    rWords = nltk.word_tokenize(result)
    currentGram = ' '.join(rWords[len(rWords)-n:len(rWords)])

print(result)
def find_bigrams(words):
  bigram_list = []
  for i in range(len(words)-1):
      bigram_list.append((words[i], words[i+1]))
  return bigram_list
find_bigrams(words)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Sample Data
dataset = ["The amount of polution is increasing day by day",
           "The concert was just great",
           "I love to see Gordon Ramsay cook",
           "Google is introducing a new technology",
           "AI Robots are examples of great technology present today",
           "All of us were singing in the concert",
           "We have launch campaigns to stop pollution and global warming"]

dataset = [line.lower() for line in dataset]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset)

# Visualizing the Tfidf Model
print(X[0])
# Creating the SVD
lsa = TruncatedSVD(n_components = 4, n_iter = 4000)
lsa.fit(X)
# First row of V
row1 = lsa.components_[0] ## the first concept. it has 4 concepts. so row1
# is displaying the probability of the diff words to be in concept 0.
len(row1)
row1
# Visualizing the concepts
concept_words = {}
terms = vectorizer.get_feature_names()## the 42 words
for i,comp in enumerate(lsa.components_):
    componentTerms = zip(terms,comp)
    sortedTerms = sorted(componentTerms,key=lambda x:x[1],reverse=True)
    sortedTerms = sortedTerms[:10]
    concept_words["Concept "+str(i)] = sortedTerms
    print("\nConcept",i,":")
    for term in sortedTerms:
        print(term)
concept_words
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
    print("\n"+key+":")
    for sentence_score in sentence_scores:
        print(sentence_score)
from nltk.corpus import wordnet

# Initializing the list of synnonyms and antonyms
synonyms = []
antonyms = []

for syn in wordnet.synsets("correct"):
    for s in syn.lemmas():
        synonyms.append(s.name())
        for a in s.antonyms():
            antonyms.append(a.name())
            
            
# Displaying the synonyms and antonyms
print('Synonyms are',set(synonyms))
print('Antonyms are',set(antonyms))
sentence = "I was not sad with the team's performance"

words = nltk.word_tokenize(sentence)

new_words = []

temp_word = ''
for word in words:
    if word == 'not':
        temp_word = 'not_'
    elif temp_word == 'not_':
        word = temp_word + word
        temp_word = ''
    if word != 'not':
        new_words.append(word)

sentence = ' '.join(new_words)
sentence
sentence = "I was not sad with the team's performance"

words = nltk.word_tokenize(sentence)

new_words = []

temp_word = ''
for word in words:
    antonyms = []
    if word == 'not':
        temp_word = 'not_'
    elif temp_word == 'not_':
        for syn in wordnet.synsets(word):
            for s in syn.lemmas():
                for a in s.antonyms():
                    antonyms.append(a.name())
        if len(antonyms) >= 1:
            word = antonyms[0]
        else:
            word = temp_word + word # when antonym is not found, it will
                                    # remain not_happy
        temp_word = ''
    if word != 'not':
        new_words.append(word)

sentence = ' '.join(new_words)
sentence
