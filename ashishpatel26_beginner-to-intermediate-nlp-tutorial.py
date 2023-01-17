import nltk

def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})

format_sentence("Life is beautiful so Enjoy everymoment you have.")
pos = []
with open("../input/sentimental-analysis-nlp/pos_tweets.txt") as f:
    for i in f: 
        pos.append([format_sentence(i), 'pos'])
        
pos[0]
neg = []
with open("../input/sentimental-analysis-nlp/neg_tweets.txt") as f:
    for i in f: 
        neg.append([format_sentence(i), 'neg'])
        
neg[0]
training = pos[:int((.9)*len(pos))] + neg[:int((.9)*len(neg))]
test = pos[int((.1)*len(pos)):] + neg[int((.1)*len(neg)):]
from nltk.classify import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(training)
classifier.show_most_informative_features()
example1 = "this workshop is awesome."
example2 = "This workshop is not good"

print(classifier.classify(format_sentence(example1)))
print(classifier.classify(format_sentence(example2)))
from nltk.classify.util import accuracy

print(accuracy(classifier, test))
import re
re1 = re.compile('python')
print(bool(re1.match('Python')))
import nltk 

text = nltk.word_tokenize("Python is an awesome language!")
nltk.pos_tag(text)
nltk.help.upenn_tagset('JJ')
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
unigram_tagger.tag(brown_sents[2007])
bigram_tagger = nltk.BigramTagger(brown_tagged_sents)
bigram_tagger.tag(brown_sents[2007])
raw = "OMG, Natural Language Processing is SO cool and I'm really enjoying this workshop!"
tokens = nltk.word_tokenize(raw)
tokens = [i.lower() for i in tokens]
tokens
lancaster = nltk.LancasterStemmer()
stems = [lancaster.stem(i) for i in tokens]
stems
porter = nltk.PorterStemmer()
stem = [porter.stem(i) for i in tokens]
stem
from nltk import WordNetLemmatizer

lemma = nltk.WordNetLemmatizer()
text = "Women in technology are amazing at coding"
ex = [i.lower() for i in text.split()]
lemmas = [lemma.lemmatize(i) for i in ex]
lemmas
from nltk.corpus import wordnet as wn
print(wn.synsets('motorcar'))
print(wn.synset('car.n.01').lemma_names())
print(wn.synset('car.n.01').definition())
from nltk.corpus import sentiwordnet as swn
cat = swn.senti_synset('cat.n.03')
cat.pos_score()
cat.neg_score()
cat.obj_score()
cat.unicode_repr()
import nltk 
sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"), ("dog", "NN"), ("barked", "VBD"), ("at", "IN"), ("the", "DT"), ("cat", "NN")]
pattern = "NP: {<DT>?<JJ>*<NN>}" 
NPChunker = nltk.RegexpParser(pattern)
result = NPChunker.parse(sentence) 
result.draw
import spacy
import pandas as pd
nlp = spacy.load('en')
review = "Columbia University was founded in 1754 as King's College by royal charter of King George II of England. It is the oldest institution of higher learning in the state of New York and the fifth oldest in the United States. Controversy preceded the founding of the College, with various groups competing to determine its location and religious affiliation. Advocates of New York City met with success on the first point, while the Anglicans prevailed on the latter. However, all constituencies agreed to commit themselves to principles of religious liberty in establishing the policies of the College. In July 1754, Samuel Johnson held the first classes in a new schoolhouse adjoining Trinity Church, located on what is now lower Broadway in Manhattan. There were eight students in the class. At King's College, the future leaders of colonial society could receive an education designed to 'enlarge the Mind, improve the Understanding, polish the whole Man, and qualify them to support the brightest Characters in all the elevated stations in life.'' One early manifestation of the institution's lofty goals was the establishment in 1767 of the first American medical school to grant the M.D. degree."
doc = nlp(review)
sentences = [sentence.orth_ for sentence in doc.sents] # list of sentences
print("There were {} sentences found.".format(len(sentences)))
nounphrases = [[np.orth_, np.root.head.orth_] for np in doc.noun_chunks]
print("There were {} noun phrases found.".format(len(nounphrases)))
entities = list(doc.ents) # converts entities into a list
print("There were {} entities found".format(len(entities)))
orgs_and_people = [entity.orth_ for entity in entities if entity.label_ in ['ORG','PERSON']]
pd.DataFrame(orgs_and_people)
import nltk
import re
content = "Starbucks has not been doing well lately"
tokenized = nltk.word_tokenize(content)
tagged = nltk.pos_tag(tokenized)
print(tagged)
namedEnt = nltk.ne_chunk(tagged)
namedEnt.draw
IN = re.compile(r'.*\bin\b(?!\b.+ing)')
for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
    for rel in nltk.sem.relextract.extract_rels('ORG', 'LOC', doc,corpus='ieer', pattern = IN):
        print (nltk.sem.relextract.rtuple(rel))
import urllib.request
test_file = '../input/sentimental-analysis-nlp/test_data.csv'
train_file = '../input/sentimental-analysis-nlp/train_data.csv'
# test_data_f = urllib.request.urlretrieve(test_file)
# train_data_f = urllib.request.urlretrieve(train_file)
import pandas as pd

test_data_df = pd.read_csv(test_file, header=None, delimiter="\t", quoting=3)
test_data_df.columns = ["Text"]
train_data_df = pd.read_csv(train_file, header=None, delimiter="\t", quoting=3)
train_data_df.columns = ["Sentiment","Text"]
test_data_df.head()
train_data_df.head()
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer        
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = [stemmer.stem(item) for item in tokens]
    return(stemmed)
def tokenize(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return(stems)
vectorizer = CountVectorizer(
analyzer = 'word',
tokenizer = tokenize,
lowercase = True,
stop_words = 'english',
max_features = 85
)
features = vectorizer.fit_transform(
train_data_df.Text.tolist() + test_data_df.Text.tolist())
features_nd = features.toarray()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(features_nd[0:len(train_data_df)], train_data_df.Sentiment,train_size=0.85, random_state=1234)
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
y_pred
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
log_model = LogisticRegression()
log_model = log_model.fit(X=features_nd[0:len(train_data_df)], y=train_data_df.Sentiment)
test_pred = log_model.predict(features_nd[len(train_data_df):])
test_pred
import random
spl = random.sample(range(len(test_pred)), 10)
for text, sentiment in zip(test_data_df.Text[spl], test_pred[spl]):
    print(sentiment, text)