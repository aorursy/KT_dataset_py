import nltk

from bs4 import BeautifulSoup

import urllib.request 



import pandas as pd
response = urllib.request.urlopen('http://php.net/')

html = response.read()



soup = BeautifulSoup(html, "html5lib")

text = soup.get_text(strip=True)
tokens = [t for t in text.split()]
freq = nltk.FreqDist(tokens)
freq.plot(20, cumulative=False)
from nltk.corpus import stopwords
clean_tokens = tokens[:]

sr = stopwords.words('english')



for token in tokens:

    if token in sr:

        clean_tokens.remove(token)
freq_clean = nltk.FreqDist(clean_tokens)

freq_clean.plot(20, cumulative=False)
from nltk.tokenize import sent_tokenize
mytext = "Hello Mr. Adam, how are you? I hope everything is going well.  Today is a good day, see"

mytest_sent = sent_tokenize(mytext)



print(mytest_sent)
from nltk.tokenize import word_tokenize
mytext_word = word_tokenize(mytext)

print(mytext_word)
mytext = "Bonjour M. Adam, comment allez-vous? J'espère que tout va bie"

my_text_french = sent_tokenize(mytext, "french")



print(my_text_french)
from nltk.corpus import wordnet
syn = wordnet.synsets("pain")



print(syn[0].definition())

print(syn[0].examples())
syn = wordnet.synsets('NLP')

print(syn[0].definition())



syn = wordnet.synsets('Python')

print(syn[0].definition())
synonyms = []



for syn in wordnet.synsets('computer'):

    for lemma in syn.lemmas():

        synonyms.append(lemma.name())

    
print(synonyms)
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

stemmed_word = stemmer.stem('playing')
print(stemmed_word)
from nltk.stem import WordNetLemmatizer
lemmmatizer = WordNetLemmatizer()

word_lemma = lemmmatizer.lemmatize('playing')
print(word_lemma)
word_lemma_verb = lemmmatizer.lemmatize('playing', pos="v")
print(word_lemma_verb)
word_lemma_verb = lemmmatizer.lemmatize('playing', pos="v")

word_lemma_noun = lemmmatizer.lemmatize('playing', pos="n")

word_lemma_a = lemmmatizer.lemmatize('playing', pos="a")

word_lemma_r = lemmmatizer.lemmatize('playing', pos="r")
print(word_lemma_verb)

print(word_lemma_noun)

print(word_lemma_a)

#print(word_lemma_r)
print(stemmer.stem('calculates'))

print(lemmmatizer.lemmatize('calculates'))



print(stemmer.stem('purple'))

print(lemmmatizer.lemmatize('purple'))
from nltk import pos_tag

from nltk import RegexpParser
text = "learn NLP from kaggle kernel and make implementation easy".split()

print("after split: ",text)
tokens_tag = pos_tag(text)

print("after token: ",tokens_tag)
patterns = """mychunk:{<NN.?>*<VBD.?>*<JJ.?>*<CC>?}"""

chunker = RegexpParser(patterns)
output = chunker.parse(tokens_tag)



print("After Chunking: ",output)
text = "Temperature of NewYork"

tokens = nltk.word_tokenize(text)

tag = nltk.pos_tag(tokens)



grammar = "NP: {<DT>?<JJ>*<NN>}"

cp = nltk.RegexpParser(grammar)

result = cp.parse(tag)
print(result)
!pip install svgling

import svgling
svgling.draw_tree(result)
from nltk import load_parser

cp = load_parser('grammars/book_grammars/sql0.fcfg')
query = 'What cities are located in China'



# parsing the above statement to meaningful format(parse a query into SQL)

trees = list(cp.parse(query.split()))



answer = trees[0].label()['SEM']

answer = [s for s in answer  if s]

q = ' '.join(answer)

print(q)
from nltk.sem import chat80
rows = chat80.sql_query('corpora/city_database/city.db',q)

for r in rows:

    print(r[0], end=" ")
read_dexpr = nltk.sem.DrtExpression.fromstring
drs1 = read_dexpr('([x,y], [angus(x), dog(y), own(x,y)])')

drs2 = read_dexpr('([u,z] ,[PRO(u), irene(z), bite(u,z)])')
drs_final = drs1 + drs2

print(drs_final.simplify().resolve_anaphora())
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()



data = ['Kaggle is best place to learn data science. Kaggle also provide oppourtunity to compete.']



vocabulary = vectorizer.fit(data)

x = vectorizer.transform(data)



print(x.toarray())

print(vocabulary.get_feature_names())
import gensim

from nltk.corpus import abc
model = gensim.models.Word2Vec(abc.sents())



X = list(model.wv.vocab)

data = model.most_similar('computer')



print(data)
# importing all necessary modules

from nltk.tokenize import sent_tokenize, word_tokenize 

import warnings 



warnings.filterwarnings(action = 'ignore')
sample = open('../input/test-text/test_text/alice.txt')

s = sample.read()



#replace escape character with space

f = s.replace("\n", " ")
data = []



#iterate through each sentence in the file

for i in sent_tokenize(f):

    temp = []

    

    #tokenize the sentence into words

    for j in word_tokenize(i):

        temp.append(j.lower())

    data.append(temp)
#Create CBOW model

model1 = gensim.models.Word2Vec(data, min_count=1, size=100, window=5)

model1['alice']
#create the Skip-Gram model

model2 = gensim.models.Word2Vec(data, min_count=1, size=100, window=5, sg=1)

model2['alice']
#similarity between words using CBOW

print("CBOW Cosine similarity between alica and wonderland {}".format(model1.similarity('alice','wonderland')))

print("CBOW Cosine similarity between pool and tears {}".format(model1.similarity('pool','tears')))



print('\n')



#similarity between words using skip-gram

print("SKIPGRAM Cosine similarity between alica and wonderland {}".format(model2.similarity('alice','wonderland')))

print("SKIPGRAM Cosine similarity between pool and tears {}".format(model2.similarity('pool','tears')))
model1.similar_by_word('caterpillar')
from sklearn.manifold import TSNE



def tsne_plot(model):

    labels = []

    tokens = []

    

    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

        

    

    tsne_model = TSNE(perplexity=40, n_components=3, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)

    

    x = []

    y = []

    z = []

    

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        z.append(value[2])

    

    return x,y,z,labels  
x,y,z,labels = tsne_plot(model2)
#make dataframe of x,y,z,words

import pandas as pd



df = pd.DataFrame()



df['pos1'] = x

df['pos2'] = y

df['pos3'] = z

df['pos_avg'] = (df['pos1']+df['pos2']+df['pos3'])/3

df['word'] = labels
import plotly.express as px





#requires huge memory, so may not load if more words visualize in 3d scatter plot

# 2d scatter plot is able to visualize whole words

fig = px.scatter_3d(df[:500], x="pos1",y="pos2", z= "pos3", text="word",color='pos_avg')



fig.update_traces(

    marker_coloraxis = None,

)



fig.update_layout(

    title_text = 'Visualization of Document',

    height=1000

)



fig.show()