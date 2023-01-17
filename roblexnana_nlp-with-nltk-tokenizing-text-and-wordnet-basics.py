# we can start by creating a paragraph of text:

para = "Hello World. It's good to see you. Thanks for buying this book."



from nltk.tokenize import sent_tokenize

sent_tokenize(para)
import nltk.data

tokenizer = nltk.data.load('../tokenizers/punkt/english.pickle')

tokenizer.tokenize(para)
spanish_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')

spanish_tokenizer.tokenize('Hola amigo. Estoy bien.')
from nltk.tokenize import word_tokenize

word_tokenize('Hello World.')
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

tokenizer.tokenize('Hello World.')
from nltk.tokenize import WordPunctTokenizer

tokenizer = WordPunctTokenizer()

tokenizer.tokenize("Can't is a contraction.")
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer("[\w']+")

tokenizer.tokenize("Can't is a contraction.")
from nltk.tokenize import regexp_tokenize

regexp_tokenize("Can't is a contraction.", "[\w']+")
tokenizer = RegexpTokenizer('\s+', gaps=True)

tokenizer.tokenize("Can't is a contraction.")
from nltk.tokenize import PunktSentenceTokenizer

from nltk.corpus import webtext

text = webtext.raw('overheard.txt')

sent_tokenizer = PunktSentenceTokenizer(text)
sents1 = sent_tokenizer.tokenize(text)

sents1[0]
from nltk.tokenize import sent_tokenize

sents2 = sent_tokenize(text)

sents2[0]
sents1[678]
sents2[678]
from nltk.corpus import stopwords

english_stops = set(stopwords.words('english'))

words = ["Can't", 'is', 'a', 'contraction']

[word for word in words if word not in english_stops]
from nltk.corpus import wordnet

syn = wordnet.synsets('cookbook')[0]

syn.name()
syn.definition()
from nltk.corpus import wordnet

syn = wordnet.synsets('cookbook')[0]

lemmas = syn.lemmas()

len(lemmas)
lemmas[0].name()
lemmas[1].name()
lemmas[0].synset() == lemmas[1].synset()
[lemma.name() for lemma in syn.lemmas()]
synonyms = []

for syn in wordnet.synsets('book'):

    for lemma in syn.lemmas():

        synonyms.append(lemma.name())

len(synonyms)
len(set(synonyms))
gn2 = wordnet.synset('good.n.02')

gn2.definition()
evil = gn2.lemmas()[0].antonyms()[0]

evil.name
evil.synset().definition()
ga1 = wordnet.synset('good.a.01')

ga1.definition()
bad = ga1.lemmas()[0].antonyms()[0]

bad.name()
bad.synset().definition()
from nltk.corpus import wordnet

cb = wordnet.synset('cookbook.n.01')

ib = wordnet.synset('instruction_book.n.01')

cb.wup_similarity(ib)
ref = cb.hypernyms()[0]

cb.shortest_path_distance(ref)
ib.shortest_path_distance(ref)
cb.shortest_path_distance(ib)
from nltk.corpus import webtext

from nltk.collocations import BigramCollocationFinder

from nltk.metrics import BigramAssocMeasures

words = [w.lower() for w in webtext.words('grail.txt')]

bcf = BigramCollocationFinder.from_words(words)

bcf.nbest(BigramAssocMeasures.likelihood_ratio, 4)
from nltk.corpus import stopwords

stopset = set(stopwords.words('english'))

filter_stops = lambda w: len(w) < 3 or w in stopset

bcf.apply_word_filter(filter_stops)

bcf.nbest(BigramAssocMeasures.likelihood_ratio, 4)
from nltk.collocations import TrigramCollocationFinder

from nltk.metrics import TrigramAssocMeasures

words = [w.lower() for w in webtext.words('singles.txt')]

tcf = TrigramCollocationFinder.from_words(words)

tcf.apply_word_filter(filter_stops)

tcf.apply_freq_filter(3)

tcf.nbest(TrigramAssocMeasures.likelihood_ratio, 4)