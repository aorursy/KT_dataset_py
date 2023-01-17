import nltk
nltk.corpus.gutenberg.fileids()
nltk.corpus.gutenberg.raw("shakespeare-hamlet.txt")
hamlet = nltk.corpus.gutenberg.raw('carroll-alice.txt')
print(hamlet[1000:1300])
nltk.corpus.machado.fileids()
macn001 = nltk.corpus.machado.raw("contos/macn001.txt")
print(macn001[1000:1300])
nltk.corpus.nps_chat.fileids()
nltk.corpus.nps_chat.posts("11-06-adults_706posts.xml")[500:700]
from nltk.corpus import brown
brown.categories()
brown.fileids()
ca01 = brown.raw('ca01')
ca01[1000:1300]
from nltk import pos_tag

words = "He will race the car. When will the race start?".split(" ")

#words = "He will race the car when will the race start".split(" ")

pos_tags = pos_tag(words)

print(pos_tags)
!python -m spacy download pt_core_news_sm
import spacy

#from spacy.lang.pt import Portuguese

import pt_core_news_sm

nlp = pt_core_news_sm.load()
#nlp = spacy.load('pt_core_news_sm')
tokens = nlp("Eles não vão trabalhar hoje para estudar para a prova. Espero que não seja em vão")
for t in tokens:
    print(t.text, t.pos_)
from nltk.corpus import wordnet as wn

wn.synsets('dog')

wn.synsets('dog', pos=wn.NOUN)
wn.synset('dog.n.01').definition()
wn.synset('frump.n.01').examples()
wn.synset('frump.n.01').definition()
wn.synset('chase.v.01').examples()
wn.synset('dog.n.01').lemma_names('por')
wn.synsets('cão', lang='por')
print("Hypernyms:")
print(wn.synset('dog.n.01').hypernyms())

print("Hyponyms:")
print(wn.synset('dog.n.01').hyponyms())

print("Holonyms")
print(wn.synset('dog.n.01').member_holonyms())
print("Hypernyms:")
print(wn.synset('canine.n.02').lemma_names())

print("Hyponyms:")
print(wn.synset('hunting_dog.n.01').lemma_names())
print(wn.synset('poodle.n.01').lemma_names())

print("Holonyms")
print(wn.synset('canis.n.01').lemma_names())
print(wn.synset('pack.n.06').lemma_names())

dog = wn.synset('dog.n.01')
cat = wn.synset('cat.n.01')
fork = wn.synset('fork.n.01')
print(dog.path_similarity(cat))
print(dog.path_similarity(fork))

print(dog.path_similarity(dog))

dog3 = wn.synset('dog.n.03')
print(dog.path_similarity(dog3))

print(dog3.lemma_names())
print(dog3.definition())

