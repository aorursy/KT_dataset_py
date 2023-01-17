import nltk

from nltk.book import *

from nltk.corpus import gutenberg

from nltk.corpus import brown

from nltk.corpus import webtext

from nltk.corpus import state_union

from nltk.corpus import wordnet as wn

from nltk.corpus import stopwords

from nltk.corpus import udhr

import random

import matplotlib.pyplot as plt
phrase = ['hello','natural','language','processing']

phrase.append('python')

print(phrase + phrase)

print(phrase * 5)

print(phrase[0])

print(phrase[1:])

print(sorted(phrase))
# howmany word tokens and howmany word types

persuasion = gutenberg.words('austen-persuasion.txt')

print('word_tokens:',len([word for word in persuasion if word.isalpha()]))

print('word_types:',len(set([word.lower() for word in persuasion if word.isalpha()])))
print(nltk.corpus.brown.categories()[5:7])

print(nltk.corpus.webtext.fileids()[3])

print(nltk.corpus.brown.words(categories = 'hobbies'))

print(nltk.corpus.brown.words(categories = 'humor'))

print(nltk.corpus.webtext.words(fileids = 'pirates.txt'))
cfd = nltk.ConditionalFreqDist(((target,fileid[:4]) for fileid in state_union.fileids()

                     for words in state_union.words(fileid)

                     for target in ['men','women','people']

                     if words.lower().startswith(target)

 ))



cfd.plot()
nltk.Text(persuasion).concordance('However')
names = nltk.corpus.names

cfd = nltk.ConditionalFreqDist((fileid,name[0])

                               for fileid in names.fileids()

                               for name in names.words(fileid)

                              

                              

                              )

cfd.plot()
news_text = nltk.corpus.brown.words(categories = 'news')

romance_text = nltk.corpus.brown.words(categories = 'romance')

print('Vocabulary of news',len(set(news_text)))

print('Vocabulary of romance',len(set(romance_text)))

print('Vocabulary richness of news',len(set(news_text))/len(news_text))

print('Vocabulary richness of romance',len(set(romance_text))/len(romance_text))

print('Similarity of address in both categories')

print(nltk.Text(news_text).similar('address'))

print(nltk.Text(romance_text).similar('address'))
modals = ['can','could','may','might','must','will']



cfd = nltk.ConditionalFreqDist(((genre,words.lower())

 for genre in brown.categories()

 for words in brown.words(categories = genre)

))
genres = ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government','hobbies']

cfd.tabulate(conditions = ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government','hobbies'],samples = modals)
prondict = nltk.corpus.cmudict.dict() #we use dict to count distinct keywords

print('Distinct Words in Prondict are',len(prondict))

wordPron = 0

for key in prondict:

    if len(prondict[key]) > 1:

        wordPron += 1

print('The Fraction is',wordPron/len(prondict))

        
#what percent of noun_synsets have no hyponyms

# wn.all_synsets('n')

length = len(list(wn.all_synsets('n')))

cnt = 0

for synset in wn.all_synsets('n'):

    if(synset.hyponyms() == []):

        cnt += 1

print(cnt/length)
s = wn.synsets('screen')[0]



defis = ' '

defis = defis + s.name() + ' ' + s.definition() + '\n'

for synset in s.hypernyms():

    defis = defis + synset.name() + ' ' + synset.definition() + '\n'

for synset in s.hyponyms():

    defis = defis + synset.name() + ' ' + synset.definition() + '\n'

print(defis)
s = nltk.corpus.wordnet.synsets('computer')[0]

defis = ' '

defis = defis + s.name() + ' ' + s.definition() + '\n\n'

for synset in s.hypernyms():

    defis = defis + synset.name() + ' ' + synset.definition() + '\n\n'

for synset in s.hyponyms():

    defis = defis + synset.name() + ' ' + synset.definition() + '\n\n'

print(defis)
def supergloss(s):

    defis = ' '

    defis = defis +  s.name() + 'definition:' + s.definition() + '\n\n'

    for synset in s.hypernyms():

        defis = defis + synset.name() + 'definition:' + synset.definition() + '\n\n'

    for synset in s.hyponyms():

        defis = defis + synset.name() + 'definition:' + synset.definition() + '\n\n'

    return defis
sets = nltk.corpus.wordnet.synsets('eating')

for s in sets:

    print(supergloss(s))
brown_c = nltk.corpus.brown.words()

freq_dist = nltk.FreqDist(w.lower() for w in brown_c if w.isalpha())

wordSet = []

for key in freq_dist:

    if freq_dist[key] >= 3:

        wordSet.append(key)
categories = nltk.corpus.brown.categories()

score_arr = []

cat_arr = []

for category in categories:

    token = len(nltk.corpus.brown.words(categories = category))

    types = len(set(brown.words(categories = category)))

    score = types/token

    cat_arr.append(category)

    score_arr.append(score)   
import pandas as pd

df = pd.DataFrame({'category':cat_arr,'score':score_arr})
df[df['score'] == df['score'].min()]
# 50 most frequently occuring words in a text that are not stopwords

text = nltk.corpus.nps_chat.words()

from nltk.corpus import stopwords

stopwords = stopwords.words('english')

fdist = nltk.FreqDist(words.lower() for words in text if words not in stopwords)

fdist.most_common(50)
def find_50_most_common(text):

    fdist = nltk.FreqDist(words.lower() for words in text if words not in nltk.corpus.stopwords.words('english'))

    return fdist.most_common(50)
print(find_50_most_common(nltk.corpus.nps_chat.words()))
text = nltk.corpus.gutenberg.words('chesterton-ball.txt')

stopwords = nltk.corpus.stopwords.words('english')

bigram = list(nltk.bigrams(text))

bigram = list(nltk.bigrams(text))

fd = nltk.FreqDist(b for b in bigram if b[0].isalpha() and b[0] not in stopwords and b[1] not in stopwords)
def find_50_most_common_bigrams(text):

    stopwords = nltk.corpus.stopwords.words('english')

    bigram = list(nltk.bigrams(text))

    feq_dist = nltk.FreqDist(b for b in bigram if b[0].isalpha() and b[0] not in stopwords and b[1] not in stopwords)

    return feq_dist.most_common(50)

    
print(find_50_most_common_bigrams(nltk.corpus.gutenberg.words('chesterton-ball.txt')))
cfd = nltk.ConditionalFreqDist(

    (genre,words)

    for genre in brown.categories()

    for words in brown.words(categories = genre)

)

samples = ['love','like','peace','hate','war','fight','battle']

genres = brown.categories()

cfd.tabulate(comditions = genres,samples = samples)
text = nltk.corpus.gutenberg.words('chesterton-ball.txt')

prondict = nltk.corpus.cmudict.dict()

number = 0

for word in text:

    if word.lower() in prondict.keys():

        number += len(prondict[word.lower()][0])

print(number)
text = nltk.corpus.gutenberg.words()

prondict = nltk.corpus.cmudict.dict()

number = 0

for word in text:

    if word in prondict.keys():

        number += len(prondict[word][0])

print(number)
def number_of_syllables(text):

    prondict = nltk.corpus.cmudict.dict()

    number = 0

    for words in text:

        if words in prondict.keys():

            number += len(prondict[words][0])

    return number
print(number_of_syllables(nltk.corpus.gutenberg.words()))
text = nltk.corpus.gutenberg.words('chesterton-ball.txt')

new_version = list(text)

for i in range(3,len(text)+len(text)//3,3):

    new_version.insert(i,'like')

nltk.Text(new_version)
def hedge(text):

    new_version  = list(text)

    for i in range(2,len(text)+len(text)//3,3):

        new_version.insert(i,'like')

    return nltk.Text(new_version)
text = nltk.corpus.gutenberg.words('chesterton-ball.txt')

fdist = nltk.FreqDist((words.lower() for words in text if words.isalpha()))

fdist = fdist.most_common()

rank = []

freq = []

n = 1

for i in range(len(fdist)):

    freq.append(fdist[i][1])

    

    rank.append(n)

    n += 1

    

plt.plot(rank,freq)

plt.xscale('log')

# zips law suggests that in a corpora when the words are arranged in decreasing order of their frequencies the product of their rank and freq

#uency remains a constant or rank and frequency have a inverse relation



def zip_law(text):

    fdist = nltk.FreqDist((word.lower() for word in text if word.isalpha()))

    fdist = fdist.most_common()

    freq = []

    rank = []

    n = 1

    for i in range(len(fdist)):

        freq.append(fdist[i][1])



        rank.append(n)

        n += 1



    plt.plot(rank,freq)

    plt.xscale('log')

    
zip_law(nltk.corpus.brown.words())
import random

randomText = ' '

for i in range(100000):

    randomText = randomText + random.choice('abcdefg ')
randomText.split()
zip_law(randomText.split())