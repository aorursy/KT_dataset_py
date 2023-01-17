import pandas as pd

import re

import nltk

import string

#for list of puctuations

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.tokenize import RegexpTokenizer

#tokenizer

from nltk.stem import WordNetLemmatizer

#lemmatizer

from nltk.stem import PorterStemmer

#stemmer

import numpy



pd.options.display.max_colwidth = 400
#since the data was imported from web a lot of utf-8 characters are added

#these are not required by us so we keep only the ascii characters

def removeNonAscii(text):

    text = re.sub('\n' , ' ' , text)

    return "".join(i for i in text if ord(i)<128)



def remove_punctuation(text):

    no_punct="".join([c for c in text if c not in set(string.punctuation)-{'!'}])

    return no_punct







def remove_numbers(text):

    words = [w for w in text if w.isalpha() or w=='!']

    return words



def lower_case(text):

    words = [[w.lower() , w][w.isupper() and len(w)>1] for w in text]

    return words

#doesn't tamper with the words which are entirely in uppercase

#but if single letter words like 'a' occur at the beginning of the sentence

#this converts it to lower case for better grouping



lemmatizer = WordNetLemmatizer()

def word_lemmatizer(text):

    lem_text = [lemmatizer.lemmatize(i) for i in text]

    return lem_text



stemmer = PorterStemmer()

def word_stemmer(text):

    #stem_text = " ".join([stemmer.stem(i) for i in text])

    stem_text = [[stemmer.stem(i) , i][i.isupper()] for i in text]

    return stem_text



def stopword_remover(text):

    irrelevant = ['card','credit_card','thi_card','wa','bank','hi','get','thi','nan','also','got','ha']

    stop_words = set(stopwords.words('english'))

    no_stop = [i for i in text if not i in stop_words]

    result = [i for i in no_stop if not i in irrelevant]

    return result



def joiner(text):

    joined = " ".join(i for i in text)

    return joined
data=pd.read_csv("../input/reviews-before-cleaning/reviews.csv", index_col=0)





data['Reviews'] = data['Reviews'].apply(lambda x: remove_punctuation(x))

data['Reviews'] = data['Reviews'].apply(lambda x: removeNonAscii(x))



#tokenizer = RegexpTokenizer(r'\w+')

tokenizer = nltk.tokenize

data['Reviews'] = data['Reviews'].apply(lambda x: tokenizer.word_tokenize(x))



data['Reviews'] = data['Reviews'].apply(lambda x: remove_numbers(x))

data['Reviews'] = data['Reviews'].apply(lambda x: lower_case(x))

data['Reviews'] = data['Reviews'].apply(lambda x: word_lemmatizer(x))

data['Reviews'] = data['Reviews'].apply(lambda x: word_stemmer(x))



unlist_comments = [item for items in data['Reviews'] for item in items]

#print(data['Reviews'].head(15))
bigrams = nltk.collocations.BigramAssocMeasures()

trigrams = nltk.collocations.TrigramAssocMeasures()



bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(unlist_comments)

trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(unlist_comments)



bigram_freq = bigramFinder.ngram_fd.items()

bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram','freq']).sort_values(by='freq', ascending=False)

#print(bigramFreqTable.head().reset_index(drop=True))

#print(bigramFreqTable[:10])



en_stopwords = set(stopwords.words('english'))



#function to filter for ADJ/NN bigrams

def rightTypesBi(ngram):

    if '-pron-' in ngram or '' in ngram or ' 'in ngram or 't' in ngram:

        return False

    for word in ngram:

        if word in en_stopwords:

            return False

    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')

    second_type = ('NN', 'NNS', 'NNP', 'NNPS')

    tags = nltk.pos_tag(ngram)

    if tags[0][1] in acceptable_types and tags[1][1] in second_type:

        return True

    else:

        return False



#filter bigrams

filtered_bi = bigramFreqTable[bigramFreqTable.bigram.map(lambda x: rightTypesBi(x))]

#print(filtered_bi[:10])

bigr=list(filtered_bi[:20].bigram)



bigram_lookup = []

for element in bigr:

    bigram_lookup.append(list(element))



#print(bigram_lookup)



trigram_freq = trigramFinder.ngram_fd.items()

trigramFreqTable = pd.DataFrame(list(trigram_freq), columns=['trigram','freq']).sort_values(by='freq', ascending=False)

trigramFreqTable.head().reset_index(drop=True)



#print(trigramFreqTable[:10])



def rightTypesTri(ngram):

    if '-pron-' in ngram or '' in ngram or ' 'in ngram or '  ' in ngram or 't' in ngram:

        return False

    for word in ngram:

        if word in en_stopwords:

            return False

    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')

    third_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')

    tags = nltk.pos_tag(ngram)

    if tags[0][1] in first_type and tags[2][1] in third_type:

        return True

    else:

        return False



filtered_tri = trigramFreqTable[trigramFreqTable.trigram.map(lambda x: rightTypesTri(x))]

#print(filtered_tri[:10])

trigr=list(filtered_tri[:15].trigram)



trigram_lookup = []



for element in trigr:

    trigram_lookup.append(list(element))



#print(trigram_lookup)





def bigram_join(text):

    collocator=[]

    check=None

    for x, y in zip(text[0::], text[1::]):

        l=[x,y]

        if x==check:

            check=None

            continue

        if l in bigram_lookup:

            x = "_".join(l)

            check=y

        collocator.append(x)

    return collocator





def trigram_join(text):

    collocator=[]

    count=3

    for x, y, z in zip(text[0::], text[1::], text[2::]):

        l=[x,y,z]

        if count==2:

            count=1

            continue

        if count==1:

            count=3

            continue

        if l in trigram_lookup:

            x = "_".join(l)

            count=count-1

        collocator.append(x)

    return collocator
data['Reviews'] = data['Reviews'].apply(lambda x: trigram_join(x))

data['Reviews'] = data['Reviews'].apply(lambda x: bigram_join(x))



data['Reviews'] = data['Reviews'].apply(lambda x: stopword_remover(x))

data['Reviews'] = data['Reviews'].apply(lambda x: joiner(x))



#sample output file data

print(data['Reviews'].head(15))



data.to_csv("reviews_1.csv")