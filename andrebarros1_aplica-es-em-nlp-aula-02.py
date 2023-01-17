import nltk



nltk.download("gutenberg")
hamlet_raw = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')

print(hamlet_raw[:1000])
from nltk.tokenize import sent_tokenize



sentences = sent_tokenize(hamlet_raw)



print(sentences[:10])

from nltk.tokenize import word_tokenize



words = word_tokenize(sentences[0])



print(words)
from nltk.corpus import stopwords



stopwords_list = stopwords.words('english')



print(stopwords_list)
non_stopwords = [w for w in words if not w.lower() in stopwords_list]

print(non_stopwords)
import string

punctuation = string.punctuation

print(punctuation)
non_punctuation = [w for w in non_stopwords if not w in punctuation]



print(non_punctuation)
from nltk import pos_tag



pos_tags = pos_tag(words)



print(pos_tags)
from nltk.stem.snowball import SnowballStemmer



stemmer = SnowballStemmer('english')



sample_sentence = "He has already gone"

sample_words = word_tokenize(sample_sentence)



stems = [stemmer.stem(w) for w in sample_words]



print(stems)
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet



lemmatizer = WordNetLemmatizer()



pos_tags = nltk.pos_tag(sample_words)



lemmas = []

for w in pos_tags:

    if w[1].startswith('J'):

        pos_tag = wordnet.ADJ

    elif w[1].startswith('V'):

        pos_tag = wordnet.VERB

    elif w[1].startswith('N'):

        pos_tag = wordnet.NOUN

    elif w[1].startswith('R'):

        pos_tag = wordnet.ADV

    else:

        pos_tag = wordnet.NOUN

        

    lemmas.append(lemmatizer.lemmatize(w[0], pos_tag))

    

print(lemmas)
from nltk import word_tokenize



frase = 'o cachorro correu atrás do gato'





ngrams = ["%s %s %s" % (nltk.word_tokenize(frase)[i], \

                      nltk.word_tokenize(frase)[i+1], \

                      nltk.word_tokenize(frase)[i+2]) \

          for i in range(len(nltk.word_tokenize(frase))-2)]



print(ngrams)

non_punctuation = [w for w in words if not w.lower() in punctuation]



n_grams_3 = ["%s %s %s"%(non_punctuation[i], non_punctuation[i+1], non_punctuation[i+2]) for i in range(0, len(non_punctuation)-2)]



print(n_grams_3)
from sklearn.feature_extraction.text import CountVectorizer



count_vect = CountVectorizer(ngram_range=(3,3))



import numpy as np



arr = np.array([sentences[0]])



print(arr)



n_gram_counts = count_vect.fit_transform(arr)



print(n_gram_counts.toarray())



print(count_vect.vocabulary_)
arr = np.array(sentences)



n_gram_counts = count_vect.fit_transform(arr)



print(n_gram_counts.toarray()[:20])



print([k for k in count_vect.vocabulary_.keys()][:20])