import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer







documentA = 'the man went out for a walk'

documentB = 'the children sat around the fire'

bagOfWordsA = documentA.split(' ')

bagOfWordsB = documentB.split(' ')

print("Split")

print(bagOfWordsA, bagOfWordsB,"\n")





uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))

print("Unique Set of Words")

print(uniqueWords,"\n")



numOfWordsA = dict.fromkeys(uniqueWords, 0)

for word in bagOfWordsA:

    numOfWordsA[word] += 1

print("Numbers of WordsA")

print(numOfWordsA)

numOfWordsB = dict.fromkeys(uniqueWords, 0)

for word in bagOfWordsB:

    numOfWordsB[word] += 1

print("Numbers of WordsB")

print(numOfWordsB,"\n")



from nltk.corpus import stopwords

stopwords.words('english')



def computeTF(wordDict, bagOfWords):

    tfDict = {}

    bagOfWordsCount = len(bagOfWords)

    for word, count in wordDict.items():

        tfDict[word] = count / float(bagOfWordsCount)

    return tfDict

tfA = computeTF(numOfWordsA, bagOfWordsA)

tfB = computeTF(numOfWordsB, bagOfWordsB)

print("Term Frequency for First Document")

print(tfA)

print("Term Frequency for Second Document")

print(tfB)





def computeIDF(documents):

    import math

    N = len(documents)



    idfDict = dict.fromkeys(documents[0].keys(), 0)

    for document in documents:

        for word, val in document.items():

            if val > 0:

                idfDict[word] += 1



    for word, val in idfDict.items():

        idfDict[word] = math.log(N / float(val))

    return idfDict

'''The IDF is computed once for all documents.'''

idfs = computeIDF([numOfWordsA, numOfWordsB])



def computeTFIDF(tfBagOfWords, idfs):

    tfidf = {}

    for word, val in tfBagOfWords.items():

        tfidf[word] = val * idfs[word]

    return tfidf

tfidfA = computeTFIDF(tfA, idfs)

tfidfB = computeTFIDF(tfB, idfs)

df = pd.DataFrame([tfidfA, tfidfB])





vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform([documentA, documentB])

feature_names = vectorizer.get_feature_names()

dense = vectors.todense()

denselist = dense.tolist()

df = pd.DataFrame(denselist, columns=feature_names)

print("\n",df)