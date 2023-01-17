import spacy
import nltk
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

nlp = spacy.load("en_core_web_lg")
plt.figure(figsize=(20,5))
plt.bar(range(300),nlp('dog').vector)
tokens = nlp("cherry pie tree digital information")

#for token in tokens:
#    print(token.vector)
for token1 in tokens:
    print("\t%10s" % (token1.text),end="")
print()

for token1 in tokens:
    print("%-10s" % (token1.text),end="")
    for token2 in tokens:
        print("\t%-10.3f" % token1.similarity(token2),end="")
    print()
def sim(v1,v2):
    return 1.0 - cosine(v1,v2)
dogv = nlp('dog').vector
catv = nlp('cat').vector

sim(dogv,catv)
nlp('dog').similarity(nlp('cat'))
# import the PCA module from sklearn
from sklearn.decomposition import PCA

# converts a list of words into their word vectors
def get_word_vectors(words):
    return [nlp(word).vector for word in words]

def plotpca(words):
    '''plot vector of words in 2D plot by using PCA projection'''
    # intialise pca model and tell it to project data down onto 2 dimensions
    pca = PCA(n_components=2)

    # fit the pca model to our 300D data, this will work out which is the best 
    # way to project the data down that will best maintain the rel
    pca.fit(get_word_vectors(words))

    # Used fitted data to project word vectors to 2D
    word_vecs_2d = pca.transform(get_word_vectors(words))
    # create a nice big plot 
    plt.figure(figsize=(10,10))

    # plot the scatter plot of where the words will be
    plt.scatter(word_vecs_2d[:,0], word_vecs_2d[:,1])

    # for each word and coordinate pair: draw the text on the plot
    for word, coord in zip(words, word_vecs_2d):
        x, y = coord
        plt.text(x, y, word, size= 15)

    # show the plot
    plt.show()
words = "brother sister uncle aunt man woman king queen boy girl".split(" ")
plotpca(words)
words = "big bigger biggest large larger largest strong stronger strongest quick quicker quickest".split(" ")
plotpca(words)
w1 = nlp("boy").vector
w2 = nlp("doctor").vector
w3 = nlp("girl").vector
w4 = w1 - w2 + w3
from scipy.spatial import distance
def closestWord(vec):
    '''Return word that is closest to a vector'''
    p = np.array([vec])
    # Format the vocabulary for use in the distance function
    ids = [x for x in nlp.vocab.vectors.keys()]
    vectors = [nlp.vocab.vectors[x] for x in ids]
    vectors = np.array(vectors)

    # Find the closest word below
    closest_index = distance.cdist(p, vectors, metric='cosine').argmin()
    word_id = ids[closest_index]
    output_word = nlp.vocab[word_id].text
    # output_word is identical, or very close, to the input word
    return output_word
n1 = "Molly Amy Claire Emily Katie Madeline Katelyn Emma Abigail Carly Jenna Heather Katherine Caitlin Kaitlin Holly Allison Kaitlyn Hannah Kathryn"
n2 = "Imani Ebony Shanice Aaliyah Precious Nia Deja Diamond Asia Aliyah Jada Tierra Tiara Kiara Jazmine Jasmin Jazmin Jasmine Alexus Raven"
n3 = "Connor Tanner Wyatt Cody Dustin Luke Jack Scott Logan Cole Lucas Bradley Jacob Garrett Dylan Maxwell Hunter Brett Colin"
n4 = "DeShawn DeAndre Marquis Darnell Terrell Malik Trevon Tyrone Willie Dominique Demetrius Reginald Jamal Maurice Jalen Darius Xavier Terrance Andre Darryl"


w1 = "happy love laughter pleasure"
w2 = "sad abuse hatred pain"

wlists = [n1,n2,n3,n4,w1,w2]
vecs = [nlp(w) for w in wlists]
for x in wlists:
    print(closestWord(nlp(x).vector))
for token1 in vecs:
    for token2 in vecs:
        print("\t%-10.3f" % token1.similarity(token2),end="")
    print()
    
   

