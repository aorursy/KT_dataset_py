#Assignment #4 from https://machinelearningmastery.com/crash-course-deep-learning-natural-language-processing/



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy# linear algebra

import pandas# data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import nltk

from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec

from sklearn.decomposition import PCA

from matplotlib import pyplot



filename = '../input/genesis/genesis/english-kjv.txt'



open_file = open(filename, 'rt')



text = open_file.read()



open_file.close()



from nltk.tokenize import word_tokenize



#print(text.splitlines())



#This makes a list of sentences, then tokenises the words in a list

tokens = [word_tokenize(a) for a in text.splitlines()]

#thus creating a list of sentences, each of which is a list of "words"



#NEED TO NEST THIS, DUMMY

#Okay, I nested it - the code may be slightly hard to parse, but for each sentence

#we look at each word, and if the first character is a letter we keep it,

#then reconstruct the all_tokens as a list of sentences,

#each of which is a list of "actual" word

all_tokens = [[b for b in a if b[0].isalpha()] for a in tokens]





#tokens = [a for a in tokens if a[0].isalpha() and len(a) > 5]



#print(all_tokens)



model = Word2Vec(all_tokens, min_count=1)

# fit a 2D PCA model to the vectors

X = model[model.wv.vocab]

pca = PCA(n_components=4)

result = pca.fit_transform(X)

#print(result)

# create a scatter plot of the projection





words = list(model.wv.vocab)

print(len(words))

TO_LIST=25#Otherwise, plot is too busy



pyplot.figure(figsize=(8,8))

pyplot.scatter(result[:TO_LIST, 0], result[:TO_LIST, 1])





print(words[:TO_LIST])

for i, word in enumerate(words):

    if i < TO_LIST: pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))



pyplot.show()

#print(words)