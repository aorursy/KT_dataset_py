# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

my_sent = "Machine Learning Machine Learning (ML) is defined as the use algorithms and computational statistics to learn from data without being explicitly programmed. It is a subsection of the artificial intelligence domain within computer science. While the field of machine learning did not explode until more recently, the term was first coined in 1959 and the most foundational research was done throughout the 70’s and 80's. Machine learning’s rise to prominence today has been enabled by the abundance of data, more efficient data storage, and faster computers.Depending on what you are trying to accomplish, there are many different ways to get a computer to learn from data. These various ways can be categorized into three main subsections of machine learning: supervised learning, unsupervised learning and reinforcement learning.Supervised Learning A supervised learning algorithm takes labeled data and creates a model that can make predictions given new data.These can be either a classification problem or a regression problem. In a classification problem, there might be test data consisting of photos of animals, each one labeled with its corresponding name. The model would be trained on this test data and then the model would be used to classify unlabeled animal photos with the correct name. In a regression problem, there is a relationship trying to be determined among many different variables. Usually, this takes place in the form of historical data being used to predict future quantities. An example of this would be predicting the future price of a stock based on past prices movements.Unsupervised Learning Unsupervised learning is when we are dealing with data that has not been labeled or categorized. The goal is to find patterns and create structure in data in order to derive meaning.Two forms of unsupervised learning are clustering and dimensionality reduction. Clustering is grouping alike data. Data in one group should have similar properties or features to one another. However, when compared to data of another group they should have highly dissimilar properties. Dimensionality reduction is the compressing of data by removing random variables and holding onto principle ones without losing the structure and meaningfulness of the dataset. The usefulness of dimensionality reduction comes from making the data easier to store, quicker to run computations over, and easier to view in data visualizations. It is also helpful in removing false signals or noise from a model which helps improve its performance. Reinforcement Learning Reinforcement learning uses a reward system and trial-and-error in order to maximize the long-term reward. Take, for example, a game of Pac-man. Pac would explore the maze eating up dots. With each dot, it receives a point/reward. As it navigates its environment it evaluates the probability of a reward in each state. Pac would learn to not double back over previously charted sections of the maze because the dots have already been consumed and there would be no immediate reward. Great! But what if a fruit worth 25 points appeared in a previously chartered section? It would no longer be maximizing the potential reward. As you can see, there is an exploration and exploitation tradeoff. In order to handle this, reinforcement learning algorithms integrate a level of randomness called an epsilon-greedy strategy. Epsilon is the percentage of states where the agent, Pac, would take a random route and knowingly miss out on a reward. Generally, reinforcement learning algorithms will begin more explorative and as the reward systems of the game are better understood, the algorithm will then lean towards exploitation.In the above reinforcement problem, the act of reevaluating the probability in each state is known as a Markov Decision Process (MDP). Nearly, all reinforcement problems are classified as such. Deep Learning Amazingly enough, the area of machine learning has seen the most significant results has done so by mimicking the human brain. Deep learning utilizes neural networks which, just like the human brain, contain interconnected neurons that can be activated or deactivated. Deep learning can fall into supervised and unsupervised learning subsections of ML. How it works: Input or multiple inputs are passed into a neural network which then processes them into one or many outputs. The neural network itself is a network of neurons grouped into layers. Deep learning gets its name from the depth or number of layers. Matrix operations are performed on the data layer by layer. Multiply the input by a weight, add a bias, and then apply an activation function to the result. Pass it along to the next neuron and repeat until it gets to the end. With each subsequent pass through the entire network, a cost function adjusts the weighted connections in order to reduce error and improve the model. This continuous error reduction is also known as gradient descent. A simple neural network and gradient descent Neural networks have achieved significant results in image recognition, speech recognition, natural language processing which have endless commercial applications. Some Fascinating Research. Using Instagram photos, machine learning researchers were able to determine whether or not users suffered from depression with greater accuracy than an in-person diagnosis from a general practitioner. You can read more about it here. OpenAI has been teaching a robot how to manipulate physical objects with human-like dexterity. Check it out here. In another OpenAI study, researchers gave two AI agents a small choice of words as well as some communication goals. The agents then began to develop their own language. Check it out here Applications of Machine Learning. The applications for machine learning are endless. No profession or industry will be left unaltered. The most common machine learning that the average consumers will interface with are recommender systems. Whether its Amazon recommending a product, Netflix recommending your next binge series, or Facebook showing you another dog video. Each one of these recommendations is personalized for you based on your personal data. In the healthcare space, machine learning already has numerous applications for medical imaging and diagnosis and even drug discovery. Algorithms can learn to recognize the most intricate of details from x-rays, brain scans, and photos of skin that even the most seasoned doctors could miss. Drug discovery is amazingly inefficient. It is estimated, that on average it takes 10 years and $2.6 Billion to get a major drug from its initial discovery to the marketplace. Machine learning can help reduce those atrocious numbers by means of bioactivity prediction, molecular design, and synthesis prediction. In finance, machine learning will increase access to loans and bring portfolio management services to everyone, not just high-net-worth individuals. In the United States alone, car crashes cost the economy over $500 billion each year and countless lives. Autonomous vehicles that rely on machine learning, will help tackle these issues. Additionally, autonomous vehicles will reduce the cost of living for many. Not only will there be a reduction in the cost of transportation, but many will choose to live outside city centers, in suburbs where real estate prices are cheaper.The Future of Machine Learning. Machine learning is unpredictable and yet it will have an immense impact on every aspect of the professional and personal lives of all humans. Everyone from financiers, doctors, to customer service reps and salespeople can expect to be retooled and more productive than ever. If you are tech professional you can expect machine learning to be integrated into every layer of the software engineering stack. Ultimately, machine learning is a means to get to true artificial intelligence. Today, all artificial intelligence programs are deemed artificial narrow intelligence (ANI). These are limited in scope, single application programs. Artificial general intelligence (AGI) are programs that can be applied to a diverse range of problem sets, much like in the way humans are able to solve varying problems with limited knowledge. AGI would ideally be able to reprogram itself. When artificial intelligence surpasses human-level intelligence, it reaches the last stage artificial superintelligence (ASI). Nick Bostrom, author of Superintelligence, define ASI as any intellect that greatly exceeds the cognitive performance of humans in virtually all domains of interest. Lastly, a quote from Max Tegmark, the president Future of Life Institute a non-profit research organization that aims to mitigate the risks of existential threats to humanity. Everything we love about civilization is a product of intelligence, so amplifying our human intelligence with artificial intelligence has the potential of helping civilization flourish like never before — as long as we manage to keep the technology beneficial."

from nltk import sent_tokenize, word_tokenize
tokens = word_tokenize(my_sent)
print(tokens)
print(len(tokens))
sent_tokenize(my_sent)
for sent in sent_tokenize(my_sent):
    print(word_tokenize(sent))
for sent in sent_tokenize(my_sent):
    # It's a little in efficient to loop through each word,
    # after but sometimes it helps to get better tokens.
    print([word.lower() for word in word_tokenize(sent)])
    # Alternatively:
    #print(list(map(str.lower, word_tokenize(sent))))
my_sent_tokenized_lowered = list(map(str.lower, word_tokenize(my_sent)))
print(my_sent_tokenized_lowered)
stopwords_en = set(stopwords.words('english')) # Set checking is faster in Python than list.

# List comprehension.
print([word for word in my_sent_tokenized_lowered if word not in stopwords_en])
from string import punctuation
# It's a string so we have to them into a set type
print('From string.punctuation:', type(punctuation), punctuation)
stopwords_en_withpunct = stopwords_en.union(set(punctuation))

stopwords_en_withpunct.add('’')
stopwords_en_withpunct.add('0')
stopwords_en_withpunct.add('1')
stopwords_en_withpunct.add('2')
stopwords_en_withpunct.add('3')
stopwords_en_withpunct.add('4')
stopwords_en_withpunct.add('5')
stopwords_en_withpunct.add('6')
stopwords_en_withpunct.add('7')
stopwords_en_withpunct.add('8')
stopwords_en_withpunct.add('9')
print(stopwords_en_withpunct)   
my_sent_tokenized_lowered= [word for word in my_sent_tokenized_lowered if word not in stopwords_en_withpunct]
print(my_sent_tokenized_lowered)
from nltk.stem import PorterStemmer
porter = PorterStemmer()

for word in ['walking', 'walks', 'walked']:
    print(porter.stem(word))
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

for word in ['walking', 'walks', 'walked']:
    print(wnl.lemmatize(word))
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
lem=[]
for word in my_sent_tokenized_lowered:
    lem.append(wnl.lemmatize(word))
    print(wnl.lemmatize(word))
print(lem)
nltk.pos_tag(lem)
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
from nltk import word_tokenize
from nltk.collocations import BigramCollocationFinder
text = my_sent
finder = BigramCollocationFinder.from_words(word_tokenize(text))
finder.nbest(bigram_measures.pmi, 5)
import nltk
bigrams = nltk.collocations.BigramAssocMeasures()
trigrams = nltk.collocations.TrigramAssocMeasures()
bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(tokens)
import nltk
bigrams = nltk.collocations.BigramAssocMeasures()
trigrams = nltk.collocations.TrigramAssocMeasures()
bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(lem)
trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(lem)
#bigrams
bigram_freq = bigramFinder.ngram_fd.items()
bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram','freq']).sort_values(by='freq', ascending=False)
#trigrams
trigram_freq = trigramFinder.ngram_fd.items()
trigramFreqTable = pd.DataFrame(list(trigram_freq), columns=['trigram','freq']).sort_values(by='freq', ascending=False)
#get english stopwords
#en_stopwords = set(stopwords.words('english'))
en_stopwords =stopwords_en_withpunct
#function to filter for ADJ/NN bigrams
def rightTypes(ngram):
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords or word.isspace():
            return False
    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_type = ('NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in acceptable_types and tags[1][1] in second_type:
        return True
    else:
        return False
#filter bigrams
filtered_bi = bigramFreqTable[bigramFreqTable.bigram.map(lambda x: rightTypes(x))]
#function to filter for trigrams
def rightTypesTri(ngram):
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords or word.isspace():
            return False
    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    third_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in first_type and tags[2][1] in third_type:
        return True
    else:
        return False
#filter trigrams
filtered_tri = trigramFreqTable[trigramFreqTable.trigram.map(lambda x: rightTypesTri(x))]
print(filtered_tri)
bigramTtable = pd.DataFrame(list(bigramFinder.score_ngrams(bigrams.student_t)), columns=['bigram','t']).sort_values(by='t', ascending=False)
trigramTtable = pd.DataFrame(list(trigramFinder.score_ngrams(trigrams.student_t)), columns=['trigram','t']).sort_values(by='t', ascending=False)
#filters
filteredT_bi = bigramTtable[bigramTtable.bigram.map(lambda x: rightTypes(x))]
filteredT_tri = trigramTtable[trigramTtable.trigram.map(lambda x: rightTypesTri(x))]
print(filteredT_bi)
