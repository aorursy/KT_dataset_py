print('Hello Python!')
Print('Hello Python!')
p = 'hello python'

print(p)
a = 1 + 1

print(a)
b = '1' + '1'

print(b)
c = b*2  # equal "11" * 2

print(c)

print('abc'*2)
d = 'hello'

e = 'python'

print((d+e+'\n')*2)
with open('../input/WeatherAnimalsSports.csv', 'rb') as file:

    f = file.read().decode('utf8','ignore')

print(f)
print("The type of f is: ",type(f))

print("The length of f is: ", len(f) )
print(type(1))

print(type(1.1))

print(type('abc'))

print(type([1,2,3]))

print(type({"name":"Jack"}))

print(type(print))
import pandas as pd

data=pd.read_csv('../input/WeatherAnimalsSports.csv')

data.head()
data['Target_Subject']
data[0:5]
data.loc[0,'TextField']
data.iloc[0,1]
j = 0

for i in range(3):

    print(j)

    j += 1
for i in data['TextField'][:3]:

    print(i)
subtext = [i for i in data['TextField'][:3]]

print(subtext)
text1 = data.loc[0,'TextField']

print(text1)
token1 = text1.split(' ')  # split text by spaces

print(token1)
tokens = [tx.split(' ') for tx in data.loc[:, 'TextField']]  # use for-loop to tokenize every text in the file

print(tokens[:5])  # print the first 5 tokens
from nltk.corpus import stopwords  # use stop words from nltk library

stopword = stopwords.words(['english'])  # define stopword
cleaned_tokens = []  # create a new list to store result

for token in tokens:  # look through all the element in tokens

    cleaned_token = [word.lower() for word in token]  # lowercase

    cleaned_token = [word for word in cleaned_token if word not in stopword]  # delete stopword in each token

    cleaned_token = [word for word in cleaned_token if word.isalpha()]  # delte non alphabet word

    cleaned_tokens.append(cleaned_token)  # put each result into new list

print(cleaned_tokens[:5]) # check first 5 result
wordlists = []  # creat a empty list for storing the result

for t in cleaned_tokens:  # look through all the element in tokens

    wordlists += t  # add every token into list

print(wordlists)
print(len(wordlists))  # check how many words in total, we have duplicates in the list which need to be deleted
wordlist = list(set(wordlists))  # remove duplicate words from wordlist

print(len(wordlist))  # check words number after removing duplicate
wordcounts = []  # creat a empty list for storing the whole result

for token in cleaned_tokens:  # look through all the element in tokens

    wordcount = []  # creat a empty list for storing each result temparorily, notice every loop, this list will be emptified

    for word in wordlist: # look through all the element in wordlist

        count = token.count(word)

        wordcount.append(count)

    wordcounts.append(wordcount)

print(len(wordcounts))

print(len(wordcounts[0]))

print(len(cleaned_tokens))

print(count)

print(wordlist)

print(wordcount)

print(cleaned_tokens[-1])

wordmatrix = pd.DataFrame(data=wordcounts, columns=wordlist)  # creat a dataframe to help you look the result

wordmatrix.head()  # show first 5 documents
wordmatrix['row_total'] = wordmatrix.aggregate('sum',axis=1) # add a sum column (total number of words in each document)

wordmatrix.head()
N = len(wordmatrix)

n = wordmatrix.astype('bool').sum()  

print(n)
import math



for row in range(len(wordmatrix)):  # go through every row

    for col in wordmatrix.columns[:-1]:  # go through every column exclude 'row_total'

        wordmatrix.loc[row,col] = wordmatrix.loc[row,col]/wordmatrix.loc[row,'row_total']*math.log10(N/n[col])

        

wordmatrix.head()
from sklearn.decomposition import TruncatedSVD



svd = TruncatedSVD(n_components=15, n_iter=30, random_state=0)

X = svd.fit_transform(wordmatrix.drop('row_total',axis=1))

print(X)
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3)

gmm.fit(X)

result = gmm.predict(X)

print(result)
from sklearn.cluster import SpectralClustering

clustering = SpectralClustering(n_clusters=3, random_state=0)

clustering.fit(X)

result0 = clustering.labels_

print(result0)
data['cluster'] = result

data['cluster0'] = result0

data
docmatrix = wordmatrix.drop('row_total',axis=1).transpose()

docmatrix.head(20)
svdT = TruncatedSVD(n_components=5, n_iter=30, random_state=0)

XT = svd.fit_transform(docmatrix)

print(XT)
print(X.shape)

print(XT.shape)
topic = GaussianMixture(n_components=6)

topic.fit(XT)

topic_label = topic.predict(XT)

print(topic_label)
topic_table = pd.DataFrame([topic_label], columns=docmatrix.index)

topic_table.index = ['topic']

topic_table = topic_table.sort_values(by='topic',axis=1)

topic_table
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, n_iter=300)

coordinates = tsne.fit_transform(docmatrix)

print(coordinates)
wordmap = pd.DataFrame(coordinates, columns=['x','y'])

wordmap.head()
from sklearn.cluster import KMeans

word_label = KMeans(n_clusters=3).fit(wordmap).labels_
wordmap['word'] = docmatrix.index

wordmap['cluster'] = word_label

wordmap.head()
import seaborn as sns 

from matplotlib import pyplot as plt

plt.figure(figsize=(20, 20))

sns.scatterplot('x','y',hue='cluster',palette="Set1",s=150, data=wordmap)

for n in range(len(wordmap)):

        plt.annotate(wordmap['word'][n],

                     xy=(wordmap['x'][n],wordmap['y'][n]),

                     xytext=(2,5), textcoords='offset points', fontsize=16)
import pandas as pd



train_data = pd.read_csv('../input/WeatherAnimalsSports.csv')

score_data = pd.read_csv('../input/Score_WeatherAnimalSports.csv')

train_data.head()
score_data.head()
train_blob = list(zip(train_data['TextField'],train_data['Target_Subject']))  # transform format

train_blob[:5]  # check format
from textblob.classifiers import NaiveBayesClassifier

cl = NaiveBayesClassifier(train_blob)
score0 = cl.classify(score_data['TextField'][0])

print(score_data['TextField'][0])

print(score0)
scores = [cl.classify(sentence) for sentence in score_data['TextField']]

score_data['prediction'] = scores

score_data
text0 = score_data['TextField'][1]

print(text0)
from textblob import TextBlob

blob0 = TextBlob(text0)

blob0.sentiment.polarity