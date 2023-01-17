#imports



import numpy as np

import pandas as pd

import nltk

import collections as co

from io import StringIO

import matplotlib.pyplot as plt

import warnings

from IPython.display import display, HTML, Markdown, display



#constants

%matplotlib inline

def printmd(string):

    display(Markdown(string))

alphaLev = .5
#load in dataset

complaintFrame = pd.read_csv("../input/consumer_complaints.csv")
#consider only narrative observations

complaintNarrativeFrame = complaintFrame[complaintFrame["consumer_complaint_narrative"].notnull()]

# build a fast way to get strings

# adapted from 

# https://www.kaggle.com/mchirico/d/cfpb/us-consumer-finance-complaints/analyzing-text-in-consumer-complaints

s = StringIO()

complaintNarrativeFrame["consumer_complaint_narrative"].apply(lambda x: s.write(x))

k=s.getvalue()

s.close()

k=k.lower()

k=k.split()
# Next only want valid strings

words = co.Counter(nltk.corpus.words.words())

stopWords =co.Counter( nltk.corpus.stopwords.words() )

k=[i for i in k if i in words and i not in stopWords]

c = co.Counter(k)

printmd("We see that we $" + str(len(k)) + "$ legal word tokens in our corpus. There are $" + str(

        len(list(c.most_common())))

       + "$ legal non-stopword types in our corpus.")
wordFrequencyFrame = pd.DataFrame(c.most_common(len(c)),columns = ["Word","Frequency"])

#plot frequency on rank

fig, (ax1, ax2) = plt.subplots(1, 2)

fig.set_size_inches(18, 7)

#freq-rank

ax1.plot(wordFrequencyFrame.index,wordFrequencyFrame["Frequency"])

ax1.set_title("Frequency on Rank of Vocabulary")

ax1.set_xlabel("Rank")

ax1.set_ylabel("Frequency")

#freq-logRank

ax2.plot(np.log(wordFrequencyFrame.index + 1),np.log(wordFrequencyFrame["Frequency"]))

ax2.set_title("Log-Frequency on Log-Rank of Vocabulary")

ax2.set_xlabel("Log Rank")

ax2.set_ylabel("Frequency")

plt.show()

printmd("_Figure 1: Frequency-Rank Graphs of Our Vocabulary._")

#get 15 most common

top15FrequencyFrame = wordFrequencyFrame.iloc[0:15,:]

display(top15FrequencyFrame)

printmd("_Table 1: The $15$ most frequent words with their frequencies_")

#get 15 least common

bottom15FrequencyFrame = wordFrequencyFrame.iloc[(wordFrequencyFrame.shape[0]-15):wordFrequencyFrame.shape[0],:]

display(bottom15FrequencyFrame)

printmd("_Table 2: The $15$ least frequent words with their frequencies_")
#get token-type list

typeSet = set([]) #we will add to this over time

typeTokenList = [] #we will add tuples to this

for i in range(len(k)):

    givenToken = k[i]

    if (givenToken not in typeSet): #we should get a new type count

        typeSet.add(givenToken)

    #then add information to type-token list

    typeTokenList.append((i+1,len(typeSet)))
#then plot

typeTokenFrame = pd.DataFrame(typeTokenList,columns = ["Token Count","Type Count"])

plt.plot(typeTokenFrame["Token Count"],typeTokenFrame["Type Count"])

plt.xlabel("Token Count")

plt.ylabel("Type Count")

plt.title("Token Count on Type Count")

plt.show()

printmd("_Figure 2: Type-Token Graph for full vocabulary._")
productCountFrame = complaintNarrativeFrame.groupby("product")["consumer_complaint_narrative"].count()

#from pylab import *

#val = 3+10*rand(5)    # the bar lengths

pos = np.arange(productCountFrame.shape[0])+.5    # the bar centers on the y axis



plt.barh(pos,productCountFrame, align='center')

plt.yticks(pos,productCountFrame.index)

plt.xlabel('Count')

plt.ylabel("Product Type")

plt.title('Distribution of Product Type')

plt.grid(True)

plt.show()

printmd("_Figure 3: Distribution of product types._")

printmd("The number of narratives of the product type 'Other financial service' is $" + str(

        productCountFrame["Other financial service"]) + "$.")
#declare functions before making type-token procedures

def makeTypeTokenFrame(tokenList):

    #helper that makes our type-token frame for a given token list

    typeSet = set([]) #we will add to this over time

    typeTokenList = [] #we will add tuples to this

    for i in range(len(tokenList)):

        givenToken = tokenList[i]

        if (givenToken not in typeSet): #we should get a new type count

            typeSet.add(givenToken)

        #then add information to type-token list

        typeTokenList.append((i+1,len(typeSet)))

    return pd.DataFrame(typeTokenList,columns = ["Token Count","Type Count"])



def makeTokenList(consumerComplaintFrame):

    #helper that makes token list from the given complaint frame

    s = StringIO()

    consumerComplaintFrame["consumer_complaint_narrative"].apply(lambda x: s.write(x))

    k = s.getvalue() #gets string of unprocessed words

    s.close()

    #get actual unprocessed words

    #k = k.lower()

    k = k.split()

    k = [i for i in k if i in words and i not in stopWords] #only consider legal words

    return k



def getTokenTypeFrameForProduct(consumerComplaintFrame,productName):

    #helper that gets our token-type frame for narratives of a given product name

    #get observations with this product name

    givenProductComplaintFrame = consumerComplaintFrame[consumerComplaintFrame["product"] == productName]

    #then get token list

    tokenList = makeTokenList(givenProductComplaintFrame)

    #then make type-token frame

    return makeTypeTokenFrame(tokenList)
#run through our observations

typeTokenFrameDict = {} #we will adds to this

for productName in productCountFrame.index:

    typeTokenFrameDict[productName] = getTokenTypeFrameForProduct(complaintNarrativeFrame,productName)
cmap = plt.get_cmap('Dark2')

colorList = [cmap(i) for i in np.linspace(0, 1, len(typeTokenFrameDict))]

for i in range(len(typeTokenFrameDict)):

    productName = list(typeTokenFrameDict)[i]

    givenProductTokenTypeFrame = typeTokenFrameDict[productName]

    plt.plot(givenProductTokenTypeFrame["Token Count"],

             givenProductTokenTypeFrame["Type Count"],label = productName,

            c = colorList[i])

plt.legend(bbox_to_anchor = (1.6,1))

plt.xlabel("Token Count")

plt.ylabel("Type Count")

plt.title("Token-Type Graph\nBy Product Name")

plt.show()

printmd("_Figure 4: Token-Type Graph By Product Name._")
def getTokenTypeFrameForDispute(consumerComplaintFrame,disputeLev):

    #helper that gets our token-type frame for narratives of a dispute level

    #get observations with this product name

    givenDisputeComplaintFrame = consumerComplaintFrame[consumerComplaintFrame["consumer_disputed?"] == disputeLev]

    #then get token list

    tokenList = makeTokenList(givenDisputeComplaintFrame)

    #then make type-token frame

    return makeTypeTokenFrame(tokenList)
consumerDisputeDict = {} #we will add to this

for disputeLev in complaintNarrativeFrame["consumer_disputed?"].unique():

    consumerDisputeDict[disputeLev] = getTokenTypeFrameForDispute(complaintNarrativeFrame,disputeLev)
for disputeLev in consumerDisputeDict:

    DisputeTokenTypeFrame = consumerDisputeDict[disputeLev]

    plt.plot(DisputeTokenTypeFrame["Token Count"],

             DisputeTokenTypeFrame["Type Count"],label = disputeLev)

plt.legend(bbox_to_anchor = (1.3,1))

plt.xlabel("Token Count")

plt.ylabel("Type Count")

plt.title("Token-Type Graph\nBy Whether Consumer Disputed")

plt.show()

printmd("_Figure 5: Token-Type Graph By whether the consumer disputed._")
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#make mappable for vocabulary

counterList = c.most_common()

vocabDict = {} #we will add to this

for i in range(len(counterList)):

    vocabWord = counterList[i][0]

    vocabDict[vocabWord] = i

#make array of tf-idf counts

vectorizer = TfidfVectorizer(min_df=1,stop_words = stopWords,vocabulary = vocabDict)

unigramArray = vectorizer.fit_transform(complaintNarrativeFrame["consumer_complaint_narrative"])
#generate our language matrix

languageFrame = pd.DataFrame(unigramArray.toarray(),columns = vectorizer.get_feature_names())

printmd("The number of features extracted is $" + str(languageFrame.shape[1]) + "$.")