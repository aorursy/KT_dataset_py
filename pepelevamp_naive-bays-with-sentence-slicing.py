# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from collections import defaultdict

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random

import math

import re as re

#global variables



trainingdata = pd.read_csv("../input/abstracts/trg.csv")#, skiprows=0,nrows=1000)

testingdata = pd.read_csv("../input/abstracts/tst.csv")#, skiprows=0,nrows=700)

englishdata= pd.read_csv("../input/english-wc/wordcount.csv", skiprows=0,nrows=2000)







#testingdata  = pd.read_csv("../input/tst.csv")

classcount = {"B":0,"A":0,"E":0,"V":0} # how many papers in each class

classnames={"B","A","E","V"}

global_wordcount=defaultdict(int)

A_wordcount=defaultdict(int) # word:count wordcount for A class

B_wordcount=defaultdict(int) # word:count wordcount for B class

E_wordcount=defaultdict(int)

V_wordcount=defaultdict(int)

english_wordcount=defaultdict(int)



#build up english word count into dictionary

print("building english dictionary")



for index,row in englishdata.iterrows():

    tab="\t"

    w=row[0].split(tab)[0]

    freq=row[0].split(tab)[1]

    #print(w,":",freq)

    english_wordcount[w]=int(freq)

print ("english dictionary complete")



def resetGlobalVariables():

    A_wordcount.clear()

    B_wordcount.clear()

    E_wordcount.clear()

    V_wordcount.clear()

    global_wordcount.clear()

def createTrainAndTestSet(df, size) :

    train = df.head(size)

    test = df.tail(len(df)-size)

    return train, test

def createTenFoldValidation(df):

    # yield a 1x size and a 9x size.

    for c in range(0,10):

        trb = getAllButTenChunk(df,c)

        tst = getTenChunk(df,c)

        yield trn,tst



# these two lovely methods allow us to easily split a dataset into a 1/10th and 9/10th

def getTenChunk(df,chunkNumber):

    tenth = int(len(df)/10);

    startIndex=chunkNumber*tenth

    endIndex=startIndex+tenth -1

    #print("asked for 10th of dataset of size ",len(df),", which is chunk size:",tenth,", and startIndex:",startIndex," and endIndex:",endIndex )

    chunk=df.tail(len(df)-startIndex)

    chunk=chunk.head(tenth)  #cull to one tenth size.

    return chunk

def getAllButTenChunk(df,chunkNumber):

    tenth = int(len(df)/10)

    startIndex=chunkNumber*tenth

    chunk=df.head(startIndex)  #first part of dataset 

    #stuff in between is startIndex to endIndex and we dont want this

    chunk = chunk.append(df.tail(len(df)-(startIndex+tenth))) #last part of dataset

    return chunk

        

        

# because i can't do multi-dimensional keyed hash things & suck at python, heres the utility functions:

#all class names, excluding one

def allClassNamesExcluding(ex):

    for c in classnames:

        if not c==ex:

            yield c



# return the wordcount bin given the supplied argument

def classWordCount(classname):

    if classname == 'A' : countbin=A_wordcount;

    if classname == 'B' : countbin=B_wordcount;

    if classname == 'E' : countbin=E_wordcount;

    if classname == 'V' : countbin=V_wordcount;

    return countbin 



#probability of a word occuring given a certain class

def ratioOfWordOccuringInClass(w,c):

    # word 'balls' occurs in 5 c-papers out of all c-papers.

    # word 'balls' occurs in 5 virus emails out of 125 virus papers.

    return (0.01+classWordCount(c)[w]) / classcount[c];

def ratioOfWordOccuringGlobally(w):

    return (0.01+global_wordcount[w]) / len(global_wordcount);

def ratioOfWordOccuringInEnglish(w): #returns 0-1 popularity in english

    return (0.01+english_wordcount[w]) / len(english_wordcount)

def iterateAllClassWordCounts():

    yield classWordCount('A')

    yield classWordCount('B')

    yield classWordCount('E')

    yield classWordCount('V')

def isDigits(w):

    try:

        int(w)

        return True

    except ValueError:

        return False



def filterWordsInAbstract(sentence,allowDuplicates=False,allowCommonWords=False,allowNumerals=False,groupWords=False,bothGroupsAndSingles=False):

    #return a list of words allowed to be processed. allow duplicates of word? who knows.

    blacklist=["the","and","or","of","to","&","an","is","not","as","by","a","be","in","with","was","when","all","that","those","for","they","have","from","this","word"]

    countedWords = []

    for w in re.split("[,- \!?:]+", sentence):

        countWord=True

        if (not allowDuplicates) and (w in countedWords): countWord=False

        if (not allowCommonWords) and (w in blacklist): countWord=False

        if (not allowCommonWords) and (len(w) <=2): countWord=False

        if (not allowNumerals) and isDigits(w): countWord=False

        if countWord:

            countedWords.append(w)

    returnList=countedWords

    if groupWords or bothGroupsAndSingles:

        # group into groups of 2 words

        phrases=[]

        previousWord=""

        for w in countedWords:

            phrases.append(previousWord+w)

            previousWord=w+" "

            

        if bothGroupsAndSingles:

            phrases.extend(countedWords)

        return phrases

    return countedWords

    

def filterWordsInAbstractByPopularitySplits(sentence):

    phrases=[]

    phrase=""

    rollingAverage=0;

    started=False

    for w in re.split("[- \!?:]+", sentence):

        popularity=ratioOfWordOccuringInEnglish(w)

        if not started:

            rollingAverage=popularity

            started=True# default value of average popularity is the first word's populartiy right

        rollingAverage=(rollingAverage*0.5) + (popularity * 0.5) #smoothing function

        transientSpike = (math.log(popularity) -math.log(rollingAverage)) ;

        if abs(transientSpike) >5:

            #print("juice:",w,"\t junk:",phrase)

            phrases.append(w)

            phrase=""

        else:

            phrase+=" "+w

            #print("word:",w,", current popularity:",math.log(popularity),", rolling average:",math.log(rollingAverage))

            #print("word",w,". Transient against rolling-average popularity:",transientSpike)

    return phrases

def keyWithBiggestValue(dic):

    maxkey=None

    maxv=0;

    for k,v in dic.items():

        if v > maxv:

            maxv=v

            maxkey=k

    return maxkey

def train_on_dataset(whichDataFrame):

    

    for index,row in whichDataFrame.iterrows():

        classname=row['class']

        abstract = row['abstract']

        trainOnSentence(abstract,classname)



def trainOnSentence(abstract,classname):

    # increment total occurances of papers in this class.

    classcount[classname]+=1;







    #determine which wordcount bin to use to count up each word (one for each class)

    wordcountbin=classWordCount(classname)



    # iterate through each word, add it the right wordcount for the current class. view each word once

    #abstractWords=filterWordsInAbstract(abstract,allowDuplicates=False,allowCommonWords=True,allowNumerals=False,groupWords=False,bothGroupsAndSingles=False)

    abstractWords=filterWordsInAbstractByPopularitySplits(abstract)

    

    for w in abstractWords:

        wordcountbin[w]+=1

        global_wordcount[w]+=1

        #print ("\n",w , " occured in a ",classname, " abstract. count is now:", classWordCount(classname)[w])

        #print("counts for word ",w," in other classes:")



        #for c in iterateAllClassWordCounts():

        #    print (c[w])

def classifyFilteredAbstractWords(abstract):

    # go through each word,  up the virusness of each word, the bacterianess of each word, etc

    classScores = {}

    for cl in classnames:



        #print ("\n\ncalculating score for class:",cl)

        probably_in_cl=1



        for w in abstract:

            #how many emails in class cl does w appear?

            r = ratioOfWordOccuringInClass(w,cl)

            probably_in_cl+=(math.log(r)-ratioOfWordOccuringGlobally(w) ) #weaken influence by popularity

            

            #print(w," appears in class ",cl);

            # what about the other classes? we need to apply the negative effect to say 'definitely not this class' if the word dont match cl, but matches a different class.

            for other_cl in allClassNamesExcluding(cl):

                other_r=ratioOfWordOccuringInClass(w,other_cl)

                probably_in_cl-=(math.log(other_r)-ratioOfWordOccuringGlobally(w)) # weaken influence by popularity



            

        classScores[cl]=probably_in_cl;



    # get biggest score

    #print(classScores)

    winningClass= keyWithBiggestValue(classScores)

    return winningClass

    

def verify_on_dataset(whichDataFrame,dontCollectStatsCosTheyDontExist=False):

    totalPapers = whichDataFrame.shape[0]

    #read in test set

    #whichDataFrame['computed_class']=[random.randint(0,1000) for x in range(whichDataFrame.shape[0])]

    

    outputdata = pd.DataFrame(columns=["id", "class"])

    correctAnswers=0

    wrongAnswers=0

    displayPercent=0;

    n=0

    for index,row in whichDataFrame.iterrows():

        n+=1;

        if (n* 100 / totalPapers > displayPercent):

            print(displayPercent,"% ",end=None)

            displayPercent+=10

        aid      = row['id']

        abstract = row['abstract']

        

        

        #HOW DO WE FILTER THE ABSTRACT DURING EVALUATION TIME? ALLOW DUPLICATE WORDS? ETC

        #abstractwords = filterWordsInAbstract(abstract,allowDuplicates=False,allowCommonWords=True,allowNumerals=False,groupWords=False,bothGroupsAndSingles=False)

        abstractwords=filterWordsInAbstractByPopularitySplits(abstract)

        

        

        # go through each word,  up the virusness of each word, the bacterianess of each word, etc

        winningClass=classifyFilteredAbstractWords(abstractwords)

        

        

        if not dontCollectStatsCosTheyDontExist:

            classname=row['class']

        else:

            classname=winningClass # were not collecting stats cos its probably the real deal. so fake success.

            

        #if (winningClass != classname): print("id:",aid,", PredictedClass:",winningClass,", ActualClass:",classname)    

        if winningClass==classname:

            correctAnswers+=1

        else:

            wrongAnswers+=1

        outputdata = outputdata.append({ "id": aid, "class": winningClass}, ignore_index=True)

        

        #*********RETRAIN ON DATA. this is a great way to overfit

        #trainOnSentence(abstract,winningClass)

        

    print()    

    return outputdata,correctAnswers,wrongAnswers







# print the number of emails the word 'the' appears in a virus email, against the total number of virus emails. (likelihood of 'the' appearing in virus email)

# train dataset (build the word counts)            









def benchmark():

    #benchmarkResults = pd.DataFrame(columns=["split", "correct","incorrect","accuracy"])

    #benchmarkResults = benchmarkResults.append({ "id": aid, "class": winningClass}, ignore_index=True)

    #benchmarkResults.to_csv('benchmark_naive_nooptimizations.csv', index = False)





    averageAccuracy=0;

    for n in range(0,10):

        # make sets

        resetGlobalVariables()

        trn = getAllButTenChunk(trainingdata,n)

        tst = getTenChunk(trainingdata,n)

        print("Training on segment ",n, " of 10...")

        train_on_dataset(trn)

        print("Evlaluating on all but segment ",n,"...",end="")

        finaloutput,correctAnswers,wrongAnswers=verify_on_dataset(tst)

        accuracyPercent =  (100* correctAnswers/(correctAnswers+wrongAnswers))

        averageAccuracy+=accuracyPercent

        #print("right:",correctAnswers,". wrong:",wrongAnswers, ", accuracy:",accuracyPercent)

    averageAccuracy/=10

    return averageAccuracy

def produceAssignmentResults():

    resetGlobalVariables()

    train_on_dataset(trainingdata)

    

    finaloutput,correctAnswers,wrongAnswers=verify_on_dataset(testingdata,dontCollectStatsCosTheyDontExist=True)

    finaloutput.to_csv('testing_data_results.csv', index = False)

    print ("size of finaloutput:", finaloutput.shape[0])



print("averageAccuracy:",benchmark())

#produceAssignmentResults()





#for rid,row in trainingdata.iterrows():

#    abstract=row['abstract']

#    filterWordsInAbstractByPopularitySplits(abstract)

#print("done");



#print (ratioOfWordOccuringInClass("the","V") )

#print ( V_wordcount["the"] , " / " , classcount['V'])

#print ( classWordCount('V')['the'] / classcount['V'] )