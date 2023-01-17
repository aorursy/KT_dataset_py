import numpy as np

import pandas as pd

import os

import csv

import nltk

from nltk.corpus import wordnet as wn

from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
#radix tree generation



#each NODE

class Node:

    

    def __init__(self):

        self.nxt = []

        self.endmark = False

        self.context = []

        

        for i in range(26):

            self.nxt.append(None)

        

    def addContext(self, genSense):

        self.context.append(genSense)

        



##initiate tree root





##inserting each word in tree

def insert(root,string,length,context):

    curr = root

    for i in range(length):

       ## print(string[i])

        index = ord(string[i]) - ord('a')

        if index >= 0 and index<26:

            if curr.nxt[index] is None:

                curr.nxt[index] = Node()

            curr = curr.nxt[index]

        else:

            return

    curr.endmark = True

    if context not in curr.context:

        curr.addContext(context)

    



#search for a word

def search(root,string,length):

    curr = root

    #print('root')

    for i in range(length):

        index = ord(string[i]) - ord('a')

        if index >= 0 and index<26:

            if curr.nxt[index] is None:

                return False

            curr = curr.nxt[index]

        else:

            return False 

    return curr.context



root = Node()

##add context info from csv table

import csv

import re

def train(traincsv):

    eCnt = 0

    cnt = 0

    with open(traincsv) as csvfile:

        readCSV = csv.reader(csvfile, delimiter=',')

        c =0

        c1 = 1

        r = []

        for row in readCSV:

            #print(row)

            c1 += 1

            

            for i in range(0,len(row)):

                if i%2 == 0:       #Second line is missing

                    if (i+3) < len(row):    #Ensuring there is a sense annotated word after current word

                        #print(i+3,row[i+3])

                        r.clear();

                        r = re.findall(r"[\w'%:]+", row[i+3])

                    try:

                        for nextWordSense in r:

                            gs  = wn.lemma_from_key(nextWordSense).synset()

                            cnt+=1

                            insert(root,row[i].lower(),len(row[i]),gs.name())

                        r.clear()



                    except Exception as ex:

                        template = "1 ERRROR!!! An exception of type {0} occurred. Arguments:\n{1!r}"

                        message = template.format(type(ex).__name__, ex.args)

                        #print(message)

                        eCnt +=1

                        pass



                    if (i-1) >= 0:

                        #print(i-1,row[i-1])

                        r.clear()

                        r = re.findall(r"[\w'%:]+", row[i-1])

                        #print(i-1,r)



                    try:

                        for prevWordSense in r:

                            gs  = wn.lemma_from_key(prevWordSense).synset()

                            cnt+=1

                            insert(root,row[i].lower(),len(row[i]),gs.name())

                        r.clear()

                    except Exception as ex:

                        template = "2 ERROR!!! An exception of type {0} occurred. Arguments:\n{1!r}"

                        message = template.format(type(ex).__name__, ex.args)

                        #print(message)

                        eCnt += 1

                        pass

            

        print('Context table inserts : ',cnt - eCnt,'\n')

i = 0

for i in range(16):

    traincsv = '../input/masc-train-final-1/masc_train_'+str(i)+'.csv'

    print("Training Dataset : ",i+1)

    train(traincsv)    

s = 'stay'

print(search(root,s,len(s)))
testcsv = '../input/all-masc-test-datasets/masc_test_0.csv'

def genTestString(testcsv):

    with open(testcsv) as csvfile:

        readCSV = csv.reader(csvfile, delimiter=',')

        oddRow = -1  # each even row blank in the csv file

        oddCol = 1   # each odd col contains a word

        s = ''

        for row in readCSV:

            #print(row)

            if(oddRow==-1):

                oddRow = 1

            else:

                for word in row:

                    if oddCol==1:

                        s += word + ' '

                        oddCol = 0

                    else:

                        oddCol = 1

                if oddRow==1:

                    s += " "

                    oddRow = 0

                else:

                    oddRow = 1

        

        return s

s = genTestString(testcsv)

print(s)
def sentenceTokenize(s):

    st = s

    sentences = nltk.sent_tokenize(st.lower())

    return sentences



sentences = sentenceTokenize(s)
sentences




def removePunctuation(sentences):

    tt="" 

    sentences2=[]

    for x in sentences:

        tokenizer = RegexpTokenizer('\w+')

        text2 = tokenizer.tokenize(x)

        cnt=1   

        for x2 in text2:

            if cnt==1:   # no space before first word

                tt+=x2

                cnt=0

            else:

                tt+=" "+x2  # space before other words  

        sentences2.append(tt)

        tt=""

        

    return sentences2
sentences2=removePunctuation(sentences)

sentences2
def stopWords(sentences2):

    stop_words = set(stopwords.words("english"))

    context_tab=[]

    for sentence in sentences2:

        words = nltk.word_tokenize(sentence)

        without_stop_words = [word for word in words if not word in stop_words]

        context_tab.append(without_stop_words)

    return context_tab
context_tab =stopWords(sentences2)

context_tab
def lemmatization(context_tab):

    lemma=[]

    wl=WordNetLemmatizer()

    for x in context_tab:

        m2=[]

        for x2 in x:

            x3=wl.lemmatize(x2,wn.VERB)

            x3=wl.lemmatize(x3,wn.NOUN)

            x3=wl.lemmatize(x3,wn.ADJ)

            x3=wl.lemmatize(x3,wn.ADV)

            m2.append(x3)

        lemma.append(m2)

    return lemma
lemma=lemmatization(context_tab)

lemma
def tagPos(lemma):

    pos=[]

    for n in lemma:

        pos.append(nltk.pos_tag(n))

    return pos    

wordPos=tagPos(lemma)

wordPos


def wordOfInterest(pos):

    wn_pos=['NN','VB','JJ','JJR','JJS','NNP','VBG','RB','VBD','VBP']



    woi1=[]



    for x in pos:

        arr=[]

        for y in x:

            if y[1] in wn_pos:

                arr.append(y)

        woi1.append(arr) 

    woi=[]

    for i in woi1:

        arr2=[]

        for j in i:



            if j[1]=='VBD' or j[1]=='VB' or j[1]=='VBP':

                tup=(j[0],'v')

                arr2.append(tup)

            elif j[1]=='VBG':

                tup=(j[0],'n')

                arr2.append(tup)

            elif j[1]=='NN' or j[1]=='NNP':

                tup=(j[0],'n')

                arr2.append(tup)

            elif j[1]== 'JJ' or j[1]=='JJR' or j[1]=='JJS':

                tup=(j[0],'a')

                arr2.append(tup)

            elif j[1]=='RB':

                tup=(j[0],'r')

                arr2.append(tup)

        woi.append(arr2)       

            

    return woi
woi=wordOfInterest(wordPos)

woi
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

  

# X = input("Enter first string: ").lower()

# Y = input("Enter second string: ").lower()

def cosineSimilarity(X,Y):

    # tokenization

    X_list = word_tokenize(X) 

    Y_list = word_tokenize(Y)



    # sw contains the list of stopwords

    sw = stopwords.words('english') 

    l1 =[];l2 =[]



    # remove stop words from string

    X_set = {w for w in X_list if not w in sw} 

    Y_set = {w for w in Y_list if not w in sw}



    # form a set containing keywords of both strings 

    rvector = X_set.union(Y_set) 

    for w in rvector:

        if w in X_set: l1.append(1) # create a vector

        else: l1.append(0)

        if w in Y_set: l2.append(1)

        else: l2.append(0)

    c = 0



    # cosine formula 

    for i in range(len(rvector)):

            c+= l1[i]*l2[i]

    cosine = c / float((sum(l1)*sum(l2))**0.5)

    return cosine
outcome=[]

def WSD(woi):

    lineNo=0;

    for line in woi:

        wordSenseAr=[]

        for word in line:

            senseAr=[]

            temp=[]

            senseAr = wn.synsets(word[0], word[1])

            for i in senseAr:

                score=0

                count=0

                for j in i.examples():

                    score=score+cosineSimilarity(j,sentences[lineNo])

                    count=count+1

                if count ==0:

                    score=0

                else:

                    score = score/count

                temp.append((i,score))

            temp = sorted(temp, key=lambda tup: tup[1],reverse=True)

            wordSenseAr.append(temp[0:int(len(temp)/2+1)])

        outcome.append(wordSenseAr)

    #print(outcome)

    print('\n')

    return outcome







                

outcome = WSD(woi)

outcome


def test(testcsv):

       

    print('Testing .. ')

    testOutput = []

    count = 0

    count2 = 0

    count3 = 0

    count4 = 0

    

    with open(testcsv) as csvfile:

        readCSV = csv.reader(csvfile, delimiter=',')

        for row in readCSV:

            testOutput.append(row)

    

    lineCount=0

    wordCount = 0

    output = [[]]

    outputLine = []

    

    for line in woi:   # each line of Ambiguous words

        wordCount=0

        

        #print(count,count4)

        

        for word in line:   # each ambiguous words

            ok=0

            

            # find the context words from the whole input text

            

            contextWord=[]

            for line1 in range(0,len(woi)):

                for word1 in range(0,len(woi[line1])):

                    if word==woi[line1][word1]:

                        if(word1-1>=0):

                            contextWord.append(woi[line1][word1-1])   # add prev word as context word

                        if (word1+1<len(woi[line1])):

                            contextWord.append(woi[line1][word1+1])   # add next word as context word



            

            for w in contextWord:   

                

                result=search(root,w[0],len(w[0]))   # search each context words in the context table

                

                if result != False:  # if found

                    

                    if ok==0:

                        ok=1    # flag that a context word is found in context table

                        maxx=-1

                        j=None

                        prob = 0

                        

                        for sense in outcome [lineCount][wordCount]:   # each sense from previous step

                            p=0

                            s=sense[0]

                            for genSenses in result:      # each general senses from context table

                                gs=wn.synset(genSenses)

                                a=s.path_similarity(gs)

                                b=gs.path_similarity(s)

                                if a is None:

                                    a=0

                                if b is None:

                                    b=0

                                similarity1 = (a+b) / 2

                                

                                if(similarity1>p):

                                    p = similarity1      # take the max simililarity



                            prob=p+sense[1]    # add with similarity from previous step

                            

                            if(prob>maxx):    # find out which sense has max score

                                maxx=prob

                                j=s

                        

                        if j is not None:

                            outputLine.append(word[0])

                            outputLine.append(j.name())

                            count2+=1



            if(ok==0):  # if no context word found in the context table

                try:

                    if len(outcome[lineCount][wordCount])!= 0:

                        count2+=1

                        outputLine.append(word[0])

                        outputLine.append(wn.synsets(woi[lineCount][wordCount][0],woi[lineCount][wordCount][1])[0].name())

                except:

                    pass

            wordCount=wordCount+1

        

        

        

        #accuracy test

        



        if 2*(lineCount+1) < len(testOutput):

            wordCount += len(outputLine)/2

            for out in range(len(outputLine)):

                br = 0

                for testline in range(len(testOutput)):

                    if(br==1):

                        br = 0

                        break

                    if(testline%2==0):

                        for test in range(len(testOutput[testline])):

                                                   

                            if(out%2 == 0 and test%2==0):

                                t = testOutput[testline][test+1]

                                if(t!='NONE'):

                                    count3 += 1

                                    if(outputLine[out] == testOutput[testline][test].lower()):

                                        print("Ambiguous word: ",outputLine[out])

                                        print('\nOutput: ',wn.synset(outputLine[out+1]).definition(),'\n','\nTest senses: ')

                                        

                                        count+=1

                                        testSenses = re.findall(r"[\w'%:\(\)]+", t)

                                        

                                        correct = 0

                                        

                                        for testSense  in testSenses:

                                            

                                            try:

                                                print('>>>',wn.lemma_from_key(testSense).synset().definition())

                                                if(outputLine[out+1] == wn.lemma_from_key(testSense).synset().name()):

                                                    count4 += 1

                                                    

                                                    correct = 1

                                            except:

                                                pass

                                        br = 1

                                        if(correct == 1):

                                            print('\n********Correct*********\n')

                                            correct = 0

                                        else:

                                            print('\n********Incorrect*********\n')

                                        

                                        break









        lineCount=lineCount+1



        output.append(outputLine)

        outputLine.clear()

        

    

    

    count1 = 0



    i = 0

    for testLine in testOutput:

        for test in range(len(testLine)):

            if test%2==0 and testLine[test+1]!='NONE':

                count1 += 1



    print('Total Ambiguous words: ',count)

    print('Correctly disambiguated: ',count4)

    print('Accuracy :  ',(count4+1)/(count+1)*100,'%')

    

    

#testcsv = "../input/all-masc-test-datasets/masc_test_1.csv"

test(testcsv)