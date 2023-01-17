# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import nltk

from nltk.corpus import wordnet as wn

from nltk.tokenize import RegexpTokenizer



from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



# Any results you write to the current directory are saved as output.
str = 'He works in a bank. He is a man of words. he opened me a bank account.'

def sentTokenize(str):

    sentences = nltk.sent_tokenize(str)

    for sentence in sentences:

        print(sentence)

    return sentences
sentences = sentTokenize(str)

sentences




def removePunctuation(sentences):

    

    tt=""

    sentences2=[]

    for x in sentences:

        tokenizer = RegexpTokenizer('\w+')

        text2=tokenizer.tokenize(x)

       

        cnt=1

        for x2 in text2:

            if cnt==1:

                tt+=x2

                cnt=0

            else:

                tt+=" "+x2    

        sentences2.append(tt)

        tt=""

    return sentences2

    

puncRemoved=removePunctuation(sentences)

puncRemoved




def stopWords(sentences2):

    stop_words = set(stopwords.words("english"))

    context_tab=[]

    for sentence in sentences2:

        words = nltk.word_tokenize(sentence)

        without_stop_words = [word for word in words if not word in stop_words]

        context_tab.append(without_stop_words)

    return context_tab 

    

   
stopWordsRemoved =stopWords(puncRemoved)

stopWordsRemoved




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


def sentence_similarity(sentence1WOI,exampleSen):

    print('sentence1WOI ',sentence1WOI,'exampleSen ',exampleSen)

    exampleSentence=[exampleSen]

    examples=removePunctuation(exampleSentence)

    examples=stopWords(examples)

    examples=lemmatization(examples)

    examples=tagPos(examples)

    examples=wordOfInterest(examples)

    

    exsynset=[]



    for i in examples:

        for j in i:

            if len(wn.synsets(j[0], j[1])) != 0:

                exsynset.append(wn.synsets(j[0], j[1])[0])

    

   

    score, count = 0.0, 0

    score1, count1 = 0.0, 0

    # For each word in the first sentence

    for synset in sentence1WOI:

        print('synset ',synset)

        # Get the similarity value of the most similar word in the other sentence

        best_score = max([synset.path_similarity(ss)!=None for ss in exsynset])

 

        # Check that the similarity could have been computed

        if best_score is not None:

            score += best_score

            count += 1

 

    # Average the values

    score /= count

    

    for synset in exsynset:

        # Get the similarity value of the most similar word in the other sentence

        best_score = max([synset.path_similarity(ss)!=None for ss in sentence1WOI])

 

        # Check that the similarity could have been computed

        if best_score is not None:

            score1 += best_score

            count1 += 1

 

    # Average the values

    score1 /= count1

    return (score+score1)/2

    



    

    

def sentence_Similarity(sentence1WOI,exampleSen):

    

    exampleSentence=[exampleSen]

    examples=removePunctuation(exampleSentence)

    examples=stopWords(examples)

    examples=lemmatization(examples)

    examples=tagPos(examples)

    examples=wordOfInterest(examples)

    

    exsynset=[]



    for i in examples:

        for j in i:

            if len(wn.synsets(j[0], j[1])) != 0:

                exsynset.append(wn.synsets(j[0], j[1])[0])

    

   

    score, count = 0.0, 0

    score1, count1 = 0.0, 0

    # For each word in the first sentence

    for synset in sentence1WOI:

        print('synset ',synset)

        # Get the similarity value of the most similar word in the other sentence

        best_score = max([synset.path_similarity(ss)!=None for ss in exsynset])

 

        # Check that the similarity could have been computed

        if best_score is not None:

            score += best_score

            count += 1

 

    # Average the values

    score /= count

    

    for synset in exsynset:

        # Get the similarity value of the most similar word in the other sentence

        best_score = max([synset.path_similarity(ss)!=None for ss in sentence1WOI])

 

        # Check that the similarity could have been computed

        if best_score is not None:

            score1 += best_score

            count1 += 1

 

    # Average the values

    score1 /= count1

    return (score+score1)/2
woi
for aword in woi:

    print(aword)

    arr=[]

    cnt = 0

    for j in aword:

        arr.append(wn.synsets(j[0], j[1]))

    for senses in arr:

        for sense in senses:

            val = sentence_similarity(sentences[cnt],example)

            print(val)

    

    
def WSD1(woi):

    ##for each sentence i woi, arr = [synsets of the words]

    for aword in woi:

        print(aword)

        arr=[]

        for j in aword:

            arr.append(wn.synsets(j[0], j[1]))

        from itertools import product

        list1=arr[0]

        for a in range (1,len(arr)):

            print (a)

            list1 = [(a,b) for a, b in product(arr[a], list1) ]        

        print('Total combenation ',len(list1))

        c=0

        outcome=[]

        for i in list1:

            word_c=0

            probability=0.0

            for j in i:

                val=[]

                for e in j.examples():

                    ar=[]

                    for a in i:

                        ar.append(a)

                    

                    val.append(sentence_similarity(ar,e))

                    

                if len(val)!=0:

                    probability=probability+max(val)

                    word_c=word_c+1

                else:

                    word_c=word_c+1

            probability=probability/word_c

            #print(probability)

            outcome.append((i,probability))

        outcome = sorted(outcome, key=lambda tup: tup[1],reverse=True)

        #print(outcome)

        c=0

        for i in outcome:

            if i[1]==1:

                for p in i[0]:

                    print(p,p.definition())

                print('#########################')

                c=c+1

        print('**************************************************************************************',c,'**********')

        for i in outcome:

            if i[1]==1:

                hyponym.append(i[0][0])





    
WSD1([woi[0]])



hyponym

hypo=set()

for i in hyponym:

    hypo.add(i.hypernyms()[0])

print(hypo) 

context=[]

context.append(wn.synsets('open',pos='a')[0])

context.append(wn.synsets('account',pos='n')[0])

print(context[0])

p=0

for i in hypo:

    for s in context:

        a=i.path_similarity(s)

        b=s.path_similarity(i)

        if a is None:

            a=0

        if b is None:

            b=0

        a=a+b

        p=p+(a/2)

    print(i,i.definition(),p/2)
#path_similarity

sense=[]



for j in syns2:

    for i in syns1:

        if i.wup_similarity(j)!=None:

            p=int((j.wup_similarity(i)+i.wup_similarity(j))*100)

        else:

            p=int((j.wup_similarity(i))*100)

        tup=(j.definition(),i.definition(),p)

        sense.append(tup)





        
sense
sense = sorted(sense, key=lambda tup: tup[2],reverse=True)
sense
#each NODE

class Node:

    endmark = False

    

    context = None

    def __init__(self):

        self.nxt = []

        self.endmark = False

        for i in range(26):

            self.nxt.append(None)



##initiate tree root





##inserting each word in tree

def insert(root,string,length,context):

    curr = root

    print('\n')

    for i in range(length):

        print(string[i])

        index = ord(string[i]) - ord('a')

        if curr.nxt[index] is None:

            curr.nxt[index] = Node()

        curr = curr.nxt[index]

    curr.endmark = True

    curr.context = context

    print(context)



#insert('slope',5,'sloping land')



#search for a word

def search(root,string,length):

    curr = root

    print('root')

    for i in range(length):

        index = index = ord(string[i]) - ord('a')

        print('index: ',index)

        if curr.context is not None:

            print(curr.context)

        if curr.nxt[index] is None:

            return False

        curr = curr.nxt[index]

        print(string[i])

    print(curr.context)    

    return curr.endmark,curr.context



#search('slope',5)

    
##input to context table:

contextTable = [

['depository_financial_institution.n.01',['deposite','money','lend','credit','finance',

                          'institute','capital','market','reserve','payment','account']],

['bank.n.01',['slope','land','water','canoe','river','current','riverbank','riverside',

                 'waterside','incline','side','stream']],

['',['university','education','preschool','primary school']]]
print(contextTable)
root = Node()



for context in contextTable:

    for word in context[1]:

        insert(root,word,len(word),context[0])

        print(word,context[0])





    
a = search(root,'account',len('account'))

print(a[1])

print(wn.synset(a).definition())
rt = Node()

s = 'preschool'

search(root,s,len(s))
rt.nxt[17].context

for i in range(26):

    if root.nxt[i] is not None:

        print(i,root.nxt[i].context,rt.nxt[i].context)


for i in wn.synsets('car'):

    print(i,i.hypernyms())

    for c in i.hypernyms():

        print(c.hypernyms())

        

wn.synsets('vehicle')[0].hyponyms()

##        print(i, i.hyponyms())
