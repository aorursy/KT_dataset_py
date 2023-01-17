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
str = 'Bank is a financial institution.My friend works in a bank.He opened me a bank account.He even gave me a lift in his a car.'

sentences = nltk.sent_tokenize(str.lower())

for sentence in sentences:

    print(sentence)
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
w=woi[0][1][0]

tag=woi[0][1][1]



synsB=wn.synsets(w,pos=tag)

print(synsB)



w=woi[0][0][0]

tag=woi[0][0][1]



for i in wn.synsets(w,pos=tag):

    print(i,i.definition(),i.examples())


def sentence_similarity(sentence1WOI,exampleSen):

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
hyponymB=[]

hyponymW=[]
def WSD1(woi):

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

                hyponymB.append(i[0][0])

                hyponymW.append(i[0][1])

                





    
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

    print(outcome)

    print('\n')







                

WSD(woi)
outcome
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

    ##print('\n')

    for i in range(length):

       ## print(string[i])

        index = ord(string[i]) - ord('a')

        if curr.nxt[index] is None:

            curr.nxt[index] = Node()

        curr = curr.nxt[index]

    curr.endmark = True

    curr.context = context

    ##print(context)



#insert('slope',5,'sloping land')



#search for a word

def search(root,string,length):

    curr = root

    ##print('root')

    for i in range(length):

        index = index = ord(string[i]) - ord('a')

        #print('index: ',index)

        if curr.context is not None:

            print(curr.context)

        if curr.nxt[index] is None:

            return False

        curr = curr.nxt[index]

       ## print(string[i])

    ##print(curr.context)    

    return curr.context
##input to context table:

contextTable = [

['financial_institution.n.01',['deposite','money','lend','credit','finance',

                          'institute','capital','market','reserve','payment','account','institution']],

['vehicle.n.01',['car','engine','road','drive','wheel','lift']]

]
root = Node()



for context in contextTable:

    for word in context[1]:

        insert(root,word,len(word),context[0])

        





    
lineCount=0

for line in woi:

    wordCount=0

    for word in line:

        ok=0

        contextWord=[]

        for line1 in range(0,len(woi)):

            for word1 in range(0,len(woi[line1])):

                if word==woi[line1][word1]:

                    if(word1-1>=0):

                        contextWord.append(woi[line1][word1-1])

                    if (word1+1<len(woi[line1])):

                        contextWord.append(woi[line1][word1+1])

                    

        print(word,'\n',"Context words ==>>",contextWord)

        for w in contextWord:

            result=search(root,w[0],len(w[0]))

            if result!= False:

                if ok==0:

                    ok=1

                    maxx=-1

                    j=None

                    for sense in outcome [lineCount][wordCount]:

                        #print(sense[0].definition(),sense[1])

                        s=sense[0]

                        gs=wn.synset(result)

                        a=s.path_similarity(gs)

                        b=gs.path_similarity(s)

                        if a is None:

                            a=0

                        if b is None:

                            b=0

                        p=(a+b)/2

                       # 

                        p=p+sense[1]

                        p=p/2

                        print("***",sense[0],p)

                        if(p>maxx):

                            maxx=p

                            j=s

                    if j is not None:

                        print(">",j,j.definition()) 

                        print('\n')

        if(ok==0):

            if len(outcome[lineCount][wordCount])!= 0:

                #outcome[lineCount][wordCount][0][0].definition()

                print("==>",wn.synsets(woi[lineCount][wordCount][0],woi[lineCount][wordCount][1])[0].definition())

                print('\n')

                    

        wordCount=wordCount+1

    lineCount=lineCount+1

                

            

            
from nltk.corpus import wordnet

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords





class SimplifiedLesk:

    """Implement Lesk algorithm."""



    def __init__(self):

        self.stopwords = set(stopwords.words('english'))



    def disambiguate(self, word, sentence):

        """Return the best sense from wordnet for the word in given sentence.

        Args:

            word (string)       The word for which sense is to be found

            sentence (string)   The sentence containing the word

        """

        word_senses = wordnet.synsets(word)

        best_sense = word_senses[0]  # Assume that first sense is most freq.

        max_overlap = 0

        context = set(word_tokenize(sentence))

        for sense in word_senses:

            signature = self.tokenized_gloss(sense)

            overlap = self.compute_overlap(signature, context)

            if overlap > max_overlap:

                max_overlap = overlap

                best_sense = sense

        return best_sense



    def tokenized_gloss(self, sense):

        """Return set of token in gloss and examples"""

        tokens = set(word_tokenize(sense.definition()))

        for example in sense.examples():

            tokens.union(set(word_tokenize(example)))

        return tokens



    def compute_overlap(self, signature, context):

        """Returns the number of words in common between two sets.

        This overlap ignores function words or other words on a stop word list

        """

        gloss = signature.difference(self.stopwords)

        return len(gloss.intersection(context))





# Sample Driver code:



sentence = ("bank is a finantial institution"

            "my friend works in a bank"

            "he opened me a bank account"

            "he even gave me a lift in his a car"

           )

word = "bank"

lesk = SimplifiedLesk()

array=['bank','financial','institution','friend','work','open','account','even','give','lift','car']



for i in range(0,11):

    print ("Word :", array[i])

    print ("Best sense: ", lesk.disambiguate(array[i], sentence).definition())

    print ('')