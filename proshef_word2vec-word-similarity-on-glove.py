import numpy as np

import pandas as pd

import keras

import re

from keras.layers import Embedding, Flatten, Dense

from sklearn.metrics.pairwise import cosine_similarity

!pip install glove_python

import nltk

import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectFromModel

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict

from collections import Counter 

from gensim.models import Word2Vec

from gensim.models import FastText
from glove import Corpus, Glove

from sklearn.metrics.pairwise import cosine_similarity
!ls /kaggle/input/songsdata
# read data

data= pd.read_csv("/kaggle/input/songsdata/Indian_songs.csv" )



data.head()
doc=[]

for i in range(data.shape[0]):

    doc.append(nltk.word_tokenize(re.sub('[^a-zA-z\s]','',data['songLyrics'][i].lower())))
len(doc)
train=[]

strr=''

for i in doc:

    srr=""

    for j in i:

        srr=srr+' '+j

        strr=strr+' '+j

    train.append(srr)
def preprocess_lyrics():

    count_vec1=CountVectorizer()

    features1=count_vec1.fit_transform(train)

  

    #stemming

    stem3=['ing','ana','yan','ane','eya','ies','aaa','aan','ake']

    stem2=['on','ai','na','an','un','ey','en','se','aa']

    stem1=['a','e','i','o','u','y','n','s']

    dictt={}

    for i in count_vec1.get_feature_names():

        dictt[i]=i

    

    allwords=[]

    for i in dictt:

        allwords.append(i)





    for i in range(len(allwords)):

        j=i+1

        while j<len(allwords):

            if len(allwords[i])>3 and (len(allwords[j])-len(allwords[i])==3) and allwords[i]==allwords[j][0:-3] and (allwords[j][-3:] in stem3) and dictt[allwords[j]]==allwords[j]:

                dictt[allwords[j]]=dictt[allwords[i]]

#                 print(allwords[j],dictt[allwords[i]])



            elif len(allwords[i])>3 and (len(allwords[j])-len(allwords[i])==2) and allwords[i]==allwords[j][0:-2] and (allwords[j][-2:] in stem2) and dictt[allwords[j]]==allwords[j]:

                dictt[allwords[j]]=dictt[allwords[i]]

#                 print(allwords[j],dictt[allwords[i]])





            elif len(allwords[i])>3 and (len(allwords[j])-len(allwords[i])==1) and allwords[i]==allwords[j][0:-1] and (allwords[j][-1] in stem1) and dictt[allwords[j]]==allwords[j]:

                dictt[allwords[j]]=dictt[allwords[i]]

#                 print(allwords[j],dictt[allwords[i]])

            j=j+1



    afterstemming=[]

    for i in dictt:

        afterstemming.append(dictt[i])

    afterstemming=set(afterstemming)

    print('length after stemming', len(afterstemming))

    

    stemdictt={}

    for i in dictt:

        stemdictt[dictt[i]]=dictt[i]

    

    #spelling errors

    for i in afterstemming:

        if 'aa' in i:

            word=i.replace('aa','a')

            if word in afterstemming:

                stemdictt[i]=word



        if 'ee' in i:

            word=i.replace('ee','i')

            if word in afterstemming:

                stemdictt[i]=word



        if 'w' in i:

            word=i.replace('w','v')

            if word in afterstemming:

                stemdictt[i]=word



        if 'j' in i:

            word=i.replace('j','z')

            if word in afterstemming:

                stemdictt[i]=word



        if 'oo' in i:

            word=i.replace('oo','u')

            if word in afterstemming:

                stemdictt[i]=word

#         if stemdictt[i]!=i:

#             print(i, stemdictt[i])

            

    afterspell=[]

    

    for i in stemdictt:

        afterspell.append(stemdictt[i])

    afterspell=set(afterspell)

    print('length after spell', len(afterspell))

    

    

    for i in dictt:

        dictt[i]=stemdictt[dictt[i]]

    return dictt
vocab_d=preprocess_lyrics()

print(len(vocab_d))

vocab_d['chhori']='chhori'

vocab_d['munda']='munda'

vocab_d['mundey']='mundey'

vocab_d['chhora']='chhora'
docp=[]

for i in range(len(doc)):

    k=[]

    for j in range(len(doc[i])):

        if doc[i][j] in vocab_d:

            k.append(vocab_d[doc[i][j]])

        else:

            k.append(doc[i][j])

#         print(vocab_d[doc[i][j]])

    docp.append(k)
corpus = Corpus() 



#Training the corpus to generate the co occurence matrix which is used in GloVe

corpus.fit(docp, window=15)



glove = Glove(no_components=2, learning_rate=0.001) 

glove.fit(corpus.matrix, epochs=10, no_threads=4, verbose=True)

glove.add_dictionary(corpus.dictionary)

glove.save('glove.model')
female_names=['ladki','girl','gori','lady','kudi','chhori','woman']

male_names=['ladka','munda','mundey','boy','chokra','chhora','man']



color =['sanwali','saanwala','pink','pinky','red','laal','kaala','kaali','gori','gora','white','black','yellow','brown']

softAttitude=['bholi','bhola','komalkomal','heeran','hirni','nadaan','beautiful','mastani','mastana','seedhi','seedha','sharmili','sharmeela','sohni','sohna','bhali','bhala']

strongAttitude=['kukkad','bigda','bigdi','khatra','khauf','handa','jungli','badmash','gussa']

cars=['car', 'gaddi','drive','lamborghini','jaguar','gaadi','motorcycle']

clothes=['jeans','skirt','shirt','lehnga','chunni','ainak','ghagra','kurta','pajama','jacket','choodi','jhumka','chasma','chashma','kangan','top']

food=['namkeen','mithi','tikhi','teekha','khatti','makkhan','sweet','nimbu','imli','mitthe','rasmalai','mirchi','mishti','naariyal']

alcohol=['daaru','whisky','daru','pila','botal','peg','shots','drink','peeta']

bodylooks=['choti','chota','cheeks','adayein','thumka','aankhen','aankhein','nazron','charming']



att1=[color,softAttitude,strongAttitude,cars, clothes,food,alcohol,bodylooks]



attp=[]

for i in range(len(att1)):

    k=[]

    for j in range(len(att1[i])):

        if att1[i][j] in vocab_d:

            k.append(vocab_d[att1[i][j]])

        else:

            k.append(att1[i][j])

    attp.append(list(set(k)))

color =attp[0]

softAttitude=attp[1]

strongAttitude=attp[2]

cars=attp[3]

clothes=attp[4]

food=attp[5]

alcohol=attp[6]

bodylooks=attp[7]

att1=[color,softAttitude,strongAttitude,cars, clothes,food,alcohol,bodylooks]
def unit_vector(vec):

    """

    Returns unit vector

    """

    return vec / np.linalg.norm(vec)



def cosine_similarity(v1, v2):

    """

    Returns cosine of the angle between two vectors

    """

    v1_u = unit_vector(v1)

    v2_u = unit_vector(v2)

    return np.clip(np.tensordot(v1_u, v2_u, axes=(-1, -1)), -1.0, 1.0)
vec1 = glove.word_vectors[glove.dictionary['ladki']] 

vec2 = glove.word_vectors[glove.dictionary['ladka']] 



print(cosine_similarity((vec1), vec2))


def weat_test(target_one,target_two, target_one_words, attribute_one,attribute_two, attribute_one_words, target_two_words, attribute_two_words):

    cos=[]

    s=0

    s1=[]

    s2=[]

    S=[]

    n=0

        

    for i in range(0, len(target_one_words)):

            c1=[]

            c2=[]

            for k in range(0, len(attribute_one_words)):

                wt = target_one_words[i]

                at1 = attribute_one_words[k]

                try:

                    vec1 = glove.word_vectors[glove.dictionary[wt]]

                    vec2 = glove.word_vectors[glove.dictionary[at1]]

                    cos1 = cosine_similarity(vec1, vec2)

                    cos.append(cos1)

                    c1.append(cos1)

                except:

                    cos1=0

                    cos.append(cos1)

                    c1.append(cos1)

                    continue

            for k in range(0, len(attribute_two_words)):

                cos2=0

                wt = target_one_words[i]

                at2 = attribute_two_words[k]

                try:

                    vec1 = glove.word_vectors[glove.dictionary[wt]]

                    vec2 = glove.word_vectors[glove.dictionary[at2]]

                    cos2 = cosine_similarity(vec1, vec2)

                    cos.append(cos2)

                    c2.append(cos2)

                except:

                    cos2=0

                    cos.append(cos2)

                    c2.append(cos2)

                    continue

            s1.append((np.mean(c1)-np.mean(c2)))

            S.append((np.mean(c1)-np.mean(c2)))

            n=n+1

    for i in range(0, len(target_two_words)):

            c1=[]

            c2=[]

            for k in range(0, len(attribute_one_words)):

                wt = target_two_words[i]

                at1 = attribute_one_words[k]

                try:

                    vec1 = glove.word_vectors[glove.dictionary[wt]]

                    vec2 = glove.word_vectors[glove.dictionary[at1]]

                    cos1 = cosine_similarity(vec1, vec2)

                    cos.append(cos1)

                    c1.append(cos1)

                except:

                    cos1=0

                    cos.append(cos1)

                    c1.append(cos1)

                    continue

            for k in range(0, len(attribute_two_words)):

                cos2=0

                wt = target_two_words[i]

                at2 = attribute_two_words[k]

                try:

                    vec1 = glove.word_vectors[glove.dictionary[wt]]

                    vec2 = glove.word_vectors[glove.dictionary[at2]]

                    cos2 = cosine_similarity(vec1, vec2)

                    cos.append(cos2)

                    c2.append(cos2)

                except:

                    cos2=0

                    cos.append(cos2)

                    c2.append(cos2)

                    continue

            s2.append((np.mean(c1)-np.mean(c2)))

            S.append((np.mean(c1)-np.mean(c2)))

    s=np.sum(s1)-np.sum(s2)

    stdev=np.std(S)

    print(target_one + ' vs ' + target_two  + ' , ' +attribute_one + ' vs ' + attribute_two +', d = ' + str(s/(stdev*n)))

def avg_similarity(target,attribute):

   

    S=[]

    

    for i in range(0, len(target)):

        

#         cos.append(model.similarity(target[i],attribute))

#             c1=[]

        maxlist=[]    

    

        vec1 = glove.word_vectors[glove.dictionary[target[i]]] 

        

        for k in range(0, len(attribute)):

            

            vec2 = glove.word_vectors[glove.dictionary[attribute[k]]] 

            maxlist.append(cosine_similarity(vec1, vec2))

            

        maxlist.sort(reverse=True)

        

        for j in range(0,4):

            S.append(maxlist[j])

            

        ans= np.array(S).sum()/(len(target)*4)

    return ans

print("Weat test results for [] are ")

print(weat_test('female_names','male_names', female_names, 'softAttitude' ,'strongAttitude', softAttitude, male_names, strongAttitude))

att=['color','softAttitude','strongAttitude','cars','clothes','food','alcohol','bodylooks']

att1=[color,softAttitude,strongAttitude,cars, clothes,food,alcohol,bodylooks]

female_score=[]

male_score=[]



for i in att1:

    

    female_score.append(avg_similarity(female_names,i))

    male_score.append(avg_similarity(male_names,i))      

    

avg_sim=pd.DataFrame({'attribute':att,'female_names':female_score,'male_names':male_score})



print(avg_sim)

