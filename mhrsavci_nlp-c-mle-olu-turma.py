# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

# Örnek bir metin alındı.



passage="""Ülkeler eurbondlarını yurt dışı piyasalarda bulunan aracı kuruluşlar kanalıyla ihraç ederler bu aracı 

            kurumlar da eurobondları bankalara veya diğer finansal kurumlara pazarlar. Eurbondlara bu şekilde yatırım 

            yapan bankalar ve finansal kuruluşlar bu eurobondları kendi portföylerinde saklayabilir veya finansal 

            kuruluşlar kurumsal yatırmcılara veya fonlara satarak söz konusu eurrobondar için ikinci piyasa oluştururlar. 

            Sonuç olarak yurt dışındaki yatırımcı için ihraç edilen eurobondlar, eurobondu ihraç eden ülkelerde yaşayan 

            yatırımcılar tarafından ikincil piyasalardan satın alınabilir."""
wwlist=passage.split()



wwlist=[i.lower() for i in wwlist]
# "word" kelimesinin "passage" ifadesi içerisinde geçme oranı.



def Prob(word,passage): 

    wlist=passage.split()

    wlist=[i.lower() for i in wlist]

    sample_space=len(wlist)

    sample=passage.count(word)

    probability=(sample)/(sample_space)

    

    return probability



##### Prob("word",passage)
def getNGrams(wordlist,n):

    return [tuple(wordlist[i:i+n]) for i in range(len(wordlist)-(n-1))]



##### getNGrams(wwlist,2)
# "wlist" ile belirtilen kelimelerden sonra "word" kelimesinin "passage" ifadesine göre bakılarak koşullu durumdu.



def CProb(passage,wlist,word):

    

    uword=" ".join(wlist)

    

    upword=uword+" "+word

    

    p_AUB=Prob(upword,passage)

    

    p_B=Prob(uword,passage)

    

    try:

        s=(p_AUB)/(p_B)

    except ZeroDivisionError: 

        s=0

    return s
w2list=getNGrams(wwlist,2)

w22list=[]

for i in range(0,len(w2list)):

    k=" ".join(w2list[i])

    w22list.append(k)
wwlist=set(wwlist)

ProbList=[]

for i in wwlist:

    ProbList.append((i,Prob(i,passage)))

    

ProbList
wwlist=list(wwlist)

import numpy as np

CProbArray=np.zeros((len(wwlist),len(wwlist)))

for i in range(0,len(wwlist)):

    for j in range(0,len(wwlist)):

        CProbArray[i][j]=CProb(passage,[wwlist[j]],wwlist[i])

        

# CProbArray :  Sütun kelimelerinden sonra gelen satır kelimelerinin koşullu olasılığının tablosudur.
import pandas as pd



data=pd.DataFrame(CProbArray,columns=wwlist)



data.insert(0, "Kelimeler", wwlist,True)



data


def GetSentence(dataframe,wordlist):

    import random

    s=dataframe["Kelimeler"][dataframe[random.choice(wordlist)].idxmax()]

    string=""

    for i in range(15):

        s=dataframe["Kelimeler"][dataframe[s].idxmax()]

        string=string+" "+s

        

    return string

    



sentence=GetSentence(data,wwlist)

sentence