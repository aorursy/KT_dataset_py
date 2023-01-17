# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import nltk

import nltk.corpus

from nltk.tokenize import word_tokenize

from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample_submission = pd.read_csv("/kaggle/input/kaggledaysariana/Sample_Submission_kaggle.csv")

test = pd.read_csv("/kaggle/input/kaggledaysariana/test.csv")

df = pd.read_csv("/kaggle/input/kaggledaysariana/train.csv")

df1=df[0:2000]
df1.head()
df1.country.unique()
most_freq= [x for x in df1.country.value_counts().sort_values(ascending=False).head(10).index]

most_freq
#for c in most_freq:

    #df[c]=np.where(df["country"]==c,1,0)

#df[["country"]+most_freq].head(40) 
#df.drop(["country"],axis=1)
def one_hot_mostfreq(data,variable,mostfreq):

    for c in mostfreq:

        data[variable +"_" +c]=np.where(data[variable]== c , 1,0)

 #reread the data

df=pd.read_csv("/kaggle/input/kaggledaysariana/train.csv")

one_hot_mostfreq(df,"country",most_freq)

df.head(20)



#usecols=["description","country","points","price","province","designation","region_1","region_2","taster_name","taster_twitter_handle","title"])



        

        
most_freq= [x for x in df.province.value_counts().sort_values(ascending=False).head(10).index]

one_hot_mostfreq(df,"province",most_freq)

df.head(10)
df=df.drop(["country","province"],axis=1)

df.head(20)

desc =df1["description"]

desc
liste=[]

for c in desc:

    li= word_tokenize(c)

    liste.append(np.char.lower(li))

liste
#filtered_sentence = [w for w in liste if not w in stop_words]

stop_words = set(stopwords.words('english'))

filtered_sentence = []

for w in liste:

    lis=[]

    for item in w:

        if item not in stop_words:

            lis.append(item)

    filtered_sentence.append(lis)

              

filtered_sentence

  

symbols = "!,.;\"#$%&()*+-/,:;<=>?@'[\]^_`{|}~\n"

for c in symbols:

    for w in filtered_sentence:

        if c in w:

            w.remove(c)

filtered_sentence

        
for w in filtered_sentence:

    for i in w:

        if len(i)==1:

            w.remove(i)

filtered_sentence

        
from nltk.stem import PorterStemmer

from nltk.stem import LancasterStemmer

porter = PorterStemmer()

steemed_lis=[]

for w in filtered_sentence:

    li=[]

    for i in w:

        word=porter.stem(i)

        li.append(word)

    steemed_lis.append(li)

        

steemed_lis        

      

        
from nltk.stem import PorterStemmer

from nltk.stem import LancasterStemmer

porter = PorterStemmer()

steemed_lis=[]

for w in filtered_sentence:

    li=[]

    for i in w:

        word=porter.stem(i)

        li.append(word)

    steemed_lis.append(li)

        

steemed_lis        

      

        


#from num2words import num2words 

#for i in steemed_lis :

   # steemed_lis=num2words(i)

#steemed_lis    
DF = {}

for w in steemed_lis:

    li=[]

    for i in range(len(w)):

        tokens = steemed_lis[i]

        for x in tokens:

            try:

                DF[x].add(i)

            except:

                DF[x] = {i}
for i in DF:

    DF[i]=len(DF[i])

DF    

    
#ps = PorterStemmer()



#vectors=[]

#vectorizer = TfidfVectorizer()

#for c in liste:

    #vectors.append(vectorizer.fit_transform(c))

    #feature_names = vectorizer.get_feature_names()

    #dense = vectors.todense()

    #denselist = dense.tolist()

#dataf = pd.DataFrame(denselist, columns=feature_names)