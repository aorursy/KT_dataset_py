"""

@author: Mohamed Wael Bishr

"""

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from textblob import TextBlob

import nltk



from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

import collections

from collections import Counter 

from wordcloud import WordCloud, STOPWORDS 

import warnings



#Downloading wordbags from nltk



nltk.download('stopwords')

nltk.download('punkt')



# ignore warnings

warnings.filterwarnings("ignore")
S =[]

l=[]

num=0

All=[]

sent=0.0

Pos=0

Neg=0

Nat=0

Words=''

word_counter=0

POS_LST=[]

NEG_LST=[]

NAT_LST=[]

CloudPOS='';CloudNEG='';CloudNAT=''

CLOUDS=[CloudPOS,CloudNEG,CloudNAT]

df = pd.read_csv('/kaggle/input/qurancsv/Quran.csv')
print(plt.style.available) # look at available plot styles

plt.style.use('ggplot')

df.describe()
data= df.drop(columns=['OrignalArabicText', 'ArabicText','ArabicWordCount','ArabicLetterCount'],axis=1)
for i in data['SurahNo']:

    if i not in l:

        l.append(i)



for x in range(1,len(l)+1):

    for m in data['SurahNo']:

        if m == x:

            num +=1

    S.append([x,num])

    num=0

        

for j in range(len(S)):

    surah = S[j][0]

    aya =   S[j][1]

    ayas = []

    for a in range(len(data['SurahNo'])):

        if data['SurahNo'][a] == surah:

            ayas.append(data['EnglishTranslation'][a])

    All.append([surah,ayas])

    ayas=[]
for s in All:

    for a in s[1]:

        obj = TextBlob(a)

        sent = sent + obj.sentiment.polarity

        Words+=''+a

    result = sent/(len(s[1]))*100

    if result > 0:

        Pos+=1

    elif result < 0:

        Neg+=1

    else:

        Nat+=1

    sent=0    

print(f'''

          ==========================

          Happy Surahs   => {Pos}

          Sad Surahs     => {Neg}

          Natural Surhas => {Nat}

          ==========================

          ''')
stoplist = set(stopwords.words('english')) 

  

word_tokens = word_tokenize(Words) 

  

filtered_sentence = [w for w in word_tokens if not w in stoplist] 



import string

cleanList = []

string.punctuation

for i in filtered_sentence:

    cleanList.append(i.translate(str.maketrans('','',string.punctuation)))

    

for i in cleanList:

    if i=='':

        cleanList.remove(i)

        



# Pass the split_it list to instance of Counter class. 


for word in cleanList:

    obj = TextBlob(word)

    sent = obj.sentiment.polarity

    if sent >0: POS_LST.append(word)

    elif sent < 0 : NEG_LST.append(word)

    else: NAT_LST.append(word)

  

 



C_POS = Counter(POS_LST)

most_occurPOS = C_POS.most_common(50)

dfPOS = pd.DataFrame(most_occurPOS, columns = ['Word', 'Count'])

dfPOS.plot.bar(x='Word',y='Count')

C_NEG = Counter(NEG_LST)

most_occurNEG = C_NEG.most_common(50)

dfNEG = pd.DataFrame(most_occurNEG, columns = ['Word', 'Count'])

dfNEG.plot.bar(x='Word',y='Count')

C_NAT = Counter(NAT_LST)

most_occurNAT = C_NAT.most_common(50)

dfNAT = pd.DataFrame(most_occurNAT, columns = ['Word', 'Count'])

dfNAT.plot.bar(x='Word',y='Count')

# most_common() produces k frequently encountered 

# input values and their respective counts. 
print(f'''Pos: {len(POS_LST)} [{len(POS_LST)/len(word_tokens)*100}%]

          Neg: {len(NEG_LST)} [{len(NEG_LST)/len(word_tokens)*100}%] 

          Nat: {len(NAT_LST)} [{len(NAT_LST)/len(word_tokens)*100}%]''')
for word in POS_LST:

    CloudPOS +=word+' '

    

for word in NEG_LST:

    CloudNEG +=word+' '

    

for word in NAT_LST:

    CloudNAT +=word+' '

wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = stopwords.words(), min_font_size = 10).generate(CloudPOS) 

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 
wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = stopwords.words(), min_font_size = 10).generate(CloudNEG) 

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 
wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = stopwords.words(), min_font_size = 10).generate(CloudNAT) 

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 