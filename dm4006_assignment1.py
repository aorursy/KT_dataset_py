import nltk

import pandas as pd

import numpy

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.tokenize import RegexpTokenizer

from  collections import Counter

import re

import gc
file_0=open("../input/xaa",mode='r',encoding='latin-1')
file_content=file_0.read()

file_0.close()
print("file size is",len(file_content)/(1024*1024))

print("no of line in file is",len(file_content.split('\n')))

print("no of word in file is",len(file_content.replace('\n',' ').split(' ')))
temp = re.sub(r'[^a-zA-Z0-9\-\s]*', r'', file_content)

temp = re.sub(r'(\-|\s+)', ' ', temp)

del(file_content)

gc.collect()
token_nltk=nltk.word_tokenize(temp)

del(temp)

gc.collect()
two_gram=nltk.ngrams(token_nltk,2)
gm=Counter(two_gram)

token=gm.keys()

frequency=gm.values()
length_bigram=sum(frequency)

length_dist_bigram=len(token)
print("total number of  bi-gram ",length_bigram)

print("total number of distinct bi-gram ",length_dist_bigram)
two_gram=nltk.ngrams(token_nltk,2)

fdist_bigram = nltk.FreqDist(two_gram)
fdist_bigram.plot(10)
l=fdist_bigram.most_common(10)

x=[]

y=[]

for v1,v2 in l:

    x.append(str(v1))

    y.append(v2)

    

# x-coordinates of left sides of bars 

left =[1, 2, 3, 4, 5,6,7,8,9,10]



# heights of bars 

height = y 



# labels for bars 

tick_label =x



# plotting a bar chart 

plt.bar(left, height, tick_label = tick_label, color = ['red', 'green']) 

plt.xticks(left,tick_label,rotation=90)

# naming the x-axis 

plt.xlabel('token') 

# naming the y-axis 

plt.ylabel('frequency') 

# plot title 

plt.title('bi-gram frequency distribution') 



# function to show the plot 

plt.show() 

l_100=fdist_bigram.most_common(1000)

x_new=[]

y_new=[]

c=len(token_nltk)

i=1

for v1,v2 in l_100:

    x_new.append(i)

    y_new.append(v2*i/c)

    i=i+1

    

plt.plot(x_new,y_new)



plt.xlabel('token') 

# naming the y-axis 

plt.ylabel('value') 

# plot title 

plt.title('bi-gram rank vs rank*frequency') 



# function to show the plot 

plt.show() 
i=1

s=0

gram_80=int(length_bigram*0.8)+1

print("80% of the selected corpus is ",gram_80)

val=int(length_dist_bigram*.3)

for v1,v2 in fdist_bigram.most_common(val):

    s=s+v2

    if s>gram_80:

        break

    i=i+1 

    

print("no of token require to cover 80% of corpus is ",i)

print("and corpus size of {} token is {}".format(i,s))

print("{} %  tokens cover the 80% corpus".format(i/length_dist_bigram*100))
df=pd.DataFrame(columns=["bi-Gram","frequency"])
df['bi-Gram']=token

df['frequency']=frequency
df.head()
df.to_csv("bi-gram.csv")