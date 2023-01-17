# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import nltk
import pandas as pd
from nltk.stem.porter import PorterStemmer
import re
import random
from itertools import groupby
import numpy as np
#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/Tweets.csv")
#data.head()
data.loc[data["airline_sentiment"]=="positive","airline_sentiment"]=4
data.loc[data["airline_sentiment"]=="neutral","airline_sentiment"]=2
data.loc[data["airline_sentiment"]=="negative","airline_sentiment"]=0
data["tweet_created"].describe()
date=[]
time=[]
for e in data["tweet_created"]:
    lst=e.strip().split(" ")
    date.append(lst[0])
    time.append(lst[1])

data["date"]=date
data["time"]=time
def wordcount(text):
    p=PorterStemmer()
    text2=re.sub(r'^[A-Za-z @]',"",text)
    #print(text2)
    lst=text2.strip().split(" ")
    s=""
    for e in lst:
        try:
            index=stopwords.index(e.lower())
        except:
            
            s=s+" "+(e.lower())
            
    tr=s.strip().split()
    s=""
    for ele in tr:
        try:
            s=s+" "+p.stem(ele)
        except:
            continue
   
    #s2=re.sub(r'^[A-Za-z0-9@ ]','',s)
    
    lst1=s.strip().split(" ")
    
    st1=set(lst1)
    lst2=[]
    for el in st1:
        c=lst1.count(el)
        lst2.append((el,c))
    return lst2
wc=[]
for e in data["text"]:
    wc.append(wordcount(e))
#wc[0:10]
data["wordcount"]=wc
#print(wc[0:2])
columns=["airline","wordcount","date","airline_sentiment_confidence","airline_sentiment"]
data=data[columns]
date_unique=data["date"].unique()
for i in range(len(date_unique)):
    data.loc[data["date"]==date_unique[i],"date"]=i
#data["date"].unique()
#virginamerica,americanair,unit,southwestair,jetblu,usairway
x=data["airline"].unique()
flight_tokens=["virginamerica","americanair","unit","southwestair","jetblu","usairway"]
split_ratio=0.7
length=len(data)
len_train=int(0.7*length)
len_test=int(length-len_train)
train_index=[]
i=0
train=[]
train_df=pd.DataFrame(columns=columns)
test=data
while(i<len_train):
    index1=random.randint(0,length)
    flag=1
    try:
        g=train_index.index(index1)
        flag=0
    except:
        train_index.append(index1)
        i=i+1
        #k=data.iloc[index1]
        #h=[k[i] for i in range(len(k))]
        #train.append(h)
        train_df=train_df.append(data[index1:index1+1])
        test1=(test[0:index1]).append(test[index1+1:])
        test=test1
ind1=range(len_test)

train_df2=train_df.reset_index(drop=True)
train_df=train_df2
test_df=test.reset_index(drop=True)
train_df.head()
test_df.head()
train_pos=train_df[train_df["airline_sentiment"]==4]
train_neu=train_df[train_df["airline_sentiment"]==2]
train_neg=train_df[train_df["airline_sentiment"]==0]
#print(len(train_pos),"pos")
#print(len(train_neu),"neu")
#print(len(train_neg),"neg")
global prob_airline_pos,prob_airline_neg,prob_airline_neu,airline_lst
airline_lst=list(x)
prob_airline_pos=[]
prob_airline_neg=[]
prob_airline_neu=[]
for el in airline_lst:
    c=len(train_pos[train_pos["airline"]==el])
    #print(c)
    prob_airline_pos.append(c/(len(train_pos)*1.0))
    c=len(train_neu[train_neu["airline"]==el])
    #print(c)
    prob_airline_neu.append(c/(len(train_neu)*1.0))
    c=len(train_neg[train_neg["airline"]==el])
    #print(c)
    prob_airline_neg.append(c/(len(train_neg)*1.0))
pos_wordcount=[]
neg_wordcount=[]
neu_wordcount=[]
for element in train_pos["wordcount"]:
    for (k,v) in element:
        pos_wordcount.append((k,v))
for element in train_neg["wordcount"]:
    for (k,v) in element:
        neg_wordcount.append((k,v))
for element in train_neu["wordcount"]:
    for (k,v) in element:
        neu_wordcount.append((k,v))
pos_wordcount.sort()
neg_wordcount.sort()
neu_wordcount.sort()
pos_wc=[]
pos_wc_word=[]
pos_wc_count=[]
neg_wc=[]
neg_wc_word=[]
neg_wc_count=[]
neu_wc=[]
neu_wc_word=[]
neu_wc_count=[]

for key, group in groupby(pos_wordcount, lambda x: x[0]):
    val=0
    for thing in group:
        val=val+thing[1]
    pos_wc.append((key,val))
    pos_wc_word.append(key)
    pos_wc_count.append(val)
for key, group in groupby(neg_wordcount, lambda x: x[0]):
    val=0
    for thing in group:
        val=val+thing[1]
    neg_wc.append((key,val))
    neg_wc_word.append(key)
    neg_wc_count.append(val)
for key, group in groupby(neu_wordcount, lambda x: x[0]):
    val=0
    for thing in group:
        val=val+thing[1]
    neu_wc.append((key,val))   
    neu_wc_word.append(key)
    neu_wc_count.append(val)

global date_airline_pos,date_airline_neg,date_airline_neu
date_airline_pos=[]
date_airline_neg=[]
date_airline_neu=[]
#print(train_pos["date"].unique())
for airline in airline_lst:
    al=train_pos[train_pos["airline"]==airline]
    num=[]
    for date in range(9):
        #print(date)
        k=al[al["date"]==date]
        num.append(len(k))
    date_airline_pos.append(num)
for airline in airline_lst:
    al=train_neg[train_neg["airline"]==airline]
    num=[]
    for date in range(9):
        #print(date)
        k=al[al["date"]==date]
        num.append(len(k))
    date_airline_neg.append(num)
for airline in airline_lst:
    al=train_neu[train_neu["airline"]==airline]
    num=[]
    for date in range(9):
        #print(date)
        k=al[al["date"]==date]
        num.append(len(k))
    date_airline_neu.append(num)
# words common to wordcounts of the three categories (pos, neg, neu)
global common_set
common_set=[]
st_pos=set(pos_wc_word)
st_neg=set(neg_wc_word)
st_neu=set(neu_wc_word)
common_set=list(st_pos.intersection(st_neg.intersection(st_neu)))
num_pos=sum([v for (k,v) in pos_wc])
num_neg=sum([v for (k,v) in neg_wc])
num_neu=sum([v for (k,v) in neu_wc])
print(num_pos,num_neg,num_neu)


def predict(wordcount,airline,date):
    global prob_airline_pos,prob_airline_neg,prob_airline_neu,airline_lst
    global date_airline_pos,date_airline_neg,date_airline_neu,common_set
    total_words=sum([v for (k,v) in wordcount])
    pos_score=1
    neg_score=1
    neu_score=1
    airline_index=airline_lst.index(airline)
    pos_score1=date_airline_pos[airline_index][date]/(len(train_pos)*1.0)
    neg_score1=date_airline_neg[airline_index][date]/(len(train_neg)*1.0)
    neu_score1=date_airline_neu[airline_index][date]/(len(train_neu)*1.0)
    pos_score2=pos_score*prob_airline_pos[airline_index]
    neg_score2=neg_score*prob_airline_neg[airline_index]
    neu_score2=neu_score*prob_airline_neu[airline_index]
    #number of words that match
    pos_match=0
    neg_match=0
    neu_match=0
    for (token,val) in wordcount:
        try:
            jj=common_set.index(token)
        except:
            try:
                index=pos_wc_word.index(token)
                pos_score=pos_score*(pos_wc_count[index]/(num_pos*1.0))**(val) #--[1]
                pos_match=pos_match+1
            except:
                continue
    for (token,val) in wordcount:
        try:
            jj=common_set.index(token)
        except:
            try:
                index=neg_wc_word.index(token)
                neg_score=neg_score*(neg_wc_count[index]/(num_neg*1.0))**(val) #--[1]
                neg_match=neg_match+1
            except:
                continue
    for (token,val) in wordcount:
        try:
            jj=common_set.index(token)
        except:           
           
            try:
                index=neu_wc_word.index(token)
                neu_score=neu_score*(neu_wc_count[index]/(num_neu*1.0))**(val) #--[1]
                neu_match=neu_match+1
            except:
                continue
    
    #pos_score=pos_score*pos_match#*1.0/total_words
    #neg_score=neg_score*neg_match#*1.0/total_words
    #neu_score=neu_score*neu_match#*1.0/total_words
    lst=[pos_score,neg_score,neu_score]
    max_1=min(lst) #minimum since score is multiplied by power of probability --see [1]-- hence lower score better prediction
    index=lst.index(max_1)
    pred=-1
    if(index==0):
        pred=4
    elif(index==1):
        pred=0
    else:
        pred=2
    return pred
predictions=[]
for i in range(len(test_df)):
    predictions.append(predict(test_df["wordcount"][i],test_df["airline"][i],test_df["date"][i]))
acc=sum([int(test_df["airline_sentiment"][j]==predictions[j]) for j in range(len(predictions))])
accuracy=acc/(len(predictions)*1.0)
print(accuracy)
acc=sum([int(test_df["airline_sentiment"][j]==0) for j in range(len(test_df))])
accuracy=acc/(len(test_df)*1.0)
print(accuracy)
test_df["pred"]=predictions
categories=[4,0,2]
con_mat=[["label","pred-positive","pred-negative","pred-neutral"]]
test_pos=test_df[test_df["airline_sentiment"]==4]
test_neg=test_df[test_df["airline_sentiment"]==0]
test_neu=test_df[test_df["airline_sentiment"]==2]
num=np.zeros(len(categories))
test_pos=test_pos.reset_index(drop=True)
test_neg=test_neg.reset_index(drop=True)
test_neu=test_neu.reset_index(drop=True)
for i in range(len(test_pos)):
    if(test_pos["pred"][i]==4):
        num[0]=num[0]+1
    elif(test_pos["pred"][i]==0):
        num[1]=num[1]+1
    else:
        num[2]=num[2]+1
con_mat.append(["positive",num[0],num[1],num[2]])

num=np.zeros(len(categories))
for i in range(len(test_neg)):
    if(test_neg["pred"][i]==4):
        num[0]=num[0]+1
    elif(test_neg["pred"][i]==0):
        num[1]=num[1]+1
    else:
        num[2]=num[2]+1
con_mat.append(["negative",num[0],num[1],num[2]])

num=np.zeros(len(categories))
for i in range(len(test_neu)):
    if(test_neu["pred"][i]==4):
        num[0]=num[0]+1
    elif(test_neu["pred"][i]==0):
        num[1]=num[1]+1
    else:
        num[2]=num[2]+1
con_mat.append(["neutral",num[0],num[1],num[2]])  
for r in con_mat:
    print(r)
