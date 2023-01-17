import numpy as np
import pandas as pd 
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import re

%matplotlib inline
pd.options.display.max_colwidth = 1000 #to print complet verses
index=pd.read_csv('../input/bible_version_key.csv')
#Drop the columns where at least one element is missing.
index=index.dropna(axis='columns')
index
#American Standard-ASV1901
asv = pd.read_csv('../input/t_asv.csv')

#Bible in Basic English
bbe = pd.read_csv('../input/t_bbe.csv')

#Darby English Bible
dby = pd.read_csv('../input/t_dby.csv',encoding='latin-1')

#King James Version
kjv = pd.read_csv('../input/t_kjv.csv')

#Webster's Bible
wbt = pd.read_csv('../input/t_wbt.csv')

#World English Bible
web = pd.read_csv('../input/t_web.csv')

#Young's Literal Translation
ylt = pd.read_csv('../input/t_ylt.csv')
#Find verses containing "LOVE". 
love=asv[asv['t'].str.contains('love',case=False)]
sel=np.random.randint(1,love.shape[0])
print("Verse Number:",love['b'].iloc[sel],love['c'].iloc[sel])
print(love['t'].iloc[sel])
#Find verses containing "christ"
chri=asv[asv['t'].str.contains('christ',case=False)]
sel=np.random.randint(1,chri.shape[0])
print("Verse Number:",chri['b'].iloc[sel],chri['c'].iloc[sel])
print(chri['t'].iloc[sel])
ct=asv.groupby(['b'])['t'].count()
plt.bar(range(1,67),ct)
counts = dict()
for text in asv['t']:
    tokens=text.lower().split()
    tokens=[re.sub(r'[^\w\s]','',i) for i in tokens]
    for i in tokens: 
        if i in counts:
            counts[i]+=1
        else:
            counts[i]=1
sorted_counts = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
print("10 most common words:\nWord\tCount")
for word, count in sorted_counts[:10]:
    print("{}\t{}".format(word, count))

print("\n10 least common words:\nWord\tCount")
for word, count in sorted_counts[-10:]:
    print("{}\t{}".format(word, count))
for text in asv['t']:
    sentence=list(map(str.strip, re.split(r"[.?](?!$.)", text)))[:-1]
    for sent in sentence:
        list(map(str.strip, 
                       re.split("(?:(?:[^a-zA-Z]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)",sent)))
#Book index for Corinthians, Chapter and verse number
b,c,vn=46,13,4

diff=pd.DataFrame(index['version'])
ver=[asv,bbe,dby,kjv,wbt,web,ylt]
for i,v in enumerate(ver):
    diff.loc[[i],'verse'] =v[(v['b']==b) & (v['c']==c) &(v['v']==vn)]['t'].values
    
diff
def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
#Example: get certain verse without index 
asv.loc[[0],'t'].to_string(index=False)
#Compare two verses
a=diff.loc[[0],'verse'].to_string(index=False)
b=diff.loc[[4],'verse'].to_string(index=False)
get_jaccard_sim(a,b)
#Metric Matrix
jac=pd.DataFrame(index=range(7))
for it in range(7):
    jac[it]=[get_jaccard_sim(diff.loc[[it],'verse'].to_string(index=False),
                          diff.loc[[i],'verse'].to_string(index=False)) for i in range(7)]

sns.heatmap(jac, annot=True)

#Compare two books!
def com_book(b1,b2):
    if b1.shape[0]==b2.shape[0]:
        sim=[]
        for i in range(b1.shape[0]):
            a=b1.loc[[i],'t'].to_string(index=False)
            b=b2.loc[[i],'t'].to_string(index=False)    
            sim.append(get_jaccard_sim(a,b))
        return np.mean(sim)
    else:
        #print("Lengths differ. Something is wrong in the dataset :(")
        return np.nan
com_book(asv,bbe)
com_book(asv,dby)
#DataFrame Setup
ver=["asv","bbe","dby","kjv","wbt","web","ylt"]
jacsim=pd.DataFrame(index=ver)
for i in ver:
    jacsim[i]=np.nan
#Calculate Jaccard Similarity of any of the two versions.
#Could be optimized by calculating (i,j) and (j,i) once.
ver=[asv,bbe,dby,kjv,wbt,web,ylt]
for i in range(7):
    for j in range(7):
        jacsim.iloc[i,j]=com_book(ver[i],ver[j])
#sns.heatmap(jacsim, annot=True)
