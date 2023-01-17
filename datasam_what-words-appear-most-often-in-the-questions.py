%matplotlib inline



import pandas as pd 

import seaborn as sns

data=pd.read_csv('../input/questions.csv')

df=data
df.head()
word_list=[]

for i in df['question1']:

    for k in i.split(' '):

        word_list.append(k.strip('?.,():/\\\'\"@$%£!^&*_+=\"').lower())                
from collections import Counter



word_count=Counter(word_list)

df1=pd.DataFrame.from_dict(word_count,orient='index').reset_index()

df1.columns=['word', 'count']
df1.head()
top50=df1.sort_values(by='count', ascending=False)[0:50]

top50.head()
top50[0:14].plot(x='word', kind='bar', colormap='Paired');
from nltk.corpus import stopwords

stop = stopwords.words('english')

stopl=[]

for i in stop:

    stopl.append(i.lower()) 
df1ns=df1[df1['word'].isin(stopl)==False]

top50ns=df1ns.sort_values(by='count', ascending=False)[0:49]
top50ns[0:14].plot(x='word',kind='bar');
top50ns[0:14]
top50ns=top50ns.drop(380);
top50ns[0:14].plot(x='word',kind='bar');