import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
data = pd.read_csv("../input/globussoft-out-df/out.csv")
data.head()
data.info()
#data.plot()
#data['Message'] = data['Message'].astype("float64")
from collections import Counter
x=(data['Message'])
x=x.astype('str')
#getting rid of emojis..

import re

x = x.str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
z=x

z
#from nltk.tokenize import word_tokenize 

from nltk.corpus import stopwords

stop = stopwords.words('english')
#getting rid of stopwords.

clean_text = z.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
clean_text.head(10)
# define string

string = clean_text.to_string()

word = ['sell','advertise','auction','handle','hawk','market','bargain','barter','boost','contract','dispose','drum','dump','exchange','hustle','merchandise','persuade','pitch','plug','push','retail','stock','traffic','vend','wholesale','buy','acquisition','bargain','investment','purchase','closeout','deal','steal','value','good deal']

for i in word:

    substring = i

    count = string.count(substring)

    print("The count is:", i, count)
string = clean_text.to_string()

word = ['money',"Money",'sell','want',]

substring = 'money' and 'want' 

count = string.count(substring)

print("The count is:", count)
buy=[]

sell=[]

data = clean_text

for i in data:

    str1= ''.join(i)

    for r in str1.split():

        if 'buy' == r:

            buy.append(i)

        if 'purchase' == r:

            buy.append(i)

        if 'investment' == r:

            buy.append(i)

        if 'obtain' == r:

            buy.append(i)

        if 'purchase' == r:

            buy.append(i)

        if 'sell' == r:

            sell.append(i)      

        if 'wholesale' == r:

            sell.append(i)

        if 'merchandise' == r:

            sell.append(i)

        if 'advertise' == r:

            sell.append(i)

        if 'stock' == r:

            sell.append(i)

bux=len(buy)

selx=len(sell)

totx=len(clean_text)

print('buy text =',bux/totx*100,'%')

print('sell text =',selx/totx*100,'%')
print(len(buy))

buy = pd.DataFrame(buy)

buy.to_csv('buy_text_imp.csv')
print(len(sell))

sell = pd.DataFrame(sell)

sell.to_csv('sell_text_imp.csv')