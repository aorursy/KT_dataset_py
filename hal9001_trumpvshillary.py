# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


debate_df = pd.read_csv("../input/debate.csv")

debate_df.head()
debate_df.tail()
debate_df.info()
#unique speakers from dataset

debate_df['Speaker'].unique()
#separating data for hillary alone

hillary = debate_df[debate_df.Speaker == 'Clinton']

hillary
#separating data for trump

trump = debate_df[debate_df.Speaker == 'Trump']

trump
#cross talk that occured between the candidates

cross_talk = debate_df[(debate_df.Speaker == 'CANDIDATES') & (debate_df.Date.str.match('2016-09-26')) ]

cross_talk
print("No of times hillary spoke on something", hillary.shape[0])

print ("No of times Trump spoke on something ", trump.shape[0])

print ("No of times there was cross talk between both ", cross_talk.shape[0])
# Who spoke the most number of times? 



%matplotlib inline



total = pd.concat([hillary,trump,cross_talk])

import seaborn as sns

sns.set(style="whitegrid")



sns.countplot(x="Speaker", data = total)
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')



hillary['num_words'] = hillary.apply(lambda row : len(tokenizer.tokenize(row['Text'])), axis = 1)

trump['num_words'] = trump.apply(lambda row : len(tokenizer.tokenize(row['Text'])), axis = 1)
hillary.head()
trump.head()
# Number of words spoken by Hillary and trump in continuos time.





from matplotlib import pyplot as plt

plt.figure(figsize=(14,8))



hillary['num_words'].plot(legend=True,color="blue",kind='line')

trump['num_words'].plot(legend = True,color="red",kind='line')
import matplotlib.pyplot as plt



tot_words = []

tot_words.append(hillary['num_words'].sum())

tot_words.append(trump['num_words'].sum())

labels = 'Hillary','Trump'

colors = ['blue', 'red']

explode = (0.1, 0)



plt.pie(tot_words, explode =explode,colors=colors, shadow=True, startangle=90, labels=labels, 

                         autopct='%.2f')

plt.legend(labels, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()
hillary_text = ' '.join(hillary['Text']).lower()

trump_text = ' '.join(trump['Text']).lower()
from nltk.corpus import stopwords

stop = list(stopwords.words('english'))

stop.extend(['would','well','us','know','going','look','think','said','say','get','many','one','years','want','like',

             'go','things','much','really','thing','way','tax','got','make','tell'])
trump_top_words = pd.Series([_ for _ in tokenizer.tokenize(trump_text.lower()) if _ not in stop]).value_counts()[:10]

trump_top_words
hillary_top_words = pd.Series([_ for _ in tokenizer.tokenize(hillary_text.lower()) if _ not in stop]).value_counts()[:10]

hillary_top_words
plt.figure(figsize=(14,8))

plt.xticks(fontsize=20)  

trump_top_words.plot(kind='bar', stacked=True, color='red')
plt.figure(figsize=(14,8))

plt.xticks(fontsize=20) 

hillary_top_words.plot(kind='bar', stacked=True, color='blue')