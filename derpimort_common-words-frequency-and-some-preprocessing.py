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
import os

import pandas as pd

import numpy as np

import sys



DATA_DIR="/kaggle/input/english-sentences-with-noise/"



class ProgressBar():

	def __init__(self,n_files):

		self.total_files=n_files

		self.curr_counter=0

		self.multiplier=0

	def increment(self):

		self.curr_counter+=1

		self.multiplier=(self.curr_counter*30)//self.total_files

		sys.stdout.write('\r')

		sys.stdout.write("[%-30s] %d/%d" % ('='*self.multiplier, self.curr_counter,self.total_files))

		sys.stdout.flush()

		
!pip install pandarallel
clean=None

noisy=None

with open(DATA_DIR+"clean.txt") as f:

	clean=f.read().split("\n")

with open(DATA_DIR+"noised.txt") as f:

	noisy=f.read().split("\n")

print(len(clean),len(noisy))
from sklearn.model_selection import train_test_split



train_clean,test_clean, train_noisy, test_noisy=train_test_split(clean,noisy, test_size=0.25, shuffle=True, random_state=1)



df_train=pd.DataFrame({

	"Sentence":train_clean,

	"clean":1

})

df_train=pd.concat([

	df_train,

	pd.DataFrame({

		"Sentence": train_noisy,

		"clean":0

	})], ignore_index=True)

df_train=df_train.sample(frac=1, random_state=1).reset_index(drop=True)





df_test=pd.DataFrame({

	"Sentence":test_clean,

	"clean":1

})

df_test=pd.concat([

	df_test,

	pd.DataFrame({

		"Sentence": test_noisy,

		"clean":0

	})], ignore_index=True)

df_test=df_test.sample(frac=1, random_state=1).reset_index(drop=True)



df_train.head()
df_train.to_csv("train.csv", index=False)

df_test.to_csv("test.csv", index=False)
print(df_test['clean'].value_counts())

print(df_train['clean'].value_counts())
words=pd.read_csv("/kaggle/input/english-word-frequency/unigram_freq.csv")

words.head()
df_subset=df_train.loc[:10000,:].copy(deep=True)

df_subset.head()
progress=None

word_list=list(words['word'])

def sentence_metrics(row):

    sent=row.Sentence.split(" ")

    s_len=len(sent)

    similar=0

    cumulative_freq=0

    for s in sent:

        try:

            cumulative_freq+=words['count'][word_list.index(s)]

            similar+=1

        except ValueError:

            continue

    #progress.increment()

    return s_len,similar, cumulative_freq
import time

st=time.time()



#progress=ProgressBar(df_subset.shape[0])

df_subset['word_count'], df_subset['common_words_count'],df_subset['cumulative_freq']=zip(*df_subset.apply(sentence_metrics, axis=1))

print('That took {} seconds'.format(time.time() - st))
for i in df_subset:

    print(df_subset[i].isnull().value_counts())

df_subset.tail()
total_length=(10000)

list(range(0,total_length,total_length//4))
df_subset=df_train.loc[:10000,:].copy(deep=True)

df_subset.head()
from pandarallel import pandarallel



pandarallel.initialize()



st=time.time()



#progress=ProgressBar(df_subset.shape[0])

df_subset['word_count'], df_subset['common_words_count'],df_subset['cumulative_freq']=zip(*df_subset.parallel_apply(sentence_metrics, axis=1))

print('That took {} seconds'.format(time.time() - st))
df_subset.head(20)
print("Test DF")

df_test['word_count'], df_test['common_words_count'],df_test['cumulative_freq']=zip(*df_test.parallel_apply(sentence_metrics, axis=1))

df_test.to_csv("test_PP.csv", index=False)



print("Train DF")

df_train['word_count'], df_train['common_words_count'],df_train['cumulative_freq']=zip(*df_train.parallel_apply(sentence_metrics, axis=1))

df_train.to_csv("train_PP.csv", index=False)
df_train.head(10)
from IPython.display import FileLink

    

os.chdir(r'/kaggle/working')

FileLink(r'test_PP.csv')

FileLink(r'train_PP.csv')