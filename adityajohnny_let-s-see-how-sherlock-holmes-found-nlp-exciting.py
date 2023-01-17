import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
# #nlp augmentation
!pip install --quiet googletrans
from googletrans import Translator

#for fast parallel processing
from dask import bag, diagnostics
# load train data which is provided in the competition link
data_train=pd.read_csv("../input/contradictory-my-dear-watson/train.csv")

data_text=pd.read_csv("../input/contradictory-my-dear-watson/test.csv")


# here I try to plot EDA For the Given data
plt.figure(figsize=[20,10])
x=data_train.groupby(['language','label'])['label'].count()
sns.countplot(x='language',hue='label',data=data_train)
# lets see some example to understand it properly
# so let's play with sherlock Dialogues

# as we can pass list, tuple, iterator, generator in the text . so I try to pass a list of  sherlock Dialogues and try to translate it ..,.,.
# we already import the required files above .,..

# we have to create a object for the Translator

trans=Translator()

# after creating object we call the translate from trans
# 'en' to 'ja'

Translated=trans.translate([ 'You have a grand gift for silence, Watson. It makes you quite invaluable as a companion.','Partial deafness in ear.','First point of attack. '],dest='ja',src="auto")

# as we know it return list so we use iteration to see the transation

for i in Translated:
    print(i.origin ,i.text)
    print()

# 'en' to 'hi'
Translated=trans.translate(['You have a grand gift for silence, Watson. It makes you quite invaluable as a companion.','Partial deafness in ear.','First point of attack. '],dest='hi',src="auto")

for i in Translated:
    print(i.origin ,i.text)
    print()
    
# 'en' to 'hi'
Translated=trans.translate(['You have a grand gift for silence, Watson. It makes you quite invaluable as a companion.','Partial deafness in ear.','First point of attack. '],dest='is',src="auto")

for i in Translated:
    print(i.origin ,i.text)
    print()    
    

# these are the all languages supported in Googletrans
from googletrans import LANGUAGES

for  lang in LANGUAGES:
    print(LANGUAGES[lang])
# lets see some example to understand it properly
# we use the object which we created above

## BASIC USE
print("BASIC USE")
print(trans.detect('이 문장은 한글로 쓰여졌습니다.'))

print(trans.detect('आपके पास मौन, वाटसन के लिए एक भव्य उपहार है। यह आपको एक साथी के रूप में काफी अमूल्य बनाता है।'))

## ADVANCE USE OF DETECT

print('ADVANCE USE OF DETECT')

langs = trans.detect(['आपके पास मौन, वाटसन के लिए एक भव्य उपहार है। यह आपको एक साथी के रूप में काफी अमूल्य बनाता है।', '日本語', '耳の部分的な難聴。', 'le français'])

for i in langs:
    print(i.lang,i.confidence)

# Here I created two function which I talked above
from time import sleep
def inte(X):
    sleep(1)
    return(X+1)
def add(x,y):
    sleep(1)
    return(x+y)
# now here we use normal python coding for implementation
%time
x=inte(1)
y=inte(2)
z=add(x,y)

from dask import delayed
%time
x = delayed(inte)(1)
y = delayed(inte)(2)

z = delayed(add)(x,y)



z.compute()
z.visualize()
# creating a random list

data=[1,2,3,4,4,5,6,6,7,8,9]
# create a empty list and then I call inte and store in yand then append it in results all this procees occure in a secquence
%time
results = []
for x in data:
    y = inte(x)
    results.append(y)
    
total = sum(results)
# we create allthe same things created above but using dask 

%time
results=[]

for x in data:
    y=delayed(inte)(x)
    results.append(y)
total = delayed(sum)(results) 

total.compute()
total.visualize()
# take example of numpy

import numpy as np

x=np.ones(15)
x
# implementing dask array
import dask.array as da

x=da.ones(15,chunks=(5,))

x
x.visualize()
x.sum().compute()
import dask.array as da

x=da.ones((15,15,15),chunks=(5,5,5))
x
x.compute()
x.visualize()
import dask.array as da

x=da.ones((15,15),chunks=(5,5))
x
x.compute()
x.visualize()
import dask.dataframe as dd

df=dd.read_csv('../input/contradictory-my-dear-watson/train.csv')

df.head()
df.language.value_counts().compute()
df.language.value_counts().visualize(rankdir="LR", size="10, 10!")
# A SORT EXAMPLE TO UNDERSTAND THE BAG
import dask.bag as db
b = db.from_sequence(range(5))
list(b.filter(lambda x: x % 2 == 0).map(lambda x: x * 10))

# TASK GRAPG 
# here 5 chunks are made for parallel processing
b.visualize()
from concurrent.futures import ThreadPoolExecutor
from time import sleep

def task(message):
    sleep(2)
    return message

def main():
    executor=ThreadPoolExecutor(5)
    future=executor.submit(task,("completed"))
    print(future.done())
    
    sleep(2)
    
    print(future.done())

    print(future.result())

if __name__ == '__main__':
    main()
    
    
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
values = [2,3,4,5]

def square(n):
    return n * n
   

def main():
    executor=ThreadPoolExecutor(max_workers = 3)
    results = executor.map(square, values)
    for result in results:
          print(result)
            
if __name__ == '__main__':
   main()
from concurrent.futures import ProcessPoolExecutor
from time import sleep


def task(message):
    sleep(2)
    return message

def main():
    executor = ProcessPoolExecutor(5)
    future = executor.submit(task, ("Completed"))
    print(future.done())
    sleep(2)
    print(future.done())
    sleep(10)
    print(future.done())
    print(future.result())
    
    
if __name__ == '__main__':
    main()    
from dask.distributed import Client

#  # start local workers as processes
client= Client()

# or start local workers as threads

client =Client()
def inc(x):
    return x + 1

def add(x, y):
    return x + y

a=client.submit(inc,10)

a.result()
import dask
dask.config.set(scheduler='threads')
import dask.multiprocessing
dask.config.set(scheduler='processes')  # overwrite default with multiprocessing scheduler
import dask
dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler
from dask.distributed import Client
client = Client()
# or
client = Client(processes=False)
#  Now I create fuction for translation 

# As a attribute we pass sequence as a text and lang as language in which we want to translate

def translation(sequence,lang):
      #instantiate translator
      
    translator=Translator()
    
    # translate to new language and convert back it to orig.
    
    translated=translator.translate(sequence,dest= lang).text
    
    return translated


def translation_parallel(dataset, lang):
    prem_bag=bag.from_sequence(dataset['premise'].tolist()).map(lambda x: translation(x, lang = lang))
    
    hypo_bag=bag.from_sequence(dataset['hypothesis'].tolist()).map(lambda x: translation(x, lang= lang))
    
    with diagnostics.ProgressBar():
        prems = prem_bag.compute()
            
        hyps = hypo_bag.compute()

    #pair premises and hypothesis
    dataset[['premise', 'hypothesis']] = list(zip(prems, hyps))
    
    
    return dataset

train_data=translation_parallel(data_train,'en')


text_data=translation_parallel(data_text,'en')
train_data
text_data
df=train_data[['premise','hypothesis','label']]

df_text=text_data[['premise','hypothesis']]
from dask import dataframe as dd

my_dask_df = dd.from_pandas(df, npartitions=4)
my_dask_df_text = dd.from_pandas(df_text, npartitions=4)
my_dask_df.persist()
my_dask_df_text.persist()
def remove_noice(text):
    text=re.sub('[^a-zA-Z]',' ',text)
    text=text.lower()
    text=text.split()
    
    return text
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
lemmatizer=WordNetLemmatizer()
corpus=[]

def remove_stopwords(text):
    
    wn.ensure_loaded()
    text=[lemmatizer.lemmatize(word) for word in text if word not in stopwords.words('english')]
    text=' '.join(text)
    
    return text

def clean_text(df):
    df['premise']=df.premise.map(remove_noice).map(remove_stopwords)
    
    df['hypothesis']=df.hypothesis.map(remove_noice).map(remove_stopwords)
    
    return df
df=my_dask_df.map_partitions(clean_text,meta=df)
df_text=my_dask_df_text.map_partitions(clean_text,meta=df_text)
df.persist()
df_text.persist()
y= df['label'].compute()
df=df.drop('label',axis=1)
df=df.persist()


df["comb"] = df.apply(lambda x:" ".join(x), axis=1,meta=("str"))
df_text["comb"]=df_text.apply(lambda x:" ".join(x), axis=1,meta=("str"))
df.persist()
x=df["comb"].compute().tolist()

x_text=df_text["comb"].compute().tolist()
import tensorflow as tf
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy() 
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense



### vocabulary size

voc_size=10000

sent_length=60

def oneEmbed(x):
    
    onehot_repr=[one_hot(words,voc_size)for words in x]
    
    embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
    
    return embedded_docs
    
x=oneEmbed(x)
x_text=oneEmbed(x_text)
x_text
### Creating Model


    
embedding_vector_features= 40
    
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(10,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(100))

model.add(Dense(3,activation='softmax'))
    
opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.2, nesterov=True)
    
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    
    
x_final=np.array(x)
x_final_text=np.array(x_text)
y_final=np.array(y).reshape((-1,1))
from dask_ml.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_final,y_final,test_size=0.2, random_state=40)
y_train.shape

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=50,batch_size=324)
y_pred1=model.predict_classes(x_test)
print(y_pred1)
y_pred=model.predict_classes(x_final_text)
print(y_pred)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred1)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred1)
