import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, LSTM, Embedding, Flatten



from sklearn.compose import ColumnTransformer
import nltk

from re import sub



nltk.download('stopwords')



class Vocabulary:



  def __init__(self,max_words):

    self.max_words = max_words

    self.word2index = {}

    self.word2count = {}

    self.index2word = {}

    self.num_words = 0

    self.StopWords = set(nltk.corpus.stopwords.words('english'))



  def add_word(self, word):

    if word not in self.word2count:

      # First entry of word into vocabulary

      self.word2count[word] = 1

      self.num_words += 1

    else:

      # Word exists; increase word count

      self.word2count[word] += 1

          

  def add_sentence(self, sentence):

    sentence = sub(r'[^\w\s]','',sentence)

    for word in sentence.split(' '):

      to_add = word.lower()

      if to_add not in self.StopWords:

        self.add_word(to_add)



  def consolidate(self):

    self.index2word = {0 : "NULL"}



    sortedList = [k for k, v in sorted(self.word2count.items(), key=lambda item: item[1],reverse=True)]

    for idx in range(1,min(len(sortedList),self.max_words)+1):

      self.index2word[idx] = sortedList[idx-1]

    self.word2index = dict({(value,key) for (key,value) in self.index2word.items()})





  def to_word(self, index):

    return self.index2word[index]



  def to_index(self, word):

    return self.word2index[word]
def sent2list(sent,word2idx):

    idxList=[]

    for word in sent.split(' '):

        try:

            idxList.append(word2idx[word.lower()])

        except:

            pass

    return np.array(idxList)
def fake_or_not(y):

  if y>0.75: return print("Fake")

  elif y>0.5: return print("Probably Fake")

  elif y>0.25: return print("Probably True")

  else: return print("True")



from re import sub



def preprocess(sent, word2idx):

    sent = sub(r'[^\w\s]','',sent)

    return sent2list(sent, word2idx)
Fake = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")

Real = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")

Fake["Fake"]=1

Real["Fake"]=0

data = pd.concat([Fake,Real])

data.head()
num_words = 20000



voc = Vocabulary(num_words)



for sentence in data.text.values:

  voc.add_sentence(sentence)



voc.consolidate()



idx2word = voc.index2word

word2idx = voc.word2index
from sklearn.model_selection import train_test_split



y = data['Fake'].values

X = data[['title','text']].values



X[:,0] = [sent2list(sent,word2idx) for sent in X[:,0]]

X[:,1] = [sent2list(sent,word2idx) for sent in X[:,1]]



X_train, X_test, y_train, y_test = train_test_split(X,y)
from keras.preprocessing.sequence import pad_sequences



max_words = 700

#X_temp=np.array(X_temp.shape[0],max)

X_temp = pad_sequences(X_train[:,1],maxlen=max_words)

X_train = X_temp

X_temp = pad_sequences(X_test[:,1],maxlen = max_words)

X_test = X_temp
from keras.layers.convolutional import Conv1D,MaxPooling1D



model = Sequential()

model.add(Embedding(num_words+1,100,input_length=max_words))

model.add(LSTM(32, dropout=0.9, return_sequences=True))

model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))

model.add(MaxPooling1D())

model.add(Flatten())

model.add(Dropout(0.9))

model.add(Dense(1,activation='sigmoid'))



model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=512,epochs=15,validation_split=0.2)
model.evaluate(X_test,y_test)
text = 'With a flood of unemployment claims continuing to overwhelm many state agencies, economists say the job losses may be far worse than government tallies indicate. The Labor Department said Thursday that 3.8 million workers filed for unemployment benefits last week, bringing the six-week total to 30 million. But researchers say that as the economy staggers under the weight of the coronavirus pandemic, millions of others have lost jobs but have yet to see benefits. A study by the Economic Policy Institute found that roughly 50 percent more people than counted as filing claims in a recent four-week period may have qualified for benefits — with the difference representing those who were stymied in applying or didn’t even try because the process was too formidable. “The problem is even bigger than the data suggest,” said Elise Gould, a senior economist with the institute, a left-leaning research group. “We’re undercounting the economic pain.” Alexander Bick of Arizona State University and Adam Blandin of Virginia Commonwealth University found that 42 percent of those working in February had lost their jobs or suffered a reduction in earnings. By April 18, they found, up to eight million workers were unemployed but not reflected in the weekly claims data. The difficulties at the state level largely flow from the sheer volume of claims, which few agencies were prepared to handle. Many were burdened by aging computer systems that were hard to reconfigure for new federal guidelines. “We’ve known that the state unemployment insurance systems were not up to the task, yet those investments were not made,” Ms. Gould said. “The result is that the state systems are buckling under the weight of these claims.” The crush of claims is a major reason — but not the only one — that states are backlogged. Frustrated applicants who refile their applications, some as many as 20 times, slow the system as processors weed out duplicates. Some applications are missing i formation. New York analyzed a million claims and found many had been delayed because of a missing employer identification number. In such cases, each applicant has to be called back. Callers looking for updates also flood the system, increasing the wait for those who need to correct a mistake.'

sentTest = preprocess(text, word2idx)



sentTest = sentTest.reshape(1,sentTest.shape[0])

sentTest

sentTest = pad_sequences(sentTest, maxlen = max_words)



y = model.predict(sentTest)

y, fake_or_not(y)