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
embeddings = {}

with open('/kaggle/input/glove6b300dtxt/glove.6B.300d.txt') as glove:

    for line in glove:

        values = line.split()

        word = values[0]

        try:

            coef = np.asarray(values[1:],dtype='float32')

            embeddings[word.lower()] = coef

        except:

            print(word +  " Embedding Not Found")

        
df = pd.read_csv('/kaggle/input/intents/MIX_INTENT_NEW.csv')

df = df.dropna()
previous_intent = (df['previous_intent'])

current_intent = df['current_intent']

sentences = df['query']
# queries = df['query']

# label = df['label'] 
queries = []



for i in sentences:

    try:

        temp = i.split()

    except:

        print(i)

    ans=[]

    for j in temp:

        if(j not in embeddings.keys()):

            j='unk'

        ans.append(j)

    queries.append(" ".join(ans))

        
# q = []



# for i in queries:

    

#     temp = i.split()

#     ans=[]

#     for j in temp:

#         if(j not in embeddings.keys()):

#             j='unk'

#         ans.append(j)

#     q.append(" ".join(ans))

        
prev_dummies= pd.get_dummies(previous_intent)

cur_dummies= pd.get_dummies(current_intent)

#prev_dummies = prev_dummies.drop(columns=['no'],axis=1)
len(prev_dummies.columns)
from collections import Counter





W= Counter()



for w in queries:

    words = w.split()

    W.update(words)
tokenizer = {}
for ind,i in enumerate(W.keys()):

    tokenizer[i] = ind

    

    
tokenizer['unk']

tokenizer['pad'] = len(tokenizer)-1
maxlen=20
embedding_matrix = np.zeros((len(tokenizer),300))



for i in tokenizer.keys():

    

    embedding_matrix[tokenizer[i]] = embeddings[i]
queries_tokenized = []



for i in queries:

    temp = i.split()

    ans=[]

    for y in temp:

        ans.append(tokenizer[y.lower()])

    queries_tokenized.append(ans)
from keras.preprocessing.sequence import pad_sequences



queries = pad_sequences(queries_tokenized,maxlen,value = tokenizer['pad'],padding='post')
dataset = pd.DataFrame(queries)
dataset = dataset.join(prev_dummies)
# labels = pd.get_dummies(label)
X = dataset

Y = cur_dummies
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.02,random_state=1)
print(x_train.shape , y_train.shape)

print(x_test.shape , y_test.shape)
from keras.layers import Bidirectional, Dense , Embedding, Dropout , Input , GRU , Conv1D , concatenate

from keras.models import Model

from keras.layers import Bidirectional, Dense , Embedding, Dropout , Input , GRU , Conv1D , concatenate

from keras.models import Model



input1 = Input((maxlen,))

x= Embedding(len(tokenizer),300,weights = [embedding_matrix])(input1)

biout = Bidirectional(GRU(maxlen,return_sequences = False))(x)



input2 = Input((len(prev_dummies.columns),))

x = concatenate([biout,input2])

x=Dense(64,activation = "relu")(x)

x= Dropout(0.1)(x)

x=Dense(len(cur_dummies.columns),activation = 'softmax')(x)

# input1 = Input((maxlen,))

# x= Embedding(len(tokenizer),300,weights = [embedding_matrix])(input1)

# biout = Bidirectional(GRU(maxlen,return_sequences = False))(x)

# x=Dense(64,activation = "relu")(biout)

# x= Dropout(0.1)(x)

# x=Dense(len(labels.columns),activation = 'softmax')(x)

# len(labels.columns)
# model = Model(inputs= input1,outputs=x)

# model.compile(loss= 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
model = Model(inputs=[input1,input2],outputs=x)

model.compile(loss= 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
model.summary()
# model.fit(x_train,y_train,epochs=8,validation_split = 0.1)
model.fit([x_train.iloc[:,:maxlen],x_train.iloc[:,maxlen:]],y_train,epochs=8,validation_split = 0.1)
predictions=model.predict([x_test.iloc[:,:maxlen],x_test.iloc[:,maxlen:]])
for i,x in enumerate(predictions):

    print(y_test.columns[np.argmax(x)])

    
from sklearn.metrics import confusion_matrix,f1_score
y_testing = model.predict([x_test.iloc[:,:30],x_test.iloc[:,30:]], verbose = 1)



y_pred = [np.argmax(i) for i in y_testing]

score = f1_score(y_test, y_pred, average = 'micro')

print(score)
def transformQ(q):

    

    queries = []



    



    temp = q.split()

    ans=[]

    for j in temp:

        if(j not in tokenizer.keys()):

            j='unk'

        ans.append(j)

    queries.append(" ".join(ans))

    

    

    

    queries_tokenized=[]

    

    temp = queries[0].split()

    ans=[]

    for y in temp:

        ans.append(tokenizer[y.lower()])

    queries_tokenized.append(ans)

    queries_tokenized=pad_sequences(queries_tokenized,maxlen,value = tokenizer['pad'],padding='post')

    return np.array(queries_tokenized)





def transformI(i):

    one_hot = [0]*len(prev_dummies.columns)

    one_hot[i] = 1

    return np.array([one_hot])
while(True):

    query = input("Enter a query -")

    if(query=='quit'):

        break

    for i,x in enumerate(prev_dummies.columns):

        print(x,i)

    intent = int(input("Enter previous intent -"))



    q= transformQ(query)

    i = transformI(intent)



    p=model.predict([q,i])

    

    print("\n\n\nCurrent Intent -",y_test.columns[np.argmax(p[0])],np.max(p[0]))



while(True):

    query = input("Enter a query ")

    ans=""

    if(query == 'quit'):

        break

    for x in query.split():

        if(x not in tokenizer.keys()):

            x='unk'

        ans+= x + " "



    vector = []    

    for x in ans.split():



        vector.append(tokenizer[x])





    x = pad_sequences([vector],maxlen,value = tokenizer['pad'],padding='post')  



    print(y_train.columns[np.argmax(model.predict(x))])
model.save('nlu_all.h5')
model.save_weights('nlu_all_weights.h5')
from IPython.display import FileLink,FileLinks

import pickle 
with open('tokenizer.pkl', 'wb') as w:

    w.write(pickle.dumps(tokenizer))

FileLinks('.')