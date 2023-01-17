import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re 
import gensim
df = pd.read_csv("/kaggle/input/twitter-airline-sentiment/Tweets.csv")
df.head()
X = df['text'].values
def preprocess(strr):
    strr = strr.lower()
    strr = re.sub(r'\W+' , ' ' , strr )
    tokens = word_tokenize(strr)
    stop_words = stopwords.words('english')
    return [i for i in tokens if i not in stop_words ]
#Some preprocessing 
all = []
for i in range(0,len(X)):
    all.append(  preprocess(  X[i] ))
model = gensim.models.Word2Vec( all , size = 50 , window = 5 , min_count = 1, workers = 2, sg=1 )
#Here i print Most simillar words to word Good 
test_ = model.most_similar("good" , topn = 5 )
print( test_ )
# First Model Calculate Sentence Based on Sum of Word embiding
al =[]
for i in all   : 
    X = np.zeros((50))
    for j in i :
        X = X + model[j]
    al.append(X)
print(len(al))
Y = df['airline_sentiment'].values
for i in range(len(Y)):
    Y[i] = Y[i].lower()
    if Y[i] == "neutral":
        Y[i] = 2
    elif Y[i]=="positive":
        Y[i] = 1
    else:
        Y[i] = 0
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('neutral', 'positive', 'negative')
y_pos = np.arange(len(objects))
performance = [(Y==2).sum(),(Y==1).sum(),(Y==0).sum()]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Frequency')
plt.title('labels')

plt.show()
al = np.array(al)
X_train,X_test,Y_train,Y_test = train_test_split( al , Y , test_size = 0.4 )
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('neutral', 'positive', 'negative')
y_pos = np.arange(len(objects))
performance = [(Y_train==2).sum(),(Y_train==1).sum(),(Y_train==0).sum()]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Frequency')
plt.title('labels')

plt.show()
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
objects = ('neutral', 'positive', 'negative')
y_pos = np.arange(len(objects))
performance = [(Y_test==2).sum(),(Y_test==1).sum(),(Y_test==0).sum()]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Frequency')
plt.title('labels')
plt.show()
X_train = X_train.reshape( X_train.shape[0] , 1 , X_train.shape[1] )
X_test = X_test.reshape( X_test.shape[0] , 1 , X_test.shape[1] )
Y_test = keras.utils.to_categorical(Y_test  , num_classes = 3 )
Y_train = keras.utils.to_categorical(Y_train , num_classes = 3 )
print(X_train.shape)
print(Y_train.shape)
modell = keras.Sequential()
modell.add(keras.layers.LSTM(50,input_shape=(1, 50)))
modell.add(keras.layers.Dense(3, activation='softmax'))
modell.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
print(modell.summary())
history = modell.fit(X_train, Y_train , epochs = 10 , batch_size = 64 ,  verbose=1 )
print(modell.evaluate( X_test , Y_test , verbose = 0 ) )
# Model Using Average 
# First Model Calculate Sentence Based on Sum of Word embiding
al =[]
for i in all   : 
    X = np.zeros((50))
    for j in i :
        X = X + model[j]
    X /= len( i )
    al.append(X)
al = np.array(al)
X_train,X_test,Y_train,Y_test = train_test_split( al , Y , test_size = 0.4 )
X_train = X_train.reshape( X_train.shape[0] , 1 , X_train.shape[1] )
X_test = X_test.reshape( X_test.shape[0] , 1 , X_test.shape[1] )
Y_test = keras.utils.to_categorical(Y_test  , num_classes = 3 )
Y_train = keras.utils.to_categorical(Y_train , num_classes = 3 )
model2 = keras.Sequential()
model2.add(keras.layers.LSTM(50,input_shape=(1, 50)))
model2.add(keras.layers.Dense(3, activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
print(model2.summary())
history = model2.fit(X_train, Y_train , epochs = 10 , batch_size = 64 ,  verbose=1)
print(model2.evaluate( X_test , Y_test , verbose = 0 ) )
inputt = input()
preprocessed = preprocess(inputt)
al =[]
for i in preprocessed   : 
    X = np.zeros((50))
    try:
        X = X + model[i]
    except:
        continue
al.append(X)
al = np.array(al)
al = al.reshape( al.shape[0] , 1 , al.shape[1] )
F = modell.predict(al).argmax()
if F == 0 :
    print("Negative")
elif F == 1 :
    print("Poitive")
else:
    print("Neutral")








