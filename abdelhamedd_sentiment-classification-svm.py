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
from sklearn.model_selection import train_test_split
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
# Model Using Average 
# First Model Calculate Sentence Based on Sum of Word embiding
al =[]
for i in all   : 
    X = np.zeros((50))
    for j in i :
        X = X + model[j]
    X /= len( i )
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

print(X_train.shape)
print(Y_train.shape)
Y_train = np.array(Y_train)
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, Y_train)
svm_model_linear.score(X_test , Y_test)
svm_model_linear.score(X_train , Y_train)
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
F = svm_model_linear.predict(al).argmax()
if F == 0 :
    print("Negative")
elif F == 1 :
    print("Poitive")
else:
    print("Neutral")











