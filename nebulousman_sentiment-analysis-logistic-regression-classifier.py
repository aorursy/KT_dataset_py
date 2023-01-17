import numpy as np
import matplotlib.pylab as plt
import nltk
import re
import pandas as pd
from sklearn import metrics 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from string import punctuation
training = pd.read_csv('../input/imdb-dataset-sentiment-analysis-in-csv-format/Train.csv')
validation = pd.read_csv('../input/imdb-dataset-sentiment-analysis-in-csv-format/Valid.csv')
testing = pd.read_csv('../input/imdb-dataset-sentiment-analysis-in-csv-format/Test.csv')

# combine files
sentiments = pd.concat([training, validation, testing], axis=0)

# shuffle rows
sentiments = sentiments.sample(frac=1)
#build the functions

def sig(z):
    return 1/(1+np.exp(-z))

def costFunction(y, X, theta, lamda):
    m = X.shape[0]
    z = X @ theta
    h=sig(z)
    regCost = 1/m * np.sum((-y * np.log(h)) - ((1-y) * np.log(1-h)))+lamda/(2*m)*np.square(np.linalg.norm(theta))
    fPrimeConst = 1/m * (X.T @ (h-y)) [0]
    fPrimeReg = 1/m * (X.T @ (h-y))[1:] + (lamda/m)*theta[1:]
    gradient = np.vstack((fPrimeConst,fPrimeReg))
    return regCost, gradient
    
def stochasticGradientDescent(y, X, theta, lamda, eta, rounds, batch):
    sgd = []
    z = np.c_[y.reshape(len(y),-1), X.reshape(len(X),-1)]
    for i in range(rounds):
        np.random.shuffle(z)
        z=z[:batch]
        cost, gradient = costFunction(z[:,:1],z[:,1:],theta,lamda)
        theta = theta - (eta * gradient)
        sgd.append(cost)
    sgd = np.array(sgd)
    return sgd, theta   

def logitClassifer(theta, X):
    z = X @ theta
    h=sig(z)
    outcome = (h>=.5)*1
    return outcome


def textCleaner(pattern, corpus):
    clean = [re.sub(pattern,' ', c) for c in corpus]
    return clean

def LowerExcept(data):
    p_strip = lambda x: "".join(w for w in x if w not in punctuation)
    allcaps = re.findall(r"\b[A-Z][A-Z]+\b",data)
    to_lower = lambda l: " ".join( a if p_strip(a) in allcaps else a.lower() for a in l.split())
    return to_lower(data)
# clean text
sentiments['text']=textCleaner('<br\s/>|\(|\)|\/|\*',sentiments['text'])
sentiments['text']=[LowerExcept(w) for w in sentiments['text']]

# sample data
sentiments = sentiments.sample(n=1000)
#convert to list
x = sentiments['text'].values.tolist()
y = sentiments['label'].values.tolist()


vec = CountVectorizer(ngram_range=(2,2),
                      tokenizer=nltk.word_tokenize)

df = vec.fit_transform(x)
df = pd.DataFrame(df.toarray(), columns=vec.get_feature_names())

X = np.array(df)
y = np.array(y)
y = np.expand_dims(y, axis=1)

X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=1,test_size=0.5,shuffle=False)
X_train, X_val, y_train, y_val=train_test_split(X_train,y_train,random_state=1,test_size=0.2,shuffle=False)


# Validation & confusion matrix
theta0 = np.random.randn(X_train.shape[1],1) * np.sqrt(1. / X_train.shape[1])
gd,theta =  stochasticGradientDescent(y_train,X_train,theta0,0,.1,100,50)

y_pred_val = logitClassifer(theta, X_val)
print('Validation Accuracy:', np.sum((y_pred_val == y_val)*1)/df.shape[0])

confmat = metrics.confusion_matrix(y_val, y_pred_val) 
fig, ax = plt.subplots(figsize=(2.5, 2.5)) 
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3) 
for i in range(confmat.shape[0]): 
    for j in range(confmat.shape[1]): 
        ax.text(x=j, y=i, 
            s=confmat[i, j], 
                     va= 'center', ha='center') 
plt.xlabel('predicted label') 
plt.ylabel('true label') 
plt.title('Validation')
plt.show() 