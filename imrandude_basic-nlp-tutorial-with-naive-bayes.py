import pandas as pd, numpy as np, matplotlib.pyplot as plt
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
train = pd.read_csv('../input/train.csv',encoding='iso-8859-1')
train.shape
train.head(20)
train['SentimentText'][400]
lens = train.SentimentText.str.len()
lens.mean(), lens.std(), lens.max()
lens.hist();
plt.show()
labels = ['0', '1']
sizes = [train['Sentiment'].value_counts()[0],
         train['Sentiment'].value_counts()[1]
        ]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')
plt.title('Sentiment Proportion', fontsize=20)
plt.show()
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
 
data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
stopWords = set(stopwords.words('english'))
words = word_tokenize(data)
wordsFiltered = []
 
for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)
 
print(wordsFiltered)
stopwords.words('english')
#nltk.download("stopwords") 
from nltk.corpus import stopwords
train.SentimentText = [w for w in train.SentimentText if w.lower() not in stopwords.words('english')]
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
 
words = ["game","gaming","gamed","games"]
stemmer = PorterStemmer()
 
for word in words:
    print(stemmer.stem(word))
plurals = ['caresses', 'flies', 'dies', 'mules', 'denied',
            'died', 'agreed', 'owned', 'humbled', 'sized',
            'meeting', 'stating', 'siezing', 'itemization',
            'sensational', 'traditional', 'reference', 'colonizer',
            'plotted'] 
for word in plurals:
    print(stemmer.stem(word))
#nltk.download("wordnet")
ps = nltk.PorterStemmer()
train.SentimentText = [ps.stem(l) for l in train.SentimentText]
X = train.SentimentText
y = train.Sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
train1=pd.concat([X_train,y_train], axis=1)
train1.shape
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
from nltk import ngrams
sentence = 'this is a foo bar sentences and i want to ngramize it'
n = 2
bigrams = ngrams(sentence.split(), n)
for grams in bigrams:
  print (grams)
n = train1.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), #The lower and upper boundary of the range of n-values for different n-grams
                      tokenizer=tokenize,
                      min_df=3,      # ignore terms that have a df strictly lower than threshold
                      max_df=0.9,    #ignore terms that have a df strictly higher than threshold (corpus-specific stop words)
                      strip_accents='unicode', #Remove accents during the preprocessing step
                      use_idf=1,
                      smooth_idf=1,  #Smooth idf weights by adding one to document frequencies, 
                                     #as if an extra document was seen containing every term in 
                                     #the collection exactly once. Prevents zero divisions.
                      sublinear_tf=1, #Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
                      max_features=40000
                     )
trn_term_doc = vec.fit_transform(train1['SentimentText'])
test_term_doc = vec.transform(X_test)
#This creates a sparse matrix with only a small number of non-zero elements (stored elements in the representation below).
trn_term_doc, test_term_doc
#Here's the basic naive bayes feature equation:
def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)
def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=3,solver='newton-cg')
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r
x = trn_term_doc
test_x = test_term_doc

label_cols=['Sentiment']
preds = np.zeros((len(X_test), len(label_cols)))
preds
for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train1[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
y_pred=pd.DataFrame(preds.round(decimals=0), columns = label_cols)
accuracy_score(y_test, y_pred)