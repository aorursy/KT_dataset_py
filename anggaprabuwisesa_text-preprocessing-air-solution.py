!pip3 install Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
 

factory = StemmerFactory()
stemmer = factory.create_stemmer()
factory2 = StopWordRemoverFactory()
stopword = factory2.create_stop_word_remover()
def preprocessing(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = stemmer.stem(text)
    text = stopword.remove(text)
    return text
preprocessing("aku memakan tahu bulat di sekolahan")
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
doc = [
    "Hari ini akan hujan",
    "Hari ini saya tidak akan keluar",
    "Saya akan menonton pemutaran perdana season",
]
cou = CountVectorizer()
y = cou.fit_transform(doc).toarray()
print(cou.vocabulary_)
print(y)
vectorizer = TfidfTransformer(use_idf=True,smooth_idf=False)
x = vectorizer.fit_transform(y)
t = 0
for u in doc:
    print("sentance - ",t)
    for z in u.split():
        print(z ," : Count",y[t][cou.vocabulary_[z.lower()]] ," - tf ",x.toarray()[t][cou.vocabulary_[z.lower()]])
#         print(z ," : Count",y[t][cou.vocabulary_[z.lower()]], " - tf ",x.toarray()[t][cou.vocabulary_[z.lower()]])
        
    t += 1
print(x.toarray())


import numpy as np
from sklearn.preprocessing import normalize
#Kalimat Pertama
n = 3
tf1 = (1) * (np.log((n)/(3)) + 1)
tf2 = (1) * (np.log((n)/(2)) + 1)
tf3 = (1) * (np.log((n)/(1)) + 1)
tf4 = (1) * (np.log((n)/(2)) + 1)

x = normalize([[tf1,tf2,tf3,tf4,0,0,0,0,0,0,0]],norm="l2")
print(x)