import nltk
#nltk.download()
from nltk.corpus import stopwords

import pandas as pd
add = "../input/spam.csv"
data = pd.read_csv(add, encoding='latin-1')
data.head(5)
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
pd.set_option('display.max_colwidth', 0)
data.columns = ['label','text']

data.head(5)

#print(data.head(5))
import string
import re
stopword = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
def clean_text(text):
    remove_punct = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+',remove_punct)
    noStop = ([ps.stem(word) for word in tokens if word not in stopword])
    return noStop
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_Vector = TfidfVectorizer(analyzer= clean_text)
Xtfidf_Vector = tfidf_Vector.fit_transform(data['text'])
import string
def punct_percent(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/ (len(text) - text.count(" ")),3)*100
data['punct_%'] = data['text'].apply(lambda x: punct_percent(x))
data['length'] = data['text'].apply(lambda x: len(x) - x.count(" "))

pd.set_option('display.max_colwidth', 0)

print(data.head(5))
import re

#def find_num(text):
#    return re.findall('\d{7,}',text)

data['number'] = pd.DataFrame(data['text'].apply(lambda x: len(re.findall('\d{5,}',x))))
data.head(5)

#def get_currency_symbol(text):
#    pattern = r'(\D*)\d*\.?\d*(\D*)'
#    result = re.match(pattern,text).group()
#    return result
#data['currency']= pd.DataFrame(data['text'].apply(lambda x: len(get_currency_symbol(x))))

#print(data.head(5))
def web_address(t):
    if(len(re.findall('www|http|https|.co',t)) > 0):
        return 1
    else:
        return 0
    
data['url'] = pd.DataFrame(data['text'].apply(lambda x: web_address(x)))
print(data.head(5))   
import numpy as np
from matplotlib import pyplot
%matplotlib inline
bins = np.linspace(0,200,40)
pyplot.hist(data[data['label'] == 'spam']['length'],bins,alpha = 0.5,normed = True,label = 'spam')
pyplot.hist(data[data['label'] == 'ham']['length'],bins,alpha = 0.5,normed = True, label = 'ham')
pyplot.legend(loc = 'upper right')
pyplot.figure(figsize = (1000,400), dpi = 1000)
pyplot.show()
bins = np.linspace(0,50,40)
pyplot.hist(data[data['label'] == 'spam']['punct_%'], bins, alpha = 0.5,normed = True, label = 'spam')
pyplot.hist(data[data['label'] == 'ham']['punct_%'], bins, alpha = 0.5,normed = True, label = 'ham')
pyplot.legend(loc = 'upper right')
pyplot.show()
bins = np.linspace(0,5,10)
pyplot.hist(data[data['label'] == 'spam']['number'], bins,alpha = 0.5, label = 'spam')
pyplot.hist(data[data['label'] == 'ham']['number'], bins, alpha = 0.5, label = 'ham')
pyplot.legend(loc = 'upper right')
pyplot.show()
bins = np.linspace(0,5,100)
pyplot.hist(data[data['label'] == 'spam']['url'], bins,alpha = 0.5, label = 'spam')
pyplot.hist(data[data['label'] == 'ham']['url'], bins, alpha = 0.5, label = 'ham')
pyplot.legend(loc = 'upper right')
pyplot.show()
data['url'].value_counts()
Xfeatures_data = pd.concat([data['length'],data['punct_%'],data['number'],data['url'], pd.DataFrame(Xtfidf_Vector.toarray())], axis = 1)
Xfeatures_data.head(5)
from sklearn.ensemble import RandomForestClassifier
print(dir(RandomForestClassifier))
print(RandomForestClassifier())
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(Xfeatures_data, data['label'], test_size = 0.2)


rf = RandomForestClassifier(n_estimators= 50, max_depth= 20, n_jobs = -1)
rf_model = rf.fit(X_train,y_train)
sorted(zip(rf.feature_importances_,X_train.columns), reverse= True)[0:10]
y_pred = rf_model.predict(X_test)
precision,recall,fscore,support = score(y_test,y_pred,pos_label = 'spam', average = 'binary')
print('Precision: {}/ Recall: {}/ Accuracy: {}'.format(round(precision,3), round(recall,3), (y_pred == y_test).sum()/len(y_pred)))
def train_rf(n_est, depth):
    rf = RandomForestClassifier(n_estimators= n_est, max_depth= depth, n_jobs = -1)
    rf_model = rf.fit(X_train,y_train)
    y_pred = rf_model.predict(X_test)
    precision,recall,fscore,support = score(y_test,y_pred,pos_label= 'spam',average= 'binary')
    print('Est: {}/ Depth: {}/ Precision: {}/ Recall: {}/ Accuracy : {}'.format(n_est,depth, round(precision,3), round(recall,3), (y_pred == y_test).sum()/len(y_pred)))
for n_est in [10,30,50,70]:
    for depth in [20,40,60,80, None]:
        train_rf(n_est,depth)

