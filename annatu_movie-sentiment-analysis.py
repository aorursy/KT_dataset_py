import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt # 画图常用库
import pandas as pd

train = pd.read_csv('../input/labeledTrainData.tsv', delimiter="\t")
test = pd.read_csv('../input/testData.tsv', delimiter="\t")
train.head()                
print (train.shape)
print (test.shape)
y_train = train['sentiment']

import re  #正则表达式

def review_to_wordlist(review):
#   Only keep letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
#   print (review_text)
    
    words = review_text.lower()
    
    return(words)

y_train = train['sentiment']
train_data = []

for i in range(0,len(train['review'])):
    if i % 1000 == 0:
        print ('training process line: ', str(i))
    train_data.append(review_to_wordlist(train['review'][i]))
        
train_data = np.array(train_data)
test_data = []
for i in range(0,len(test['review'])):
    if i % 1000 == 0:
        print ('testing process line: ', str(i))
    test_data.append(review_to_wordlist(test['review'][i]))
    
test_data = np.array(test_data)
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = CountVectorizer()
data_train_count = vectorizer.fit_transform(train_data)
data_test_count  = vectorizer.transform(test_data)


tfidf = TfidfVectorizer(
           ngram_range=(1, 3),  # 二元文法模型
           use_idf=1,
           smooth_idf=1,
           stop_words = 'english') # 去掉英文停用词


data_train_count_tf = tfidf.fit_transform(train_data)
data_test_count_tf  = tfidf.transform(test_data)

# use Multinomial NB
from sklearn.naive_bayes import MultinomialNB 
from sklearn.model_selection import cross_val_score
import numpy as np

clf = MultinomialNB()
clf.fit(data_train_count_tf, y_train)

print ("多项式贝叶斯分类器20折交叉验证得分: ", np.mean(cross_val_score(clf, data_train_count, y_train, cv=10, scoring='accuracy')))
print ("多项式贝叶斯分类器TFIDF,20折交叉验证得分: ", np.mean(cross_val_score(clf, data_train_count_tf, y_train, cv=10, scoring='accuracy')))
pred = clf.predict(data_test_count_tf)
print (pred)

df = pd.DataFrame({"id": test['id'], "sentiment": pred})
df.to_csv('movie_submission_nb.csv', index = False, header=True)
df.head()



