import numpy as np
import pandas as pd
data_dir = '../input/'
train_data = pd.read_csv(data_dir+'labeledTrainData.tsv',delimiter="\t")
test_data = pd.read_csv(data_dir+'testData.tsv',delimiter="\t")
test_test = pd.read_csv(data_dir+'unlabeledTrainData.tsv',delimiter="\t",error_bad_lines= False)

train_score = train_data['sentiment']
train_review = train_data['review']
import re
def review_to_wordlist(review):
    review_text = re.sub("[^a-zA-Z]"," ", review)
    words = review_text.lower()
    return words

train_set = []
for i in range(0,len(train_review)):
    train_set.append(review_to_wordlist(train_review[i]))
    
test_set = []
for i in range(0,len(test_data['review'])):
    test_set.append(review_to_wordlist(test_data['review'][i]))

test_set = np.array(test_set)
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()

test_set_stem = []
for word in test_set:
    test_set_stem.append(ps.stem(word))

train_set_stem = []
for word in train_set:
    train_set_stem.append(ps.stem(word))


from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer

# vectorizer = CountVectorizer()


# data_train_count = vectorizer.fit_transform(train_data)
# data_test_count  = vectorizer.transform(test_data)

tfidf = TfidfVectorizer(
           ngram_range=(1,3), 
           use_idf=1,
           smooth_idf=1,
           stop_words = 'english') 


data_train_count_tf = tfidf.fit_transform(train_set_stem)
data_test_count_tf  = tfidf.transform(test_set_stem)

from sklearn.naive_bayes import MultinomialNB 

clf = MultinomialNB()
clf.fit(data_train_count_tf, train_score)
from sklearn.model_selection import cross_val_score
import numpy as np
print ( np.mean(cross_val_score(clf, data_train_count_tf, train_score, cv=10, scoring='accuracy')))

pred = clf.predict(data_test_count_tf)
print (pred)

df = pd.DataFrame({"id": test_data['id'],"sentiment": pred})

df.to_csv('submission_2.csv',index = False, header=True)
