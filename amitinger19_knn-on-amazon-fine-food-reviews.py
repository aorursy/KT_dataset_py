import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import nltk
con = sqlite3.connect('../input/amazon-fine-food-reviews/database.sqlite')
data = pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')
sql_data = pd.read_sql_query("""
SELECT * FROM Reviews
WHERE Score!=3

""", con)
#Now we are replacing the 4,5,with positive and 1,2 as negative

def pos_neg(x):
  if x<3:
    return 'negative'
  return 'positive'

Actual_score = sql_data['Score']
updated_score = Actual_score.map(pos_neg)

sql_data['Score'] = updated_score

sql_data['Score'].head()
# Checking for repeating user ID

(sql_data['UserId'].value_counts()>1).sum()
s = pd.read_sql_query("""
SELECT * FROM Reviews
WHERE ProfileName= 'Geetha Krishnan'
""",con)
sorted_data = sql_data.sort_values('ProductId', axis=0,inplace=False,kind='quicksort',na_position='last')
#De-duplication
deduplicated_data = sorted_data.drop_duplicates(subset = ['UserId','ProfileName','Score','Time','Summary','Text'],keep= 'first',inplace=False)
deduplicated_data.shape
etc = pd.read_sql_query("""
SELECT * FROM Reviews
WHERE HelpfulnessNumerator > HelpfulnessDenominator
 """,con)
final_data = deduplicated_data[deduplicated_data.HelpfulnessDenominator>=deduplicated_data.HelpfulnessNumerator]
final_data.shape
final_data['Score'].value_counts()
final_data.isna().sum()
#Removing html tags
def htmltags(sentence):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', sentence)
  return cleantext

#Expanding shortcut words
def wordexpand(phrase):
    phrase = re.sub(r"won't","will not",phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase  
import nltk
sw = nltk.download('stopwords')
from nltk.corpus import stopwords
sw_eng = stopwords.words("english")
w = ['no','nor','not']
for i in w:
  if i in sw_eng:
    sw_eng.remove(i)
from tqdm import tqdm
import re
from bs4 import BeautifulSoup
preprocessed_reviews = []
# tqdm is for printing the status bar
for sentence in tqdm(final_data['Text'].values):
    sentence = sentence.lower()
    sentence = htmltags(sentence)
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = BeautifulSoup(sentence, 'lxml').get_text()
    sentence = wordexpand(sentence)
    sentence = re.sub("\S*\d\S*", "", sentence).strip()
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)
    # https://gist.github.com/sebleier/554280
    sentance = ' '.join(e.lower() for e in sentence.split() if e.lower() not in sw_eng)
    preprocessed_reviews.append(sentence.strip())

# Now we have processed review texts, now we will replace the text with processed texts
preprocessed_reviews[1500]

from gensim.models import Word2Vec
list_of_sentences=[]
for sent in preprocessed_reviews:
  list_of_sentences.append(sent.split())
W2V_model = Word2Vec(list_of_sentences,min_count=1,size=50,workers=5)
print(W2V_model.wv.most_similar('joy'))

## Getting all the words in Word2Vec vocabulary

total_words = list(W2V_model.wv.vocab)
len(total_words)


avg_w2v_vec =[]

for sent in tqdm(list_of_sentences):
  sent_vec = np.zeros(50)
  c=0
  for words in sent:
    if words in total_words:
      vec=W2V_model.wv[words]
      sent_vec=sent_vec+vec
      c=c+1
    if c!=0:
      sent_vec = sent_vec/c  
  avg_w2v_vec.append(W2V_model.wv[words])



X_data = final_data.drop(['Id', 'ProductId', 'UserId', 'ProfileName','Score','Time','Summary','Text'],axis=1)
X_data_tfidf=X_data

#data_tfidf=np.hstack((X_data_tfidf,tfidf_vec))
data_w2v=np.hstack((X_data_tfidf,avg_w2v_vec))
#data_bow=np.hstack((X_data_tfidf,bow_vec))
#data_tfidf_w2v=np.hstack((X_data_tfidf,tfidf_w2v_vec))


y=[1 if i=='positive' else 0 for i in final_data['Score'].values]



from sklearn.model_selection import train_test_split
def data_split(X_data,y):
  X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.25, random_state=42)
  X_tr, X_cv, y_tr, y_cv = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
  return X_tr, y_tr, X_test, y_test, X_cv, y_cv

#X_tr_bow, y_tr_bow, X_test_bow, y_test_bow, X_cv_bow, y_cv_bow = data_split(data_bow,y)   
#X_tr_tfidf, y_tr_tfidf, X_test_tfidf, y_test_tfidf, X_cv_tfidf, y_cv_tfidf = data_split(data_tfidf,y)
X_tr_avgw2v, y_tr_avgw2v, X_test_avgw2v, y_test_avgw2v, X_cv_avgw2v, y_cv_avgw2v = data_split(data_w2v,y)   
#X_tr_tfidf_w2v, y_tr_tfidf_w2v, X_test_tfidf_w2v, y_test_tfidf_w2v, X_cv_tfidf_w2v, y_cv_tfidf_w2v = data_split(data_tfidf_w2v,y)   


'''
np.save('./w2v_train_data',X_tr_avgw2v)
np.save('./train_data_y',y_tr_avgw2v)

np.save('./w2v_test_data',X_test_avgw2v)
np.save('./test_data_y',y_test_avgw2v)

np.save('./w2v_cv_data',X_cv_avgw2v)
np.save('./cv_data_y',y_cv_avgw2v)
'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def cross_validation(X_cv,y_cv):
  from sklearn.model_selection import GridSearchCV as GCV  
  k = {'n_neighbors':[1,3,5,7,9,11]}
  model = KNeighborsClassifier() 
  clf = GCV(model, k)
  search = clf.fit(X_cv,y_cv)
  print(search.best_params_)
  return search.best_params_

k = cross_validation(X_cv_avgw2v,y_cv_avgw2v)
def test_performance(X_tr,X_test,y_tr,y_test):  
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.metrics import plot_confusion_matrix
  from sklearn.metrics import accuracy_score
  KNN = KNeighborsClassifier(n_neighbors=11)
  KNN.fit(X_tr,y_tr)
  test_acc = accuracy_score(y_test_avgw2v,KNN.predict(X_test))
  
  plot_confusion_matrix(KNN,X_test_avgw2v,y_test_avgw2v)
  plt.show()
  print(test_acc)
  return test_acc
test_acc_w2v = test_performance(X_tr_avgw2v,X_test_avgw2v,y_tr_avgw2v,y_test_avgw2v)
