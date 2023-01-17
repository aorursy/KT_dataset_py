import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,HashingVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
fake_data = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')
fake_data.head()
True_data = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')
True_data.head()
fake_data['label'] = 1
True_data['label'] =0
final_data = pd.concat([fake_data,True_data])
final_data.head()
Null = final_data.isnull().sum()
Null.sort_values(ascending = False)
final_data.info()
final_data['label'].value_counts()
plt.figure(figsize =(15,8))
sns.countplot(final_data['subject'])
text=list(final_data['text'].dropna().unique())
fig, (ax2) = plt.subplots(1,1,figsize=[17, 10])
wordcloud2 = WordCloud().generate(" ".join(text))
ax2.imshow(wordcloud2,interpolation='bilinear')
ax2.axis('off')
y = final_data['label'].values
X = final_data.drop(['label', 'date'], axis = 1)
X.head(1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.33, stratify=y_train)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_train.shape, y_train.shape)
print(X_cv.shape, y_cv.shape)
print(X_test.shape, y_test.shape)

print("="*100)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_Tfidf = TfidfVectorizer(min_df=10)
vectorizer_Tfidf.fit(X_train['title'])

vectorizer_Title_TFIDF = CountVectorizer(min_df=10,ngram_range=(1,4), max_features=50000)
vectorizer_Title_TFIDF.fit(X_train['title'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_title_tfidf = vectorizer_Title_TFIDF.transform(X_train['title'].values)
X_cv_title_tfidf = vectorizer_Title_TFIDF.transform(X_cv['title'].values)
X_test_title_tfidf = vectorizer_Title_TFIDF.transform(X_test['title'].values)

print("After vectorizations")
print(X_train_title_tfidf.shape, y_train.shape)
print(X_cv_title_tfidf.shape, y_cv.shape)
print(X_test_title_tfidf.shape, y_test.shape)
print("="*100)
print(X_train.shape, y_train.shape)
print(X_cv.shape, y_cv.shape)
print(X_test.shape, y_test.shape)

print("="*100)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_Tfidf = TfidfVectorizer(min_df=10)
vectorizer_Tfidf.fit(X_train['text'])

vectorizer_text_TFIDF = CountVectorizer(min_df=10,ngram_range=(1,4), max_features=50000)
vectorizer_text_TFIDF.fit(X_train['text'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_text_tfidf = vectorizer_text_TFIDF.transform(X_train['text'].values)
X_cv_text_tfidf = vectorizer_text_TFIDF.transform(X_cv['text'].values)
X_test_text_tfidf = vectorizer_text_TFIDF.transform(X_test['text'].values)

print("After vectorizations")
print(X_train_text_tfidf.shape, y_train.shape)
print(X_cv_text_tfidf.shape, y_cv.shape)
print(X_test_text_tfidf.shape, y_test.shape)
print("="*100)
vectorizer = CountVectorizer()
vectorizer.fit(X_train['subject'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_subject_ohe = vectorizer.transform(X_train['subject'].values)
X_cv_subject_ohe = vectorizer.transform(X_cv['subject'].values)
X_test_subject_ohe = vectorizer.transform(X_test['subject'].values)

print("After vectorizations")
print(X_train_subject_ohe.shape, y_train.shape)
print(X_cv_subject_ohe.shape, y_cv.shape)
print(X_test_subject_ohe.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)
from scipy.sparse import hstack
X_tr_tfidf = hstack((X_train_title_tfidf, X_train_text_tfidf, X_train_subject_ohe)).tocsr()
X_cr_tfidf = hstack((X_cv_title_tfidf, X_cv_text_tfidf, X_cv_subject_ohe)).tocsr()
X_te_tfidf = hstack((X_test_title_tfidf, X_test_text_tfidf, X_test_subject_ohe)).tocsr()

print("Final Data matrix")
print(X_tr_tfidf.shape, y_train.shape)
print(X_cr_tfidf.shape, y_cv.shape)
print(X_te_tfidf.shape, y_test.shape)
print("="*100)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
NaiveBayesBOW = MultinomialNB()

parameters = {'alpha':[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000] }

ClassFit = GridSearchCV(NaiveBayesBOW, parameters, cv= 10, scoring='roc_auc',return_train_score=True)

ClassFit.fit(X_tr_tfidf, y_train)
train_auc= ClassFit.cv_results_['mean_train_score']
train_auc_std = ClassFit.cv_results_['std_train_score']
test_auc = ClassFit.cv_results_['mean_test_score'] 
test_auc_std = ClassFit.cv_results_['std_test_score']
cv_auc = ClassFit.cv_results_['mean_test_score']
cv_auc_std= ClassFit.cv_results_['std_test_score']
print('='*75)
print('Best score: ',ClassFit.best_score_)
print('k value with best score: ',ClassFit.best_params_)
print('='*75)
print('Train AUC scores')
print(ClassFit.cv_results_['mean_train_score'])
print('='*75)
print('CV AUC scores')
print(ClassFit.cv_results_['mean_test_score'])
print('='*75)
from tqdm import tqdm
alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]
log_alpha =[]
for alpha in tqdm(alphas):
    alphabar = np.log10(alpha)
    log_alpha.append(alphabar)
plt.figure(figsize=(10,5))
plt.plot(log_alpha, train_auc, label='Train AUC')
plt.plot(log_alpha, cv_auc, label='CV AUC')
plt.scatter(log_alpha, train_auc, label='Train AUC points')
plt.scatter(log_alpha, cv_auc, label='CV AUC points')
plt.legend()
plt.xlabel("alpha: hyperparameter")
plt.ylabel("AUC")
plt.title("alpha: hyperparameter v/s AUC")
plt.grid()
plt.show()
