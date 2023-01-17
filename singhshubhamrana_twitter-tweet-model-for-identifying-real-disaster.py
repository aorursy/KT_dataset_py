import numpy as np

from bs4 import BeautifulSoup

import re

from tqdm import tqdm

import string

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

from matplotlib import rcParams

from wordcloud import WordCloud

from matplotlib import rc_params

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report

from sklearn.metrics import precision_recall_curve, precision_score, recall_score

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from gensim.models import Word2Vec

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB, MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from prettytable import PrettyTable
%matplotlib inline

rcParams["figure.figsize"] = 8,8

sns.set_style("darkgrid")

# plt.style.use('ggplot')
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

print(train.info())

train.head()
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

y_test = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")



# Droping id column from y_test data-set as that same id is present in test data-set in same order.

y_test = y_test.drop("id", axis=1)



# Joining test with y_test data-set to make a complete new test data-set.

test = pd.DataFrame.join(test, y_test)

print(test.info())

test.head()
print('There are {} rows and {} columns in train'.format(train.shape[0],train.shape[1]))

print('There are {} rows and {} columns in test'.format(test.shape[0],test.shape[1]))
fig, axes = plt.subplots(1, 2, figsize=(10,10))

sns.heatmap(train.isnull(), ax=axes[0]).set_title("Train Data-Frame")

sns.heatmap(test.isnull(), ax=axes[1]).set_title("Test Data-Frame")



plt.suptitle("Heatmap For Finding Both Data-Frame's Missing Values", fontsize=25)

# plt.tight_layout()

plt.show()
fig, axes = plt.subplots(1, 2, figsize=(10,10))

common_locations_train = train.location.value_counts()[:10]

common_locations_test = test.location.value_counts()[:10]

sns.barplot(x=common_locations_train, y=common_locations_train.index, ax=axes[0]).set_title("Train Data-Frame")

sns.barplot(x=common_locations_test, y=common_locations_test.index, ax=axes[1]).set_title("Test Data-Frame")

plt.suptitle("Top 10 Locations From Both Data Tweet's", fontsize=20)

plt.tight_layout(pad=6.0)

plt.show()
test = test.drop("location", axis=1)



# Droping the Missing Values 

test = test.dropna(axis=0)



# As Data's indexs are not in order so :

test = test.reset_index()



# Now, droping the old indexs as it became a column

test = test.drop("index", axis=1)

print(test.info())

test.head()
train = train.drop("location", axis=1)



# Droping the Missing Values

train = train.dropna(axis=0)



# As Data's indexs are not in order so :

train = train.reset_index()



# Now, droping the old indexs as it became a column

train = train.drop("index", axis=1)

print(train.info())

train.head()
fig, axes = plt.subplots(1, 2, figsize=(10,10))

sns.barplot(x=train.target.value_counts().index, y=train.target.value_counts(), ax=axes[0]).set_title("Train Data-Frame")

sns.barplot(x=test.target.value_counts().index, y=test.target.value_counts(), ax=axes[1]).set_title("Test Data-Frame")

plt.suptitle("Distribution Of Classes", fontsize=20)

plt.tight_layout(pad=6.0)

plt.show()
fig, axes = plt.subplots(1, 2, figsize=(10,10))

common_keywords_train = train.keyword.value_counts()[:10]

common_keywords_test = test.keyword.value_counts()[:10]

sns.barplot(x=common_keywords_train, y=common_keywords_train.index, ax=axes[0]).set_title("Train Data-Frame")

sns.barplot(x=common_keywords_test, y=common_keywords_test.index, ax=axes[1]).set_title("Test Data-Frame")

plt.suptitle("Top 10 Keywords From Both Data Tweets", fontsize=20)

plt.tight_layout(pad=6.0)

plt.show()
print(train.text[20])

print("="*100)

print(train.text[120])

print("="*100)

print(train.text[220])

print("="*100)

print(train.text[320])
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "like", "via", "u", "video", "would", "one"]
# Pre_processing all the train data :-

lemmatizer = WordNetLemmatizer()

train.text = train.text.apply(lambda a: a.lower())

preprocessed_train = []

for sentance in tqdm(train.text.values):

    sentance = re.sub(r"http\S+", "", sentance)

    sentance = BeautifulSoup(sentance, 'lxml').get_text()

    sentance = re.sub("\S*\d\S*", "", sentance).strip()

    sentance = re.sub('[^A-Za-z]+', ' ', sentance)

    

    sentance = ' '.join(lemmatizer.lemmatize(e) for e in sentance.split() if e not in stopwords)

    preprocessed_train.append(sentance.strip())
print(preprocessed_train[20])

print("="*100)

print(preprocessed_train[120])

print("="*100)

print(preprocessed_train[220])

print("="*100)

print(preprocessed_train[320])
print(test.text[20])

print("="*100)

print(test.text[120])

print("="*100)

print(test.text[220])

print("="*100)

print(test.text[320])
# Pre_processing all the test data :-

test.text = test.text.apply(lambda a: a.lower())

preprocessed_test = []

for sentance in tqdm(test.text.values):

    sentance = re.sub(r"http\S+", "", sentance)

    sentance = BeautifulSoup(sentance, 'lxml').get_text()

    sentance = re.sub("\S*\d\S*", "", sentance).strip()

    sentance = re.sub('[^A-Za-z]+', ' ', sentance)

    

    sentance = ' '.join(lemmatizer.lemmatize(e) for e in sentance.split() if e not in stopwords)

    preprocessed_test.append(sentance.strip())
print(preprocessed_test[20])

print("="*100)

print(preprocessed_test[120])

print("="*100)

print(preprocessed_test[220])

print("="*100)

print(preprocessed_test[320])
# Converting preprocessed_train List into a Series to join it back in Train Data : 

final_text_train = pd.Series(preprocessed_train)

final_text_train.name = "final_text"



# Joining Train Data with preprocessed_train Series & droping the old text(Tweet) column :

train = pd.DataFrame.join(train, final_text_train)

train = train.drop("text", axis=1)

train.info()
# Converting preprocessed_test List into a Series to join it back in Test Data :

final_text_test = pd.Series(preprocessed_test)

final_text_test.name = "final_text"



# Joining Test Data with preprocessed_test Series & droping the old text(Tweet) column :

test = pd.DataFrame.join(test, final_text_test)

test = test.drop("text", axis=1)

test.info()
df = train.append(test, ignore_index=True)



# Id & Keyword columns are not of any use for Creating Model or for any Prediction's :

df = df.drop(["id", "keyword"], axis=1)



print(df.info())

df.head()
print('Now there are {} rows & {} columns in train.'.format(train.shape[0],train.shape[1]))

print('Now there are {} rows & {} columns in test.'.format(test.shape[0],test.shape[1]))

print('Now there are {} rows & {} columns in df(The Final Data-Frame).'.format(df.shape[0],df.shape[1]))
words = []

for sentences in tqdm(df.final_text.values):

    sentences = "".join(sentences.lower())

    words.append(sentences)

words = ''.join(word for word in words)

words = nltk.word_tokenize(words)

words = pd.Series(words)
plt.figure(figsize=(10,10))

sns.barplot(x=words.value_counts()[:15], y=words.value_counts()[:15].index)

plt.title("Top 15 Frequent Words In Tweet's", fontsize=20)

plt.show()
# Defining Input & Output :

X = df["final_text"]

y = df["target"]



# Spliting Final Data-Frame Into Train, Cross-Validation, Test :

X_1, X_test, y_1, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train, X_cv, y_train, y_cv = train_test_split(X_1, y_1, test_size=0.2)
fig, axes = plt.subplots(2, 2, figsize=(10,10))

sns.barplot(x=df.target.value_counts().index, y=df.target.value_counts(), ax=axes[0,0]).set_title("Final Data-Frame")

sns.barplot(x=y_train.value_counts().index, y=y_train.value_counts(), ax=axes[0,1]).set_title("Y_Train")

sns.barplot(x=y_cv.value_counts().index, y=y_cv.value_counts(), ax=axes[1,1]).set_title("Y_CV")

sns.barplot(x=y_test.value_counts().index, y=y_test.value_counts(), ax=axes[1,0]).set_title("Y_Test")

plt.suptitle("Distribution Of Classes", fontsize=20)

plt.tight_layout(pad=6.0)

plt.show()
print("Length Of X_train :-", X_train.shape[0])

print("Length Of y_train :-", y_train.shape[0])

print("Length Of X_test :-", X_test.shape[0])

print("Length Of y_test :-", y_test.shape[0])

print("Length Of X_cv :-", X_cv.shape[0])

print("Length Of y_cv :-", y_cv.shape[0])
bow = CountVectorizer(ngram_range=(1,2), min_df=2)

X_train_bow = bow.fit_transform(X_train).toarray()

X_cv_bow = bow.transform(X_cv).toarray()

X_test_bow = bow.transform(X_test).toarray()
log = LogisticRegression()

log.fit(X_train_bow, y_train)
pre_cv_bow_log = log.predict(X_cv_bow)

pre_test_bow_log = log.predict(X_test_bow)
print("BOW CV Classification Report by Logistic Regression")

print(classification_report(y_cv, pre_cv_bow_log))

print("="*100)

print("BOW Test Classification Report by Logistic Regression")

print(classification_report(y_test, pre_test_bow_log))
preproba_cv_bow_log = log.predict_proba(X_cv_bow)[:,1]

preproba_test_bow_log = log.predict_proba(X_test_bow)[:,1]

preproba_train_bow_log = log.predict_proba(X_train_bow)[:,1]



fpr_cv_bow_log_roc, tpr_cv_bow_log_roc, threshold_cv_bow_log_roc = roc_curve(y_cv, preproba_cv_bow_log)

fpr_test_bow_log_roc, tpr_test_bow_log_roc, threshold_test_bow_log_roc = roc_curve(y_test, preproba_test_bow_log)

fpr_train_bow_log_roc, tpr_train_bow_log_roc, threshold_train_bow_log_roc = roc_curve(y_train, preproba_train_bow_log)



fig = plt.figure()

ax = plt.subplot(111)

ax.plot(fpr_test_bow_log_roc,tpr_test_bow_log_roc, label='Test ROC AUC ='+str(roc_auc_score(y_test,preproba_test_bow_log)*100))

ax.plot(fpr_cv_bow_log_roc, tpr_cv_bow_log_roc, label='CV ROC AUC ='+str(roc_auc_score(y_cv, preproba_cv_bow_log)*100))

ax.plot(fpr_train_bow_log_roc,tpr_train_bow_log_roc, label='Train ROC AUC ='+str(roc_auc_score(y_train,preproba_train_bow_log)*100))

plt.title('ROC', fontsize=20)

plt.xlabel('FPR')

plt.ylabel('TPR')

ax.legend()

plt.show()



precision_test_bow_log_pr, recall_test_bow_log_pr, threshold_test_bow_log_pr = precision_recall_curve(y_test, preproba_test_bow_log)

precision_cv_bow_log_pr, recall_cv_bow_log_pr, threshold_cv_bow_log_pr = precision_recall_curve(y_cv, preproba_cv_bow_log)

precision_train_bow_log_pr, recall_train_bow_log_pr, threshold_train_bow_log_pr = precision_recall_curve(y_train, preproba_train_bow_log)

fig_1 = plt.figure()

ax_1 = plt.subplot(111)

ax_1.plot(recall_test_bow_log_pr, precision_test_bow_log_pr, label="Test")

ax_1.plot(recall_cv_bow_log_pr, precision_cv_bow_log_pr, label="CV")

ax_1.plot(recall_train_bow_log_pr, precision_train_bow_log_pr, label="Train")

plt.title('Precision-Recall Curve', fontsize=20)

plt.xlabel('Recall')

plt.ylabel('Precision')

ax_1.legend()

plt.show()
conf_test_bow_log = confusion_matrix(y_test, pre_test_bow_log)

class_label = ["1 (Positive)", "0 (Negative)"]

df = pd.DataFrame(conf_test_bow_log, index = class_label, columns = class_label)

sns.heatmap(df, annot = True,fmt="d")

plt.title("Confusion Matrix", fontsize=20)

plt.ylabel("Predicted Label")

plt.xlabel("True Label")

plt.show()

mnb = MultinomialNB()

mnb.fit(X_train_bow, y_train)
pre_cv_bow_mnb = mnb.predict(X_cv_bow)

pre_test_bow_mnb = mnb.predict(X_test_bow)
print("BOW CV Classification Report by Multi-Nomial Naiye Bayes")

print(classification_report(y_cv, pre_cv_bow_mnb))

print("="*100)

print("BOW Test Classification Report by Multi-Nomial Naiye Bayes")

print(classification_report(y_test, pre_test_bow_mnb))
preproba_cv_bow_mnb = mnb.predict_proba(X_cv_bow)[:,1]

preproba_test_bow_mnb = mnb.predict_proba(X_test_bow)[:,1]

preproba_train_bow_mnb = mnb.predict_proba(X_train_bow)[:,1]



fpr_cv_bow_mnb_roc, tpr_cv_bow_mnb_roc, threshold_cv_bow_mnb_roc = roc_curve(y_cv, preproba_cv_bow_mnb)

fpr_test_bow_mnb_roc, tpr_test_bow_mnb_roc, threshold_test_bow_mnb_roc = roc_curve(y_test, preproba_test_bow_mnb)

fpr_train_bow_mnb_roc, tpr_train_bow_mnb_roc, threshold_train_bow_mnb_roc = roc_curve(y_train, preproba_train_bow_mnb)



fig = plt.figure()

ax = plt.subplot(111)

ax.plot(fpr_test_bow_mnb_roc,tpr_test_bow_mnb_roc, label='Test ROC AUC ='+str(roc_auc_score(y_test,preproba_test_bow_mnb)*100))

ax.plot(fpr_cv_bow_mnb_roc, tpr_cv_bow_mnb_roc, label='CV ROC AUC ='+str(roc_auc_score(y_cv, preproba_cv_bow_mnb)*100))

ax.plot(fpr_train_bow_mnb_roc, tpr_train_bow_mnb_roc, label='Train ROC AUC ='+str(roc_auc_score(y_train, preproba_train_bow_mnb)*100))

plt.title('ROC', fontsize=20)

plt.xlabel('FPR')

plt.ylabel('TPR')

ax.legend()

plt.show()
precision_test_bow_mnb_pr, recall_test_bow_mnb_pr, threshold_test_bow_mnb_pr = precision_recall_curve(y_test, preproba_test_bow_mnb)

precision_cv_bow_mnb_pr, recall_cv_bow_mnb_pr, threshold_cv_bow_mnb_pr = precision_recall_curve(y_cv, preproba_cv_bow_mnb)

precision_train_bow_mnb_pr, recall_train_bow_mnb_pr, threshold_train_bow_mnb_pr = precision_recall_curve(y_train, preproba_train_bow_mnb)

fig_1 = plt.figure()

ax_1 = plt.subplot(111)

ax_1.plot(recall_test_bow_mnb_pr, precision_test_bow_mnb_pr, label="Test")

ax_1.plot(recall_cv_bow_mnb_pr, precision_cv_bow_mnb_pr, label="CV")

ax_1.plot(recall_train_bow_mnb_pr, precision_train_bow_mnb_pr, label="Train")

plt.title('Precision-Recall Curve', fontsize=20)

plt.xlabel('Recall')

plt.ylabel('Precision')

ax_1.legend()

plt.show()
conf_test_bow_mnb = confusion_matrix(y_test, pre_test_bow_mnb)

class_label = ["1 (Positive)", "0 (Negative)"]

df = pd.DataFrame(conf_test_bow_mnb, index = class_label, columns = class_label)

sns.heatmap(df, annot = True,fmt="d")

plt.title("Confusion Matrix", fontsize=20)

plt.ylabel("Predicted Label")

plt.xlabel("True Label")

plt.show()

dtc = DecisionTreeClassifier()

dtc.fit(X_train_bow, y_train)
pre_cv_bow_dtc = dtc.predict(X_cv_bow)

pre_test_bow_dtc = dtc.predict(X_test_bow)
print("BOW CV Classification Report by Decision-Tree Classifier")

print(classification_report(y_cv, pre_cv_bow_dtc))

print("="*100)

print("BOW Test Classification Report by Decision-Tree Classifier")

print(classification_report(y_test, pre_test_bow_dtc))
preproba_cv_bow_dtc = dtc.predict_proba(X_cv_bow)[:,1]

preproba_test_bow_dtc = dtc.predict_proba(X_test_bow)[:,1]

preproba_train_bow_dtc = dtc.predict_proba(X_train_bow)[:,1]



fpr_cv_bow_dtc_roc, tpr_cv_bow_dtc_roc, threshold_cv_bow_dtc_roc = roc_curve(y_cv, preproba_cv_bow_dtc)

fpr_test_bow_dtc_roc, tpr_test_bow_dtc_roc, threshold_test_bow_dtc_roc = roc_curve(y_test, preproba_test_bow_dtc)

fpr_train_bow_dtc_roc, tpr_train_bow_dtc_roc, threshold_train_bow_dtc_roc = roc_curve(y_train, preproba_train_bow_dtc)



fig = plt.figure()

ax = plt.subplot(111)

ax.plot(fpr_test_bow_dtc_roc,tpr_test_bow_dtc_roc, label='Test ROC AUC ='+str(roc_auc_score(y_test,preproba_test_bow_dtc)*100))

ax.plot(fpr_cv_bow_dtc_roc, tpr_cv_bow_dtc_roc, label='CV ROC AUC ='+str(roc_auc_score(y_cv, preproba_cv_bow_dtc)*100))

ax.plot(fpr_train_bow_dtc_roc, tpr_train_bow_dtc_roc, label='Train ROC AUC ='+str(roc_auc_score(y_train, preproba_train_bow_dtc)*100))

plt.title('ROC', fontsize=20)

plt.xlabel('FPR')

plt.ylabel('TPR')

ax.legend()

plt.show()
precision_test_bow_dtc_pr, recall_test_bow_dtc_pr, threshold_test_bow_dtc_pr = precision_recall_curve(y_test, preproba_test_bow_dtc)

precision_cv_bow_dtc_pr, recall_cv_bow_dtc_pr, threshold_cv_bow_dtc_pr = precision_recall_curve(y_cv, preproba_cv_bow_dtc)

precision_train_bow_dtc_pr, recall_train_bow_dtc_pr, threshold_train_bow_dtc_pr = precision_recall_curve(y_train, preproba_train_bow_dtc)

fig_1 = plt.figure()

ax_1 = plt.subplot(111)

ax_1.plot(recall_test_bow_dtc_pr, precision_test_bow_dtc_pr, label="Test")

ax_1.plot(recall_cv_bow_dtc_pr, precision_cv_bow_dtc_pr, label="CV")

ax_1.plot(recall_train_bow_dtc_pr, precision_train_bow_dtc_pr, label="Train")

plt.title('Precision-Recall Curve', fontsize=20)

plt.xlabel('Recall')

plt.ylabel('Precision')

ax_1.legend()

plt.show()
conf_test_bow_dtc = confusion_matrix(y_test, pre_test_bow_dtc)

class_label = ["1 (Positive)", "0 (Negative)"]

df = pd.DataFrame(conf_test_bow_dtc, index = class_label, columns = class_label)

sns.heatmap(df, annot = True,fmt="d")

plt.title("Confusion Matrix", fontsize=20)

plt.ylabel("Predicted Label")

plt.xlabel("True Label")

plt.show()
fig = plt.figure()

ax = plt.subplot(111)

ax.plot(fpr_test_bow_log_roc,tpr_test_bow_log_roc, label='Logistic Test ROC AUC ='+str(roc_auc_score(y_test,preproba_test_bow_log)*100))

ax.plot(fpr_test_bow_mnb_roc,tpr_test_bow_mnb_roc, label='Multi-Nomial Test ROC AUC ='+str(roc_auc_score(y_test,preproba_test_bow_mnb)*100))

ax.plot(fpr_test_bow_dtc_roc,tpr_test_bow_dtc_roc, label='Decision-Tree Test ROC AUC ='+str(roc_auc_score(y_test,preproba_test_bow_dtc)*100))

plt.title('ROC Comparison of Logistic Vs Multi-Nomial Vs Decision-Tree in BOW', fontsize=20)

plt.xlabel('FPR')

plt.ylabel('TPR')

ax.legend()

plt.show()
fig = plt.figure()

ax_1 = plt.subplot(111)

ax_1.plot(recall_test_bow_log_pr, precision_test_bow_log_pr, label="Logistic Test PR Curve")

ax_1.plot(recall_test_bow_mnb_pr, precision_test_bow_mnb_pr, label="Multi-Nomial Test PR Curve")

ax_1.plot(recall_test_bow_dtc_pr, precision_test_bow_dtc_pr, label="Decision-Tree Test PR Curve")

plt.title('PR Curve Comparison of Logistic Vs Multi-Nomial Vs Decision-Tree in BOW', fontsize=20)

plt.xlabel('FPR')

plt.ylabel('TPR')

ax_1.legend()

plt.show()
tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2)

X_train_tfidf = tfidf.fit_transform(X_train).toarray()

X_cv_tfidf = tfidf.transform(X_cv).toarray()

X_test_tfidf = tfidf.transform(X_test).toarray()
log = LogisticRegression()

log.fit(X_train_tfidf, y_train)
pre_cv_tfidf_log = log.predict(X_cv_tfidf)

pre_test_tfidf_log = log.predict(X_test_tfidf)
print("TF-IDF CV Classification Report by Logistic Regresion")

print(classification_report(y_cv, pre_cv_tfidf_log))

print("="*100)

print("TF-IDF Test Classification Report by Logistic Regresion")

print(classification_report(y_test, pre_test_tfidf_log))
preproba_cv_tfidf_log = log.predict_proba(X_cv_tfidf)[:,1]

preproba_test_tfidf_log = log.predict_proba(X_test_tfidf)[:,1]

preproba_train_tfidf_log = log.predict_proba(X_train_tfidf)[:,1]



fpr_cv_tfidf_log_roc, tpr_cv_tfidf_log_roc, threshold_cv_tfidf_log_roc = roc_curve(y_cv, preproba_cv_tfidf_log)

fpr_test_tfidf_log_roc, tpr_test_tfidf_log_roc, threshold_test_tfidf_log_roc = roc_curve(y_test, preproba_test_tfidf_log)

fpr_train_tfidf_log_roc, tpr_train_tfidf_log_roc, threshold_train_tfidf_log_roc = roc_curve(y_train, preproba_train_tfidf_log)



fig = plt.figure()

ax = plt.subplot(111)

ax.plot(fpr_test_tfidf_log_roc,tpr_test_tfidf_log_roc, label='Test ROC AUC ='+str(roc_auc_score(y_test,preproba_test_tfidf_log)*100))

ax.plot(fpr_cv_tfidf_log_roc, tpr_cv_tfidf_log_roc, label='CV ROC AUC ='+str(roc_auc_score(y_cv, preproba_cv_tfidf_log)*100))

ax.plot(fpr_train_tfidf_log_roc,tpr_train_tfidf_log_roc, label='Train ROC AUC ='+str(roc_auc_score(y_train,preproba_train_tfidf_log)*100))

plt.title('ROC', fontsize=20)

plt.xlabel('FPR')

plt.ylabel('TPR')

ax.legend()

plt.show()
precision_test_tfidf_log_pr, recall_test_tfidf_log_pr, threshold_test_tfidf_log_pr = precision_recall_curve(y_test, preproba_test_tfidf_log)

precision_cv_tfidf_log_pr, recall_cv_tfidf_log_pr, threshold_cv_tfidf_log_pr = precision_recall_curve(y_cv, preproba_cv_tfidf_log)

precision_train_tfidf_log_pr, recall_train_tfidf_log_pr, threshold_train_tfidf_log_pr = precision_recall_curve(y_train, preproba_train_tfidf_log)

fig_1 = plt.figure()

ax_1 = plt.subplot(111)

ax_1.plot(recall_test_tfidf_log_pr, precision_test_tfidf_log_pr, label="Test")

ax_1.plot(recall_cv_tfidf_log_pr, precision_cv_tfidf_log_pr, label="CV")

ax_1.plot(recall_train_tfidf_log_pr, precision_train_tfidf_log_pr, label="Train")

plt.title('Precision-Recall Curve', fontsize=20)

plt.xlabel('Recall')

plt.ylabel('Precision')

ax_1.legend()

plt.show()
conf_test_tfidf_log = confusion_matrix(y_test, pre_test_tfidf_log)

class_label = ["1 (Positive)", "0 (Negative)"]

df = pd.DataFrame(conf_test_tfidf_log, index = class_label, columns = class_label)

sns.heatmap(df, annot = True,fmt="d")

plt.title("Confusion Matrix", fontsize=20)

plt.ylabel("Predicted Label")

plt.xlabel("True Label")

plt.show()

mnb = MultinomialNB()

mnb.fit(X_train_tfidf, y_train)
pre_cv_tfidf_mnb = mnb.predict(X_cv_tfidf)

pre_test_tfidf_mnb = mnb.predict(X_test_tfidf)
print("TF-IDF CV Classification Report by Multi-Nomial Naive Bayes")

print(classification_report(y_cv, pre_cv_tfidf_mnb))

print("="*100)

print("TF-IDF Test Classification Report by Multi-Nomial Naive Bayes")

print(classification_report(y_test, pre_test_tfidf_mnb))
preproba_cv_tfidf_mnb = mnb.predict_proba(X_cv_tfidf)[:,1]

preproba_test_tfidf_mnb = mnb.predict_proba(X_test_tfidf)[:,1]

preproba_train_tfidf_mnb = mnb.predict_proba(X_train_tfidf)[:,1]



fpr_cv_tfidf_mnb_roc, tpr_cv_tfidf_mnb_roc, threshold_cv_tfidf_mnb_roc = roc_curve(y_cv, preproba_cv_tfidf_mnb)

fpr_test_tfidf_mnb_roc, tpr_test_tfidf_mnb_roc, threshold_test_tfidf_mnb_roc = roc_curve(y_test, preproba_test_tfidf_mnb)

fpr_train_tfidf_mnb_roc, tpr_train_tfidf_mnb_roc, threshold_train_tfidf_mnb_roc = roc_curve(y_train, preproba_train_tfidf_mnb)



fig = plt.figure()

ax = plt.subplot(111)

ax.plot(fpr_test_tfidf_mnb_roc,tpr_test_tfidf_mnb_roc, label='Test ROC AUC ='+str(roc_auc_score(y_test,preproba_test_tfidf_mnb)*100))

ax.plot(fpr_cv_tfidf_mnb_roc, tpr_cv_tfidf_mnb_roc, label='CV ROC AUC ='+str(roc_auc_score(y_cv, preproba_cv_tfidf_mnb)*100))

ax.plot(fpr_train_tfidf_mnb_roc,tpr_train_tfidf_mnb_roc, label='Train ROC AUC ='+str(roc_auc_score(y_train,preproba_train_tfidf_mnb)*100))

plt.title('ROC', fontsize=20)

plt.xlabel('FPR')

plt.ylabel('TPR')

ax.legend()

plt.show()
precision_test_tfidf_mnb_pr, recall_test_tfidf_mnb_pr, threshold_test_tfidf_mnb_pr = precision_recall_curve(y_test, preproba_test_tfidf_mnb)

precision_cv_tfidf_mnb_pr, recall_cv_tfidf_mnb_pr, threshold_cv_tfidf_mnb_pr = precision_recall_curve(y_cv, preproba_cv_tfidf_mnb)

precision_train_tfidf_mnb_pr, recall_train_tfidf_mnb_pr, threshold_train_tfidf_mnb_pr = precision_recall_curve(y_train, preproba_train_tfidf_mnb)

fig_1 = plt.figure()

ax_1 = plt.subplot(111)

ax_1.plot(recall_test_tfidf_mnb_pr, precision_test_tfidf_mnb_pr, label="Test")

ax_1.plot(recall_cv_tfidf_mnb_pr, precision_cv_tfidf_mnb_pr, label="CV")

ax_1.plot(recall_train_tfidf_mnb_pr, precision_train_tfidf_mnb_pr, label="Train")

plt.title('Precision-Recall Curve', fontsize=20)

plt.xlabel('Recall')

plt.ylabel('Precision')

ax_1.legend()

plt.show()
conf_test_tfidf_mnb = confusion_matrix(y_test, pre_test_tfidf_mnb)

class_label = ["1 (Positive)", "0 (Negative)"]

df = pd.DataFrame(conf_test_tfidf_mnb, index = class_label, columns = class_label)

sns.heatmap(df, annot = True,fmt="d")

plt.title("Confusion Matrix", fontsize=20)

plt.ylabel("Predicted Label")

plt.xlabel("True Label")

plt.show()

dtc = DecisionTreeClassifier()

dtc.fit(X_train_tfidf, y_train)
pre_cv_tfidf_dtc = dtc.predict(X_cv_tfidf)

pre_test_tfidf_dtc = dtc.predict(X_test_tfidf)
print("TF-IDF CV Classification Report by Decision-Tree Classifier")

print(classification_report(y_cv, pre_cv_tfidf_dtc))

print("="*100)

print("TF-_IDF Test Classification Report by Decision-Tree Classifier")

print(classification_report(y_test, pre_test_tfidf_dtc))
preproba_cv_tfidf_dtc = dtc.predict_proba(X_cv_tfidf)[:,1]

preproba_test_tfidf_dtc = dtc.predict_proba(X_test_tfidf)[:,1]

preproba_train_tfidf_dtc = dtc.predict_proba(X_train_tfidf)[:,1]



fpr_cv_tfidf_dtc_roc, tpr_cv_tfidf_dtc_roc, threshold_cv_tfidf_dtc_roc = roc_curve(y_cv, preproba_cv_tfidf_dtc)

fpr_test_tfidf_dtc_roc, tpr_test_tfidf_dtc_roc, threshold_test_tfidf_dtc_roc = roc_curve(y_test, preproba_test_tfidf_dtc)

fpr_train_tfidf_dtc_roc, tpr_train_tfidf_dtc_roc, threshold_train_tfidf_dtc_roc = roc_curve(y_train, preproba_train_tfidf_dtc)



fig = plt.figure()

ax = plt.subplot(111)

ax.plot(fpr_test_tfidf_dtc_roc,tpr_test_tfidf_dtc_roc, label='Test ROC AUC ='+str(roc_auc_score(y_test,preproba_test_tfidf_dtc)*100))

ax.plot(fpr_cv_tfidf_dtc_roc, tpr_cv_tfidf_dtc_roc, label='CV ROC AUC ='+str(roc_auc_score(y_cv, preproba_cv_tfidf_dtc)*100))

ax.plot(fpr_train_tfidf_dtc_roc, tpr_train_tfidf_dtc_roc, label='Train ROC AUC ='+str(roc_auc_score(y_train, preproba_train_tfidf_dtc)*100))

plt.title('ROC', fontsize=20)

plt.xlabel('FPR')

plt.ylabel('TPR')

ax.legend()

plt.show()
precision_test_tfidf_dtc_pr, recall_test_tfidf_dtc_pr, threshold_test_tfidf_dtc_pr = precision_recall_curve(y_test, preproba_test_tfidf_dtc)

precision_cv_tfidf_dtc_pr, recall_cv_tfidf_dtc_pr, threshold_cv_tfidf_dtc_pr = precision_recall_curve(y_cv, preproba_cv_tfidf_dtc)

precision_train_tfidf_dtc_pr, recall_train_tfidf_dtc_pr, threshold_train_tfidf_dtc_pr = precision_recall_curve(y_train, preproba_train_tfidf_dtc)

fig_1 = plt.figure()

ax_1 = plt.subplot(111)

ax_1.plot(recall_test_tfidf_dtc_pr, precision_test_tfidf_dtc_pr, label="Test")

ax_1.plot(recall_cv_tfidf_dtc_pr, precision_cv_tfidf_dtc_pr, label="CV")

ax_1.plot(recall_train_tfidf_dtc_pr, precision_train_tfidf_dtc_pr, label="Train")

plt.title('Precision-Recall Curve', fontsize=20)

plt.xlabel('Recall')

plt.ylabel('Precision')

ax_1.legend()

plt.show()
conf_test_tfidf_dtc = confusion_matrix(y_test, pre_test_tfidf_dtc)

class_label = ["1 (Positive)", "0 (Negative)"]

df = pd.DataFrame(conf_test_tfidf_dtc, index = class_label, columns = class_label)

sns.heatmap(df, annot = True,fmt="d")

plt.title("Confusion Matrix", fontsize=20)

plt.ylabel("Predicted Label")

plt.xlabel("True Label")

plt.show()
fig = plt.figure()

ax = plt.subplot(111)

ax.plot(fpr_test_tfidf_log_roc,tpr_test_tfidf_log_roc, label='Logistic Test ROC AUC ='+str(roc_auc_score(y_test,preproba_test_tfidf_log)*100))

ax.plot(fpr_test_tfidf_mnb_roc,tpr_test_tfidf_mnb_roc, label='Multi-Nomial Test ROC AUC ='+str(roc_auc_score(y_test,preproba_test_tfidf_mnb)*100))

ax.plot(fpr_test_tfidf_dtc_roc,tpr_test_tfidf_dtc_roc, label='Decision-Tree Test ROC AUC ='+str(roc_auc_score(y_test,preproba_test_tfidf_dtc)*100))

plt.title('ROC Comparison of Logistic Vs Multi-Nomial Vs Decision-Tree in TF-IDF', fontsize=20)

plt.xlabel('FPR')

plt.ylabel('TPR')

ax.legend()

plt.show()
fig = plt.figure()

ax_1 = plt.subplot(111)

ax_1.plot(recall_test_tfidf_log_pr, precision_test_tfidf_log_pr, label="Logistic Test PR Curve")

ax_1.plot(recall_test_tfidf_mnb_pr, precision_test_tfidf_mnb_pr, label="Multi-Nomial Test PR Curve")

ax_1.plot(recall_test_tfidf_dtc_pr, precision_test_tfidf_dtc_pr, label="Decision-Tree Test PR Curve")

plt.title('PR Curve Comparison of Logistic Vs Multi-Nomial Vs Decision-Tree in TF-IDF', fontsize=20)

plt.xlabel('FPR')

plt.ylabel('TPR')

ax_1.legend()

plt.show()
fig_2 = plt.figure(figsize=(15,15))



# Ploting fig_2

ax_5 = plt.subplot(221)

ax_5.set_title("Comparing Logistic's", fontsize=20)

ax_5.plot(fpr_test_tfidf_log_roc,tpr_test_tfidf_log_roc, label='Logistic TF-IDF ROC AUC ='+str(roc_auc_score(y_test,preproba_test_tfidf_log)*100))

ax_5.plot(fpr_test_bow_log_roc,tpr_test_bow_log_roc, label='Logistic BOW ROC AUC ='+str(roc_auc_score(y_test,preproba_test_bow_log)*100))



ax_6 = plt.subplot(222)

ax_6.set_title("Comparing Multi-Nomial Naive Baye's", fontsize=20)

ax_6.plot(fpr_test_tfidf_mnb_roc,tpr_test_tfidf_mnb_roc, label='Multi-Nomial TF-IDF ROC AUC ='+str(roc_auc_score(y_test,preproba_test_tfidf_mnb)*100))

ax_6.plot(fpr_test_bow_mnb_roc,tpr_test_bow_mnb_roc, label='Multi-Nomial BOW ROC AUC ='+str(roc_auc_score(y_test,preproba_test_bow_mnb)*100))



ax_7 = plt.subplot(212)

ax_7.set_title("Finally Comparing Top ROC AUC Curve's", fontsize=20)

ax_7.plot(fpr_test_tfidf_log_roc,tpr_test_tfidf_log_roc, label='Logistic TF-IDF ROC AUC ='+str(roc_auc_score(y_test,preproba_test_tfidf_log)*100))

ax_7.plot(fpr_test_bow_mnb_roc,tpr_test_bow_mnb_roc, label='Multi-Nomial BOW ROC AUC ='+str(roc_auc_score(y_test,preproba_test_bow_mnb)*100))





ax_5.legend()

ax_6.legend()

ax_7.legend()

plt.suptitle('Final ROC AUC Comparison Of Logistic & Multi-Nomial In BOW Vs TF-IDF', fontsize=25)

plt.show()

x = PrettyTable()

x.field_names = ["Vectorizer", "Model", "AUC (in %)", "Precision Score (in %)", "Recall Score (in %)"]

x.add_row(["BOW", "Logistic Regression", round(roc_auc_score(y_test,preproba_test_bow_log)*100), round(precision_score(y_test,pre_test_bow_log)*100), round(recall_score(y_test,pre_test_bow_log)*100)])

x.add_row(["BOW", "Multi-Nomial Naive Bayes", round(roc_auc_score(y_test,preproba_test_bow_mnb)*100), round(precision_score(y_test,pre_test_bow_mnb)*100), round(recall_score(y_test,pre_test_bow_mnb)*100)])

x.add_row(["BOW", "Decision-Tree Classifier", round(roc_auc_score(y_test,preproba_test_bow_dtc)*100), round(precision_score(y_test,pre_test_bow_dtc)*100), round(recall_score(y_test,pre_test_bow_dtc)*100)])

x.add_row(["TF-IDF", "Logistic Regression", round(roc_auc_score(y_test,preproba_test_tfidf_log)*100), round(precision_score(y_test,pre_test_tfidf_log)*100), round(recall_score(y_test,pre_test_tfidf_log)*100)])

x.add_row(["TF-IDF", "Multi-Nomial Naive Bayes", round(roc_auc_score(y_test,preproba_test_tfidf_mnb)*100), round(precision_score(y_test,pre_test_tfidf_mnb)*100), round(recall_score(y_test,pre_test_tfidf_mnb)*100)])

x.add_row(["TF-IDF", "Decision-Tree Classifier", round(roc_auc_score(y_test,preproba_test_tfidf_dtc)*100), round(precision_score(y_test,pre_test_tfidf_dtc)*100), round(recall_score(y_test,pre_test_tfidf_dtc)*100)])

print(x)
