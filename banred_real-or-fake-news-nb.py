# Import our visual libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

# Import text clearning libraries
import re
from bs4 import BeautifulSoup
import nltk

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Input full file path for True news dataset
real = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")
real.head()
# Input full file path for Fake news dataset
fake = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")
fake.head()
print("Real news count: " + str(len(real)))
print("Fake news count: " + str(len(fake)))
print("Total available entries: " + str(len(real) + len(fake)))
# Merge the datasets
real["label"] = 1
fake["label"] = 0

frame = [real, fake]
df = pd.concat(frame)

df.head()
df.tail()
# Check to see if any columns have missing values
df.isnull().sum()
# Get basic information on dataset
df.info()
# Run this cell to avoid indexes of previous datasets from overlapping
df.reset_index(inplace = True)
df.drop("index", axis=1, inplace = True)
df.columns
df["text"][40000]
df["text"][10000]
# 0 for fake
# 1 for true
sns.set(style="darkgrid")
sns.countplot(df["label"])
df["subject"].value_counts()
# Chart to show count of subject by label
plt.figure(figsize=(12,9))
sns.set(style="darkgrid")
sns.countplot(df["subject"], hue=df["label"])
set(df["date"])
# Filter out dates with http links
httpremove = "http"
filter1 = df["date"].str.contains(httpremove)
df_ = df[filter1]
df_
df_["text"][30775]
# Dataset with wrong dates and meanliness text
df_ = df[df["date"].apply(lambda x: len(x) > 20)]
df_
# Want to remove entries which are not dates
df = df[df["date"].apply(lambda x: len(x) < 20)]
df.head()
# 44888 entries after removing non-dates
df["title"].count()
df_ = df.copy()
df_["date"]
# Transform dates to datetime
# Use to_period('M') to get datetime to month
df_['date'] = pd.to_datetime(df_['date']).dt.to_period('M')
df_.head()
# Check count of articles by Year
# Over half of the articles are from 2017
df_["date"].apply(lambda x: (str(x)[:4])).value_counts()
# Get number of articles by Year-Month
# Change date type to string from datetime format
df_["date"] = df_["date"].apply(lambda x: (str(x)[:7]))
df_.head()
# DataFrame of year of count by Year-Month
year_month = pd.DataFrame(df_["date"].value_counts()).sort_index()
year_month.reset_index(inplace=True)
year_month["index"] = year_month["index"].astype(str)
year_month
# Count of articles by Month
plt.figure(figsize=(12,9))
plt.bar(year_month["index"], year_month["date"])
plt.xticks(rotation=45)
plt.xlabel("Year and Month")
plt.ylabel("Count")
plt.title("Count of articles by Month/Year")
plt.show
# Plot count of articles by Month-Year
df_1 = df_[df_["label"]==1]
df_0 = df_[df_["label"]==0]
df_1 = pd.DataFrame(df_1["date"].value_counts()).sort_index()
df_1.rename(columns={"date": "true"}, inplace=True)
df_0 = pd.DataFrame(df_0["date"].value_counts()).sort_index()
df_0.rename(columns={"date": "false"}, inplace=True)

new_df = df_1.join(df_0, how='outer')
new_df.reset_index(inplace=True)
new_df
# Plot Count of articles by Month
plt.figure(figsize=(15,7))
plt.plot(new_df["index"], new_df["true"], label="True")
plt.plot(new_df["index"], new_df["false"], color="red", label="Fake")
plt.xticks(rotation=45)
plt.legend(facecolor='white')
plt.xlabel("Year and Month")
plt.ylabel("Count")
plt.title("Count of articles by Month/Year")
plt.show
# Create side-by-side histograms of True and Fake news text
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,9))

true_length = df[df["label"]==1]["text"].str.len()
ax1.set_title("True text")
ax1.hist(true_length, color="blue")

fake_length = df[df["label"]==0]["text"].str.len()
ax2.set_title("Fake text")
ax2.hist(fake_length, color="red")

fig.suptitle("Length of text (by characters)")
plt.show()
df["title"].count()
# Create side-by-side histograms of True and Fake news text
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,9))

true_length = df[df["label"]==1]["title"].str.len()
ax1.set_title("True titles")
ax1.hist(true_length, color="blue")

fake_length = df[df["label"]==0]["title"].str.len()
ax2.set_title("Fake titles")
ax2.hist(fake_length, color="red")

fig.suptitle("Length of title (by characters)")
plt.show()
# Merge text and title columns, remove title, subject and date.
df['text'] = df['title'] + " " + df['text']
del df['title']
del df['subject']
del df['date']
# Check if anything missing after cleaning
df.isna().sum()
df.head()
# Stopwords to remove from text will have little effect on the context of the text
# Stopwords in list already in lower case
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)
# Use this to deal with apostophes and abbreviation
# df_["news"] = df_['news'].str.replace('[^\w\s]','')
def remove_apostrophe_abbrev(text):
    return re.sub('[^\w\s]','', text)

# Function to remove stopwords
def remove_stop_words(text):
    clean_text = []
    for word in text.split():
        if word.strip().lower() not in stop_words:
            clean_text.append(word)
            
    return " ".join(clean_text)

# Text example of stop words removed
remove_stop_words(df["text"][0])
# Functions for cleaning text

# Remove html tags, using regex is bad idea
def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Remove links in text, http and https
# s? in regex means or s (case sensitive)
def remove_links(text):
    return re.sub('https?:\/\/\S+', '', text)

# Remove 's (possessive pronouns) from text
# Two kinds of apostophes found
def remove_possessive_pronoun(text):
    return re.sub("’s|'s", '', text)

# Remove between brackets and their contents
def remove_between_brackets(text):
    return re.sub('\([^]]*\)', '', text)

# Remove square brackets and their contents
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# Remove curly brackets and their contents
# Useful to remove any JavaScript scripts, though removed
# through html as above
def remove_between_curly_brackets(text):
    return re.sub('\{[^]]*\}', '', text)

def remove_n_space(text):
    return re.sub('\n', '', text)

def text_cleaner(text):
    text = remove_html_tags(text)
    text = remove_links(text)
    text = remove_possessive_pronoun(text)
    text = remove_apostrophe_abbrev(text)
    text = remove_between_brackets(text)
    text = remove_between_square_brackets(text)
    text = remove_between_curly_brackets(text)
    text = remove_n_space(text)
    text = remove_stop_words(text)
    
    return text
# Apply cleaning functions to text
df["text"] = df["text"].apply(text_cleaner)
from wordcloud import WordCloud, STOPWORDS 
# True news wordcloud
plt.figure(figsize=(15,15))
wordcloud = WordCloud(max_words = 1000 , width = 1600 , 
                      height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df["label"] == 1].text))

plt.axis("off")
plt.imshow(wordcloud)
# Fake news wordcloud
plt.figure(figsize=(15,15))
wordcloud = WordCloud(max_words = 1000 , width = 1600 , 
                      height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df["label"] == 0].text))

plt.axis("off")
plt.imshow(wordcloud)
from collections import Counter
# Get 25 most common words in True news
true_corpus = pd.Series(" ".join(df[df["label"] == 1].text))[0].split()

counter = Counter(true_corpus)
true_common = counter.most_common(25)
true_common = dict(true_common)
true_common
# as a dataframe
true_common_df = pd.DataFrame(true_common.items(), columns = ["words", "count"])
true_common_df.set_index("words")
# Histogram of 25 true common words
plt.figure(figsize=(12,9))
plt.bar(true_common.keys(), true_common.values())
plt.xticks(rotation=45)
plt.xlabel("Common words")
plt.ylabel("Count")
plt.title("25 Most common words (True)")
plt.show
# Get 25 most common words in Fake news
fake_corpus = pd.Series(" ".join(df[df["label"] == 0].text))[0].split()

counter = Counter(fake_corpus)
fake_common = counter.most_common(25)
fake_common = dict(fake_common)
fake_common
# as a dataframe
fake_common_df = pd.DataFrame(fake_common.items(), columns = ["words", "count"])
fake_common_df.set_index("words")
# Histogram of 25 fake common words
plt.figure(figsize=(12,9))
plt.bar(fake_common.keys(), fake_common.values(), color="red")
plt.xticks(rotation=45)
plt.xlabel("Common words")
plt.ylabel("Count")
plt.title("25 Most common words (Fake)")
plt.show
from nltk.util import ngrams
# Find most common bigrams in True news
text = pd.Series(" ".join(df[df["label"] == 1].text))[0]
tokenizer = nltk.RegexpTokenizer(r"\w+")
token = tokenizer.tokenize(text)

# ngrams set to 2
counter = Counter(ngrams(token,2))
most_common = counter.most_common(25)
most_common = dict(most_common)
most_common
# as a dataframe
true_common_bi = pd.DataFrame(most_common.items(), columns = ["bigram", "count"])
true_common_bi["bigram"] = true_common_bi["bigram"].apply(lambda x: " ".join(x))
true_common_bi
# Histogram of 25 common bigrams for True news
plt.figure(figsize=(12,9))
plt.bar(true_common_bi["bigram"], true_common_bi["count"]) # can do tuples
plt.xticks(rotation=90)
plt.xlabel("Common bigram")
plt.ylabel("Count")
plt.title("25 Most common bigram (True)")
plt.show
# Find most common bigrams in Fake news
text = pd.Series(" ".join(df[df["label"] == 0].text))[0]
tokenizer = nltk.RegexpTokenizer(r"\w+")
token = tokenizer.tokenize(text)

# ngrams set to 2
counter = Counter(ngrams(token,2))
most_common = counter.most_common(25)
most_common = dict(most_common)
most_common
# as a dataframe
fake_common_bi = pd.DataFrame(most_common.items(), columns = ["bigram", "count"])
fake_common_bi["bigram"] = fake_common_bi["bigram"].apply(lambda x: " ".join(x))
fake_common_bi
# Histogram of 25 common bigrams for Fakefake news
plt.figure(figsize=(12,9))
plt.bar(fake_common_bi["bigram"], fake_common_bi["count"], color="red") # can do tuples
plt.xticks(rotation=90)
plt.xlabel("Common bigram")
plt.ylabel("Count")
plt.title("25 Most common bigram (Fake)")
plt.show
# Find most common trigrams in True news
text = pd.Series(" ".join(df[df["label"] == 1].text))[0]
tokenizer = nltk.RegexpTokenizer(r"\w+")
token = tokenizer.tokenize(text)

# ngrams set to 3
counter = Counter(ngrams(token,3))
most_common = counter.most_common(25)
most_common = dict(most_common)
most_common
# as a dataframe
true_common_tri = pd.DataFrame(most_common.items(), columns = ["trigram", "count"])
true_common_tri["trigram"] = true_common_tri["trigram"].apply(lambda x: " ".join(x))
true_common_tri
# Histogram of 25 common trigrams for True news
plt.figure(figsize=(12,9))
plt.bar(true_common_tri["trigram"], true_common_tri["count"])
plt.xticks(rotation=90)
plt.xlabel("Common trigram")
plt.ylabel("Count")
plt.title("25 Most common trigram (True)")
plt.show
# Find most common trigrams in Fake news
text = pd.Series(" ".join(df[df["label"] == 0].text))[0]
tokenizer = nltk.RegexpTokenizer(r"\w+")
token = tokenizer.tokenize(text)

# ngrams set to 3
counter = Counter(ngrams(token,3))
most_common = counter.most_common(25)
most_common = dict(most_common)
most_common
# as a dataframe
fake_common_tri = pd.DataFrame(most_common.items(), columns = ["trigram", "count"])
fake_common_tri["trigram"] = fake_common_tri["trigram"].apply(lambda x: " ".join(x))
fake_common_tri
# Histogram of 25 common trigrams for True news
plt.figure(figsize=(12,9))
plt.bar(fake_common_tri["trigram"], fake_common_tri["count"], color="red")
plt.xticks(rotation=90)
plt.xlabel("Common trigram")
plt.ylabel("Count")
plt.title("25 Most common trigram (Fake)")
plt.show
from nltk.stem import WordNetLemmatizer


lemma = WordNetLemmatizer()
# Lemmatizer example
print(lemma.lemmatize("boys"))
# Function to perform lemmatization on text
def lemmatize_text(text):
    tokenize_text = nltk.word_tokenize(text)
    lemmatize_words = [lemma.lemmatize(word) for word in tokenize_text]
    join_text = ' '.join(lemmatize_words)
    
    return join_text

# Example sentence on function
lemmatize_text("There once was a boy named Naruto who was possessed by a Nine-Tailed Demon Fox")
# Copy main df dataset and lemmatize the text
lemmatized_df = df.copy()
lemmatized_df["text"] = lemmatized_df["text"].apply(lemmatize_text)
lemmatized_df.head()
lemmatized_df["text"][0]
# Machine Learning models to import
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
# Random state at 42 for reproducibility
# Go for 80:20 train:test set
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(lemmatized_df["text"], lemmatized_df["label"], test_size=test_size, random_state=42)
# Fit CountVectorizer to X_train and X_test datasets
cv_train = CountVectorizer(max_features=10000).fit(X_train)
X_vec_train = cv_train.transform(X_train)
X_vec_test = cv_train.transform(X_test)
X_vec_train
# Training on Logistic Regression model
# set LR parameter max_iter = 4000 to avoid error
lr = LogisticRegression(max_iter = 4000)
lr.fit(X_vec_train, y_train)
predicted_value = lr.predict(X_vec_test)
lr_accuracy_value = roc_auc_score(y_test, predicted_value)
# Logistic Regression Test ROC 99.68% on just lemmatized text
print("ROC: " + str(lr_accuracy_value*100) + "%")
conmat = confusion_matrix(y_test, predicted_value)
print(conmat)
print(classification_report(y_test, predicted_value))
# Visual of confusion matrix of Logistic Regression
fig = plt.subplot()
sns.heatmap(conmat, annot=True, ax=fig)
fig.set_ylabel('y_test')
fig.set_xlabel('predicted values')
# Training on Naive Bayes model
# Quick at training
nb = MultinomialNB()
nb.fit(X_vec_train, y_train)
predicted_value = nb.predict(X_vec_test)
nb_accuracy_value = roc_auc_score(y_test, predicted_value)
# Naive Bayes Training ROC 95.21% lemmatized text
print("ROC: " + str(roc_auc_score(y_train, nb.predict(X_vec_train))*100) + "%")
# Naive Bayes Test ROC 95.22% lemmatized text
print("ROC: " + str(nb_accuracy_value*100) + "%")
conmat = confusion_matrix(y_test, predicted_value)
print(conmat)
print(classification_report(y_test, predicted_value))
# Visual of confusion matrix of Naive Bayes
fig = plt.subplot()
sns.heatmap(conmat, annot=True, ax=fig)
fig.set_ylabel('y_test')
fig.set_xlabel('predicted values')
# Training on Support Vector Machine
# Very slow at training
# time complexity O(no.features * no.of samples**2)
svm = SVC()
svm.fit(X_vec_train, y_train)
predicted_value = svm.predict(X_vec_test)
svm_accuracy_value = roc_auc_score(y_test, predicted_value)
# SVM Training ROC 99.9% lemmatized text
# svm predict on training set took very long time
print("ROC: " + str(roc_auc_score(y_train, svm.predict(X_vec_train))*100) + "%")
# SVM Test ROC 99.53% lemmatized text
print("ROC: " + str(svm_accuracy_value*100) + "%")
conmat = confusion_matrix(y_test, predicted_value)
print(conmat)
print(classification_report(y_test, predicted_value))
# Visual of confusion matrix of SVM
fig = plt.subplot()
sns.heatmap(conmat, annot=True, ax=fig)
fig.set_ylabel('y_test')
fig.set_xlabel('predicted values')
# Training on Random Forest
rf = RandomForestClassifier()
rf.fit(X_vec_train, y_train)
predicted_value = rf.predict(X_vec_test)
rf_accuracy_value = roc_auc_score(y_test, predicted_value)
# Random Forest Training ROC 100% lemmatized text
print("ROC: " + str(roc_auc_score(y_train, rf.predict(X_vec_train))*100) + "%")
# Random Forest Test ROC 99.67% lemmatized text
print("ROC: " + str(rf_accuracy_value*100) + "%")
conmat = confusion_matrix(y_test, predicted_value)
print(conmat)
print(classification_report(y_test, predicted_value))
# Visual of confusion matrix of Random Forest
fig = plt.subplot()
sns.heatmap(conmat, annot=True, ax=fig)
fig.set_ylabel('y_test')
fig.set_xlabel('predicted values')
# Training on Gradient Boosting 
gbc = GradientBoostingClassifier()
gbc.fit(X_vec_train, y_train)
predicted_value = gbc.predict(X_vec_test)
gbc_accuracy_value = roc_auc_score(y_test, predicted_value)
# Gradient Boost Training ROC 99.64% lemmatized text
print("ROC: " + str(roc_auc_score(y_train, gbc.predict(X_vec_train))*100) + "%")
# Gradient Boost Test ROC 99.5% lemmatized text
print("ROC: " + str(gbc_accuracy_value*100) + "%")
conmat = confusion_matrix(y_test, predicted_value)
print(conmat)
print(classification_report(y_test, predicted_value))
# Visual of confusion matrix of Gradient Boosting
fig = plt.subplot()
sns.heatmap(conmat, annot=True, ax=fig)
fig.set_ylabel('y_test')
fig.set_xlabel('predicted values')
import pickle
# Save model
model_file = "gbc.pkl"
with open(model_file,mode='wb') as model_f:
    pickle.dump(gbc,model_f)
# Open the model, print result for sanity check
with open("gbc.pkl",mode='rb') as model_f:
    model = pickle.load(model_f)
    predict = model.predict(X_vec_test)
    result = roc_auc_score(y_test, predict)
    print("result:",result*100, "%")