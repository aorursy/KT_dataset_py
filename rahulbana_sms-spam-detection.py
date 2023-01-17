import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle

pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
data = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv", encoding="latin")
data.head(2)
data = data[['v2', 'v1']]
data.shape
data.head(5)
data[data['v1'] == 'spam']['v2']
data.isna().sum()
data['v1'].value_counts()
data['v1'].value_counts(normalize=True)*100
data['txt_len'] = data['v2'].apply(lambda x: len(str(x)))
data.head(2)
data.groupby(['v1'])['txt_len'].agg({'min', 'max', 'mean', 'median', 'std'})
wordcloud = WordCloud().generate(' '.join(data[data['v1'] == 'spam']['v2'].tolist()))

# Display the generated image:
plt.figure( figsize=(20,10) )
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud = WordCloud().generate(' '.join(data[data['v1'] == 'ham']['v2'].tolist()))

# Display the generated image:
plt.figure( figsize=(20,10) )
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordlist_spam = ' '.join(data[data['v1'] == 'spam']['v2'].tolist()).split(" ")
wordlist_ham = ' '.join(data[data['v1'] == 'ham']['v2'].tolist()).split(" ")

un_wordlist_spam = set(wordlist_spam)
un_wordlist_ham = set(wordlist_ham)

wordfreq_spam = []
wordfreq_ham = []
                        
for w in un_wordlist_spam:
    wordfreq_spam.append(wordlist_spam.count(w))

                        
for w in un_wordlist_ham:
    wordfreq_ham.append(wordlist_ham.count(w))
    

list_freq_spam = list(zip(un_wordlist_spam, wordfreq_spam))
list_freq_ham = list(zip(un_wordlist_ham, wordfreq_ham))

list_freq_spam.sort(key = lambda x: x[1], reverse=True)
list_freq_ham.sort(key = lambda x: x[1], reverse=True)
df_ham_20 = pd.DataFrame(list_freq_ham[:30], columns =['word', 'freq']) 

a4_dims = (20, 6)
fig, ax = plt.subplots(figsize=a4_dims)

plt.title("Top 30 words for ham sms")
sns.barplot(x='word', y='freq', data=df_ham_20)
plt.xlabel("Words")
plt.ylabel("word count")
plt.xticks(rotation='45')
plt.show()
df_spam_20 = pd.DataFrame(list_freq_spam[:30], columns =['word', 'freq']) 

a4_dims = (20, 6)
fig, ax = plt.subplots(figsize=a4_dims)

plt.title("Top 30 words for spam sms")
sns.barplot(x='word', y='freq', data=df_spam_20)
plt.xlabel("Words")
plt.ylabel("word count")
plt.xticks(rotation='45')
plt.show()
features, labels = data['v2'].tolist(), data['v1'].tolist()
x_train, x_test, y_train, y_test = train_test_split(features, 
                                                   labels, 
                                                   shuffle=True, 
                                                   test_size=0.2, 
                                                   stratify=labels,
                                                   random_state=42)
label_encoder = LabelEncoder()
label_encoder.fit(y_train)

y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

#pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))
label_encoder.inverse_transform
cv = CountVectorizer()
cv_transformer = cv.fit(x_train)

#pickle.dump(cv_transformer, open("count_transformer.pkl", "wb"))
x_train = cv.transform(x_train)
x_test = cv.transform(x_test)
x_train.shape,x_test.shape
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

print("train \ test score")
print(lr_model.score(x_train, y_train), lr_model.score(x_test, y_test))
print("\n")
print('classification report')

ypred = lr_model.predict(x_test)
print(classification_report(y_test, ypred))
print("\n")

print("confusion matrix")
print(confusion_matrix(y_test, ypred))
nb_model = MultinomialNB()
nb_model.fit(x_train, y_train)

print("train \ test score")
print(nb_model.score(x_train, y_train), lr_model.score(x_test, y_test))
print("\n")
print('classification report')

ypred = nb_model.predict(x_test)
print(classification_report(y_test, ypred))
print("\n")

print("confusion matrix")
print(confusion_matrix(y_test, ypred))
#pickle.dump(nb_model, open("sms_spam_classifier_model.pkl", "wb"))
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)

print("train \ test score")
print(dt_model.score(x_train, y_train), lr_model.score(x_test, y_test))
print("\n")
print('classification report')

ypred = dt_model.predict(x_test)
print(classification_report(y_test, ypred))
print("\n")

print("confusion matrix")
print(confusion_matrix(y_test, ypred))
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

print("train \ test score")
print(rf_model.score(x_train, y_train), lr_model.score(x_test, y_test))
print("\n")
print('classification report')

ypred = rf_model.predict(x_test)
print(classification_report(y_test, ypred))
print("\n")

print("confusion matrix")
print(confusion_matrix(y_test, ypred))


