import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True)
import re
df = pd.read_csv('../input/onion-or-not/OnionOrNot.csv')
df.head()
df.isnull().sum()
df.shape
df['label'].value_counts(normalize=True)*100
df.duplicated().sum()
df['WordCount'] = df['text'].apply(lambda x: len(str(x).split(" ")))
df.head()
df['CharCount'] = df['text'].str.len()
df.head()
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
stop = stopwords.words('english')
df['Stopwords'] = df['text'].apply(lambda x: len([x for x in x.split() if x in stop]))
df.head()
df['text'] = df['text'].map(lambda x: re.sub(r'[^a-zA-Z\s]', '',x,re.I|re.A))
df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.strip()
df.head()
df['text'] = df['text'].str.replace('[^\w\s]','')
df.head()
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df.head()
import en_core_web_sm

sp = en_core_web_sm.load()

def lemma(input_str):
    s = sp(input_str)
    
    input_list = []
    for word in s:
        w = word.lemma_
        input_list.append(w)
        
    output = ' '.join(input_list)
    return output

df['text'] = df['text'].apply(lambda x: lemma(x))


df.head()
from wordcloud import WordCloud
onion = df[df['label'] == 1].text
news = df[df['label'] == 0].text
# WordCloud for Onion
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 ).generate(" ".join(onion))
plt.imshow(wc , interpolation = 'nearest')
# WordCloud for News
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 ).generate(" ".join(news))
plt.imshow(wc , interpolation = 'nearest')
plt.figure(figsize=(20,20))
sns.pairplot(df, hue="label")
plt.show()
grouped = df.groupby('label').mean().reset_index()
grouped
df.groupby('label').mean().plot.bar(subplots=True, figsize=(10,10), grid=True)
freq = pd.Series(' '.join(df['text']).split()).value_counts()[0:300]
freq = list(freq.index)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=len(freq), lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))
df_vect = tfidf.fit_transform(df['text'])
text_df = pd.DataFrame(df_vect.toarray(), columns=tfidf.get_feature_names())

text_df.head()
text_df['WordCount'] = df['WordCount']
text_df['CharCount'] = df['CharCount']
text_df['Stopwords'] = df['Stopwords']
text_df.info()
X = text_df
y = df['label']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
    
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_lr_pred = lr.predict(X_test)
print('Accuracy: %.4f' % accuracy_score(y_test, y_lr_pred))
print(confusion_matrix(y_test, y_lr_pred))
print(classification_report(y_test,y_lr_pred))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_rfc_pred = rfc.predict(X_test)
print('Accuracy: %.4f' % accuracy_score(y_test, y_rfc_pred))
print(confusion_matrix(y_test, y_rfc_pred))
print(classification_report(y_test,y_rfc_pred))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_dtc_pred = dtc.predict(X_test)
print('Accuracy: %.4f' % accuracy_score(y_test, y_dtc_pred))
print(confusion_matrix(y_test, y_dtc_pred))
print(classification_report(y_test,y_dtc_pred))
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_mnb_pred = mnb.predict(X_test)
print('Accuracy: %.4f' % accuracy_score(y_test, y_mnb_pred))
print(confusion_matrix(y_test, y_mnb_pred))
print(classification_report(y_test,y_mnb_pred))
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
y_ada_pred = ada.predict(X_test)
print('Accuracy: %.4f' % accuracy_score(y_test, y_ada_pred))
print(confusion_matrix(y_test, y_ada_pred))
print(classification_report(y_test,y_ada_pred))
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train_std, y_train)
y_xgb_pred = xgb.predict(X_test_std)
print('Accuracy: %.4f' % accuracy_score(y_test, y_xgb_pred))
print(confusion_matrix(y_test, y_xgb_pred))
print(classification_report(y_test,y_xgb_pred))
