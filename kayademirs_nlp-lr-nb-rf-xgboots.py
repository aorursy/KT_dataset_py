# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set_style("white")

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from PIL import Image

import nltk

from nltk.corpus import stopwords

import textblob

from textblob import TextBlob

from textblob import Word

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

train = train.drop(["location"], axis=1)

train.head()
train.shape
train.isnull().sum()
train['text'] = train['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

train["keyword"] = train["keyword"].str.replace('\d','')

train["keyword"] = train["keyword"].str.replace('[^\w\s]','')

keyword = pd.unique(train.keyword)[1:]
keys = []

for text in train.text:

    t_k = []

    for key in keyword:

        if text.find(str(key)) != -1:

            t_k.append(key)

        else:

            continue

    keys.append(t_k)

train["keyword"] = keys
df = train.copy()
#capitalization conversion

df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

#punctuation

df['text'] = df['text'].str.replace('[^\w\s]','')

df['keyword'] = df['keyword'].apply(lambda x: str(x))

df["keyword"] = df["keyword"].str.replace('[^\w\s]','')

#numbers

df['text'] = df['text'].str.replace('\d','')

df["keyword"] = df["keyword"].str.replace('\d','')

#stopwords

import nltk

#nltk.download('stopwords')

from nltk.corpus import stopwords

sw = stopwords.words('english')

df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

del_word = pd.Series(' '.join(df['text']).split()).value_counts()[1500:]

#erasure sparse

df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in del_word))

#lemmi

from textblob import Word

#nltk.download('wordnet')

df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 
features = keyword

counts = []



for i in features:    

    counts.append(df["text"].apply(lambda x: len([x for x in x.split() if x.startswith(i)])).sum())



df_fre = pd.DataFrame(columns=["names", "frequency"])

df_fre["names"] = features

df_fre["frequency"] = counts

df_fre.head()
import plotly.express as px

df_class = pd.value_counts(df['target'], sort = True).sort_index()

fig = px.bar(df_class)

fig.show()
import plotly.express as px

fig = px.bar(df_fre, x=df_fre.names, y=df_fre.frequency,  title="Frequency of Keyword")

fig.show()
import plotly.express as px

fig = px.pie(df_fre, values="frequency", names="names", title='Population of Keywords')

fig.update_traces(textposition='inside')

fig.update_layout(uniformtext_minsize=7, uniformtext_mode='hide')

fig.show()
words  = " ".join(x for x in df.keyword)

plt.figure(figsize=(20,10))

wordcloud = WordCloud(background_color="white",  width = 2000, height = 800).generate(words)

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
words  = " ".join(x for x in df.text)

plt.figure(figsize=(20,10))

wordcloud = WordCloud(background_color="white", width = 2000, height = 800).generate(words)

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
# Train and Test Split

from sklearn import model_selection, preprocessing

x_train, x_test, y_train, y_test = model_selection.train_test_split(df["text"], df["target"], random_state=21, test_size=0.33)
encoder = preprocessing.LabelEncoder()

y_train = encoder.fit_transform(y_train)

y_test = encoder.fit_transform(y_test)
# Count

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=500)

vectorizer.fit(x_train)

x_train_count = vectorizer.transform(x_train)

x_test_count = vectorizer.transform(x_test)
## TF- IDF

from sklearn.feature_extraction.text import  TfidfVectorizer

tf_idf_vectorizer = TfidfVectorizer(max_features=500)

tf_idf_vectorizer.fit(x_train)

x_train_tf_idf = tf_idf_vectorizer.transform(x_train)

x_test_tf_idf = tf_idf_vectorizer.transform(x_test)
models_results = []

model_names = []
from sklearn import linear_model

from sklearn.model_selection import cross_val_score

log_reg = linear_model.LogisticRegression()

log_model = log_reg.fit(x_train_count, y_train)

acc = model_selection.cross_val_score(

    log_model,

    x_test_count,

    y_test,

    cv=7).mean()



model_names.append("Logistic Regression | Count Vectorizer")

models_results.append(acc)

print("Accuracy Count:", acc, "\n\n\n")



y_pred = log_model.predict(x_test_count)

from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7,7))

plt.title("Logistic Regression | Count Vectorizer")

sns.heatmap(cm, annot=True, fmt="d", linewidths=0.7 ,cbar=False)

plt.show()
from sklearn import linear_model

from sklearn.model_selection import cross_val_score

log_reg = linear_model.LogisticRegression()

log_model = log_reg.fit(x_train_tf_idf, y_train)

acc = model_selection.cross_val_score(

    log_model,

    x_test_tf_idf,

    y_test,

    cv=7).mean()



model_names.append("Logistic Regression | TF-IDF")

models_results.append(acc)

print("Accuracy TF-IDF:", acc, "\n\n\n")



y_pred = log_model.predict(x_test_tf_idf)

from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7,7))

plt.title("Logistic Regression | TF-IDF")

sns.heatmap(cm, annot=True, fmt="d", linewidths=0.7 ,cbar=False)

plt.show()
from sklearn import naive_bayes

nb = naive_bayes.MultinomialNB()

nb_model  = nb.fit(x_train_count, y_train)

acc = model_selection.cross_val_score(

    nb_model,

    x_test_count,

    y_test,

    cv=7).mean()



model_names.append("Naive Bayes | Count Vectorizer")

models_results.append(acc)

print("Accuracy Count:", acc, "\n\n\n")



y_pred = nb_model.predict(x_test_count)

from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7,7))

plt.title("Naive Bayes | Count Vectorizer")

sns.heatmap(cm, annot=True, fmt="d", linewidths=0.7 ,cbar=False)

plt.show()
from sklearn import naive_bayes

nb = naive_bayes.MultinomialNB()

nb_model  = nb.fit(x_train_tf_idf, y_train)

acc = model_selection.cross_val_score(

    nb_model,

    x_test_tf_idf,

    y_test,

    cv=7).mean()



model_names.append("Naive Bayes | TF-IDF")

models_results.append(acc)

print("Accuracy TF-IDF:", acc, "\n\n\n")



y_pred = log_model.predict(x_test_tf_idf)

from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7,7))

plt.title("Naive Bayes | TF-IDF")

sns.heatmap(cm, annot=True, fmt="d", linewidths=0.7 ,cbar=False)

plt.show()
from sklearn import ensemble

rf = ensemble.RandomForestClassifier()

rf_model = rf.fit(x_train_count, y_train)

acc = model_selection.cross_val_score(

    rf_model,

    x_test_count,

    y_test,

    cv=7).mean()



model_names.append("Random Forest | Count Vectorizer")

models_results.append(acc)

print("Accuracy Count:", acc, "\n\n\n")



y_pred = nb_model.predict(x_test_count)

from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7,7))

plt.title("Random Forest | Count Vectorizer")

sns.heatmap(cm, annot=True, fmt="d", linewidths=0.7 ,cbar=False)

plt.show()
from sklearn import ensemble

rf = ensemble.RandomForestClassifier()

rf_model = rf.fit(x_train_tf_idf, y_train)

acc = model_selection.cross_val_score(

    rf_model,

    x_test_tf_idf,

    y_test,

    cv=7).mean()

model_names.append("Random Forest | TF-IDF")

models_results.append(acc)

print("Accuracy TF-IDF:", acc, "\n\n\n")



y_pred = log_model.predict(x_test_tf_idf)

from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7,7))

plt.title("Random Forest | TF-IDF")

sns.heatmap(cm, annot=True, fmt="d", linewidths=0.7 ,cbar=False)

plt.show()
import xgboost

xgb = xgboost.XGBClassifier()

xgb_model = xgb.fit(x_train_count, y_train)

acc = model_selection.cross_val_score(

    xgb_model,

    x_test_count,

    y_test,

    cv=7).mean()



model_names.append("XGBoost | Count Vectorizer")

models_results.append(acc)

print("Accuracy Count:", acc, "\n\n\n")



y_pred = nb_model.predict(x_test_count)

from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7,7))

plt.title("XGBoost | Count Vectorizer")

sns.heatmap(cm, annot=True, fmt="d", linewidths=0.7 ,cbar=False)

plt.show()
import xgboost

xgb = xgboost.XGBClassifier()

xgb_model = xgb.fit(x_train_tf_idf, y_train)

acc = model_selection.cross_val_score(

    xgb_model,

    x_test_tf_idf,

    y_test,

    cv=7).mean()

model_names.append("XGBoost | TF-IDF")

models_results.append(acc)

print("Accuracy TF-IDF:", acc, "\n\n\n")



y_pred = log_model.predict(x_test_tf_idf)

from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7,7))

plt.title("XGBoost | TF-IDF")

sns.heatmap(cm, annot=True, fmt="d", linewidths=0.7 ,cbar=False)

plt.show()
type(models_results)
models_df = pd.DataFrame(columns=["Models", "Accuracy"])

models_df["Models"] = model_names

models_df["Accuracy"] = [x*100 for x in models_results] 

models_df.head(10)
plt.figure(figsize=(20,15))

sns.barplot(x='Accuracy', y='Models', data=models_df)

plt.xlabel('Accuracy')

plt.title('Accuracy Ratios of Models');
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

test = test.drop(["keyword", "location"], axis=1)

test.head()
#capitalization conversion

test['text'] = test['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

#punctuation

test['text'] = test['text'].str.replace('[^\w\s]','')

#numbers

test['text'] = test['text'].str.replace('\d','')

#stopwords

import nltk

#nltk.download('stopwords')

from nltk.corpus import stopwords

sw = stopwords.words('english')

test['text'] = test['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

del_word = pd.Series(' '.join(test['text']).split()).value_counts()[7000:]

#erasure sparse

test['text'] = test['text'].apply(lambda x: " ".join(x for x in x.split() if x not in del_word))

#lemmi

from textblob import Word

#nltk.download('wordnet')

test['text'] = test['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 
data_test = tf_idf_vectorizer.transform(test.text)
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

submission.head()
submission = test.id.copy().to_frame()

submission['target'] = log_model.predict(data_test)
submission.to_csv("submission.csv", index=False)