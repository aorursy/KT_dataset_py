import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import re

import string

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize, RegexpTokenizer

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score , confusion_matrix, f1_score



# Models

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import xgboost as xgboost

from lightgbm import LGBMClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier



# Visualization

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

import seaborn as sns



# Plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) # #do not miss this line

import plotly as py

import plotly.graph_objs as go



from wordcloud import WordCloud,STOPWORDS



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_train = pd.read_csv('../input/nlp-getting-started/train.csv')

data_test = pd.read_csv('../input/nlp-getting-started/test.csv')

submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
display(data_train.head())

display(data_test.head())

display(submission.head())
data_train.info()
data_train.columns
print("Train set contains {} rows and {} cols".format(data_train.shape[0],data_train.shape[1]))

print("Test set contains {} rows and {} cols".format(data_test.shape[0],data_test.shape[1]))
data_train.location.value_counts()
data_train.keyword.value_counts()
# Missing values in train set

data_train.isnull().sum()
# Missing values in test set

data_test.isnull().sum()
train = data_train[['text', 'target']]

train.head()
train.target.value_counts()
test = data_test[['text']]

test.head()
def clean_tweets(text):

    text = re.sub('https?://[A-Za-z0-9./]*','', text) # Remove https..(URL)

    text = re.sub('[0-9]*','', text) # Removed digits

    text = re.sub('RT @[\w]*:','', text) # Removed RT 

    text = re.sub('@[A-Za-z0-9]+', '', text) # Removed @mention

    text = re.sub('&amp; ','',text) # Removed &(and) 

    return text



def remove_punctuations(text):

    text = ' '.join([i for i in text if i not in frozenset(string.punctuation)])

    return text



stop = stopwords.words('english')

stop_list = ['u','û_']

for i in range(len(stop_list)):

    stop.append(stop_list[i])



def remove_stopword(text):

    words = [w for w in text if w not in stop]

    return words
train['cleaned_text'] = train['text'].apply(clean_tweets)

train['cleaned_text'] = train['cleaned_text'].apply(lambda x: x.lower()) 

tokenizer = RegexpTokenizer(r'\w+')

train['cleaned_text'] = train['cleaned_text'].apply(lambda x: tokenizer.tokenize(x)) # word tokenize

train['cleaned_text'] = train['cleaned_text'].apply(remove_stopword) 

train['cleaned_text'] = train['cleaned_text'].apply(remove_punctuations) 

train.head()
test['cleaned_text'] = test['text'].apply(clean_tweets)

test['cleaned_text'] = test['cleaned_text'].apply(lambda x: x.lower()) 

tokenizer = RegexpTokenizer(r'\w+')

test['cleaned_text'] = test['cleaned_text'].apply(lambda x: tokenizer.tokenize(x)) # word tokenize

test['cleaned_text'] = test['cleaned_text'].apply(remove_stopword) 

test['cleaned_text'] = test['cleaned_text'].apply(remove_punctuations) 

test.head()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

lemmatizer = WordNetLemmatizer()



def lemmatize_text(text):

    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
train['cleaned_text'] = train['cleaned_text'].apply(lemmatize_text)

train['cleaned_text'] = train['cleaned_text'].apply(remove_punctuations) 

train.head()
test['cleaned_text'] = test['cleaned_text'].apply(lemmatize_text)

test['cleaned_text'] = test['cleaned_text'].apply(remove_punctuations) 

test.head()
fig, ax = plt.subplots(figsize=(5, 4), subplot_kw=dict(aspect="equal"))

labels=['Not-Disaster', 'Disaster']

wedges, texts, autotexts = ax.pie(data_train.target.value_counts(),autopct="%1.2f%%", colors=['#66b3ff','#cc1d00'], 

                                            explode = (0,0.07), startangle=90,

                                            textprops={'fontsize': 15, 'color':'#f5f5f5'})

plt.title('The Target Distribution', fontsize=16, weight="bold")

ax.legend(wedges, labels,

          title="Ingredients",

          loc="center left",

          bbox_to_anchor=(1.2, 0, 0, 1))



plt.setp(autotexts, weight="bold")

plt.show()
fig = go.Figure([go.Bar(x=['Disaster', 'Not-Disaster'], 

                        y=[len(data_train[data_train['target']== 1]),len(data_train[data_train['target']== 0])])])

fig.update_traces(marker_color='indianred', marker_line_color='rgb(58,48,107)',

                  marker_line_width=1.5, opacity=0.7)

fig.update_layout(title_text='The Target Distribution',autosize=False,width=400,height=500)

fig.show()
disaster = train[train['target']==1]['cleaned_text']

non_disaster = train[train['target']==0]['cleaned_text']
print('Disaster Tweet: {} \nNot-Disaster Tweet: {}'.format(disaster.values[2],non_disaster.values[2]))
from PIL import Image

path = '../input/twitter-logo/twitter_logo.png'

mask = np.array(Image.open(path).convert('L'))

mask.shape
def grey_color_func(word, font_size, position, orientation, **kwargs):

    return "hsl(0, 0%%, %d%%)" % np.random.randint(60, 100)



fig, (plt1, plt2) = plt.subplots(1, 2, figsize=[14, 6])

wordcloud = WordCloud(

                        background_color='#123456',

                        random_state = 42,

                        max_words= 50,

                        mask=mask,

                        contour_width=1,

                        contour_color="#b5b5b5"

                     ).generate(''.join(disaster))



plt1.imshow(wordcloud.recolor(color_func=grey_color_func,random_state=3), interpolation="bilinear")

plt1.axis("off")

plt1.set_title('Disaster Tweets',fontsize=30);



wordcloud = WordCloud(

                        background_color='#123456',

                        random_state = 42,

                        max_words= 50,

                        mask=mask,

                        contour_width=1,

                        contour_color="#b5b5b5"

                     ).generate(''.join(non_disaster))



plt2.imshow(wordcloud.recolor(color_func=grey_color_func,random_state=3), interpolation="bilinear")

plt2.axis("off")

plt2.set_title('Not-Disaster Tweets',fontsize=30);
max_features=300

count_vectorizer = CountVectorizer(max_features=max_features,stop_words=stop)

train_vectors = count_vectorizer.fit_transform(train['text']).toarray()
print('In Train Set, the most common {} words:\n{} '.format(max_features,count_vectorizer.get_feature_names()))
# Term Frequency

tf = (train.cleaned_text).apply(lambda x : pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ['words','frequence']

tf.head()
tfreq = tf[tf['frequence']>100.0]

plt.subplots(figsize = (18,5))

chart = sns.barplot(x=tfreq.words, y=tfreq.frequence, palette=sns.color_palette("coolwarm",7), edgecolor=".3")

chart.set_xticklabels(chart.get_xticklabels(), rotation=75)

chart.set_title('Frequencies of the Most Common Words');
tf_idf_ngram = TfidfVectorizer(ngram_range=(1,2))

tf_idf_ngram.fit(train.cleaned_text)

x_train_tf_bigram = tf_idf_ngram.transform(train.cleaned_text) #.todense()

x_test_tf_bigram = tf_idf_ngram.transform(test.cleaned_text)
print(x_train_tf_bigram.shape,x_test_tf_bigram.shape)
tf_idf_ngram.get_feature_names()[:5]
X = x_train_tf_bigram

y = train.target.values



# Train-Test Splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Train Data splitted successfully')

X_train.shape, X_test.shape, y_train.shape, y_test.shape
df_accuracy = pd.DataFrame(columns=["Model","Accuracy","F1_score"])
# Fitting Train set

clf_LR = LogisticRegression(C=2,dual=True, solver='liblinear',random_state=0)

clf_LR.fit(X_train,y_train)



# Predicting 

y_pred_LR = clf_LR.predict(X_test)



# Calculating Model Accuracy and F1_score

accuracy = accuracy_score(y_test, y_pred_LR) * 100

f1score = f1_score(y_test, y_pred_LR) * 100

print("Logistic Regression Accuracy: {0:.3f} %".format(accuracy))

print("Logistic Regression F1 Score: {0:.3f} %".format(f1score))

df_accuracy = df_accuracy.append({'Model':'LogisticRegression','Accuracy':accuracy, 'F1_score': f1score },ignore_index=True)
# Fitting Train set

clf_NB = MultinomialNB()

clf_NB.fit(X_train,y_train)



# Predicting 

y_pred_NB = clf_NB.predict(X_test)



# Calculating Model Accuracy and F1_score

accuracy = accuracy_score(y_test, y_pred_NB) * 100

f1score = f1_score(y_test, y_pred_NB) * 100

print("MultinomialNB Accuracy: {0:.3f} %".format(accuracy))

print("MultinomialNB F1 Score: {0:.3f} %".format(f1score))

df_accuracy = df_accuracy.append({'Model':'NaiveBayes','Accuracy':accuracy, 'F1_score': f1score },ignore_index=True)
# Fitting Train set

clf_KNN = KNeighborsClassifier(n_neighbors = 7,weights = 'distance')

clf_KNN.fit(X_train, y_train)



# Predicting 

y_pred_KNN = clf_KNN.predict(X_test)



# Calculating Model Accuracy and F1_score

accuracy = accuracy_score(y_test, y_pred_KNN) * 100

f1score = f1_score(y_test, y_pred_KNN) * 100

print("K-Nearest Neighbors Accuracy: {0:.3f} %".format(accuracy))

print("K-Nearest Neighbors F1 Score: {0:.3f} %".format(f1score))

df_accuracy = df_accuracy.append({'Model':'K-NearestNeighbors','Accuracy':accuracy, 'F1_score': f1score },ignore_index=True)
# Fitting Train set

clf_RF = RandomForestClassifier(random_state=0)

clf_RF.fit(X_train,y_train) 



# Predicting 

y_pred_RF = clf_RF.predict(X_test)



# Calculating Model Accuracy and F1_score

accuracy = accuracy_score(y_test, y_pred_RF) * 100

f1score = f1_score(y_test, y_pred_RF) * 100

print("Random Forest Accuracy: {0:.3f} %".format(accuracy))

print("Random Forest F1 Score: {0:.3f} %".format(f1score))

df_accuracy = df_accuracy.append({'Model':'RandomForest','Accuracy':accuracy, 'F1_score': f1score },ignore_index=True)
# Fitting Train set

clf_DT = DecisionTreeClassifier(criterion= 'entropy', random_state=0)

clf_DT.fit(X_train,y_train) 



# Predicting 

y_pred_DT = clf_DT.predict(X_test)



# Calculating Model Accuracy and F1_score

accuracy = accuracy_score(y_test, y_pred_DT) * 100

f1score = f1_score(y_test, y_pred_DT) * 100

print("Decision Tree Accuracy: {0:.3f} %".format(accuracy))

print("Decision Tree F1 Score: {0:.3f} %".format(f1score))

df_accuracy = df_accuracy.append({'Model':'DecisionTree','Accuracy':accuracy, 'F1_score': f1score },ignore_index=True)
# Fitting Train set

clf_GB = GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, max_depth=20, random_state=0)

clf_GB.fit(X_train,y_train)



# Predicting 

y_pred_GB = clf_GB.predict(X_test)



# Calculating Model Accuracy and F1_score

accuracy = accuracy_score(y_test, y_pred_GB) * 100

f1score = f1_score(y_test, y_pred_GB) * 100

print("Gradient Boosting Classifier Accuracy: {0:.3f} %".format(accuracy))

print("Gradient Boosting Classifier F1 Score: {0:.3f} %".format(f1score))

df_accuracy = df_accuracy.append({'Model':'GradientBoostingClassifier','Accuracy':accuracy, 'F1_score': f1score },ignore_index=True)
# Fitting Train set

clf_XGB = xgboost.XGBClassifier(n_estimators=400, random_state=0, learning_rate=0.05, booster="gbtree",

                                n_jobs=-1, max_depth=20)

clf_XGB.fit(X_train,y_train)

# Predicting 

y_pred_XGB = clf_XGB.predict(X_test)



# Calculating Model Accuracy and F1_score

accuracy = accuracy_score(y_test, y_pred_XGB) * 100

f1score = f1_score(y_test, y_pred_XGB) * 100

print("XGBOOST Classifier Accuracy: {0:.3f} %".format(accuracy))

print("XGBOOST Classifier F1 Score: {0:.3f} %".format(f1score))

df_accuracy = df_accuracy.append({'Model':'XGBOOSTClassifier','Accuracy':accuracy, 'F1_score': f1score },ignore_index=True)
# Fitting Train set

clf_LGB = LGBMClassifier(n_estimators=1300, learning_rate=0.05, random_state=0, max_depth=20, n_jobs=-1)

clf_LGB.fit(X_train,y_train)



# Predicting 

y_pred_LGB = clf_LGB.predict(X_test)



# Calculating Model Accuracy and F1_score

accuracy = accuracy_score(y_test, y_pred_LGB) * 100

f1score = f1_score(y_test, y_pred_LGB) * 100

print("LightGB Classifier Accuracy: {0:.3f} %".format(accuracy))

print("LightGB Classifier F1 Score: {0:.3f} %".format(f1score))

df_accuracy = df_accuracy.append({'Model':'LightGBClassifier','Accuracy':accuracy, 'F1_score': f1score},ignore_index=True)
# Accuracy and F1-score Comparison of Models

trace1=go.Bar(

                x=df_accuracy.Model,

                y=df_accuracy.Accuracy,

                name="Accuracy",

                marker=dict(color = 'rgba(50, 240,120, 0.7)',

                           line=dict(color='rgb(0,0,0)',width=1.9)),

                text='Accuracy')

trace2=go.Bar(

                x=df_accuracy.Model,

                y=df_accuracy.F1_score,

                name="F1-score",

                marker=dict(color = 'rgba(240,120,10 , 0.7)', 

                           line=dict(color='rgb(0,0,0)',width=1.9)),

                text='F1-score')



edit_df=[trace1,trace2]

layout=go.Layout(barmode="group", xaxis_tickangle=-60, title="Accuracy and F1-score of Models")

fig=dict(data=edit_df,layout=layout)

iplot(fig)
submission['target'] = clf_XGB.predict(x_test_tf_bigram)

submission['target']
submission_final= submission[['id','target']]

submission_final.to_csv('submission.csv',index=False)