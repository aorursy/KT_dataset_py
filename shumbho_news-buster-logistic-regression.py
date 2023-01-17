import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import re

import math

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import nltk

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS 



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')

fake.shape
fake.head(3)
true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

true.shape
true.head(3)
true['label'] = 0

fake['label'] = 1
df = true.append(fake.reset_index(drop=True))

print(df.shape)

df = df.reset_index(drop=True)

df
df.info()
df[df.text == " "]
df.drop(df[df.text == " "].index.tolist(), inplace=True)
df.drop('label', axis=1)[df.duplicated(keep=False)]
df.drop(df[df.duplicated()].index.tolist(), inplace=True)

df.shape
df[df.duplicated(['title'])]
df.drop(df[df.duplicated(['title'])].index.tolist(), inplace=True)

df.shape
df[df.duplicated(['text'], keep=False)]
df.drop(df[df.duplicated(['text'])].index.tolist(),inplace=True)
df.shape
df.groupby("label")['title'].count().plot.bar()
df.sort_values('date').date.reset_index(drop=True).iloc[[5,10,20,50,100,200,38264,38268]]
import plotly.graph_objects as go



fig = go.Figure(layout=go.Layout(title='Day wise article count',yaxis_title='No. of news',xaxis_title='Dates'))



fig.add_trace(go.Scatter(y=df[df.label==0].groupby(['date']).label.count(),

                    mode='lines',

                    name='true'))

fig.add_trace(go.Scatter(x=df[df.label==1].groupby(['date']).label.count().index,y=df[df.label==1].groupby(['date']).label.count(),

                    mode='lines',

                    name='fake'))



fig.show()
df.drop('date', axis=1,inplace=True)
df[df.label==1].subject.value_counts()
df[df.label==0].subject.value_counts()
df.drop('subject', axis=1,inplace=True)
df['text'] = df['title'] + ' ' + df['text']

del df['title']



df.head(2)


nltk_stopwords = stopwords.words('english')

wordcloud_stopwords = STOPWORDS



nltk_stopwords.extend(wordcloud_stopwords)



stopwords = set(nltk_stopwords)

print(stopwords)



def clean(text):

    

    text = re.sub("http\S+", '', str(text))

    

    # Contractions ref: https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert

    

    text = re.sub(r"he's", "he is", str(text))

    text = re.sub(r"there's", "there is", str(text))

    text = re.sub(r"We're", "We are", str(text))

    text = re.sub(r"That's", "That is", str(text))

    text = re.sub(r"won't", "will not", str(text))

    text = re.sub(r"they're", "they are", str(text))

    text = re.sub(r"Can't", "Cannot", str(text))

    text = re.sub(r"wasn't", "was not", str(text))

    text = re.sub(r"aren't", "are not", str(text))

    text = re.sub(r"isn't", "is not", str(text))

    text = re.sub(r"What's", "What is", str(text))

    text = re.sub(r"haven't", "have not", str(text))

    text = re.sub(r"hasn't", "has not", str(text))

    text = re.sub(r"There's", "There is", str(text))

    text = re.sub(r"He's", "He is", str(text))

    text = re.sub(r"It's", "It is", str(text))

    text = re.sub(r"You're", "You are", str(text))

    text = re.sub(r"I'M", "I am", str(text))

    text = re.sub(r"shouldn't", "should not", str(text))

    text = re.sub(r"wouldn't", "would not", str(text))

    text = re.sub(r"i'm", "I am", str(text))

    text = re.sub(r"I'm", "I am", str(text))

    text = re.sub(r"Isn't", "is not", str(text))

    text = re.sub(r"Here's", "Here is", str(text))

    text = re.sub(r"you've", "you have", str(text))

    text = re.sub(r"we're", "we are", str(text))

    text = re.sub(r"what's", "what is", str(text))

    text = re.sub(r"couldn't", "could not", str(text))

    text = re.sub(r"we've", "we have", str(text))

    text = re.sub(r"who's", "who is", str(text))

    text = re.sub(r"y'all", "you all", str(text))

    text = re.sub(r"would've", "would have", str(text))

    text = re.sub(r"it'll", "it will", str(text))

    text = re.sub(r"we'll", "we will", str(text))

    text = re.sub(r"We've", "We have", str(text))

    text = re.sub(r"he'll", "he will", str(text))

    text = re.sub(r"Y'all", "You all", str(text))

    text = re.sub(r"Weren't", "Were not", str(text))

    text = re.sub(r"Didn't", "Did not", str(text))

    text = re.sub(r"they'll", "they will", str(text))

    text = re.sub(r"they'd", "they would", str(text))

    text = re.sub(r"DON'T", "DO NOT", str(text))

    text = re.sub(r"they've", "they have", str(text))

    text = re.sub(r"i'd", "I would", str(text))

    text = re.sub(r"should've", "should have", str(text))

    text = re.sub(r"where's", "where is", str(text))

    text = re.sub(r"we'd", "we would", str(text))

    text = re.sub(r"i'll", "I will", str(text))

    text = re.sub(r"weren't", "were not", str(text))

    text = re.sub(r"They're", "They are", str(text))

    text = re.sub(r"let's", "let us", str(text))

    text = re.sub(r"it's", "it is", str(text))

    text = re.sub(r"can't", "cannot", str(text))

    text = re.sub(r"don't", "do not", str(text))

    text = re.sub(r"you're", "you are", str(text))

    text = re.sub(r"i've", "I have", str(text))

    text = re.sub(r"that's", "that is", str(text))

    text = re.sub(r"i'll", "I will", str(text))

    text = re.sub(r"doesn't", "does not", str(text))

    text = re.sub(r"i'd", "I would", str(text))

    text = re.sub(r"didn't", "did not", str(text))

    text = re.sub(r"ain't", "am not", str(text))

    text = re.sub(r"you'll", "you will", str(text))

    text = re.sub(r"I've", "I have", str(text))

    text = re.sub(r"Don't", "do not", str(text))

    text = re.sub(r"I'll", "I will", str(text))

    text = re.sub(r"I'd", "I would", str(text))

    text = re.sub(r"Let's", "Let us", str(text))

    text = re.sub(r"you'd", "You would", str(text))

    text = re.sub(r"It's", "It is", str(text))

    text = re.sub(r"Ain't", "am not", str(text))

    text = re.sub(r"Haven't", "Have not", str(text))

    text = re.sub(r"Could've", "Could have", str(text))

    text = re.sub(r"youve", "you have", str(text))

    

    # Others

    text = re.sub("U.S.", "United States", str(text))

    text = re.sub("Dec", "December", str(text))

    text = re.sub("Jan.","January", str(text))

    

    # Punctuations & special characters

    text = re.sub("[^A-Za-z0-9]+"," ", str(text))

    

    # Stop word removal

    text = " ".join(str(i).lower() for i in text.split() if i.lower() not in stopwords)



    return text

    


df['text'] = df['text'].map(lambda x: clean(x))

df.text.iloc[:3]
X_train, X_test, y_train, y_test = train_test_split(df.drop('label',axis=1), df.label, test_size=0.2, random_state=42)



print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
vectorizer = TfidfVectorizer(min_df=0.01,ngram_range=(1,3))

vectorizer.fit(X_train.text)



X_tr = vectorizer.transform(X_train.text)

X_te = vectorizer.transform(X_test.text)



print(X_tr.shape, X_te.shape)
clf = SGDClassifier(loss='log')



gs = GridSearchCV(

    estimator = clf,

    param_grid = {'alpha':np.logspace(-10,5,16)},

    cv = 5,

    return_train_score = True,

    scoring = 'accuracy'

    )



gs.fit(X_tr,y_train)



results = pd.DataFrame(gs.cv_results_)



results = results.sort_values(['param_alpha'])

train_auc = results['mean_train_score']

cv_auc = results['mean_test_score']

alpha = pd.Series([ math.log(i) for i in np.array(results['param_alpha']) ]) 



plt.plot(alpha, train_auc, label='Train AUC')

plt.plot(alpha, cv_auc, label='CV AUC')

plt.scatter(alpha, train_auc)

plt.scatter(alpha, cv_auc)

plt.legend()

plt.xlabel('log(alpha): hyperparameter')

plt.ylabel('Accuracy')

plt.title('Hyperparameter vs Accuracy Plot')

plt.grid()

plt.show()



print(gs.best_params_)
clf = SGDClassifier(loss='log',alpha=1e-06, random_state=42).fit(X_tr,y_train)



print('Training score : %f' % clf.score(X_tr,y_train))

print('Test score : %f' % clf.score(X_te,y_test))



train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, clf.predict_proba(X_tr)[:,1])

test_fpr, test_tpr, te_thresholds = roc_curve(y_test, clf.predict_proba(X_te)[:,1])



plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()

plt.xlabel("FPR")

plt.ylabel("TPR")

plt.title("AUC Plot")

plt.grid()

plt.show()
print(classification_report(y_train.values, clf.predict(X_tr)))

confusion_matrix(y_train, clf.predict(X_tr))
print(classification_report(y_test.values, clf.predict(X_te)))



cm = pd.DataFrame(confusion_matrix(y_test,clf.predict(X_te)) , index = ['Fake','Not Fake'] , columns = ['Fake','Not Fake'])

sns.heatmap(cm,cmap= 'Blues', annot = True, fmt='', xticklabels = ['Fake','Not Fake'], yticklabels = ['Fake','Not Fake'])

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.title('Confusion matrix on test data')

plt.show()
coef = [abs(i) for i in clf.coef_.ravel()]

feature_names = vectorizer.get_feature_names()

feature_imp = dict(zip(feature_names,coef))

feature_imp = {k: v for k, v in sorted(feature_imp.items(), key=lambda item: item[1], reverse=True)}



top_50_features = {k: feature_imp[k] for k in list(feature_imp)[0:50]}



fig, ax = plt.subplots(figsize=(6,10))



people = top_50_features.keys()

y_pos = np.arange(len(people))

importance = top_50_features.values()



ax.barh(y_pos, importance,align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(people)

ax.invert_yaxis()

ax.set_xlabel('Importance')

ax.set_ylabel('Features')

ax.set_title('Top 50 Features')



plt.show()
feature_imp = dict(zip(feature_names,coef))

feature_imp = {k: v for k, v in sorted(feature_imp.items(), key=lambda item: item[1], reverse=False)}



bottom_50_features = {k: feature_imp[k] for k in list(feature_imp)[0:50]}



fig, ax = plt.subplots(figsize=(6,10))



people = bottom_50_features.keys()

y_pos = np.arange(len(people))

importance = bottom_50_features.values()



ax.barh(y_pos, importance,align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(people)

ax.invert_yaxis()

ax.set_xlabel('Importance')

ax.set_ylabel('Features')

ax.set_title('Least 50 important features')



plt.show()
# import pickle



# with open('vectorizer', 'wb') as f:

#     pickle.dump(vectorizer, f)

    

# with open('model', 'wb') as f:

#     pickle.dump(clf, f)