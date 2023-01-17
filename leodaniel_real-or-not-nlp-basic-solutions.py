# Installation of auxiliary libraries

!pip install missingno
# General imports

import numpy as np

import pandas as pd

import os

import missingno as msno



import warnings

warnings.filterwarnings("ignore")



# graphics import

import matplotlib.pyplot as plt

import seaborn as sns



# Natural language tool kits

import nltk

from nltk import FreqDist, ngrams

from nltk.corpus import stopwords

from wordcloud import WordCloud,STOPWORDS

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

# download stopwords

nltk.download('stopwords')





# string operations

import string 

import re



# sklearn

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.dummy import DummyClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import make_pipeline

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import f1_score



from yellowbrick.classifier import ClassificationReport

from yellowbrick.classifier import ROCAUC

from yellowbrick.classifier import PrecisionRecallCurve

from yellowbrick.model_selection import FeatureImportances

from yellowbrick.classifier import ConfusionMatrix

from yellowbrick.text import FreqDistVisualizer

from yellowbrick.text import TSNEVisualizer

from yellowbrick.contrib.classifier import DecisionViz

from yellowbrick.classifier import DiscriminationThreshold





from lime.lime_text import LimeTextExplainer



# Changing the number of characters displayed in pandas 

pd.options.display.max_colwidth = 150



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import database

df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

# submission file

df_sub = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')



# Displaying database's sample 

df.head()
df_sub.head()
df.info()
msno.matrix(df)
df['text_len'] = df['text'].str.len()



plt.figure(figsize=(15,5))

ax = sns.distplot(df['text_len'])

ax.set(xlabel='Text length', ylabel='Freq.')

plt.show()
fig = plt.figure(figsize=(15,5))

ax = sns.distplot(df[df['target'] == 1]['text_len'])

ax = sns.distplot(df[df['target'] == 0]['text_len'])

ax.set(xlabel='Text length', ylabel='Freq.')

fig.legend(labels=['Real','Fake'])

plt.show()
df['word_count'] = df['text'].str.split().map(lambda x:len(x))



plt.figure(figsize=(15,5))

ax = sns.distplot(df['word_count'])

ax.set(xlabel='Word count', ylabel='Freq.')

plt.show()
fig = plt.figure(figsize=(15,5))

ax = sns.distplot(df[df['target'] == 1]['word_count'])

ax = sns.distplot(df[df['target'] == 0]['word_count'])

ax.set(xlabel='Text length', ylabel='Freq.')

fig.legend(labels=['Real','Fake'])

plt.show()
vectorizer = CountVectorizer()

docs       = vectorizer.fit_transform(df['text'].tolist())

features   = vectorizer.get_feature_names()



visualizer = FreqDistVisualizer(features=features, orient='h', n=10)

visualizer.fit(docs)

visualizer.show()
vectorizer = CountVectorizer(ngram_range=(2, 2))

docs       = vectorizer.fit_transform(df['text'].tolist())

features   = vectorizer.get_feature_names()



visualizer = FreqDistVisualizer(features=features, orient='h', n=10)

visualizer.fit(docs)

visualizer.show()
fig, axes = plt.subplots(1, 2, figsize=(15, 8))

sns.barplot(df['target'].value_counts().index,df['target'].value_counts(), ax=axes[0])



axes[1].pie(df['target'].value_counts(),

            autopct='%1.2f%%',

            explode=(0.05, 0),

            startangle=60)



plt.show()
df_real = df[df['target']==1]['text']



wordcloud1 = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=2500,

                      height=2000

                      ).generate(" ".join(df_real))



plt.figure(1,figsize=(15, 15))

plt.imshow(wordcloud1)

plt.axis('off')

plt.show()
df_fake = df[df['target']==0]['text']



wordcloud1 = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=2500,

                      height=2000

                      ).generate(" ".join(df_fake))



plt.figure(1,figsize=(15, 15))

plt.imshow(wordcloud1)

plt.axis('off')

plt.show()
df = df[['text', 'target']]

# applying in submission dataset

df_sub = df_sub[['id','text']]





df.head(20)
df['text_lw'] = df['text'].str.lower()

# applying in submission dataset

df_sub['text_lw'] = df_sub['text'].str.lower()





df[['text','text_lw']].head(10)
def clean_text(text):

    # remove 

    text = re.sub('\[.*?\]', '', text)

    # remove links

    text = re.sub('https?://\S+|www\.\S+', '', text)

    # remove tags

    text = re.sub('<.*?>+', '', text)

    # remove punctuation

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    # remove breaklines

    text = re.sub('\n', '', text)

    # remove numbers

    text = re.sub('\w*\d\w*', '', text)

    

    # transform text into token

    text_token = nltk.word_tokenize(text)

    

    # remove stopwords

    words = [w for w in text_token if w not in stopwords.words('english')]

    

    

    return ' '.join(words)
df['text_cl'] = df['text_lw'].apply(clean_text)



# applying in submission dataset

df_sub['text_cl'] = df_sub['text_lw'].apply(clean_text)





df[['text','text_lw','text_cl']].head(20)
lemmatizer = WordNetLemmatizer() 



# function to convert nltk tag to wordnet tag

def nltk_tag_to_wordnet_tag(nltk_tag):

    if nltk_tag.startswith('J'):

        return wordnet.ADJ

    elif nltk_tag.startswith('V'):

        return wordnet.VERB

    elif nltk_tag.startswith('N'):

        return wordnet.NOUN

    elif nltk_tag.startswith('R'):

        return wordnet.ADV

    else:          

        return None



def lemmatize_sentence(sentence):

    #tokenize the sentence and find the POS tag for each token

    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  

    #tuple of (token, wordnet_tag)

    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)

    lemmatized_sentence = []

    for word, tag in wordnet_tagged:

        if tag is None:

            #if there is no available tag, append the token as is

            if len(word) > 2:

                lemmatized_sentence.append(word)

        else:        

            #else use the tag to lemmatize the token

            lemma = lemmatizer.lemmatize(word, tag)

            if len(lemma) > 2:

                lemmatized_sentence.append(lemma)

    return " ".join(lemmatized_sentence)
df['text_lm'] = df['text_cl'].apply(lemmatize_sentence)



# applying in submission dataset

df_sub['text_lm'] = df_sub['text_cl'].apply(lemmatize_sentence)





df[['text','text_lw','text_cl', 'text_lm']].head(20)
fig, axes = plt.subplots(2, 1, figsize=(15, 10))





df['text_len'] = df['text'].str.len()

plt.figure(figsize=(15,5))

axes[0].set_title('Before transformation')

ax=sns.distplot(df['text_len'], ax=axes[0])

axes[0].set(xlabel='Text length', ylabel='Freq.')



df['text_len'] = df['text_lm'].str.len()

plt.figure(figsize=(15,5))

axes[1].set_title('After transformation')

sns.distplot(df['text_len'], ax=axes[1])

axes[1].set(xlabel='Text length', ylabel='Freq.')

plt.show()
fig, axes = plt.subplots(2, 1, figsize=(15, 10))



df['text_len'] = df['text'].str.len()

axes[0].set_title('Before transformation')

ax = sns.distplot(df[df['target'] == 1]['text_len'], ax=axes[0])

ax = sns.distplot(df[df['target'] == 0]['text_len'], ax=axes[0])

ax.set(xlabel='Text length', ylabel='Freq.')



df['text_len'] = df['text_lm'].str.len()

axes[1].set_title('After transformation')

ax = sns.distplot(df[df['target'] == 1]['text_len'], ax=axes[1])

ax = sns.distplot(df[df['target'] == 0]['text_len'], ax=axes[1])

ax.set(xlabel='Text length', ylabel='Freq.')





fig.legend(labels=['Real','Fake'])

plt.show()
fig, axes = plt.subplots(2, 1, figsize=(15, 10))



df['word_count'] = df['text'].str.split().map(lambda x:len(x))

axes[0].set_title('Before transformation')

plt.figure(figsize=(15,5))

ax = sns.distplot(df['word_count'], ax=axes[0])

ax.set(xlabel='Word count', ylabel='Freq.')



df['word_count'] = df['text_lm'].str.split().map(lambda x:len(x))

axes[1].set_title('After transformation')

plt.figure(figsize=(15,5))

ax = sns.distplot(df['word_count'], ax=axes[1])

ax.set(xlabel='Word count', ylabel='Freq.')





plt.show()
fig, axes = plt.subplots(2, 1, figsize=(15, 10))



df['word_count'] = df['text'].str.split().map(lambda x:len(x))

axes[0].set_title('Before transformation')

ax = sns.distplot(df[df['target'] == 1]['word_count'], ax=axes[0])

ax = sns.distplot(df[df['target'] == 0]['word_count'], ax=axes[0])

ax.set(xlabel='Text length', ylabel='Freq.')

fig.legend(labels=['Real','Fake'])



df['word_count'] = df['text_lm'].str.split().map(lambda x:len(x))

axes[1].set_title('After transformation')

ax = sns.distplot(df[df['target'] == 1]['word_count'], ax=axes[1])

ax = sns.distplot(df[df['target'] == 0]['word_count'], ax=axes[1])

ax.set(xlabel='Text length', ylabel='Freq.')

fig.legend(labels=['Real','Fake'])





plt.show()
vectorizer = CountVectorizer()

docs       = vectorizer.fit_transform(df['text_lm'].tolist())

features   = vectorizer.get_feature_names()



visualizer = FreqDistVisualizer(features=features, orient='h', n=10)

visualizer.fit(docs)

visualizer.show()
vectorizer = CountVectorizer(ngram_range=(2, 2))

docs       = vectorizer.fit_transform(df['text_lm'].tolist())

features   = vectorizer.get_feature_names()



visualizer = FreqDistVisualizer(features=features, orient='h', n=10)

visualizer.fit(docs)

visualizer.show()
fig, axes = plt.subplots(1, 2, figsize=[20, 10])



df_real = df[df['target']==1]['text']



wordcloud1 = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=2500,

                      height=2000

                      ).generate(" ".join(df_real))



axes[0].imshow(wordcloud1)

axes[0].axis('off')

axes[0].set_title('Before transformation')



df_real = df[df['target']==1]['text_lm']



wordcloud2 = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=2500,

                      height=2000

                      ).generate(" ".join(df_real))



axes[1].imshow(wordcloud2)

axes[1].axis('off')

axes[1].set_title('After transformation')



plt.show()
fig, axes = plt.subplots(1, 2, figsize=[20, 10])



df_real = df[df['target']==0]['text']



wordcloud1 = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=2500,

                      height=2000

                      ).generate(" ".join(df_real))



axes[0].imshow(wordcloud1)

axes[0].axis('off')

axes[0].set_title('Before transformation')



df_real = df[df['target']==0]['text_lm']



wordcloud2 = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=2500,

                      height=2000

                      ).generate(" ".join(df_real))



axes[1].imshow(wordcloud2)

axes[1].axis('off')

axes[1].set_title('After transformation')





plt.show()
X_train, X_test, y_train, y_test = train_test_split(df[['text_lm']], df['target'], test_size=0.20, random_state=42, stratify=df['target'])
vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train['text_lm'])

X_test_vec = vectorizer.transform(X_test['text_lm'])



X_sub = vectorizer.transform(df_sub['text_lm'])

# Transform into dataframe

X_train_vec_df = pd.DataFrame(columns=vectorizer.get_feature_names(), data=X_train_vec.toarray())

X_test_vec_df = pd.DataFrame(columns=vectorizer.get_feature_names(), data=X_test_vec.toarray())



X_sub_vec_df = pd.DataFrame(columns=vectorizer.get_feature_names(), data=X_sub.toarray())



# some sample

X_train_vec_df[(X_train_vec_df['get']>0) | (X_train_vec_df['like'] > 0) | (X_train_vec_df['fire'] > 0)][['get', 'like', 'fire']].head(10)
tsne = TSNEVisualizer()

tsne.fit(X_train_vec_df, y_train)

tsne.show()
clf = DummyClassifier().fit(X_train_vec_df, y_train)

y_pred = clf.predict(X_test_vec_df)

visualizer = ConfusionMatrix(clf, percent=True)

visualizer.score(X_test_vec_df, y_test)

visualizer.show()



print(f'Acc = {accuracy_score(y_test, y_pred)}')

print(f'F1 = {f1_score(y_test, y_pred)}')
visualizer = ROCAUC(clf)

visualizer.score(X_test_vec_df, y_test)

visualizer.show()

visualizer = PrecisionRecallCurve(clf)

visualizer.fit(X_test_vec_df, y_test)

visualizer.score(X_test_vec_df, y_test)

visualizer.show()
visualizer = ClassificationReport(clf)

visualizer.score(X_test_vec_df, y_test)

visualizer.show()
clf = MultinomialNB().fit(X_train_vec_df, y_train)

visualizer = ConfusionMatrix(clf, percent=True)

visualizer.score(X_test_vec_df, y_test)

visualizer.show()



y_pred = clf.predict(X_test_vec_df)

print(f'Acc = {accuracy_score(y_test, y_pred)}')

print(f'F1 = {f1_score(y_test, y_pred)}')
visualizer = ROCAUC(clf)

visualizer.score(X_test_vec_df, y_test)

visualizer.show()
visualizer = PrecisionRecallCurve(clf)

visualizer.fit(X_train_vec_df, y_train)

visualizer.score(X_test_vec_df, y_test)

visualizer.show()
visualizer = ClassificationReport(clf)

visualizer.score(X_test_vec_df, y_test)

visualizer.show()
c = make_pipeline(vectorizer, clf)

explainer = LimeTextExplainer(class_names=['Fake', 'Real'])
idx = 2



s = X_test.iloc[idx]['text_lm']

print('Original:', df.loc[7515]['text'])

print('Treated:', s)



print(f'Outcome {y_test.iloc[idx]}')





exp = explainer.explain_instance(s, c.predict_proba)

exp.show_in_notebook()
idx = 3



s = X_test.iloc[idx]['text_lm']

print('Original:', df.loc[1294]['text'])

print('Treated:', s)



print(f'Outcome {y_test.iloc[idx]}')





exp = explainer.explain_instance(s, c.predict_proba)

exp.show_in_notebook()
visualizer = DiscriminationThreshold(clf)

visualizer.fit(X_test_vec_df, y_test)

visualizer.show() 
thr = 0.40



best_f1 = 0

best_thr = 0



y_pred_proba = clf.predict_proba(X_test_vec_df)[:, 1]

y_pred = [1 if y >= thr else 0 for y in y_pred_proba]





for t in np.linspace(0, 1, 101):

    new_y_pred = [1 if y >= t else 0 for y in y_pred_proba]    

    f1 = f1_score(y_test, new_y_pred)

    if f1 > best_f1:

        best_f1 = f1

        best_thr = t



new_y_pred = [1 if y >= best_thr else 0 for y in y_pred_proba]

        

print(f'Acc after= {accuracy_score(y_test, new_y_pred)}')

print(f'F1 after = {f1_score(y_test, new_y_pred)}')

print(f'Best thr = {best_thr}')
y_pred_proba = clf.predict_proba(X_sub_vec_df)[:, 1]

y_pred = [1 if y >= best_thr else 0 for y in y_pred_proba]



df_sub['target'] = y_pred

df_sub = df_sub[['id', 'target']]

df_sub.to_csv('submission.csv', index=False, header=True)

df_sub.head(10)