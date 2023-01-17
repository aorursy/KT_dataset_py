import numpy as np
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

df_angry = pd.read_csv('/kaggle/input/emotion/Emotion(angry).csv')
df_happy = pd.read_csv('/kaggle/input/emotion/Emotion(happy).csv')
df_sad = pd.read_csv('/kaggle/input/emotion/Emotion(sad).csv')
df_main = pd.concat([df_angry, df_happy, df_sad])
df_main = df_main.reset_index(drop=True)

# Remove duplicates
df_main = df_main.drop_duplicates(subset=['content', 'sentiment'])
df_main = df_main.reset_index(drop=True)

# Remove empty data
df_main['content'].replace('', np.nan, inplace=True)
df_main = df_main.dropna(subset = ['content'])
df_main = df_main.reset_index(drop=True)

print(df_main.head())
print('\n')
print(df_main.info())
angry_text = df_main[df_main["sentiment"] == 'angry']["content"].values
print(angry_text[0])
print("\n")
print(angry_text[1])
print("\n")
print(angry_text[2])
print("\n")
print(angry_text[3])
print("\n")
print(angry_text[4])
print("\n")
happy_text = df_main[df_main["sentiment"] == 'happy']["content"].values
print(happy_text[0])
print("\n")
print(happy_text[1])
print("\n")
print(happy_text[2])
print("\n")
print(happy_text[3])
print("\n")
print(happy_text[4])
print("\n")
sad_text = df_main[df_main["sentiment"] == 'sad']["content"].values
print(sad_text[0])
print("\n")
print(sad_text[1])
print("\n")
print(sad_text[2])
print("\n")
print(sad_text[3])
print("\n")
print(sad_text[4])
print("\n")
tot = df_main.shape[0]
vc = df_main['sentiment'].value_counts()

num_angry = vc['angry']
num_happy = vc['happy']
num_sad = vc['sad']

slices = [num_angry, num_happy, num_sad]
labeling = ['Angry','Happy', 'Sad']
explode = [0.1, 0.1, 0.1]
plt.pie(slices,explode=explode,shadow=True,autopct='%1.1f%%',labels=labeling,wedgeprops={'edgecolor':'black'})
plt.title('Sentiment of Content')
plt.tight_layout()
plt.show()
import re
import string
from textblob import TextBlob
from tqdm.notebook import tqdm

contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are",
"thx"   : "thanks"
}


def clean(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<,*?>+"','',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*','',text)
    text = re.sub("xa0'", '', text)
    text = re.sub(u"\U00002019", "'", text) # IMPORTANT: Their apostrophe character was not the usual one...
    words = text.split()
    for i in range(len(words)):
        if words[i].lower() in contractions.keys():
            words[i] = contractions[words[i].lower()]
    text = " ".join(words)
    #text = TextBlob(text).correct()
    return text

df_main['content'] = df_main['content'].apply(lambda x: clean(x))

# Remove empty data
df_main['content'].replace('', np.nan, inplace=True)
df_main = df_main.dropna(subset = ['content'])
df_main = df_main.reset_index(drop=True)
from wordcloud import WordCloud,STOPWORDS 

plt.style.use('fivethirtyeight')
stopwords = set(STOPWORDS) 
stop_word= list(stopwords) + ['http','co','https','wa','amp','û','Û','HTTP','HTTPS']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[26, 8])
wordcloud1 = WordCloud( background_color='white',stopwords = stop_word,
                        width=600,
                        height=400).generate(" ".join(df_main[df_main['sentiment']=='angry']['content']))
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Angry Content',fontsize=40)

wordcloud2 = WordCloud( background_color='white',stopwords = stop_word,
                        width=600,
                        height=400).generate(" ".join(df_main[df_main['sentiment']=='happy']['content']))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Happy Content',fontsize=40)

wordcloud3 = WordCloud( background_color='white',stopwords = stop_word,
                        width=600,
                        height=400).generate(" ".join(df_main[df_main['sentiment']=='sad']['content']))
ax3.imshow(wordcloud3)
ax3.axis('off')
ax3.set_title('Sad Content',fontsize=40)
plt.show()
plt.style.use('fivethirtyeight')

df_main['word_count'] = df_main['content'].apply(lambda x: len(x.split()))

fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,5))

df_angry = df_main[df_main['sentiment'] == 'angry']
word = df_angry['word_count']
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='red', kde=False)
ax1.set_title('Angry')

df_happy = df_main[df_main['sentiment'] == 'happy']
word = df_happy['word_count']
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='green', kde=False)
ax2.set_title('Happy')

df_sad = df_main[df_main['sentiment'] == 'sad']
word = df_sad['word_count']
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax3,color='blue', kde=False)
ax3.set_title('Sad')

fig.suptitle('Average word length by sentiment')
plt.show()
import nltk
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))


def remove_stopwords(text):
    text = text.split()
    words = [w for w in text if w not in stopwords.words('english')]
    return " ".join(words)

df_main['content_no_sw'] = df_main['content'].apply(lambda x : remove_stopwords(x))

from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import sent_tokenize,word_tokenize


lemmatizer = WordNetLemmatizer()
statuses = df_main['content'].values.copy()

for i in range(len(statuses)):
    a = statuses[i]
    sentences = sent_tokenize(statuses[i])
    word_list = []
    for sent in sentences:
        words = word_tokenize(sent)
        for word in words:
            if words not in word_list:
                word_list.append(word)
    word_list = [lemmatizer.lemmatize(w) for w in word_list if w not in stop_words]
    statuses[i] = ' '.join(w for w in word_list)
    
from nltk.stem import PorterStemmer
porter = PorterStemmer()

for i in range(len(statuses)):
    sentences = sent_tokenize(statuses[i])
    word_list = []
    for sent in sentences: 
        words = word_tokenize(sent)
        for word in words: 
            if words not in word_list:
                word_list.append(word)
    word_list = [porter.stem(w) for w in word_list if w not in stop_words]
    statuses[i] = ' '.join(w for w in word_list)

    
df_main['content_lemm_stem_no_sw'] = statuses
df_main.head()
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, plot_confusion_matrix

def show_cm(classifier, X_test, y_test):
    plt.style.use('default')
    class_names = clf.classes_
    titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
        plt.title(title)
        plt.show()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import f1_score

count_vectorizer0 = CountVectorizer(ngram_range = (1,2))
count_vectorizer1 = CountVectorizer(ngram_range = (1,2))
count_vectorizer2 = CountVectorizer(ngram_range = (1,2))

vecs0 = count_vectorizer0.fit_transform(df_main['content'])
vecs1 = count_vectorizer1.fit_transform(df_main['content_no_sw'])
vecs2 = count_vectorizer2.fit_transform(df_main['content_lemm_stem_no_sw'])

clf0 = linear_model.RidgeClassifier().fit(vecs0, df_main["sentiment"])
print("Content (Unigrams and Bigrams): No changes")
print("Percent correctly labeled comments by Ridge Classifier :")
print(clf0.score(vecs0, df_main["sentiment"]))
show_cm(clf0, vecs0, df_main['sentiment'])

clf1 = linear_model.RidgeClassifier().fit(vecs1, df_main["sentiment"])
print("Content (Unigrams and Bigrams): No stop words")
print("Percent correctly labeled comments by Ridge Classifier :")
print(clf1.score(vecs1, df_main["sentiment"]))
show_cm(clf1, vecs1, df_main['sentiment'])

clf2 = linear_model.RidgeClassifier().fit(vecs2, df_main["sentiment"])
print("Content (Unigrams and Bigrams): No stop words, lemmatized and stemmed")
print("Percent correctly labeled comments by Ridge Classifier :")
print(clf2.score(vecs2, df_main["sentiment"]))
show_cm(clf2, vecs2, df_main['sentiment'])
predict = clf1.predict(vecs1)
error_a_h = 0
error_a_s = 0
error_h_a = 0
error_h_s = 0
error_s_a = 0
error_s_h = 0
for i in range(len(predict)):
    prediction = predict[i]
    actual = df_main.loc[i, 'sentiment']
    if actual == 'angry' and prediction == 'happy' and error_a_h == 0:
        print("Angry status mislabeled as Happy:")
        print(df_main.loc[i, 'content'])
        print('\n')
        error_a_h += 1
    elif actual == 'angry' and prediction == 'sad' and error_a_s == 0:
        print("Angry status mislabeled as Sad:")
        print(df_main.loc[i, 'content'])
        print('\n')
        error_a_s += 1
    elif actual == 'happy' and prediction == 'angry' and error_h_a == 0:
        print("Happy status mislabeled as Angry:")
        print(df_main.loc[i, 'content'])
        print('\n')
        error_h_a += 1
    elif actual == 'happy' and prediction == 'sad' and error_h_s == 0:
        print("Happy status mislabeled as Sad:")
        print(df_main.loc[i, 'content'])
        print('\n')
        error_h_s += 1
    elif actual == 'sad' and prediction == 'angry' and error_s_a == 0:
        print("Sad status mislabeled as Angry:")
        print(df_main.loc[i, 'content'])
        print('\n')
        error_s_a += 1
    elif actual == 'sad' and prediction == 'happy' and error_s_h == 0:
        print("Sad status mislabeled as Happy:")
        print(df_main.loc[i, 'content'])
        print('\n')
        error_s_h += 1
        