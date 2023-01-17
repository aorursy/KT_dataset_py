import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

%matplotlib inline
plt.rcParams["figure.figsize"] = (16,9)
sns.set_style('whitegrid')
train_df = pd.read_csv('../input/train.tsv',delimiter='\t')
test_df = pd.read_csv('../input/test.tsv',delimiter='\t')
train_df.head()
train_df.shape
train_df.isnull().sum()
sns.countplot(x='Sentiment',data=train_df)
#label mapping
labels = ["Negative","Somewhat negative","Neutral","Somewhat positive","Positive"]
sentiment_code = [0,1,2,3,4]

labels_df = pd.DataFrame({"Label":labels,"Code":sentiment_code})
labels_df
train_df['Phrase Length'] = train_df['Phrase'].apply(len)
sns.distplot(train_df['Phrase Length'],bins=80,kde=False,hist_kws={"edgecolor":"blue"})
train_df.hist(column='Phrase Length',by='Sentiment',bins=80,edgecolor='black')
ps = PorterStemmer()
def text_processing(comment):
    nopunc = [char for char in comment if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_text = [text for text in nopunc.split() if text.lower() not in stopwords.words('english')]
    #final_text = [ps.stem(text) for text in clean_text]
    return clean_text
train_df['Phrase'].head(5).apply(text_processing)
bow_transformer = CountVectorizer(analyzer=text_processing).fit(train_df['Phrase'])
phrases_bow = bow_transformer.transform(train_df['Phrase'])
tfidf_transformer = TfidfTransformer().fit(phrases_bow)
phrases_tfidf = tfidf_transformer.transform(phrases_bow)
X_train, X_test, y_train, y_test = train_test_split(phrases_tfidf, train_df['Sentiment'], test_size=0.3, random_state=42)
sentiment_detect_model = MultinomialNB().fit(X_train, y_train)
predictions = sentiment_detect_model.predict(X_test)
print (classification_report(y_test, predictions))
test_df.head()
test_transformer = CountVectorizer(analyzer=text_processing).fit(test_df['Phrase'])
test_bow = bow_transformer.transform(test_df['Phrase'])

test_transformer = TfidfTransformer().fit(test_bow)
test_tfidf = test_transformer.transform(test_bow)
test_predictions = sentiment_detect_model.predict(test_tfidf)
test_df['Sentiment'] = test_predictions
submission_df = test_df[['PhraseId','Sentiment']]
submission_df.to_csv('submission.csv',index=False)
