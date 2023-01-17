import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sms_data=pd.read_csv('../input/spam.csv',encoding='latin-1')
sms_data.head()
sms_data.rename(columns={'v1':'label','v2':'message'},inplace=True)
sms_data.head()
sms_df=sms_data[['label','message']]
sms_df.head()
sms_df.info()
sms_df.describe()
sms_df.groupby('label').describe()
sms_df['message_length']=sms_df['message'].apply(len)
sms_df.head()
plt.figure(figsize=(14,6))
sns.set_style('whitegrid')
sns.distplot(sms_df['message_length'],bins=50,hist_kws={'edgecolor':'green'},kde=False)
sms_df.message_length.describe()
sms_df[sms_df['message_length']==910]['message'].iloc[0]
sms_df.hist(column='message_length',by='label',bins=50,figsize=(14,6),edgecolor="green")
import nltk
import string
from nltk.corpus import stopwords
stopwords.words('english')[0:10]
def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
sms_df['message'].head(5).apply(text_process)
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(sms_df['message'])
print(len(bow_transformer.vocabulary_))
messages_bow = bow_transformer.transform(sms_df['message'])
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, sms_df['label'])
all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)
from sklearn.metrics import classification_report
print (classification_report(sms_df['label'], all_predictions))
