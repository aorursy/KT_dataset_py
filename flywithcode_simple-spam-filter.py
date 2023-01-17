import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
%matplotlib inline
emails = pd.read_csv('../input/emails.csv')
emails.head()
#Lets read a single email 

emails.get_value(58,'text')
emails.shape
#Total 5728 emails
#Checking class distribution
emails.groupby('spam').count()
#23.88% emails are spam which seems good enough for our task
#Lets see the distribution of spam using beautiful seaborn package

label_counts = emails.spam.value_counts()
plt.figure(figsize = (12,6))
sns.barplot(label_counts.index, label_counts.values, alpha = 0.9)

plt.xticks(rotation = 'vertical')
plt.xlabel('Spam', fontsize =12)
plt.ylabel('Counts', fontsize = 12)
plt.show()
#Lets check if email length is coorelated to spam/ham
emails['length'] = emails['text'].map(lambda text: len(text))

emails.groupby('spam').length.describe()
#emails length have some extreme outliers, lets set a length threshold & check length distribution
emails_subset = emails[emails.length < 1800]
emails_subset.hist(column='length', by='spam', bins=50)

#Nothing much here, lets process the contents of mail now for building spam filter
emails['tokens'] = emails['text'].map(lambda text:  nltk.tokenize.word_tokenize(text)) 
#Lets check tokenized text from first email

print(emails['tokens'][1])
#Removing stop words

stop_words = set(nltk.corpus.stopwords.words('english'))
emails['filtered_text'] = emails['tokens'].map(lambda tokens: [w for w in tokens if not w in stop_words]) 
#Every mail starts with 'Subject :' lets remove this from each mail 

emails['filtered_text'] = emails['filtered_text'].map(lambda text: text[2:])
#Lets compare an email with stop words removed

print(emails['tokens'][3],end='\n\n')
print(emails['filtered_text'][3])

#many stop words like 'the', 'of' etc. were removed
#Mails still have many special charater tokens which may not be relevant for spam filter, lets remove these
#Joining all tokens together in a string
emails['filtered_text'] = emails['filtered_text'].map(lambda text: ' '.join(text))

#removing apecial characters from each mail 
emails['filtered_text'] = emails['filtered_text'].map(lambda text: re.sub('[^A-Za-z0-9]+', ' ', text))
wnl = nltk.WordNetLemmatizer()
emails['filtered_text'] = emails['filtered_text'].map(lambda text: wnl.lemmatize(text))
#Lets check one of the mail again after all these preprocessing steps
emails['filtered_text'][4]
#Wordcloud of spam mails
spam_words = ''.join(list(emails[emails['spam']==1]['filtered_text']))
spam_wordclod = WordCloud(width = 512,height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wordclod)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()
#Wordcloud of non-spam mails
spam_words = ''.join(list(emails[emails['spam']==0]['filtered_text']))
spam_wordclod = WordCloud(width = 512,height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wordclod)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()
count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(emails['filtered_text'].values)
print(counts.shape)
classifier = MultinomialNB()
targets = emails['spam'].values
classifier.fit(counts, targets)
#Predictions on sample text
examples = ['cheap Viagra', "Forwarding you minutes of meeting"]
example_counts = count_vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)
tfidf_vectorizer = TfidfTransformer().fit(counts)
tfidf = tfidf_vectorizer.transform(counts)
print(tfidf.shape)
classifier = MultinomialNB()
targets = emails['spam'].values
classifier.fit(counts, targets)
#Predictions on sample text
examples = ['Free Offer Buy now',"Lottery from Nigeria","Please send the files"]
example_counts = count_vectorizer.transform(examples)
example_tfidf = tfidf_vectorizer.transform(example_counts)
predictions_tfidf = classifier.predict(example_tfidf)
print(predictions_tfidf)