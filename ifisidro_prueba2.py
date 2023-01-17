# The data comes both as CSV files and a SQLite database

import sqlite3
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
sql_conn = sqlite3.connect('../input/database.sqlite')
# MetadataTo - Email TO field (from the FOIA metadata)
# MetadataFrom - Email FROM field (from the FOIA metadata)
# ExtractedBodyText - Attempt to only pull out the text in the body that the email sender wrote (extracted from the PDF)
data = sql_conn.execute('SELECT MetadataTo, MetadataFrom, ExtractedBodyText FROM Emails')
# https://docs.python.org/3/library/sqlite3.html
showfirst = 8
l =0
Senders = []
for email in data:
 if l<showfirst:
  print(email)
  Senders.append(email[1].lower())
  l+=1
 else:
  break
print('\n',Senders)
[sender for sender in Senders if sender!='']
df_aliases = pd.read_csv('../input/Aliases.csv', index_col=0)
df_emails = pd.read_csv('../input/Emails.csv', index_col=0)
df_email_receivers = pd.read_csv('../input/EmailReceivers.csv', index_col=0)
df_persons = pd.read_csv('../input/Persons.csv', index_col=0) 
df_emails.columns
top = df_email_receivers.PersonId.value_counts().head(n=10).to_frame()
top.columns = ['Emails received']
top = pd.concat([top, df_persons.loc[top.index]], axis=1)
top.plot(x='Name', kind='barh', figsize=(12, 8), grid=True, color='green')
# Data cleaning
df_persons['Name'] = df_persons['Name'].str.lower()
df_emails = df_emails.dropna(how='all').copy()
print(len(df_emails)) 
person_id = df_persons[df_persons.Name.str.contains('hillary')].index.values # identificadores de hillary
df_emails = df_emails[(df_emails['SenderPersonId']==person_id[0])] #
print(u'Hillarys emails:', len(df_emails))
df_emails['MetadataDateSent'] = pd.to_datetime(df_emails['MetadataDateSent'])
df_emails = df_emails.set_index('MetadataDateSent')
df_emails['dayofweek'] = df_emails.index.dayofweek
#http://pandas.pydata.org/pandasdocs/stable/generated/pandas.DatetimeIndex.dayofweek.html
df_emails.MetadataTo.value_counts().head(n=10)
sns.set_style('white')
t_labels = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
ax = sns.barplot(x=np.arange(0,7), y=df_emails.groupby('dayofweek').SenderPersonId.count(),\
 label=t_labels, palette="RdBu")
sns.despine(offset=10)
ax.set_xticklabels(t_labels)
ax.set_ylabel('Message Count')
ax.set_title('Hillary\'s Sent Emails') 
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
import re
def cleanEmailText(text):
    text = re.sub(r"-", " ", text) # Replace hypens with spaces
    text = re.sub(r"\d+/\d+/\d+", "", text)# Removes dates
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text) # Removes times
    text = re.sub(r"[\w]+@[\.\w]+", "", text)# Removes email addresses
    text = re.sub(r"/[a-zA-Z]*[:\/\/]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text) # Removes web addresses
    clndoc = ''
    for eachLetter in text:
        if eachLetter.isalpha() or eachLetter == ' ':
            clndoc += eachLetter
        text = ' '.join(clndoc.split()) # Remove any bad characters
    return text 
np.random.seed(13)
data = df_emails['RawText']
data = data.apply(lambda s: cleanEmailText(s))
vectorizer = TfidfVectorizer(max_df=0.6, max_features=500,stop_words='english', use_idf=True)
X = vectorizer.fit_transform(data) 
svd = TruncatedSVD(100)
lsa = make_pipeline(svd, Normalizer(copy=False))
X = lsa.fit_transform(X) 
x_vect = np.arange(3,100,5)
y_vect = np.zeros(x_vect.shape)
for i, cl in enumerate(x_vect):
    km = KMeans(n_clusters=cl, init='k-means++', max_iter=100, n_init=1, verbose=0)
    km.fit(X)
    dist = np.min(cdist(X,km.cluster_centers_,'euclidean'),axis=1)
    y_vect[i] = np.sum(dist)/X.shape[0]

plt.plot(x_vect,y_vect,marker="o")
plt.ylim([0,1])
k = 30
km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1,
 verbose=0)
km.fit(X)
original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(k):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()