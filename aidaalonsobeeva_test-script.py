%matplotlib inline
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
df_aliases = pd.read_csv('../input/Aliases.csv', index_col=0)
df_emails = pd.read_csv('../input/Emails.csv', index_col=0)
df_email_receivers = pd.read_csv('../input/EmailReceivers.csv', index_col=0)
df_persons = pd.read_csv('../input/Persons.csv', index_col=0) 
df_emails.head(1)
df_emails.describe()
top = df_email_receivers.PersonId.value_counts().head(n=10).to_frame()
top.columns = ["Emails received"]
top = pd.concat([top, df_persons.loc[top.index]], axis=1)
top.plot(x='Name', kind='barh', figsize=(12,8), grid=True, color='purple')
top.plot(x='Name', kind='barh', figsize=(12,8), grid=True, color='purple')
# Data cleaning
df_persons['Name'] = df_persons['Name'].str.lower()
df_emails = df_emails.dropna(how='all').copy()
print(len(df_emails))
person_id = df_persons[df_persons.Name.str.contains('hillary')].index.values
# identificadores de hillary
df_emails = df_emails[(df_emails['SenderPersonId']==person_id[0])]
print(u'Hillarys emails:', len(df_emails))
df_emails['MetadataDateSent'] = pd.to_datetime(df_emails['MetadataDateSent'])
df_emails = df_emails.set_index('MetadataDateSent')
df_emails['dayofweek'] = df_emails.index.dayofweek 
sns.set_style('white')
t_labels = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
ax = sns.barplot(x=np.arange(0,7), y=df_emails.groupby('dayofweek').SenderPersonId.count(),\
 label=t_labels, palette="RdBu")
sns.despine(offset=10)
ax.set_xticklabels(t_labels)
ax.set_ylabel('Message Count')
ax.set_title('Hillary\'s Sent Emails') 