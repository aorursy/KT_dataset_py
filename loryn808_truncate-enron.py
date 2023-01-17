import os, sys, email
import numpy as np 
import pandas as pd
# Plotting
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns; sns.set_style('whitegrid')
#import plotly
#plotly.offline.init_notebook_mode()
#import plotly.graph_objs as go
import wordcloud

# Network analysis
import networkx as nx
# NLP
from nltk.tokenize.regexp import RegexpTokenizer

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Read the data into a DataFrame
emails_df = pd.read_csv('../input/emails.csv')
print(emails_df.shape)
emails_df.head()
# A single message looks like this
print(emails_df['message'][0])
## Helper functions
def get_text_from_email(msg):
    '''To get the content from email objects'''
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append( part.get_payload() )
    return ''.join(parts)

def split_email_addresses(line):
    '''To separate multiple email addresses'''
    if line:
        addrs = line.split(',')
        addrs = frozenset(map(lambda x: x.strip(), addrs))
    else:
        addrs = None
    return addrs

print ('DONE')

# Parse the emails into a list email objects
messages = list(map(email.message_from_string, emails_df['message']))
emails_df.drop('message', axis=1, inplace=True)
# Get fields from parsed email objects
keys = messages[0].keys()
for key in keys:
    emails_df[key] = [doc[key] for doc in messages]
# Parse content from emails
emails_df['content'] = list(map(get_text_from_email, messages))
# Split multiple email addresses
emails_df['From'] = emails_df['From'].map(split_email_addresses)
emails_df['To'] = emails_df['To'].map(split_email_addresses)

# Extract the root of 'file' as 'user'
emails_df['user'] = emails_df['file'].map(lambda x:x.split('/')[0])
del messages

emails_df.head()
#print('shape of the dataframe:', emails_df.shape)
    
emails_filtered = emails_df.filter(items=['Date', 'From', 'To', 'Subject', 'content'])
#emails_filtered['Date'] = pd.to_datetime()

emails_filtered['new_date'] = pd.to_datetime(emails_filtered['Date'])

#for col in emails_filtered.columns:
#    print(col, emails_df[col].nunique())
    
#emails_name_filtered = emails_filtered.filter(like='phillip.allen@enron.com', axis=0)
email_truncated = emails_filtered.head(1000)
email_truncated.shape

email_truncated.to_csv('enron_test.csv', index=False)