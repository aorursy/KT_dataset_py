# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re # regex python

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")



from nltk.tokenize import word_tokenize



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
filename = '/kaggle/input/enron-email-dataset/emails.csv'

n = 50  # every 100th line = read 1% of the emails (total emails = 517400)

df = pd.read_csv(filename, header=0, skiprows=lambda i: i % n != 0)



# df = pd.read_csv(filename)

print("shape of the dataset:",df.shape)

df.head()
# for i in range(20):

#     print("################################################ EMAIL CONTENT NUMBER:",i,"############################################################################")

#     print(df.message[i])
# Dropping the file column:

df = df.drop(['file'], axis=1)
# Splitting:

df['pre_info']= df.message.map(lambda r: r.split('\n\n', 1)[0])

df['content']= df.message.map(lambda r: r.split('\n\n', 1)[1])

df = df.drop(['message'], axis=1)

df.head()
# Check the pre-info part:

print(df.pre_info[0])

# Keep the message id for indexing later on:

# df['message_id'] = df.pre_info.map(lambda r: r.split('\n')[0].split('Message-ID: ')[1])

# df = df.drop(['pre_info'], axis=1)
# #Investigating the content first:

# for i in range(25):

#     print("################################################ EMAIL CONTENT NUMBER:",i,"############################################################################")

#     print(df.content[i])
df.content.str.contains('[- ]*Forwarded by').value_counts()
# Test the deal with one sample email:

email = df.content[df.content.str.contains('[- ]*Forwarded by')].iloc[0]

print(email)

print("############################ END OF EMAIL ################################################################")

condition = '[- ]*Forwarded by[\S\s]*Subject:[\S\t ]*'

print(re.sub(condition, '', email).strip())



# Do it for all the others:

def deal_forwarded(row):

    condition = '[- ]*Forwarded by[\S\s]*Subject:[\S\t ]*'

    return re.sub(condition, '', row).strip()

df['content1'] = df.content.map(deal_forwarded)
print(df.content1.str.contains('[- ]*Forwarded by').value_counts())
# for email in df.content1[df.content1.str.contains('[- ]*Forwarded by')]:

#     print(email)

#     print("############################ END OF EMAIL ################################################################")
# Test for one email:

email = df.content1[df.content1.str.contains('[- ]*Forwarded by')].iloc[0]

print(email)

print("############################ END OF EMAIL ################################################################")

condition = '[- ]*Forwarded by[\S\s]*---[-]+' 

print(re.sub(condition, '', email).strip())



# DO it for all the others:

def deal_forwarded_patternless(row):

    condition = '[- ]*Forwarded by[\S\s]*[-]+'

    return re.sub(condition, '', row).strip()

df['content2'] = df.content1.map(deal_forwarded_patternless)
print(df.content2.str.contains('[- ]Forwarded by').value_counts())
# for i in range(10,50):

#     print("################################################ EMAIL CONTENT NUMBER:",i,"############################################################################")

#     print(df.content2[i])
df.content2.str.contains('[- ]Original Message').value_counts()
# Test the deal with one sample email:

email = df.content2[df.content2.str.contains('[- ]Original Message')].iloc[0]

print(email)

print("############################ END OF EMAIL ################################################################")

condition = '[- ]*Original Message[\S\s]*Subject:[\S\t ]*'

print(re.sub(condition, '', email).strip())



# Do it for all emails:

def deal_originals(row):

    condition = '[- ]*Original Message[\S\s]*Subject:[\S\t ]*'

    return re.sub(condition, '', str(row)).strip()

df['content3'] = df.content2.map(deal_originals)
df.content3.str.contains('[- ]*Original Message').value_counts()
# for email in df.content3[df.content3.str.contains('[- ]*Original Message')]:

#     print(email)

#     print("############################ END OF EMAIL ################################################################")
# Test the deal with one sample email:

email = df.content3[df.content3.str.contains('[- ]*Original Message')].iloc[0]

print(email)

print("############################ END OF EMAIL ################################################################")

condition = '[- ]*Original Message[\S\s]*(Sent:[\S\t ]*|Date:[\S\t ]*)'

print(re.sub(condition, '', email).strip())



# Do it for all emails:

def deal_originals_new(row):

    condition = '[- ]*Original Message[\S\s]*(Sent:[\S\t ]*|Date:[\S\t ]*)'

    return re.sub(condition, '', str(row)).strip()

df['content4'] = df.content3.map(deal_originals_new)
# Check again:

# print(df.content4.str.contains('[- ]*Original Message').value_counts())

# for email in df.content4[df.content4.str.contains('[- ]*Original Message')]:

#     print(email)

#     print('############################################## END OF EMAIL ###############################################')
# for i in range(50,90):

#     print("################################################ EMAIL CONTENT NUMBER:",i,"############################################################################")

#     print(df.content4[i])
# emails containing the pattern:

df.content4.str.contains('From:[\S\s]*Subject:[\S \t]*').value_counts()
# Test the deal with one sample email:

email = df.content4[df.content4.str.contains('From:[\S\s]*Subject:[\S \t]*')].iloc[0]

print(email)

print("############################ END OF EMAIL ################################################################")

condition = 'From:[\S\s]*Subject:[\S \t]*'

print(re.sub(condition, '', email).strip())



# Do it for all emails:

def deal_from(row):

    condition = 'From:[\S\s]*Subject:[\S \t]*'

    return re.sub(condition, '', str(row)).strip()

df['content5'] = df.content4.map(deal_from)
df.content5.str.contains('From:').value_counts()
# for email in df.content5.loc[df.content5.str.contains('From:')]:

#     print(email)

#     print("############################ END OF EMAIL ################################################################")
# emails containing `To`:

df.content5.str.contains('To:[\S\s]*Subject:[\S\t ]*').value_counts()
# emails containing `To` on a new line:

df.content5.str.contains('\n[\S \t]*To:[\S\s]*Subject:[\S\t ]*').value_counts()
# emails containing zero or one line before `To`:

df.content5.str.contains('\n[\S \t]*\n[\S \t]*To:[\S\s]*Subject:[\S\t ]*').value_counts()
# emails containing zero or one or two lines before `To`:

df.content5.str.contains('\n[\S \t]*\n[\S \t]*\n[\S \t]*To:[\S\s]*Subject:[\S\t ]*').value_counts()
# # Investigate +_+

# for email in df.content5.loc[df.content5.str.contains('\n[\S \t]*\n[\S \t]*\n[\S \t]*To:[\S\s]*Subject:[\S\t ]*')]:

#     print(email)

#     print("############################ END OF EMAIL ################################################################")
# Test the deal with one sample email:

email = df.content5.loc[df.content5.str.contains('\n[\S \t]*\n[\S \t]*\n[\S \t]*To:[\S\s]*Subject:[\S\t ]*')].iloc[0]

print(email)

print("############################ END OF EMAIL ################################################################")

condition = '\n[\S \t]*\n[\S \t]*\n[\S \t]*To:[\S\s]*Subject:[\S \t]*'

print(re.sub(condition, '', email).strip())



# Do it for all emails:

def deal_to(row):

    condition = '\n[\S \t]*\n[\S \t]*\n[\S \t]*To:[\S\s]*Subject:[\S \t]*'

    return re.sub(condition, '', str(row)).strip()

df['content6'] = df.content5.map(deal_to)
df.content6.str.contains('To:[\S\s]*Subject:[\S\t ]*').value_counts()
# for email in df.content6.loc[df.content6.str.contains('To:[\S\s]*Subject:[\S\t ]*')]:

#     print(email)

#     print("############################ END OF EMAIL ################################################################")
# Test the deal with one sample email:

email = df.content6.loc[df.content6.str.contains('To:[\S\s]*Subject:[\S\t ]*')].iloc[0]

print(email)

print("############################ END OF EMAIL ################################################################")

condition = '[\S\t ]*\nTo:[\S\s]*Subject:[\S \t]*'

print(re.sub(condition, '', email).strip())



# Do it for all emails:

def deal_to_new(row):

    condition = '[\S\t ]*\nTo:[\S\s]*Subject:[\S \t]*'

    return re.sub(condition, '', str(row)).strip()

df['content7'] = df.content6.map(deal_to_new)
df.content7.str.contains('To:').value_counts()
# for email in df.content7.loc[df.content7.str.contains('To:')]:

#     print(email)

#     print("############################ END OF EMAIL ################################################################")
print(df.content7.str.contains('=20').value_counts())

print(df.content7.str.contains('=20|=10|=09|=01').value_counts())

print(df.content7.str.contains('=\d\d').value_counts())
# for email in df.content7.loc[df.content7.str.contains('=20|=10|=09|=01')]:

#     print(email)

#     print("############################ END OF EMAIL ################################################################")
# Test the deal with one email:

email = df.content7.loc[df.content7.str.contains('=20|=10|=09|=01')].iloc[1]

print(email)

print("############################ END OF EMAIL ################################################################")

# condition = '[=]+[\n\t =]*\d\d'

# |[=\n]*[=10]|[=\n]*[=01]|[=\n]*[=09]'

# [\w]=[\w]'

condition1 = '[=]+\d\d'

condition2 = '[=]+[ \n]+'

email = re.sub(condition1, '', email)

print("############################## AFTER COND 1 ###############################################################")

print(re.sub(condition, '', email).strip())

email = re.sub(condition2, '', email)

print("############################## AFTER COND 2 ###############################################################")

print(re.sub(condition, '', email).strip())





# Do this for all emails:

def deal_equalsto(row):

    condition1 = '[=]+\d\d'

    condition2 = '[=]+[ \n]+'

    row = re.sub(condition1, '', str(row))

    return re.sub(condition2, '', str(row)).strip()

df['content8'] = df.content7.map(deal_equalsto)
print(df.content8.str.contains('=20').value_counts())

print(df.content8.str.contains('=20|=10|=09|=01').value_counts())

# print(df.content8.str.contains('=\d\d').value_counts())
for i in range(1000, 1050):

    print("################################################ EMAIL CONTENT NUMBER:",i,"############################################################################")

    print(df.content8[i])
# Getting the final content8 out:

df_nlp = df[['pre_info', 'content8']]

# Removing the emails that are empty:

df_nlp = df_nlp.loc[~(df_nlp.content8=='')]

df_nlp = df_nlp.rename(columns={'content8':'content'}).reset_index(drop=True)

print(df_nlp.shape)

df_nlp.head()
# # Tokenize using your fav tokenizer:

# # This step is quite time consuming: 

df_nlp['bert_tokens'] = df_nlp['content'].map(lambda r: tokenizer.tokenize(r))

df_nlp['nltk_tokens'] = df_nlp['content'].map(lambda r: word_tokenize(r))
df_nlp.to_csv('enron-pre-processed-nlp.csv')
# Saving a file for BERT transformer models:

df_out_final = df_nlp.content.loc[df_nlp['bert_tokens'].map(lambda r: (len(r)>10 and len(r)<512))].reset_index(drop=True)

print(df_out_final.shape)

# Save the final contents:

df_out_final.to_csv('enron-processed.tsv',sep='\t')
token_list = [token for row in df_nlp['nltk_tokens'] for token in row]

# token_list = [token for row in df_nlp['bert_tokens'] for token in row]

token_series = pd.Series(token_list)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also

    print(token_series.value_counts())