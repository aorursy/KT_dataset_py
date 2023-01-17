# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/enron-email-dataset/emails.csv')
df.info()

df.head()
def message_id(x):

    message_id = x.split('\n')[0]

    return message_id.split(':')[1]

df['Message_ID'] = df['message'].apply(message_id)
def date(x):

    date = x.split('\n')[1]

    date = date.split(':')[1].split(',')[1]

    date = date[:-3]

    return date

df['Date'] = df['message'].apply(date)
def month(x):

    date = x.split('\n')[1]

    return date.split(':')[1].split(',')[1].split(' ')[2]

df['Month'] = df['message'].apply(month)
def year(x):

    date = x.split('\n')[1]

    return date.split(':')[1].split(',')[1].split(' ')[3]

df['Year'] = df['message'].apply(year)



df.head(2)
def day(x):

    date = x.split('\n')[1]

    return date.split(':')[1].split(',')[0]

df['Day'] = df['message'].apply(day)

    

     
def sender(x):

    sender = x.split('\n')[2]

    return sender.split(':')[1]

df['Sender_Email'] = df['message'].apply(sender)
def receiver(x):

    receiver = x.split('\n')[3]

    return receiver.split(':')[1]

df['Receiver_Email'] = df['message'].apply(receiver)
def subject(x):

    subject = x.split('\n')[4]

    return subject.split(':')[-1]

df['Subject'] = df['message'].apply(subject)
def mime_version(x):

    mime_version = x.split('\n')[5]

    return mime_version.split(':')[-1]

df['Mime_Version'] = df['message'].apply(mime_version)
df['Mime_Version'].isnull().count()
def content_type(x):

    content_type = x.split('\n')[6]

    return content_type.split(':')[-1].split(';')[0]

df['Content-Type'] = df['message'].apply(content_type)



def ascii_set(x):

    content_type = x.split('\n')[6]

    return content_type.split(':')[-1].split(';')[-1]

df['Char_Set']= df['message'].apply(ascii_set)
def content_encoding(x):

    content_encoding = x.split('\n')[7]

    return content_encoding.split(':')[-1]

df['Content-Encoding'] = df['message'].apply(content_encoding)
def x_from(x):

    x_from = x.split('\n')[8]

    return x_from.split(':')[-1]

df['Sender'] = df['message'].apply(x_from)
def x_to(x):

    x_to = x.split('\n')[9]

    return x_to.split(':')[-1]

df['Receiver'] = df['message'].apply(x_to)
def x_cc(x):

    x_cc = x.split('\n')[10]

    return x_cc.split(':')[-1]

df['CC'] = df['message'].apply(x_cc)
def x_bcc(x):

    x_bcc = x.split('\n')[11]

    return x_bcc.split(':')[-1]

df['BCC'] = df['message'].apply(x_bcc)
def x_folder(x):

    x_folder = x.split('\n')[12]

    return x_folder.split(':')[-1]

df['Folder'] = df['message'].apply(x_folder)
def x_origin(x):

    x_origin = x.split('\n')[13]

    return x_origin.split(':')[-1]

df['Origin'] = df['message'].apply(x_origin)
def x_filename(x):

    x_filename = x.split('\n')[14]

    return x_filename.split(':')[-1]

df['File_Name'] = df['message'].apply(x_filename)
df.head(10)