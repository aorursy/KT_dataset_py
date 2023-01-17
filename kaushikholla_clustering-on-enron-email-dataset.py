# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import  email
# Importing the dataset
df = pd.read_csv("../input/enron-email-dataset/emails.csv", nrows = 35000)
df.shape
df.head()
df['message'][500]
#list(email.parser.Parser().parsestr(df['message'][500]))
email = list(map(email.parser.Parser().parsestr, df['message']))
keys = email[0].keys()
keys
for key in keys:
    df[key] = [mail[key] for mail in email]
df.head()
for i in email[0].walk():
    print(i)
#df['file'].unique
# Extracting the body of the mail
def get_raw_text(emails):
    email_text = []
    
    for mail in emails.walk():
        if mail.get_content_type() == "text/plain":
            email_text.append(mail.get_payload())
            
    return ''.join(email_text)
df['body'] = list(map(get_raw_text, email))
#df['body'][1]
# Getting user name
df['user'] = df['file'].map(lambda x: x.split('/')[0])
df.dtypes
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format = True, utc = True)
df.dtypes
df.head()
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Day'] = df['Date'].dt.dayofweek
