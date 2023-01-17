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
df=pd.read_csv("/kaggle/input/spam-ham-emails/emails.csv")
df
df.loc[df["spam"]=="ham","spam"]=0   #assigning the value 1 to spam and then value 0 to ham
df.loc[df["spam"]=="spam","spam"]=1
df.describe()
# seprating the target and features
x=df.text
y=df.spam
# performing the test train split
from sklearn.model_selection import train_test_split as tts
train_x,test_x,train_y,test_y=tts(x,y,test_size=0.2)
# returns us a vector count of the sommon words in spam mails
from sklearn.feature_extraction.text import CountVectorizer
# model we would use to train
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline   # instead of writing code and making a countvectoriser and then applying multinomial nb we create a pipeline 
clf=Pipeline([
    ("vectoriser",CountVectorizer()),
    ("nb",MultinomialNB())
])
clf.fit(train_x,train_y)
train_pred=clf.predict(train_x)
train_=f1_score(train_y,train_pred)
train_
test_pred=clf.predict(test_x)
test_=f1_score(test_y,test_pred)
test_
emails=[
    "this offer is specially for you avail it to get the discounts",
    "hey raj, can we watch football game tommorow?"]
email_count=v.transform(emails)
model.predict(email_count)
clf.predict(emails)
