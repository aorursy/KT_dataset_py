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
data=pd.read_csv("../input/sentiment-train/sentiment_train.csv")
print(data)

data.groupby('label').size()
import matplotlib.pyplot as plt
import seaborn as s
s.countplot(data['label'])

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(lowercase=True,stop_words='english')
bow = cv.fit_transform(data['sentence'])
df_bow = pd.DataFrame(bow.A,columns=cv.get_feature_names())
print(df_bow)

for col in df_bow.columns:
    d=df_bow[col].sum()
    if(d<=2):
        df_bow=df_bow.drop(col,axis=1,inplace=True)
print(df_bow)      

y = data['label']
X = df_bow
print(y)
print(X)