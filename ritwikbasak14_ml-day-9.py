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
data=pd.read_csv("/kaggle/input/sentiment_train.csv")
data.shape
data.columns
len(data[data["label"]==1])
len(data[data["label"]==0])
3204/(2464+3204)
import seaborn as sns
sns.countplot(data=data,x="label")
from sklearn.feature_extraction.text import CountVectorizer
vf=CountVectorizer(lowercase=True,stop_words="english")

bag_of_words=vf.fit_transform(data["sentence"])
data2=pd.DataFrame(bag_of_words.A,columns=vf.get_feature_names())
data3=pd.DataFrame(data2)
data3 is data2
for i in data2.columns:

    if len(data2[data2[i]>0])==1:

        data3.drop(i,axis=1,inplace=True)
data3.shape
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data3,data.drop(["sentence"],axis=1,inplace=False),test_size=0.2,random_state=20)
model=GaussianNB()

model.fit(x_train,y_train)
model.score(x_test,y_test)
from sklearn.metrics import confusion_matrix

predictions=model.predict(x_test)

cm=confusion_matrix(y_test,predictions)

cm