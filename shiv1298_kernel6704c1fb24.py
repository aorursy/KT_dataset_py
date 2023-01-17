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
df=pd.read_csv('/kaggle/input/document-classification/file.txt')
df
#column 5485 contains the labels with in it

list_1=[]

for i in df['5485']:

    list_1.append(int(i[0]))
#creating a new column with lables

df['Target']=list_1
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(binary=True)

df
df.isnull().sum()
df=df.rename({'5485':'Text'},axis=1)
df
#remove the labels from the first column

df['Text']=df['Text'].str[1:]
df
import string

exclude = set(string.punctuation)

def remove_punctuation(x):

    """

    Helper function to remove punctuation from a string

    x: any string

    """

    try:

        x = ''.join(ch for ch in x if ch not in exclude)

    except:

        pass

    return x
df['Text']=df['Text'].apply(remove_punctuation)

df['Text'] = df['Text'].str.replace('\d+', '')
df
x=df['Text']

y=df['Target']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

cv=CountVectorizer(min_df=3,ngram_range=(1,2))

x_vect=cv.fit_transform(x_train)

lr=LogisticRegression()

lr.fit(x_vect,y_train)

pred_y=lr.predict(cv.transform(x_test))
pred_y
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,pred_y))
feature_to_coef = {

    word: coef for word, coef in zip(

        cv.get_feature_names(), lr.coef_[0]

    )

}

for best_positive in sorted(

    feature_to_coef.items(), 

    key=lambda x: x[1], 

    reverse=True)[:5]:

    print (best_positive)

    
for best_negative in sorted(

    feature_to_coef.items(), 

    key=lambda x: x[1])[:5]:

    print (best_negative)
new_df=pd.DataFrame({'actual':y_test,

                    'predicted':pred_y})
new_df