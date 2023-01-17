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
test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

train = pd.read_csv("../input//tweet-sentiment-extraction/train.csv")
test.head()
train.head()
neutral = train[train['sentiment']=="neutral"]
print(neutral.shape[0]/train.shape[0]*100,"% of the data is neutral")
neutral = neutral.drop(['sentiment','textID'],axis=1).reset_index(drop= True)
same = neutral[neutral['text'] == neutral ['selected_text']].reset_index(drop= True)
print(same.shape)

print(neutral.shape)
same.shape[0]/neutral.shape[0]
diff = neutral[neutral['text'] != neutral ['selected_text']].reset_index(drop= True)
diff.head(20)
print(diff.loc[0,'text'],"**", diff.loc[0,'selected_text'])
print(len(diff.loc[0,'text']),"**", len(diff.loc[0,'selected_text']))
def compare(a, b):

    for x, y in zip(a, b):

        if x != y:

            print("Should be ",x," But is ", y)
compare(diff.loc[0,'text'], diff.loc[0,'selected_text'])
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
jaccard(diff.loc[0,'text'], diff.loc[0,'selected_text'])