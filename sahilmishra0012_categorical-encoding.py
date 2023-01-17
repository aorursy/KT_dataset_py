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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data=pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')
data.shape
data.head()
data.hist()

plt.show()
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer

def categorical_text_encoding(df,tr,te,col):

    my_counter = Counter()

    for word in df[col].values:

        my_counter.update(word.split())



    dictionary = dict(my_counter)

    sorted_dict = dict(sorted(dictionary.items(), key=lambda kv: kv[1]))

    

    vectorizer = CountVectorizer(vocabulary=list(sorted_dict.keys()), lowercase=False, binary=True)

    vectorizer.fit(tr[col].values)

    train = vectorizer.transform(tr[col].values)

    test = vectorizer.transform(te[col].values)

    print(vectorizer.get_feature_names())

    return train,test,vectorizer

data.dtypes
data['bin_3']=data['bin_3'].replace('T',1)

data['bin_3']=data['bin_3'].replace('F',0)
data['bin_4']=data['bin_4'].replace('Y',1)

data['bin_4']=data['bin_4'].replace('N',0)
data['nom_0'].value_counts()
data['nom_0']=data['nom_0'].replace('Green',0)

data['nom_0']=data['nom_0'].replace('Blue',1)

data['nom_0']=data['nom_0'].replace('Red',2)
data['nom_1'].value_counts()
data['nom_1']=data['nom_1'].replace('Trapezoid',0)

data['nom_1']=data['nom_1'].replace('Square',1)

data['nom_1']=data['nom_1'].replace('Star',2)

data['nom_1']=data['nom_1'].replace('Circle',3)

data['nom_1']=data['nom_1'].replace('Polygon',4)

data['nom_1']=data['nom_1'].replace('Triangle',5)
data['nom_2'].value_counts()
data['nom_2']=data['nom_2'].replace('Lion',0)

data['nom_2']=data['nom_2'].replace('Cat',1)

data['nom_2']=data['nom_2'].replace('Snake',2)

data['nom_2']=data['nom_2'].replace('Dog',3)

data['nom_2']=data['nom_2'].replace('Axolotl',4)

data['nom_2']=data['nom_2'].replace('Hamster',5)
data['nom_3'].value_counts()
data['nom_3']=data['nom_3'].replace('Russia',0)

data['nom_3']=data['nom_3'].replace('Canada',1)

data['nom_3']=data['nom_3'].replace('China',2)

data['nom_3']=data['nom_3'].replace('Finland',3)

data['nom_3']=data['nom_3'].replace('Costa Rica',4)

data['nom_3']=data['nom_3'].replace('India',5)
data['nom_4'].value_counts()
data['nom_4']=data['nom_4'].replace('Oboe',0)

data['nom_4']=data['nom_4'].replace('Piano',1)

data['nom_4']=data['nom_4'].replace('Bassoon',2)

data['nom_4']=data['nom_4'].replace('Theremin',3)
data['nom_5'].value_counts()
data['nom_6'].value_counts()
data['nom_7'].value_counts()
data['nom_8'].value_counts()
data['nom_9'].value_counts()
data['ord_1'].value_counts()
data['ord_2'].value_counts()
data['ord_3'].value_counts()
data['ord_4'].value_counts()
data['ord_5'].value_counts()