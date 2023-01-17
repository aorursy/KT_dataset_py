# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import chi2, SelectKBest

from sklearn.svm import LinearSVC



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_original = pd.read_csv('/kaggle/input/hierarchical-text-classification/train_40k.csv')
# the only columns I care about are these two, as I'll be using the 'Text' column to predict the feature 'Cat1'



columns = ['Text', 'Cat1']



df = shuffle(df_original[columns])
df.Cat1.value_counts()
# data needs cleaning

df
#remove special characters from df using regular expressions



import re

p = re.compile(r'[^\w\s]+')



df['Text'] = [p.sub('', x) for x in df['Text'].tolist()]

# make all characters .lower()



df.apply(lambda x: x.astype(str).str.lower())
# train test split

x,y = df.Text, df.Cat1

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

# build a pipeline 



pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,2), stop_words='english', sublinear_tf=True)),

                     ('chi', SelectKBest(chi2, k=10000)),

                     ('clf', LinearSVC(C=1.0, penalty='l1',max_iter=3000, dual=False))

                    ])

# fit pipline to our training data



model = pipeline.fit(train_x, train_y)
# check accuracy



print('accuracy score: '+ str(model.score(test_x, test_y)))
print(model.predict(['bone lasted forever, will buy again']))
print(model.predict(['nice lipstick']))