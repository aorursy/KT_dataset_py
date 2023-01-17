# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
con = sqlite3.connect('/kaggle/input/lebowski.db')

df = pd.read_sql('select line_number,speaker, dialog from dialog',con)

df.head()
my_model = Pipeline([('cv',CountVectorizer()),('nb',MultinomialNB())])

le = LabelEncoder()

y = le.fit_transform(df['speaker'])

my_model.fit(df['dialog'],y)

print('score: {}'.format(my_model.score(df["dialog"],y)))

le.inverse_transform(

    my_model.predict(

        [

            'obviously youre not a golfer',

            'you are out of your element',

            'my marmot carries an uzi and likes to eat lingenberry pancakes. I put my whites in my briefcase after I put my undies through the ringer and beat it out of them',

            'vee vill cut off your lingenberry pancakes',

            'hes a good man and thorough batting an eye does the very word',

            'i can get you a toe mark it zero',

            'dick rod johnson'

        ]

    )

)