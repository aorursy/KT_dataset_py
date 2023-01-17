# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# objective is to predict a number of listing enquiries based on features

train = pd.read_json("../input/train.json", "r")

test = pd.read_json("../input/test.json", "r")

sample_sub = pd.read_csv("../input/sample_submission.csv")
sample_sub.head()

# the above is what our submission is supposed to look like
train = train[['price', 'listing_id', 'bathrooms', 'bedrooms', 'interest_level', 'latitude', 'longitude']]

test = test[['price', 'listing_id', 'bathrooms', 'bedrooms', 'latitude', 'longitude']]
train_target = train['interest_level']
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
train.index = train['listing_id']

train = train.drop('interest_level', 1)

model = gnb.fit(train, train_target)
test.index = test['listing_id']
y = model.predict_proba(test)
y_dat = pd.DataFrame(y)
#y_dat.copy(deep = False)

y_dat.loc[:,'listing_id'] = test.index
y_dat.rename(columns = {'0':'medium', '1':'low', '2':'high'}, inplace = True)
y_dat.columns = ['medium', 'low', 'high', 'listing_id']
data = y_dat[['listing_id', 'high', 'medium', 'low']]
data.head()
#medium, low, high

#writer = pd.ExcelWriter('/Users/reshmasekar/Desktop/sub.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.

data.to_csv("sub_rf_4.csv", index = False)

#y_dat.to_excel("/Users/reshmasekar/Desktop")
# improving predictive accuracy
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# objective is to predict a number of listing enquiries based on features

train = pd.read_json("../input/train.json", "r")

test = pd.read_json("../input/test.json", "r")

sample_sub = pd.read_csv("../input/sample_submission.csv")
train.head()
# splitting words from description

description=train['description']

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

words_total=""

for word in description:

    words_total = words_total +word

tokens=tokenizer.tokenize(words_total)

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer(tokens)

dtm=vectorizer.fit_transform(train['description'])

#Need to remove stop words and also use three tokens? 
