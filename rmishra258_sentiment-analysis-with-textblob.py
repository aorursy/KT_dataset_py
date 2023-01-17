# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split #to split the training and testing data
from textblob.classifiers import NaiveBayesClassifier #classifier used to create model


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#load the dataset
data = pd.read_csv('../input/amazon_alexa.tsv', delimiter='\t')
data.head()

data = data[['verified_reviews', 'feedback']]

#convert 1 and 0 to pos and neg 
data['feedback'] = np.where(data['feedback'].isin([1]), 'pos', 'neg')

#create x and y and split them to vlaidate the model
x = data['verified_reviews']
y = data['feedback']


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.3)

#create tuples
train = [x for x in zip(x_train,y_train)]
test = [x for x in zip(x_test, y_test)]

#train the classifier
clf = NaiveBayesClassifier(train)

#test the accuracy
print(clf.accuracy(test))

#update new data to the model
clf.update([('amazon is very nice', 'pos')])
