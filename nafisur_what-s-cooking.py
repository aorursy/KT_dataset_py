# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_json('../input/train.json')
test_df = pd.read_json('../input/test.json')

train_df['seperated_ingredients'] = train_df['ingredients'].apply(','.join)
test_df['seperated_ingredients'] = test_df['ingredients'].apply(','.join)
print(train_df.shape)
print(test_df.shape)
print('Maximum Number of Ingredients in a Dish: ',train_df['ingredients'].str.len().max())
print('Minimum Number of Ingredients in a Dish: ',train_df['ingredients'].str.len().min())
train_df.head()
test_df.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(ngram_range=(1,2),binary=True)
tfidf.fit(train_df.seperated_ingredients.values)
X_train_vectorized = tfidf.transform(train_df['seperated_ingredients'].values)
X_test_vectorized=tfidf.transform(test_df['seperated_ingredients'].values)
print(X_train_vectorized.shape)
print(X_test_vectorized.shape)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder = LabelEncoder()
y_transformed = encoder.fit_transform(train_df.cuisine)
#y=y_transformed.reshape(-1, 1)
y=y_transformed
# onehot=OneHotEncoder()
# y=onehot.fit_transform(y)
y.shape
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X_train_vectorized,y,test_size=0.2)
print(X_train.shape,y_train.shape)
print(X_val.shape,y_val.shape)
print(X_test_vectorized.shape)
from sklearn.svm import SVC,LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
clf=LinearSVC()
%%time
clf.fit(X_train, y_train)
clf.score(X_val,y_val)


