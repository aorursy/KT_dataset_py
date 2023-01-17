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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score
review = pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')

review.head()
print('number of total entries are equal to ', review.shape[0])
#checking null values

review.isnull().sum()
# dropping the null values

review.dropna(inplace=True)
# checking whether the null values are dropped or not

review.isnull().sum()


#dropping the neutral values

review = review[review['Score'] != 3]
review['Positivity'] = np.where(review['Score'] > 3, 1, 0)

review.head()
sns.countplot(review['Positivity'])

plt.show()
review.info(memory_usage='deep')
# split the data into training and testing data



# text will be used for training

# positivity is what we are predicting

X_train, X_test, y_train, y_test = train_test_split(review['Text'], review['Positivity'], random_state = 0)
print('X_train first entry: \n\n', X_train[0])

print('\n\nX_train shape: ', X_train.shape)
vect = CountVectorizer().fit(X_train)

vect
# checking the features

feat = vect.get_feature_names()

feat
cloud = WordCloud(width=1440, height=1080).generate(" ".join(feat))
# larger the size of the word, more the times it appears

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')

plt.show()
# checking the length of features

len(vect.get_feature_names())
X_train_vectorized = vect.transform(X_train)
model = LogisticRegression()

model.fit(X_train_vectorized, y_train)
# accuracy

predictions = model.predict(vect.transform(X_test))
accuracy_score(y_test, predictions)
# area under the curve

roc_auc = roc_auc_score(y_test, predictions)

print('AUC: ', roc_auc)

fpr, tpr, thresholds = roc_curve(y_test, predictions)
plt.title('ROC for logistic regression on bag of words', fontsize=20)

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate', fontsize = 20)

plt.xlabel('False Positive Rate', fontsize = 20)

plt.show()
# coefficient determines the weight of a word (positivity or negativity)

# checking the top 10 positive and negative words



# getting the feature names

feature_names = np.array(vect.get_feature_names())



# argsort: Integer indicies that would sort the index if used as an indexer

sorted_coef_index = model.coef_[0].argsort()



print('Smallest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:10]]))

print('Largest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:-11:-1]]))
vect = TfidfVectorizer(min_df = 5).fit(X_train)

len(vect.get_feature_names())
# check the top 10 features for positive and negative

# reviews again, the AUC has improved

feature_names = np.array(vect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()



# print('Smallest Coef: \n{}\n'.format(feature_names[sorted_coef_index][:10]))

# print('Largest Coef: \n{}\n'.format(feature_names[sorted_coef_index][:-11:-1]))
feat = vect.get_feature_names()
cloud = WordCloud(width=1440, height=1080).generate(" ".join(feat))
# larger the size of the word, more the times it appears

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')

plt.show()
X_train_vectorized = vect.transform(X_train)
model = LogisticRegression()

model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test))
accuracy_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)

print('AUC: ', roc_auc)

fpr, tpr, thresholds = roc_curve(y_test, predictions)
plt.title('ROC for logistic regression on TF-IDF', fontsize=20)

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate', fontsize = 20)

plt.xlabel('False Positive Rate', fontsize = 20)

plt.show()
# even though we reduced the number of features considerably

# the AUC did not change much



# let us test our model

new_review = ['The food was delicious', 'The food was not good']



print(model.predict(vect.transform(new_review)))
#BOW AS OUR REVIEWS CAN CONTAIN 'NOT WORD ' IS ALSO INCLUDED

vect = CountVectorizer(min_df = 5, ngram_range = (1,2)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

len(vect.get_feature_names())
feat = vect.get_feature_names()
cloud = WordCloud(width=1440, height=1080).generate(" ".join(feat))
# larger the size of the word, more the times it appears

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')

plt.show()
# the number of features has increased again

# checking for the AUC

model = LogisticRegression()

model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test))
accuracy_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)

print('AUC: ', roc_auc)

fpr, tpr, thresholds = roc_curve(y_test, predictions)
plt.title('ROC for logistic regression on Bigrams', fontsize=20)

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate', fontsize = 20)

plt.xlabel('False Positive Rate', fontsize = 20)

plt.show()
# check the top 10 features for positive and negative

# reviews again, the AUC has improved

feature_names = np.array(vect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()



# print('Smallest Coef: \n{}\n'.format(feature_names[sorted_coef_index][:10]))

# print('Largest Coef: \n{}\n'.format(feature_names[sorted_coef_index][:-11:-1]))
new_review = ['this is not good,this is wonderful']

print(model.predict(vect.transform(new_review)))