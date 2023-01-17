# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
review_data = pd.read_csv('../input/amazon-alexa-reviews/amazon_alexa.tsv', delimiter='\t', encoding='utf-8')
review_data.head()
review_data.isnull().sum()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
num_rating5 = review_data['rating'][review_data['rating']==5].count()
num_rating4 = review_data['rating'][review_data['rating']==4].count()
num_rating3 = review_data['rating'][review_data['rating']==3].count()
num_rating2 = review_data['rating'][review_data['rating']==2].count()
num_rating1 = review_data['rating'][review_data['rating']==1].count()
x = ['5','4','3','2','1']
ratio_rating = [num_rating5 / len(review_data['rating']), num_rating4 / len(review_data['rating']), num_rating3 / len(review_data['rating']), 
                num_rating2 / len(review_data['rating']), num_rating1 / len(review_data['rating'])]
sns.barplot(x, ratio_rating)
plt.title("Consumer's rating")
plt.xlabel("Rating")
plt.ylabel("Ratio of rating")
plt.show()
var_rate = pd.pivot_table(review_data, index = ['variation'])
var_rate.head()
plt.figure(figsize = (30,10))
sns.barplot(x='variation',y='rating', data=review_data)
plt.show()
plt.figure(figsize = (30,10))
sns.violinplot(x='variation',y='rating', data=review_data)
plt.show()
var_rate['feedback'].idxmax(), var_rate['rating'].idxmax()
var_rate['feedback'].idxmin(), var_rate['rating'].idxmin()
# from google.cloud import language
# from google.cloud.language import enums
# from google.cloud.language import types
# path = ''  # FULL path to your service account key
# client = language.LanguageServiceClient.from_service_account_json(path)
# senti_score = list()
# senti_mag = list()
# for i in range(len(review_data['verified_reviews'])):
#     text = review_data['verified_reviews'][i]
#     document = types.Document(
#         content = text,
#         type    = enums.Document.Type.PLAIN_TEXT)
#     # Detects the sentiment of the text
#     sentiment = client.analyze_sentiment(document=document).document_sentiment
#     senti_score.append(sentiment.score)
#     senti_mag.append(sentiment.magnitude)
#     print('{} is completed'.format(i))
# review_data['sentiment_score'] = senti_score
# review_data['sentiment_magnitude'] = senti_mag
# review_data.head()
# review_data.to_csv('Amazon_review.csv')
review = pd.read_csv('../input/amazon-alexa-review-with-sentiment-analysis/Amazon_review.csv')
review.head()
plt.figure(figsize = (30,10))
sns.barplot(x='variation',y='sentiment_score', data=review)
plt.show()
review_norm = pd.pivot_table(review, index=['variation'])
rating_max = review_norm['rating'].max()
review_norm['rating'] = review_norm['rating'] / rating_max * 100
score_max = review_norm['sentiment_score'].max()
review_norm['sentiment_score'] = review_norm['sentiment_score'] / score_max * 100
magnitude_max = review_norm['sentiment_magnitude'].max()
review_norm['sentiment_magnitude'] = review_norm['sentiment_magnitude'] / magnitude_max * 100
feedback_max = review_norm['feedback'].max()
review_norm['feedback'] = review_norm['feedback'] / feedback_max * 100
review_norm.head()
review_norm_sort = review_norm.sort_values(by='rating', ascending=False)
target_col = ['feedback','rating','sentiment_score','sentiment_magnitude']

plt.figure()
sns.heatmap(review_norm_sort[target_col], annot=True, fmt='f', linewidths=.5)
plt.show()
minus_review = review[review['sentiment_score'] < 0]
minus_review.head()
pd.pivot_table(minus_review, index=['variation'], values=['sentiment_score'], aggfunc=[np.mean, len], margins=False)
minus_review.groupby(['variation'])['sentiment_score'].count().sort_values(ascending=False)
plus_review = review[review['sentiment_score'] > 0]
plus_review.head()
pd.pivot_table(plus_review, index=['variation'], values=['sentiment_score'], aggfunc=[np.mean, len], margins=False)
plus_review.groupby(['variation'])['sentiment_score'].count().sort_values(ascending=False)
from scipy.stats import pearsonr
corr = pd.DataFrame()
corr['rating'] = review['rating']
corr['feedback'] = review['feedback']
corr['sentiment_score'] = review['sentiment_score']
corr['sentiment_magnitude'] = review['sentiment_magnitude']
corr.head()
corr.corr()
plt.figure() 
sns.heatmap(corr.corr(), cmap='BuGn')
cols = corr.columns
mat = corr.values
arr = np.zeros((len(cols),len(cols)), dtype=object)
for xi, x in enumerate(mat.T):
    for yi, y in enumerate(mat.T[xi:]):
        arr[xi, yi+xi] = pearsonr(x,y)[1]
        arr[yi+xi, xi] = arr[xi, yi+xi]
p_value = pd.DataFrame(arr, index=cols, columns=cols)
p_value
feature = review
del_col = ['date','variation','verified_reviews']
feature = feature.drop(del_col, axis=1)
feature.head()
from sklearn.model_selection import train_test_split
y = feature['rating']
x_feature = ['feedback','sentiment_score','sentiment_magnitude']
x = feature[x_feature]
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 42)
from sklearn.tree import DecisionTreeClassifier
DT_clf = DecisionTreeClassifier()
DT_clf.fit(X_train, y_train)
DT_ypred = DT_clf.predict(X_test)
DT_accuracy = DT_clf.score(X_test, y_test)
DT_accuracy
from sklearn.ensemble import RandomForestClassifier
RF_clf = RandomForestClassifier(n_estimators = 100)
RF_clf.fit(X_train, y_train)
RF_ypred = RF_clf.predict(X_test)
RF_accuracy = RF_clf.score(X_test, y_test)
RF_accuracy
