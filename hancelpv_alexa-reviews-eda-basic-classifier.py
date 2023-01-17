# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")
pd.set_option('chained_assignment',None)

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/amazon_alexa.tsv", sep='\t')
print("Shape of Dataframe is {}".format(data.shape))
data.isnull().sum()
data.head()
test = data.copy(deep=True)
test.loc[test['feedback'] == 1, 'feedback'] = 'Positive'
test.loc[test['feedback'] == 0, 'feedback'] = 'Negative'
plt.figure(figsize=(12, 7))
sns.scatterplot(x="rating", y="rating", hue="feedback",data=test)
plt.title("Relation between Rating and Overall Feedback");
fig, axs = plt.subplots(1, 2, figsize=(24, 10))

data.feedback.value_counts().plot.barh(ax=axs[0])
axs[0].set_title(("Class Distribution - Feedback {1 (positive) & 0 (negative)}"));

data.rating.value_counts().plot.barh(ax=axs[1])
axs[1].set_title("Class Distribution - Ratings");
data.variation.value_counts().plot.barh(figsize=(12, 7))
plt.title("Class Distribution - Variation");
data.groupby('variation').mean()[['rating']].plot.barh(figsize=(12, 7))
plt.title("Variation wise Mean Ratings");
data['review_length'] = data.verified_reviews.str.len()
pd.DataFrame(data.review_length.describe()).T
data['review_length'].plot.hist(bins=200, figsize=(16, 7))
plt.title("Histogram of Review Lengths");
data.groupby('rating').mean()[['review_length']].plot.barh(figsize=(12, 7))
plt.title("Mean Length of Reviews - Grouped by Ratings");
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english')
cv.fit_transform(data.verified_reviews);
vector = cv.fit_transform(data.verified_reviews)
sum_words = vector.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
freq_df = pd.DataFrame(words_freq, columns=['word', 'freq'])
freq_df.head(15).plot(x='word', y='freq', kind='barh', figsize=(20, 12))
plt.title("Most Frequently Occuring Words - Top 15");
from wordcloud import WordCloud
wordcloud = WordCloud(background_color='white',width=800, height=500).generate_from_frequencies(dict(words_freq))
plt.figure(figsize=(10,8))
plt.imshow(wordcloud)
plt.title("WordCloud - Vocabulary from Reviews", fontsize=22);
features = pd.DataFrame(vector.toarray(), columns=list(sorted(cv.vocabulary_)))
features = features.join(data[['review_length', 'variation']], rsuffix='_base')
features = pd.get_dummies(features)
target = data[['feedback']].astype(int)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
model = RandomForestClassifier()
params = {
    'bootstrap': [True],
    'max_depth': [80, 100],
    'min_samples_split': [8, 12],
    'n_estimators': [100, 300]
}
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
cv_object = StratifiedKFold(n_splits=5)

grid = GridSearchCV(estimator=model, param_grid=params, cv=cv_object, verbose=0, return_train_score=True)
grid.fit(x_train, y_train.values.ravel())
pd.crosstab(y_train['feedback'], grid.predict(x_train), rownames=['True'], colnames=['Predicted'], margins=True)
print("Best Parameter Combination : {}".format(grid.best_params_))
print("Mean Cross Validation Accuracy - Train Set : {}".format(grid.cv_results_['mean_train_score'].mean()*100))
print("="*70)
print("Mean Cross Validation Accuracy - Validation Set : {}".format(grid.cv_results_['mean_test_score'].mean()*100))
feature_imp_df = pd.DataFrame([grid.best_estimator_.feature_importances_], columns=list(x_train.columns)).T
feature_imp_df.columns = ['imp']
feature_imp_df.sort_values('imp', ascending=False, inplace=True)
feature_imp_df.head(15).plot.barh(figsize=(16, 9))
plt.title("15 Most Important Features");
y_test['pred'] = grid.predict(x_test)
from sklearn.metrics import accuracy_score
print("Accuracy Score for Test Set : {}".format(accuracy_score(y_test.feedback, y_test.pred)*100))
pd.crosstab(y_test['feedback'], grid.predict(x_test), rownames=['True'], colnames=['Predicted'], margins=True)