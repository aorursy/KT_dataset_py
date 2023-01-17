import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
products = pd.read_csv('../input/amazon_baby.csv')
products.head()
import string
products['review_clean']=products['review'].str.replace('[{}]'.format(string.punctuation), '')
products.head()
products = products.fillna({'review':''})  # fill in N/A's in the review column
products = products[products['rating'] != 3]
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
products.sample(5)
from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(products,test_size = 0.20)
print('Size of train_data is :', train_data.shape)
print('Size of test_data is :', test_data.shape)
import gc
from sklearn.feature_extraction.text import HashingVectorizer

vectorizer = HashingVectorizer(token_pattern=r'\b\w+\b')

train_matrix = vectorizer.transform(train_data['review_clean'].values.astype('U'))
test_matrix = vectorizer.transform(test_data['review_clean'].values.astype('U'))
gc.collect()
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
sentiment_model = clf.fit(train_matrix,train_data['sentiment'])
y_pred = sentiment_model.predict(test_matrix)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(sentiment_model.score(test_matrix, test_data.sentiment)))
model_coef = pd.DataFrame(sentiment_model.coef_)
model_coef
from scipy.stats import norm
plt.figure(figsize = (15,7))
sns.distplot(model_coef, fit = norm);

pos_coef = np.sum(sentiment_model.coef_>-0)
print('Positive Coefficients are : ',pos_coef)
sample_test_data = test_data[10:13]
sample_test_data
sample_test_data.iloc[0]['review']
sample_test_data.iloc[2]['review']
sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = sentiment_model.decision_function(sample_test_matrix)
print(scores)
for i in scores:
    print(1/(1+np.exp(-i)))
test_scores = sentiment_model.decision_function(test_matrix)
print(test_scores)
positive_reviews = np.argsort(-test_scores)[:20]
print(positive_reviews)
print(test_scores[positive_reviews[0]])
test_data.iloc[positive_reviews]
negative_reviews = np.argsort(test_scores)[:20]
print(negative_reviews)
print(test_scores[negative_reviews[0]])
test_data.iloc[negative_reviews]
predicted_y = sentiment_model.predict(test_matrix)
correct_num = np.sum(predicted_y == test_data['sentiment'])
total_num = len(test_data['sentiment'])
print("correct_num: {}, total_num: {}".format(correct_num, total_num))
accuracy = correct_num * 1./ total_num
print(accuracy)
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_word_subset = CountVectorizer(vocabulary=significant_words) # limit to 20 words
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'].astype('U'))
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'].astype('U'))
simple_model = LogisticRegression()
simple_model.fit(train_matrix_word_subset, train_data['sentiment'])
simple_model_coef_table = pd.DataFrame({'word':significant_words,
                                         'coefficient':simple_model.coef_.flatten()})
#simple_model_coef_table
simple_model_coef_table.sort_values(['coefficient'], ascending=False)
len(simple_model_coef_table[simple_model_coef_table['coefficient']>0])
model_coef_table = pd.DataFrame({'word':significant_words,
                                         'coefficient':simple_model.coef_.flatten()})
#simple_model_coef_table
simple_model_coef_table.sort_values(['coefficient'], ascending=False)
vectorizer_word_subset.get_feature_names()
train_predicted_y = sentiment_model.predict(train_matrix)
correct_num = np.sum(train_predicted_y == train_data['sentiment'])
total_num = len(train_data['sentiment'])
print("correct_num: {}, total_num: {}".format(correct_num, total_num))
train_accuracy = correct_num * 1./ total_num
print("sentiment_model training accuracy: {}".format(train_accuracy))

train_predicted_y = simple_model.predict(train_matrix_word_subset)
correct_num = np.sum(train_predicted_y == train_data['sentiment'])
total_num = len(train_data['sentiment'])
print("correct_num: {}, total_num: {}".format(correct_num, total_num))
train_accuracy = correct_num * 1./ total_num
print("simple_model training accuracy: {}".format(train_accuracy))
test_predicted_y = sentiment_model.predict(test_matrix)
correct_num = np.sum(test_predicted_y == test_data['sentiment'])
total_num = len(test_data['sentiment'])
print("correct_num: {}, total_num: {}".format(correct_num, total_num))
test_accuracy = correct_num * 1./ total_num
print("sentiment_model training accuracy: {}".format(test_accuracy))

test_predicted_y = simple_model.predict(test_matrix_word_subset)
correct_num = np.sum(test_predicted_y == test_data['sentiment'])
total_num = len(test_data['sentiment'])
print("correct_num: {}, total_num: {}".format(correct_num, total_num))
test_accuracy = correct_num * 1./ total_num
print("simple_model training accuracy: {}".format(test_accuracy))
positive_label = len(test_data[test_data['sentiment']>0])
negative_label = len(test_data[test_data['sentiment']<0])
print("positive_label is {}, negative_label is {}".format(positive_label, negative_label))
baseline_accuracy = positive_label*1./(positive_label+negative_label)
print("baseline_accuracy is {}".format(baseline_accuracy))



