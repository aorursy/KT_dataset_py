%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics



from tqdm import tqdm

import os

from prettytable import PrettyTable
table = PrettyTable()

table.field_names = ["Vectoriser","Optimal Alpha","summary feat incl/not incl","AUC score"]
final = pd.read_csv('../input/preprocessed_reviews.csv')

final.head()
final = final.sort_values(by='Time')

final['Score'].value_counts()
x,x_test,y,y_test = train_test_split(final['preprocessed_reviews'],final['Score'],test_size = 0.3, shuffle = False)

x_train,x_validate,y_train,y_validate = train_test_split(x,y,test_size = 0.3, shuffle = False)
#bi-gram, tri-gram and n-gram



count_vect = CountVectorizer(ngram_range=(1,2)) #in scikit-learn

final_bigram_counts_x_train = count_vect.fit_transform(x_train.astype('U'))

final_bigram_counts_x_validate = count_vect.transform(x_validate.astype('U'))

final_bigram_counts_x_test = count_vect.transform(x_test.astype('U'))



print("the type of count vectorizer - x test",type(final_bigram_counts_x_test))

print("the shape of out text BOW bigram vectorizer - x test",final_bigram_counts_x_test.get_shape())

print("="*50)

print("the type of count vectorizer - x train",type(final_bigram_counts_x_train))

print("the shape of out text BOW bigram vectorizer - x train",final_bigram_counts_x_train.get_shape())

print("="*50)

print("the type of count vectorizer - x validate",type(final_bigram_counts_x_validate))

print("the shape of out text BOW bigram vectorizer - x validate",final_bigram_counts_x_validate.get_shape())
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2)) #in scikit-learn

final_tf_idf_x_train = tf_idf_vect.fit_transform(x_train.astype('U'))

final_tf_idf_x_validate = tf_idf_vect.transform(x_validate.astype('U'))

final_tf_idf_x_test = tf_idf_vect.transform(x_test.astype('U'))



print("the type of count vectorizer - x test",type(final_tf_idf_x_test))

print("the shape of out text TFIDF vectorizer - x test",final_tf_idf_x_test.get_shape())

print("="*50)

print("the type of count vectorizer - x train",type(final_tf_idf_x_train))

print("the shape of out text TFIDF vectorizer - x train",final_tf_idf_x_train.get_shape())

print("="*50)

print("the type of count vectorizer - x validate",type(final_tf_idf_x_validate))

print("the shape of out text TFIDF vectorizer - x validate",final_tf_idf_x_validate.get_shape())
auc_scores_validate = []

auc_scores_train = []

alphas = [0.00001, 0.00005, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 1, 1.1, 1.5, 10, 100, 250, 500, 1000] 



for alp in tqdm(alphas):

    MultiNB = MultinomialNB(alpha = alp)

    #fitting Multinomial naive bayes classifier with train data

    MultiNB.fit(final_bigram_counts_x_train,y_train)

    #predicting the log probability scores

    y_prob_pred = MultiNB.predict_log_proba(final_bigram_counts_x_train)

    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_prob_pred[:,1])

    auc_scores_train.append(metrics.auc(fpr,tpr))

    

    y_prob_pred = MultiNB.predict_log_proba(final_bigram_counts_x_validate)

    fpr, tpr, thresholds = metrics.roc_curve(y_validate, y_prob_pred[:,1])

    auc_scores_validate.append(metrics.auc(fpr,tpr))
plt.plot(np.log(alphas),auc_scores_train,label='Train')

plt.plot(np.log(alphas),auc_scores_validate,label='Validate')

plt.xlabel('log of Alpha values')

plt.ylabel('AUC scores')

plt.title('Bigram Bow - Multinomial Naive Bayes')

plt.legend()

plt.show()
optimal_alpha = alphas[auc_scores_validate.index(max(auc_scores_validate))]

MultiNB = MultinomialNB(alpha = optimal_alpha)

MultiNB.fit(final_bigram_counts_x_train,y_train)

y_prob_pred = MultiNB.predict_log_proba(final_bigram_counts_x_test)

fpr_test, tpr_test, thresholds = metrics.roc_curve(y_test, y_prob_pred[:,1])

y_prob_pred = MultiNB.predict_log_proba(final_bigram_counts_x_train)

fpr_train, tpr_train, thresholds = metrics.roc_curve(y_train, y_prob_pred[:,1])

plt.plot(fpr_test,tpr_test,label='Test AUC is %0.2f' %(metrics.auc(fpr_test,tpr_test)))

table.add_row(["Bigram BOW",optimal_alpha,"summary not included",metrics.auc(fpr_test,tpr_test)])

plt.plot(fpr_train,tpr_train,label='Train AUC is %0.2f' %(metrics.auc(fpr_train,tpr_train)))

plt.xlabel('true positive rate')

plt.ylabel('false positive rate')

plt.title('Bigram BOW for optimal alpha = %f' %optimal_alpha)

plt.legend()

plt.show()
import seaborn as sns

from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_test,MultiNB.predict(final_bigram_counts_x_test)).ravel()

ax = sns.heatmap([[fn,tn],[fp,tp]],yticklabels=["Actual 0","Actual 1"],\

                 xticklabels=["Predicted 0","Predicted 1"],annot = True,fmt='d')

ax.set_title('Bigram BOW for optimal alpha = %f' %optimal_alpha)
features = pd.DataFrame(data = MultiNB.feature_log_prob_.T,index=count_vect.get_feature_names(), columns=["0","1"])
positive_features = features.sort_values(by='1', ascending=False)[0:50]

print(positive_features)
negative_features = features.sort_values(by='0', ascending=False)[0:50]

print(negative_features)
auc_scores_validate = []

auc_scores_train = []

alphas = [0.00001, 0.00005, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 250, 500, 1000]



for alp in tqdm(alphas):

    MultiNB = MultinomialNB(alpha = alp)

    #fitting multinomial naive bayes classifier with train data

    MultiNB.fit(final_tf_idf_x_train,y_train)

    #predicting the log probability scores

    y_prob_pred = MultiNB.predict_log_proba(final_tf_idf_x_train)

    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_prob_pred[:,1])

    auc_scores_train.append(metrics.auc(fpr,tpr))

    

    y_prob_pred = MultiNB.predict_log_proba(final_tf_idf_x_validate)

    fpr, tpr, thresholds = metrics.roc_curve(y_validate, y_prob_pred[:,1])

    auc_scores_validate.append(metrics.auc(fpr,tpr))
plt.plot(np.log(alphas),auc_scores_train,label='Train')

plt.plot(np.log(alphas),auc_scores_validate,label='Validate')

plt.xlabel('log of Alpha values')

plt.ylabel('AUC scores')

plt.title('Tfidf - Multinomial Naive Bayes')

plt.legend()

plt.show()
optimal_alpha = alphas[auc_scores_validate.index(max(auc_scores_validate))]

MultiNB = MultinomialNB(alpha = optimal_alpha)

MultiNB.fit(final_tf_idf_x_train,y_train)

y_prob_pred = MultiNB.predict_log_proba(final_tf_idf_x_test)

fpr_test, tpr_test, thresholds = metrics.roc_curve(y_test, y_prob_pred[:,1])

y_prob_pred = MultiNB.predict_log_proba(final_tf_idf_x_train)

fpr_train, tpr_train, thresholds = metrics.roc_curve(y_train, y_prob_pred[:,1])

plt.plot(fpr_test,tpr_test,label='Test AUC is %0.2f' %(metrics.auc(fpr_test,tpr_test)))

table.add_row(["tfidf",optimal_alpha,"summary not included",metrics.auc(fpr_test,tpr_test)])

plt.plot(fpr_train,tpr_train,label='Train AUC is %0.2f' %(metrics.auc(fpr_train,tpr_train)))

plt.xlabel('true positive rate')

plt.ylabel('false positive rate')

plt.title('Tfidf for optimal alpha = %f' %optimal_alpha)

plt.legend()

plt.show()
tn, fp, fn, tp = confusion_matrix(y_test,MultiNB.predict(final_tf_idf_x_test)).ravel()

ax = sns.heatmap([[fn,tn],[fp,tp]],yticklabels=["Actual 0","Actual 1"],\

                 xticklabels=["Predicted 0","Predicted 1"],annot = True,fmt='d')

ax.set_title('Tfidf for optimal alpha = %f' %optimal_alpha)
features = pd.DataFrame(data = MultiNB.feature_log_prob_.T,index=tf_idf_vect.get_feature_names(), columns=["0","1"])
positive_features = features.sort_values(by='1', ascending=False)[0:50]

print(positive_features)
negative_features = features.sort_values(by='0', ascending=False)[0:50]

print(negative_features)
x,x_test,y,y_test = train_test_split(final['preprocessed_reviews'].astype(str)+" "+ final['preprocessed_reviews_summary'],final['Score'],test_size = 0.3, shuffle = False)

x_train,x_validate,y_train,y_validate = train_test_split(x,y,test_size = 0.3, shuffle = False)
#bi-gram, tri-gram and n-gram



count_vect = CountVectorizer(ngram_range=(1,2)) #in scikit-learn

final_bigram_counts_x_train = count_vect.fit_transform(x_train.astype('U'))

final_bigram_counts_x_validate = count_vect.transform(x_validate.astype('U'))

final_bigram_counts_x_test = count_vect.transform(x_test.astype('U'))



print("the type of count vectorizer - x test",type(final_bigram_counts_x_test))

print("the shape of out text BOW bigram vectorizer - x test",final_bigram_counts_x_test.get_shape())

print("="*50)

print("the type of count vectorizer - x train",type(final_bigram_counts_x_train))

print("the shape of out text BOW bigram vectorizer - x train",final_bigram_counts_x_train.get_shape())

print("="*50)

print("the type of count vectorizer - x validate",type(final_bigram_counts_x_validate))

print("the shape of out text BOW bigram vectorizer - x validate",final_bigram_counts_x_validate.get_shape())





auc_scores_validate = []

auc_scores_train = []

alphas = [0.00001, 0.00005, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 1, 1.1, 1.5, 10, 100, 250, 500, 1000] 

for alp in tqdm(alphas):

    MultiNB = MultinomialNB(alpha = alp)

    #fitting multinomial naive bayes classifier with train data

    MultiNB.fit(final_bigram_counts_x_train,y_train)

    #predicting the log probability scores

    y_prob_pred = MultiNB.predict_log_proba(final_bigram_counts_x_train)

    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_prob_pred[:,1])

    auc_scores_train.append(metrics.auc(fpr,tpr))

    

    y_prob_pred = MultiNB.predict_log_proba(final_bigram_counts_x_validate)

    fpr, tpr, thresholds = metrics.roc_curve(y_validate, y_prob_pred[:,1])

    auc_scores_validate.append(metrics.auc(fpr,tpr))

    

plt.plot(np.log(alphas),auc_scores_train,label='Train')

plt.plot(np.log(alphas),auc_scores_validate,label='Validate')

plt.xlabel('log of Alpha values')

plt.ylabel('AUC scores')

plt.title('Bigram BOW - multinomial naive bayes')

plt.legend()

plt.show()



optimal_alpha = alphas[auc_scores_validate.index(max(auc_scores_validate))]

MultiNB = MultinomialNB(alpha = optimal_alpha)

MultiNB.fit(final_bigram_counts_x_train,y_train)

y_prob_pred = MultiNB.predict_log_proba(final_bigram_counts_x_test)

fpr_test, tpr_test, thresholds = metrics.roc_curve(y_test, y_prob_pred[:,1])

y_prob_pred = MultiNB.predict_log_proba(final_bigram_counts_x_train)

fpr_train, tpr_train, thresholds = metrics.roc_curve(y_train, y_prob_pred[:,1])

plt.plot(fpr_test,tpr_test,label='Test AUC is %0.2f' %(metrics.auc(fpr_test,tpr_test)))

table.add_row(["Bigram BOW",optimal_alpha,"summary included",metrics.auc(fpr_test,tpr_test)])

plt.plot(fpr_train,tpr_train,label='Train AUC is %0.2f' %(metrics.auc(fpr_train,tpr_train)))

plt.xlabel('true positive rate')

plt.ylabel('false positive rate')

plt.title('Bigram BOW for optimal alpha = %f' %optimal_alpha)

plt.legend()

plt.show()



tn, fp, fn, tp = confusion_matrix(y_test,MultiNB.predict(final_bigram_counts_x_test)).ravel()

ax = sns.heatmap([[fn,tn],[fp,tp]],yticklabels=["Actual 0","Actual 1"],\

                 xticklabels=["Predicted 0","Predicted 1"],annot = True,fmt='d')

ax.set_title('Bigram BOW for optimal alpha = %f' %optimal_alpha)
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2)) #in scikit-learn

final_tf_idf_x_train = tf_idf_vect.fit_transform(x_train.astype('U'))

final_tf_idf_x_validate = tf_idf_vect.transform(x_validate.astype('U'))

final_tf_idf_x_test = tf_idf_vect.transform(x_test.astype('U'))

print("the type of count vectorizer - x test",type(final_tf_idf_x_test))

print("the shape of out text TFIDF vectorizer - x test",final_tf_idf_x_test.get_shape())

print("="*50)

print("the type of count vectorizer - x train",type(final_tf_idf_x_train))

print("the shape of out text TFIDF vectorizer - x train",final_tf_idf_x_train.get_shape())

print("="*50)

print("the type of count vectorizer - x validate",type(final_tf_idf_x_validate))

print("the shape of out text TFIDF vectorizer - x validate",final_tf_idf_x_validate.get_shape())



auc_scores_validate = []

auc_scores_train = []

alphas = [0.00001, 0.00005, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 250, 500, 1000]

for alp in tqdm(alphas):

    MultiNB = MultinomialNB(alpha = alp)

    #fitting multinomial naive bayes classifier with train data

    MultiNB.fit(final_tf_idf_x_train,y_train)

    #predicting the log probability scores

    y_prob_pred = MultiNB.predict_log_proba(final_tf_idf_x_train)

    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_prob_pred[:,1])

    auc_scores_train.append(metrics.auc(fpr,tpr))

    

    y_prob_pred = MultiNB.predict_log_proba(final_tf_idf_x_validate)

    fpr, tpr, thresholds = metrics.roc_curve(y_validate, y_prob_pred[:,1])

    auc_scores_validate.append(metrics.auc(fpr,tpr))

    

plt.plot(np.log(alphas),auc_scores_train,label='Train')

plt.plot(np.log(alphas),auc_scores_validate,label='Validate')

plt.xlabel('log of Alpha values')

plt.ylabel('AUC scores')

plt.title('Tfidf - Multinomial Naive Bayes')

plt.legend()

plt.show()





optimal_alpha = alphas[auc_scores_validate.index(max(auc_scores_validate))]

MultiNB = MultinomialNB(alpha = optimal_alpha)

MultiNB.fit(final_tf_idf_x_train,y_train)

y_prob_pred = MultiNB.predict_log_proba(final_tf_idf_x_test)

fpr_test, tpr_test, thresholds = metrics.roc_curve(y_test, y_prob_pred[:,1])

y_prob_pred = MultiNB.predict_log_proba(final_tf_idf_x_train)

fpr_train, tpr_train, thresholds = metrics.roc_curve(y_train, y_prob_pred[:,1])

plt.plot(fpr_test,tpr_test,label='Test AUC is %0.2f' %(metrics.auc(fpr_test,tpr_test)))

table.add_row(["tfidf",optimal_alpha,"summary included",metrics.auc(fpr_test,tpr_test)])

plt.plot(fpr_train,tpr_train,label='Train AUC is %0.2f' %(metrics.auc(fpr_train,tpr_train)))

plt.xlabel('true positive rate')

plt.ylabel('false positive rate')

plt.title('Tfidf for optimal alpha = %f' %optimal_alpha)

plt.legend()

plt.show()





tn, fp, fn, tp = confusion_matrix(y_test,MultiNB.predict(final_tf_idf_x_test)).ravel()

ax = sns.heatmap([[fn,tn],[fp,tp]],yticklabels=["Actual 0","Actual 1"],\

                 xticklabels=["Predicted 0","Predicted 1"],annot = True,fmt='d')

ax.set_title('Tfidf for optimal alpha = %f' %optimal_alpha)
print(table)