# Import the necessary libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
data = pd.read_csv("/kaggle/input/newsgroup_sports_0514.csv",encoding='utf8')

data_df = data[['Clean Article', 'Target Label', 'Target Name','Article']] 

X = np.array(data_df['Clean Article'].fillna(' ')) 

y = np.array(data_df['Target Label'])
from sklearn.model_selection import train_test_split 



# Your target variable is the 'Target Label'

# Your indpendent varaibles will be derived from the text in the "Clean Article" column

# you can "Clean Article" to X using something like this data_df['Clean Article']



train_corpus, test_corpus, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.feature_extraction.text import CountVectorizer



### build BOW features on train_corpus using the CountVectortizer

cv = CountVectorizer(binary=False, min_df=0.0, max_df=1.0)

cv.fit(train_corpus)



# transform train articles into features

cv_train_features = cv.transform(train_corpus)

# transform test articles into features

cv_test_features = cv.transform(test_corpus)



print('BOW model:> Train features shape:', cv_train_features.shape, 'Test features shape:', cv_test_features.shape)
# Naïve Bayes Classifier

from sklearn.naive_bayes import MultinomialNB



# fit the model with the y_train as the target and the BOW featurs on the train_corpus as input

mnb = MultinomialNB(alpha=1)

mnb.fit(cv_train_features, y_train)
# Predict the testing output using the BOW features on the test_corpus

y_pred = mnb.predict(cv_test_features)
from sklearn.metrics import confusion_matrix

# Evaluate the model using the confusion matrix



labels = np.unique(y_test)

cm = confusion_matrix(y_test, y_pred, labels=labels)



print ('Confusion Matrix:')



pd.DataFrame(cm, index=labels, columns=labels)
# calculate the accuracy, precison, recall and F1 - score

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score



# accuracy: (tp + tn) / (p + n)

accuracy = accuracy_score(y_pred, y_test)

print('Accuracy: %f' % accuracy)



# precision tp / (tp + fp)

precision = precision_score(y_pred, y_test)

print('Precision: %f' % precision)



# recall: tp / (tp + fn)

recall = recall_score(y_pred, y_test)

print('Recall: %f' % recall)



# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_pred, y_test)

print('F1 score: %f' % f1)
# Predict probability (using predict_proba method) of 1 - i.e. probably that the news is classified as baseball news

y_score = mnb.predict_proba(cv_test_features)[:,1]



# Compute ROC curve and AUC for the predicted class "1"



from sklearn.metrics import roc_curve

from sklearn.metrics import auc



fpr = dict()

tpr = dict()

roc_auc = dict()

fpr, tpr, threshold = roc_curve(y_test, y_score)



# Compute Area Under the Curve (AUC) using the trapezoidal rule

mnb_roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr)

print("AUC score = {}".format(mnb_roc_auc))
# verify



fpr, tpr, threshold = roc_curve(y_test, y_score)

lr_roc_auc = auc(fpr, tpr)



fig = plt.figure(figsize=(10,6))

ax = fig.add_subplot(111)

ax.set_title('Receiver Operating Characteristic')

ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % lr_roc_auc)

ax.legend(loc = 'lower right')

ax.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

ax.set_ylabel('True Positive Rate')

ax.set_xlabel('False Positive Rate')
# Support Vector Machines

from sklearn.svm import SVC



svm = SVC(kernel='poly',probability=True)



# svm = LinearSVC(penalty='l2', C=1, random_state=42)

# svm = SVC(probability=True)

svm.fit(cv_train_features, y_train)



#Predict will give either 0 or 1 as output

y_pred = svm.predict(cv_test_features)
# confusion matrix

labels = np.unique(y_test)

cm = confusion_matrix(y_test, y_pred, labels=labels)



print ('Confusion Matrix:')

pd.DataFrame(cm, index=labels, columns=labels)
# accuracy: (tp + tn) / (p + n)

accuracy = accuracy_score(y_pred, y_test)

print('Accuracy: %f' % accuracy)



# precision tp / (tp + fp)

precision = precision_score(y_pred, y_test)

print('Precision: %f' % precision)



# recall: tp / (tp + fn)

recall = recall_score(y_pred, y_test)

print('Recall: %f' % recall)



# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_pred, y_test)

print('F1 score: %f' % f1)
y_score = svm.predict_proba(cv_test_features)[:,1]



# Compute ROC curve and AUC for the predicted class "1"



fpr = dict()

tpr = dict()

roc_auc = dict()

fpr, tpr, threshold = roc_curve(y_test, y_score)



# Compute Area Under the Curve (AUC) using the trapezoidal rule

svm_roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr)

print("AUC score = {}".format(svm_roc_auc))
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()

lr_model.fit(cv_train_features, y_train)
#Predict class of 0 or 1 as output

y_pred = lr_model.predict(cv_test_features)
# confusion matrix

cm = confusion_matrix(y_test, y_pred, labels=labels)

print ('Confusion Matrix:')

pd.DataFrame(cm, index=labels, columns=labels)
# accuracy: (tp + tn) / (p + n)

accuracy = accuracy_score(y_pred, y_test)

print('Accuracy: %f' % accuracy)



# precision tp / (tp + fp)

precision = precision_score(y_pred, y_test)

print('Precision: %f' % precision)



# recall: tp / (tp + fn)

recall = recall_score(y_pred, y_test)

print('Recall: %f' % recall)



# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_pred, y_test)

print('F1 score: %f' % f1)
y_score = lr_model.predict_proba(cv_test_features)[:,1]
fpr = dict()

tpr = dict()

roc_auc = dict()

fpr, tpr, threshold = roc_curve(y_test, y_score)



# Compute Area Under the Curve (AUC) using the trapezoidal rule

lr_roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr)

print("AUC score = {}".format(lr_roc_auc))
from sklearn.feature_extraction.text import TfidfVectorizer



### build BOW features on train_corpus using the TfidfVectortizer

tfid = TfidfVectorizer(binary=False, min_df=0.0, max_df=1.0)

tfid.fit(train_corpus)



### transform the test_corpus using the transform method



tfid_train_features = tfid.transform(train_corpus)

tfid_test_features = tfid.transform(test_corpus)



print('TFIDF model:> Train features shape:', cv_train_features.shape, 'Test features shape:', cv_test_features.shape)
mnb.fit(tfid_train_features, y_train)
# Predict the testing output using the BOW features on the test_corpus

y_pred = mnb.predict(tfid_test_features)
# Evaluate the model using the confusion matrix

labels = np.unique(y_test)

cm = confusion_matrix(y_test, y_pred, labels=labels)



print ('Confusion Matrix:')

pd.DataFrame(cm, index=labels, columns=labels)
# accuracy: (tp + tn) / (p + n)

accuracy = accuracy_score(y_pred, y_test)

print('Accuracy: %f' % accuracy)



# precision tp / (tp + fp)

precision = precision_score(y_pred, y_test)

print('Precision: %f' % precision)



# recall: tp / (tp + fn)

recall = recall_score(y_pred, y_test)

print('Recall: %f' % recall)



# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_pred, y_test)

print('F1 score: %f' % f1)
# Predict probability (using predict_proba method) of 1 - i.e. probably that the news is classified as baseball news

y_score = mnb.predict_proba(tfid_test_features)[:,1]



# Compute ROC curve and AUC for the predicted class "1"

fpr = dict()

tpr = dict()

roc_auc = dict()

fpr, tpr, threshold = roc_curve(y_test, y_score)



# Compute Area Under the Curve (AUC) using the trapezoidal rule

mnb_roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr)

print("AUC score = {}".format(mnb_roc_auc))
svm.fit(tfid_train_features, y_train)

y_pred = svm.predict(tfid_test_features)
# confusion matrix

labels = np.unique(y_test)

cm = confusion_matrix(y_test, y_pred, labels=labels)



print ('Confusion Matrix:')

pd.DataFrame(cm, index=labels, columns=labels)
# accuracy: (tp + tn) / (p + n)

accuracy = accuracy_score(y_pred, y_test)

print('Accuracy: %f' % accuracy)



# precision tp / (tp + fp)

precision = precision_score(y_pred, y_test)

print('Precision: %f' % precision)



# recall: tp / (tp + fn)

recall = recall_score(y_pred, y_test)

print('Recall: %f' % recall)



# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_pred, y_test)

print('F1 score: %f' % f1)
y_score = svm.predict_proba(tfid_test_features)[:,1]



# Compute ROC curve and AUC for the predicted class "1"



fpr = dict()

tpr = dict()

roc_auc = dict()

fpr, tpr, threshold = roc_curve(y_test, y_score)



# Compute Area Under the Curve (AUC) using the trapezoidal rule

svm_roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr)

print("AUC score = {}".format(svm_roc_auc))
lr_model.fit(tfid_train_features, y_train)

y_pred = lr_model.predict(tfid_test_features)
# confusion matrix

cm = confusion_matrix(y_test, y_pred, labels=labels)

print ('Confusion Matrix:')

pd.DataFrame(cm, index=labels, columns=labels)
# accuracy: (tp + tn) / (p + n)

accuracy = accuracy_score(y_pred, y_test)

print('Accuracy: %f' % accuracy)



# precision tp / (tp + fp)

precision = precision_score(y_pred, y_test)

print('Precision: %f' % precision)



# recall: tp / (tp + fn)

recall = recall_score(y_pred, y_test)

print('Recall: %f' % recall)



# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_pred, y_test)

print('F1 score: %f' % f1)
y_score = lr_model.predict_proba(tfid_test_features)[:,1]

fpr = dict()

tpr = dict()

roc_auc = dict()

fpr, tpr, threshold = roc_curve(y_test, y_score)



# Compute Area Under the Curve (AUC) using the trapezoidal rule

lr_roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr)

print("AUC score = {}".format(lr_roc_auc))