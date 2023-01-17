import pandas as pd

sms = pd.read_csv("../input/spam.csv",encoding='latin-1')

sms.shape
sms.head()
for col in sms.columns:

    if 'Unnamed' in col:

        del sms[col]
sms.head()
sms = sms.rename(columns={'v1': 'label', 'v2': 'message'})

sms.head()
sms.shape
# Examine the class distribution

sms.label.value_counts()
# convert the label to numeric variable

sms['label_num']=sms.label.map({"ham":0,"spam":1})

# checking the conversion

sms.head()
# Required way to use x and y for countvectorizer

x = sms.message

y = sms.label_num

print(x.shape)

print(y.shape)
# split x and y into traing and testing set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)

print(x_train.shape)

print(x_test.shape)
# import and instatiate the countvectorizer with default parameter

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()

# Creating a document term matrix

x_train_dtm = vect.fit_transform(x_train)

x_train_dtm
# Transform testing data into document term matrix

x_test_dtm = vect.transform(x_test)

x_test_dtm
# Import and instatiate the naive bayes model

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
# Train the model using x_train_dtm

%time nb.fit(x_train_dtm,y_train)
# make class prediction for x_test_dtm

y_pred_class = nb.predict(x_test_dtm)
# calculate the accuarcy of class prediction

from sklearn import metrics

metrics.accuracy_score(y_test,y_pred_class)
# print the confusion matrix

metrics.confusion_matrix(y_test,y_pred_class)
# print the message for false positive(meaning they were incorrectly classified as spam)

x_test[y_test<y_pred_class]
# print the message for false negative(meaning they were incorrectly classified as ham)

x_test[y_test > y_pred_class]
# calculate the predicted probablities of x_test_dtm(poorly calibarated)

y_pred_prob = nb.predict_proba(x_test_dtm)[:,1]

y_pred_prob
# calculate AUC

metrics.roc_auc_score(y_test,y_pred_prob)
# Import and instaiate logistic regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

# Train the model using x_train_dtm

%time logreg.fit(x_train_dtm,y_train)
# make class prediction fot x_test

y_pred_class = logreg.predict(x_test_dtm)
# calculate the accuarcy of class prediction

metrics.accuracy_score(y_test,y_pred_class)
# predicted probablities for x_test_dtm

y_pred_prob = logreg.predict_proba(x_test_dtm)[:,1]

y_pred_prob
# Calculate Auc

metrics.roc_auc_score(y_test,y_pred_class)