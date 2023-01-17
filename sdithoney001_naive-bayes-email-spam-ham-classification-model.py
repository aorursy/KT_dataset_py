import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

email_data = pd.read_csv("/kaggle/input/spam-mails-dataset/spam_ham_dataset.csv")
print(email_data)
email_data
# Checking for null values



email_data.isnull().sum()
# dropping unwanted columns

email_data = email_data.drop("Unnamed: 0", axis = 1)

email_data = email_data.drop("label_num", axis = 1)
email_data
# Changing type of ham/ spam to 1/0 as sklearn library deals with 0 and 1



email_data['type'] = email_data.label.map({'ham': 1 , 'spam' : 0}) 
email_data
spam_ham_value = email_data.label.value_counts()

spam_ham_value
# Calculate the ratio of spam data

print("Spam % is ",(spam_ham_value[1]/float(spam_ham_value[0]+spam_ham_value[1]))*100)
# Seperate data into variables as X and y, So that we can use Bags of Words Representation to store the data. 

X=email_data.text

y=email_data.type
X
y
print(X.shape)

print(y.shape)
# splitting into test and train



from sklearn.model_selection  import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
X_train.head()
from sklearn.feature_extraction.text import CountVectorizer



# vectorising the text

vect = CountVectorizer(stop_words='english')
vect.fit(X_train)
vect.vocabulary_
vect.get_feature_names()
# transform

X_train_transformed = vect.transform(X_train)

X_test_tranformed =vect.transform(X_test)
print(X_test[:1])
print(X_test_tranformed)
from sklearn.naive_bayes import BernoulliNB



# instantiate bernoulli NB object

bnb = BernoulliNB()



# fit 

bnb.fit(X_train_transformed,y_train)



# predict class

y_pred_class = bnb.predict(X_test_tranformed)



# predict probability

y_pred_proba =bnb.predict_proba(X_test_tranformed)



# accuracy

from sklearn import metrics

metrics.accuracy_score(y_test, y_pred_class)
bnb
metrics.confusion_matrix(y_test, y_pred_class)
confusion = metrics.confusion_matrix(y_test, y_pred_class)

print(confusion)

#[row, column]

TN = confusion[0, 0]

FP = confusion[0, 1]

FN = confusion[1, 0]

TP = confusion[1, 1]
sensitivity = TP / float(FN + TP)

print("sensitivity",sensitivity)
specificity = TN / float(TN + FP)



print("specificity",specificity)
precision = TP / float(TP + FP)



print("precision",precision)

print(metrics.precision_score(y_test, y_pred_class))
print("precision",precision)

print("PRECISION SCORE :",metrics.precision_score(y_test, y_pred_class))

print("RECALL SCORE :", metrics.recall_score(y_test, y_pred_class))

print("F1 SCORE :",metrics.f1_score(y_test, y_pred_class))
y_pred_proba
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_proba[:,1])

roc_auc = auc(false_positive_rate, true_positive_rate)
print (roc_auc)
import matplotlib.pyplot as plt

%matplotlib inline  

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.title('ROC')

plt.plot(false_positive_rate, true_positive_rate)
import pickle



# Exporting my model to use in a project 

Pkl_Filename = "SpamOrHam.pkl"  



with open(Pkl_Filename, 'wb') as file:  

    pickle.dump(bnb, file)
# Exporting Counter Vector to get the BOW at Runtime

# Exporting my model to use in a project 

CV_Pkl_Filename = "CounterVector.pkl"  



with open(CV_Pkl_Filename, 'wb') as file:  

    pickle.dump(vect, file)