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
#Load Dataset
col_names = ["class", "sms"]
df = pd.read_csv("/kaggle/input/sms-spam-collection-data-set/SMSSpamCollection", names = col_names, sep='\t')
df.head()
len(df)
#Count class labels
ham_spam = df['class'].value_counts()
ham_spam
# Check for null values
df.isnull().sum()
#Compare Class Label Percentage wise
print("Spam Percentage is {0}".format(round((ham_spam[1]*100)/(float(ham_spam[0]+ham_spam[1])),2)))
#Label encoding of categorical feature
df['label'] = df['class'].map({'ham':0,'spam':1})
df.head()
#Dop Categorical Column
df = df.drop('class',axis=1)
df.head()
#Plot class frequency
import matplotlib.pyplot as plt

ham_spam.plot(kind ='bar',rot=0)

plt.title('Ham Spam Email Classification')
#plt.xticks(range(2),[ham_spam[0],ham_spam[1]])
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.show()
X = df.sms
y = df.label
print(X.shape)
print(y.shape)
#Split test and train dataset
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
X_train.head()
#Build BAG-Of-Word representation of text data using CountVectorizer or TF-idf
from sklearn.feature_extraction.text import CountVectorizer

# vectorising the text
vect = CountVectorizer(stop_words='english')
vect.fit(X_train)
print(vect.vocabulary_)
print(vect.get_feature_names())
# transform
X_train_transformed = vect.transform(X_train)
X_test_transformed  = vect.transform(X_test)
X_train_transformed
#Experiment 1 - fitting first using SMOTE and then Multinomial NB
  
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report


sm = SMOTE(random_state=12)
x_train_res, y_train_res = sm.fit_sample(X_train_transformed, y_train)

print("SMOTE data distribution: {}".format(Counter(y_train_res)))
print("Normal data distribution: {}".format(Counter(y_train)))
comp = Counter(y_train_res)
print(comp)
plt.bar(comp.keys(), comp.values())
 
plt.title('Ham Spam Email Classification')
#plt.xticks(range(2),[ham_spam[0],ham_spam[1]])
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.show()
mnb = MultinomialNB()

# fit
mnb.fit(x_train_res,y_train_res)

# predict class
y_pred_class = mnb.predict(X_test_transformed)

# predict probabilities
y_pred_proba = mnb.predict_proba(X_test_transformed)
# accuracy
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)
metrics.confusion_matrix(y_test, y_pred_class)
confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(confusion)
#[row, column]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]
sensitivity = TP / float(FN + TP)
 

specificity = TN / float(TN + FP)
 
precision = TP / float(TP + FP)
 
print("PRECISION SCORE :",metrics.precision_score(y_test, y_pred_class))
print("RECALL SCORE :", metrics.recall_score(y_test, y_pred_class))
print("F1 SCORE :",metrics.f1_score(y_test, y_pred_class))
#Experiment using Sklearn Pipeline

# instantiate bernoulli NB object
classifier = MultinomialNB()

# build normal model
pipeline = make_pipeline(classifier)
model = pipeline.fit(X_train_transformed, y_train)
prediction = model.predict(X_test_transformed)

# build model with SMOTE imblearn
smote_pipeline = make_pipeline_imb(SMOTE(random_state=4), classifier)
smote_model = smote_pipeline.fit(X_train_transformed, y_train)
smote_prediction = smote_model.predict(X_test_transformed)
print()
print("normal data distribution: {}".format(Counter(y_train)))
X_smote, y_smote = SMOTE().fit_sample(X_train_transformed, y_train)

 

print("SMOTE data distribution: {}".format(Counter(y_smote)))
# classification report
print(classification_report(y_test, prediction))
print(classification_report_imbalanced(y_test, smote_prediction))
print()
print('NORMAL Pipeline Score {}'.format(pipeline.score(X_test_transformed, y_test)))
print('SMOTE Pipeline Score  {}'.format(smote_pipeline.score(X_test_transformed, y_test)))

def print_results(headline, true_value, pred):
    print(headline)
    print("accuracy: {}".format(accuracy_score(true_value, pred)))
    print("precision: {}".format(precision_score(true_value, pred)))
    print("recall: {}".format(recall_score(true_value, pred)))
    print("f1: {}".format(f1_score(true_value, pred)))


print()
print_results("normal classification", y_test, prediction)
print()
print_results("SMOTE classification", y_test, smote_prediction)
