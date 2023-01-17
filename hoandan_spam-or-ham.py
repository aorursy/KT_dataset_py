# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer
data = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv", encoding='latin-1') 

data.info()
data['v1'].head()
source = data['v2'] 

type(source)
source[:5]
data.groupby('v1').v2.count()
target = data['v1'] 

type(target)
# ham = 0, spam = 1
target = target.replace("ham", 0)
target = target.replace("spam", 1)
target[:5]
temp = pd.DataFrame(target)
temp.head()
text_data = np.array(source) 

text_data
target_data = np.array(target) 

target_data
count = CountVectorizer() 

count.fit(text_data)

bag_of_words = count.transform(text_data) 

bag_of_words
X = bag_of_words.toarray() 

X
X.shape
y = np.array(target)
y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
clf = MultinomialNB()

model = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('score Scikit learn - train: ', model.score(X_train,y_train))
print('score Scikit learn: ', model.score(X_test,y_test))
from sklearn.metrics import accuracy_score

print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"%")
# Both training and testing have a high Score
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred, labels=[0, 1])
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
print(classification_report(y_test, y_pred))
# Comment: High precision, high recall
y_prob = model.predict_proba(X_test) 

y_prob
roc_auc_score(y_test, y_prob[:, 1])
import matplotlib.pyplot as plt
# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1]) # plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

plt.plot(fpr, tpr, marker='.')

plt.show()
# High ROC

# Based on all reviews => Model matches
# save model
import pickle

pkl_filename = "ham_spam_model.pkl" 

with open(pkl_filename, 'wb') as file:

    pickle.dump(model, file)
# save model

with open(pkl_filename, 'rb') as file: 

    ham_spam_model = pickle.load(file)
x_new = np.array(['Dear Mr. Hoangdang. I will come on time.',

                'Dear dangvanconghoang.ch@gmail.com VIB credit card opening program with many attractive offersFree annual fee for lifeInterest free for the first 3 months using the card0% interest installment with more than 80 points associated with the bankUnlimited withdrawals up to 100% of the limitRefund up to 6% - equivalent to VND 12 million / year for all spending transactionsGive unlimited points up to 5 times for every transactionEarn reward miles for every spendDonate up to 500 liters of gasoline per yearInterest free up to 55 days'])

x_new = count.transform(x_new)
y_pred_new = ham_spam_model.predict(x_new) 

y_pred_new