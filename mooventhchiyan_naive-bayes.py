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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,roc_curve
df =pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv",encoding='ISO-8859-1')

df.head()
df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],axis=1,inplace=True)
df.rename(columns={"v1":"Class","v2":"sms"},inplace=True)
df["Class"].value_counts(normalize=True).plot(kind='bar')
df['Class']=df['Class'].map({"ham":0,"spam":1})
df.head()
# Its a reference code from other kernel, It looked Cool

topMessages = df.groupby("sms")["Class"].agg([len, np.max]).sort_values(by = "len", ascending = False).head(n = 10)

topMessages.head(10)
y= df["Class"]

X= df["sms"]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
# Count Vectorizer is used to convert each word to matrix

# Stop words used to remove the use of and,the...other joining words

from sklearn.feature_extraction.text import CountVectorizer

vect= CountVectorizer(stop_words="english")
X_traint = vect.fit_transform(X_train)

X_testt = vect.transform(X_test)
# Visualize the data Frame

pd.DataFrame(vect.get_feature_names(),columns=["Words"]).tail(15)
from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB()
bnb.fit(X_traint,y_train)
# Validating the train on the model

y_train_pred =bnb.predict(X_traint)

y_train_prob =bnb.predict_proba(X_traint)[:,1]



print("Accuracy Score of train", accuracy_score(y_train,y_train_pred))

print("AUC of the train ", roc_auc_score(y_train,y_train_prob))

print(" confusion matrix \n" , confusion_matrix(y_train,y_train_pred))
# Model on Test data 

y_test_pred =bnb.predict(X_testt)

y_test_prob =bnb.predict_proba(X_testt)[:,1]



print("Accuracy Score of test", accuracy_score(y_test,y_test_pred))

print("AUC od the test ", roc_auc_score(y_test,y_test_prob))

print(" confusion matrix \n" , confusion_matrix(y_test,y_test_pred))
fpr,tpr,thresholds= roc_curve(y_train,y_train_prob)

plt.plot(fpr,tpr)

plt.plot(fpr,fpr)

#plt.plot(fpr,thresholds)

plt.xlabel("fpr")

plt.ylabel("tpr")

plt.show()
# We are okay with less fpr but we need to be make sure we get high tpr

# Because we are okay to receive certain spam email but we cannot miss the normal email going into spam