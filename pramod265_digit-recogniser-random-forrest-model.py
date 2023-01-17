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
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")



print("train shape", train.shape)

print("test shape", test.shape)
train.head()
test.head()
# No columns have missing values

train.isnull().sum()
train['label'].value_counts().sort_values(ascending=True)
train.label.value_counts()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

# from sklearn.cross_validation import cross_val_score



from sklearn.metrics import accuracy_score



y = train['label']

del train['label']



X = train

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,

                                                 random_state=2,)
X_train.head()
#train the RF classifier

clf = RandomForestClassifier(n_estimators = 500, max_depth = 6)

clf.fit(X_train,y_train)



#     RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

#                 max_depth=6, max_features='auto', max_leaf_nodes=None,

#                 min_impurity_split=1e-07, min_samples_leaf=1,

#                 min_samples_split=2, min_weight_fraction_leaf=0.0,

#                 n_estimators=500, n_jobs=1, oob_score=False, random_state=None,

#                 verbose=0, warm_start=False)



clf.predict(X_test)
rfc_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

confusion_matrix = confusion_matrix(y_test, rfc_pred)

confusion_matrix
#make prediction and check model's accuracy

prediction = clf.predict(X_test)

acc =  accuracy_score(np.array(y_test),prediction)

print ('The accuracy of Random Forest is {}'.format(acc))
classification_report = classification_report(y_test, rfc_pred)

print(classification_report)
# os.remove("/kaggle/output/kaggle/working/submission.csv")



# submissions=pd.DataFrame({"ImageId": list(range(1,len(rfc_pred)+1)),

#                          "Label": rfc_pred})

# submissions.to_csv("submission.csv", index=False, header=True)
submissions.shape
#make prediction and check model's accuracy

prediction = clf.predict(test)

prediction
submissions=pd.DataFrame({"ImageId": list(range(1,len(prediction)+1)),

                         "Label": prediction})

submissions.to_csv("submission.csv", index=False, header=True)