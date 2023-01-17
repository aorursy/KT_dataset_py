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

os.listdir()

# Any results you write to the current directory are saved as output.
# import required python packages #

import warnings

warnings.filterwarnings(action="ignore")

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score

from sklearn.metrics import accuracy_score
# Load train and test data#

train_set = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

test_set = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

y=train_set['target'] #target variable
train_set = train_set.drop(['ID_code','target'],axis=1)#drop columns#

test_set = test_set.drop(['ID_code'],axis=1)#drop columns#

whole_data = pd.concat((train_set,test_set), sort=False).reset_index(drop=True)#concat#
whole_data = preprocessing.scale(whole_data)#scaling#

whole_data = preprocessing.normalize(whole_data,norm='l2')#normalisation#



X_train = whole_data[:len(y), 5:]#train data#

X_test = whole_data[len(y):, :]#test data#

#build model#

rf = RandomForestClassifier(n_estimators=100,max_depth=15,min_samples_split=5,

                           min_samples_leaf=5,max_features=None,oob_score=True,random_state=42,verbose=1)

#fit model#

rf.fit(X_train,y)
#load submission data#

pred = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')

y_test= pred['target'].tolist()

#predict test data#

y_pred = rf.predict(X_test)
#check accuracy of the model with test data#

print('Test Accuracy:',accuracy_score(y_pred, y_test) * 100)
#submit result#

test_df = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

result_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})

result_df["target"] = y_pred

result_df.to_csv("submission.csv", index=False)