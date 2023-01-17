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
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
data = pd.read_pickle('/kaggle/input/enron-data/final_project_dataset.pkl')

data.keys()
data.pop('TOTAL')
data_frame = pd.DataFrame.from_dict(data).T
data_frame.info()
features = ['salary','to_messages','deferral_payments','total_payments','bonus','restricted_stock_deferred','deferred_income','expenses','from_poi_to_this_person','exercised_stock_options','from_messages','from_this_person_to_poi','shared_receipt_with_poi','poi']
data_frame = data_frame[features].astype(np.float32)
import seaborn as sns

sns.boxplot(x = data_frame['salary'])
sns.boxplot(x = data_frame['salary'])
data_frame.fillna(0.,inplace = True)
data_frame['to_messages'] = data_frame['to_messages'].astype(np.int32)

data_frame['from_poi_to_this_person'] = data_frame['from_poi_to_this_person'].astype(np.int32)

data_frame['from_this_person_to_poi'] = data_frame['from_this_person_to_poi'].astype(np.int32)
data_frame = data_frame.loc[:,~data_frame.columns.duplicated()]
data_frame.info()
from_fraction = data_frame['from_poi_to_this_person'].astype(np.float32)

to_fraction = data_frame['from_this_person_to_poi'].astype(np.float32)

to_msg = data_frame['to_messages'].astype(np.float32)
data_frame['fraction_from_poi'] = from_fraction/to_msg
data_frame['fraction_to_poi'] = to_fraction/to_msg
data_frame.fillna(0. , inplace = True)

data_frame.info()
X = data_frame[['fraction_to_poi','expenses','salary','bonus','fraction_from_poi','shared_receipt_with_poi','deferral_payments','deferred_income']].values

Y = data_frame['poi'].values
X
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4,random_state=42)
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
clf=SVC()

clf.fit(X_train,Y_train)

prediction=clf.predict(X_test)

print("Accuracy for SVM: ",accuracy_score(prediction,Y_test))
from sklearn.naive_bayes import GaussianNB

clf=GaussianNB()

clf.fit(X_train,Y_train)

prediction =clf.predict(X_test)

print("Accuracy for GaussianNB: ",accuracy_score(prediction,Y_test))
clf=AdaBoostClassifier()

clf.fit(X_train,Y_train)

prediction = clf.predict(X_test)

print("Accuracy for AdaBoost: ",accuracy_score(prediction,Y_test))
clf=RandomForestClassifier(n_estimators = 100)

clf.fit(X_train,Y_train)

prediction=clf.predict(X_test)

print("Accuracy for RandomForest: ",accuracy_score(prediction,Y_test))