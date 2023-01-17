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
# load data

train_data = pd.read_csv('../input/leaf-classification/train.csv.zip', index_col ='id')

test_data = pd.read_csv('../input/leaf-classification/test.csv.zip')
test_ids = test_data.id

test_data = test_data.drop(['id'], axis =1)
train_data.head()
# taking care of missing values

train_data.isnull().any().sum()
test_data.isnull().any().sum()
# encoding catagorical

train_data.info()
test_data.info()
train_data['species'].nunique()
# IV and DV

x = train_data.drop('species',axis=1)

y = train_data['species']
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y_fit = encoder.fit(train_data['species'])

y_label = y_fit.transform(train_data['species']) 

classes = list(y_fit.classes_) 

classes
# splitting

from sklearn.model_selection import  train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y_label, test_size = 0.2, random_state =1)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 40)

classifier.fit(x_train, y_train)
from sklearn.metrics import classification_report

predictions = classifier.predict(x_test)

print (classification_report(y_test, predictions))
final_predictions = classifier.predict_proba(test_data)
submission = pd.DataFrame(final_predictions, columns=classes)

submission.insert(0, 'id', test_ids)

submission.reset_index()
submission.to_csv('submission.csv', index = False)