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
train_data = pd.read_csv('../input/leaf-classification/train.csv.zip',index_col='id')

train_data_cop = data.copy()

train_data
from sklearn.preprocessing import LabelEncoder

y = train_data['species']

le = LabelEncoder().fit(y)

labels = le.transform(y) 

#label_y = label_encoder.fit_transform(y)

classes = list(le.classes_) 

print(classes)

print(y.shape)

y[0:5]
from sklearn.preprocessing import StandardScaler

train_data.drop(columns='species',axis=1,inplace=True)

X = train_data

print(X.shape)

X
parameters = [{

    'n_estimators': list(range(100, 351, 50)), 

    'learning_rate': [0.2,0.5,0.7,0.9], 

    'max_depth': list(range(10, 50, 10))

}]
from sklearn.model_selection import GridSearchCV



from xgboost import XGBClassifier

from sklearn.metrics import log_loss ,accuracy_score
gsearch = GridSearchCV(estimator=XGBClassifier(),

                       param_grid = parameters, 

                       scoring= 'neg_log_loss',

                       n_jobs=4,cv=5, verbose=7)

gsearch
final_model = XGBClassifier(n_estimators=100, 

                          learning_rate=0.5, 

                          max_depth=6)

final_model.fit(X,y)
test_data = pd.read_csv('../input/leaf-classification/test.csv.zip')

test_index = test_data.id 

test_data.drop('id',axis=1,inplace=True)

pred_test = final_model.predict_proba(test_data)

pred_test
submission = pd.DataFrame(pred_test, columns=classes)

submission.insert(0, 'id', test_index)

submission.to_csv('submission_final.csv', index=False)

print('Done!')