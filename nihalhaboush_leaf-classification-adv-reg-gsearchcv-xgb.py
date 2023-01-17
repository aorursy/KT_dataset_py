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
df=pd.read_csv('../input/leaf-classification/train.csv.zip', index_col='id')

df.head()
df.isnull().sum()
df.species.unique()

len(df.species.unique())
df['species'].value_counts()
from sklearn.preprocessing import LabelEncoder



labeled_df = df.copy()



label_encoder = LabelEncoder().fit (df['species'])



labeled_species = label_encoder.transform(df['species'])



classes = list(label_encoder.classes_)  

classes
labeled_df
X=labeled_df.drop('species', axis=1)

y=labeled_df.species
parameters = {

    'n_estimators': list(range(100, 300, 100)), 

    'learning_rate': [l / 100 for l in range(5, 50, 10)], 

    'max_depth': list(range(6, 20, 10))

}

parameters
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

from sklearn.metrics import log_loss



gsearch = GridSearchCV(estimator=XGBClassifier(random_state=1,objective='multi:softprob'),

                       param_grid = parameters, 

                       scoring='neg_log_loss',

                       n_jobs=4,cv=5, verbose=1)
gsearch.fit(X, y)
best_n_estimators = 100

best_n_estimators
best_learning_rate = 0.25

best_learning_rate
best_max_depth = 6

best_max_depth
final_model = XGBClassifier(n_estimators=best_n_estimators, 

                          learning_rate=best_learning_rate, 

                          max_depth=best_max_depth)
final_model.fit(X, y)
test = pd.read_csv('../input/leaf-classification/test.csv.zip',index_col='id')

test 
preds_test = final_model.predict_proba(test)

preds_test
# Format DataFrame

submission = pd.DataFrame(preds_test, columns=classes)

submission.insert(0, 'id', test.index)

submission.reset_index()

submission.to_csv('submission.csv', index = False)

print('done!')