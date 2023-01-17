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
leaf_df = pd.read_csv('../input/leaf-classification/train.csv.zip', index_col = 'id')

leaf_df
leaf_df.isna().any().sum()
leaf_df.species
len(leaf_df.species)
leaf_df['species'].value_counts()
leaf_df.species.unique()
y = leaf_df.species

X = leaf_df.drop('species',axis=1)

X
from sklearn.preprocessing import LabelEncoder



labeled_leaf_df = leaf_df.copy()



label_encoder = LabelEncoder().fit (leaf_df['species'])



labeled_species = label_encoder.transform(leaf_df['species'])
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()



le = LabelEncoder().fit(y)

labels = le.transform(y) 

#label_y = label_encoder.fit_transform(y)

classes = list(le.classes_) 

classes
labeled_leaf_df
X=labeled_leaf_df.drop('species', axis=1)

y=labeled_leaf_df.species
parameters = {

    'n_estimators': list(range(100, 300, 100)), 

    'learning_rate': [m/ 100 for m in range(5, 30, 10)], 

    'max_depth': list(range(6, 40, 10))

}

parameters
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier



gsearch = GridSearchCV(estimator=XGBClassifier(random_state=0,objective='multi:softprob'),

                       param_grid = parameters, 

                       n_jobs=4,cv=5, verbose=7)
gsearch.fit(X, y)
best_n_estimators = gsearch.best_params_.get('n_estimators')

best_n_estimators
best_learning_rate = gsearch.best_params_.get('learning_rate')

best_learning_rate
best_max_depth = gsearch.best_params_.get('max_depth')

best_max_depth
final_model = XGBClassifier(n_estimators=best_n_estimators, 

                          learning_rate=best_learning_rate, 

                          max_depth=best_max_depth)
final_model.fit(X, y)
X_test=pd.read_csv('../input/leaf-classification/test.csv.zip')

test_ids = X_test.id 

X_test.drop('id',axis=1,inplace=True)

X_test
pred_test = final_model.predict_proba(X_test)

pred_test
'''sub_df = pd.DataFrame(data={

    'id': X_test.index,

    'species': pred_test

})

sub_df.to_csv('submission.csv', index=False)'''
submission = pd.DataFrame(pred_test, columns=classes)

submission.insert(0, 'id', test_ids)

submission.to_csv('submission.csv', index=False)

print('done!')