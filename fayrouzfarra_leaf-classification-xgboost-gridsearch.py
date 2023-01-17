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
df= pd.read_csv('../input/leaf-classification/train.csv.zip', index_col='id')

df
for col in df.columns:

    if df[col].isna().sum() > 0:

        print(col, df[col].isna().sum()   /len(df))
df.species.value_counts()
len(df.species.unique())
y=df.species

y.head()
X = df.drop(columns = 'species', axis=1)

X.head()
from sklearn.preprocessing import LabelEncoder



label_encoder= LabelEncoder().fit(y)

labeled_species= label_encoder.transform(y)
classes=list(label_encoder.classes_)

classes
parameters = parameters = {

    'n_estimators': list(range(200,401,100)),

    'learning_rate':[l/1000 for l in range (5,15,10)],

    'max_depth': list(range(6,20,5)) 

}           

parameters
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

from sklearn.metrics import log_loss



gsearch = GridSearchCV(estimator=XGBClassifier(),

                       param_grid = parameters, 

                       scoring= 'neg_log_loss',

                       n_jobs=4,cv=5, verbose=7)
gsearch.fit(X,labeled_species)
best_n_estimators = gsearch.best_params_.get('n_estimators')

best_n_estimators
best_learning_rate = gsearch.best_params_.get('learning_rate')

best_learning_rate
best_max_depth = gsearch.best_params_.get('max_depth')

best_max_depth
final_model = XGBClassifier(n_estimators=best_n_estimators,

                            learning_rate=best_learning_rate,

                            max_depth=best_max_depth)
final_model.fit(X, labeled_species)
test = pd.read_csv('../input/leaf-classification/test.csv.zip',index_col='id')
pred_test=final_model.predict_proba(test)

pred_test.shape
pred_test
output = pd.DataFrame(pred_test, columns=classes)

output.insert(0, 'id', test.index)

output.reset_index()



output.to_csv('submission.csv', index=False)

print('done')

output