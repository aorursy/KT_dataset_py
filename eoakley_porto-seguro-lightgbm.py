import lightgbm as lgb

from sklearn.cross_validation import cross_val_score
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape
test.shape

train.columns
model = lgb.LGBMClassifier(n_estimators=150)
train.head()
cat_features = [a for a in test.columns if a.endswith('cat')]

cat_features
model_vars = [a for a in train.columns if 'id' not in a and 'target' not in a]

model_vars
cross_val_score(model, train[model_vars], train.target, cv=5, fit_params=dict(categorical_feature=cat_features))
model.fit(train[model_vars], train.target, categorical_feature=cat_features)
preds = model.predict_proba(test[model_vars])[:,1]
sub = pd.DataFrame({'id': test.id, 'target': preds})

sub.head()
sub.to_csv('output_lgb_1.csv', index=False, header=True)