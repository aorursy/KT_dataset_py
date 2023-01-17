# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Try random forest for regression as use of categorical and continuous variables

# Factorise all categorical variables

cat_cols = [x for x in train.columns if x.startswith('cat')]



for col in cat_cols:

    train[col], _ = pd.factorize(train[col])

    test[col], _ = pd.factorize(test[col])

    

# Features for model are all cat and cont variables

features = [x for x in train.columns if (x.startswith('cat') or x.startswith('cont'))]



# Loss

y = train.loss
# Define RF

rfr = RandomForestRegressor(n_estimators=100, max_features='log2', oob_score=True)



# Fit model

rfr.fit(train[features],y)
# Predict model

preds = rfr.predict(test[features])
rfr.oob_score_