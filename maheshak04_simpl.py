# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')

sample=pd.read_csv('../input/sample_submission.csv')
X=train.drop(['target','ID_code'],axis=1)

y=train.target.values
from sklearn.ensemble import RandomForestRegressor
clf=RandomForestRegressor(max_depth=40,max_features=1,n_estimators=200)
clf.fit(X,y)
features=X.columns
predRF=clf.predict(test[features])
sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})

sub_df["target"] = predRF

sub_df.to_csv("submission.csv", index=False)
sub_df