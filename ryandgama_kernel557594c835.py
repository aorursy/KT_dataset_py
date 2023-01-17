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

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")
df_test = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")

from catboost import CatBoostClassifier 
from sklearn.model_selection import train_test_split
model = CatBoostClassifier(learning_rate =1, depth =3)
object
object_type = np.dtype("O")
df_train = df_train.dropna()
df_test = df_test.apply(lambda x: x.fillna("NA") if x.dtype == object_type
                                       else x.fillna(-1))

X = df_train.drop(["id","target"],axis=1)
y = df_train.target
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7)
X.dtypes

catagorical_features = np.where(X.dtypes != np.float)[0]

model.fit(X_train,y_train,cat_features = catagorical_features)
preds = model.predict(df_test.drop("id",axis=1))
preds.shape
submission = pd.DataFrame({
    "id" : df_test.id,
    "target" : preds
})
submission.head()
submission.to_csv("submission.csv",index=False)