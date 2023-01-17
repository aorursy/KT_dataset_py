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
df = pd.read_csv("/kaggle/input/credit-risk-modeling-case-study/CRM_TrainData.csv")
df.head()
df = df.dropna()
df.head()
df.dtypes
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
df["Loan Status"]= le.fit_transform(df["Loan Status"])

df["Term"]= le.fit_transform(df["Term"])

df["Years in current job"]= le.fit_transform(df["Years in current job"])

df["Home Ownership"]= le.fit_transform(df["Home Ownership"])

df["Purpose"]= le.fit_transform(df["Purpose"])
df.head()
df.dtypes
df['Monthly Debt'] = pd.to_numeric(df['Monthly Debt'].astype(str).str.replace('$',''), errors='coerce').fillna(0).astype(float)

df['Maximum Open Credit'] = df['Maximum Open Credit'].astype("float64")
df.head()
df.dtypes
y = df["Loan Status"]

X = df.drop(["Loan Status","Loan ID","Customer ID"],axis=1)
X.head()
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X,y,test_size=0.3,random_state=54)
from sklearn.linear_model import LogisticRegression

lm = LogisticRegression()
fit = lm.fit(X_train,y_train)

print(fit.score(X_train,y_train),fit.coef_)
y_pred = fit.predict(X_test)
from sklearn.metrics import r2_score

print(r2_score(y_pred,y_test))
submission = {"pred":y_pred}
submission = pd.DataFrame(submission)
submission.to_csv("submission.csv")