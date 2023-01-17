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
import matplotlib.pyplot as plt
# load data into df

df = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")
# check content

df.head()
len(df)
# describe data: simple stats

df.describe()
# data types

df.dtypes
# distributions of features and target (categorical)

categorical_features = df.columns # all columns are categorical

fig, ax = plt.subplots(len(categorical_features), 1, figsize=(6,len(categorical_features)*5))

for i, categorical_feature in enumerate(df[categorical_features]):

    df[categorical_feature].value_counts().plot(kind="bar", ax=ax[i]).set_title(categorical_feature)

fig.show()
# correlation matrix

df.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
df["class"].value_counts()
beta = 0.5 # we give recall half the importance of precision
from sklearn.metrics import fbeta_score



# fbeta_score example



y_true = [0, 1, 1, 0, 1, 1]

y_pred = [0, 0, 1, 0, 0, 1]



fbeta_score(y_true, y_pred, beta=0.5)
from sklearn.preprocessing import LabelEncoder
# encode target: we aim to predict poisonous mushrooms => we need high precision

y = df["class"].map({'p':1, 'e':0})

y
# encode features

X = df.drop(columns=["class"]).apply(LabelEncoder().fit_transform)

X.head()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y,

                                                   test_size=0.2,

                                                   random_state=0)



print("X_train: ", X_train.shape)

print("X_test: ", X_test.shape)

print("Y_train: ", Y_train.shape)

print("Y_test: ", Y_test.shape)
from xgboost import XGBClassifier
model = XGBClassifier()

model.fit(X_train, Y_train, eval_metric="auc", eval_set=[(X_test, Y_test)], verbose=False) # TODO use fbeta_score for eval_metric
pred = model.predict(X)

fbeta_score(y, pred, beta=0.5)
pred_baseline = np.ones(y.shape) # class=1

fbeta_score(y, pred_baseline, beta=beta)
pred_baseline = np.zeros(y.shape) # class=0

fbeta_score(y, pred_baseline, beta=beta)
from xgboost import plot_importance
plot_importance(model)