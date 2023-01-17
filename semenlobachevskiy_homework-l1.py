import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgbm

pd.options.display.float_format = '{:.5f}'.format
df = pd.read_csv('../input/telecom-churn/telecom_churn.csv')

df.head()
df_kor = df.corr()

plt.figure(figsize=(20,10))

sns.heatmap(df_kor, vmin=-1, vmax=1, cmap="viridis", annot=True, linewidth=0.1)
Y = df["churn"]

X = df[["total day charge", "total day minutes", "customer service calls"]]
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
%%time

lgbm_clf = lgbm.LGBMClassifier(n_estimators=1500, random_state = 42)



lgbm_clf.fit(X_train, Y_train)

lgbm_clf.fit(X_train, Y_train)

Y_pred = lgbm_clf.predict(X_test)

Y_score = lgbm_clf.predict_proba(X_test)[:,1]