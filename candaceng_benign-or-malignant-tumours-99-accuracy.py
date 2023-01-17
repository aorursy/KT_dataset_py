import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/kaggle/input/breast-cancer/Breast_Cancer.csv')
df.head()
df.info()
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
df.dtypes
sns.heatmap(df.corr())
X = df.loc[:, df.columns != 'diagnosis']
y = df.loc[:, 'diagnosis']
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)

lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)
preds = lr.predict(X_test)
accuracy_score(y_test, preds)