import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/drug-classification/drug200.csv')
df.head()
le = LabelEncoder()
for i in list(df.columns):
    if df[i].dtype=='object':
        df[i] = le.fit_transform(df[i])
df.head()
df.info()
sns.heatmap(df.corr())
sns.pairplot(df, hue='Drug')
X = df.loc[:, df.columns != 'Drug']
y = df.loc[:, 'Drug']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model1 = RandomForestRegressor().set_params(random_state=23)
model1.fit(X_train, y_train)
preds1 = model1.predict(X_test)

accuracy_score(y_test, preds1.astype(int))
df['log_Na_to_K'] = np.log(df['Na_to_K'] + 1)
df.drop('Na_to_K', axis=1, inplace=True)
sns.pairplot(df, hue='Drug')
X = df.loc[:, df.columns != 'Drug']
y = df.loc[:, 'Drug']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model2 = LogisticRegression(max_iter=10000)
model2.fit(X_train, y_train)
preds2 = model2.predict(X_test)

accuracy_score(y_test, preds2.astype(int))
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
pred_2 = rfc.predict(X_test)
score_2 = accuracy_score(y_test,pred_2.astype(int))
score_2