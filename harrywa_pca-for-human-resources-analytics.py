import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/HR_comma_sep.csv')
df.head()
df.info()
sns.heatmap(df.isnull(), yticklabels=False, cbar=False);
df['sales'].unique()
df.groupby('sales').mean()['satisfaction_level'].plot(

    kind='barh', figsize=(7,5), title='Mean Satisfaction Level of each Department');
df['salary'].unique()
salary = pd.get_dummies(df['salary'], drop_first=True)
left = df['left']

df.drop(['left', 'sales', 'salary'], axis=1, inplace=True)
df = pd.concat([df, salary], axis=1)
df.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df, left, test_size=0.4, random_state=42)
from sklearn.linear_model import LogisticRegression



log_model = LogisticRegression()

log_model.fit(X_train, y_train)

predictions = log_model.predict(X_test)
from sklearn.metrics import classification_report



print(classification_report(y_test, predictions))
df.drop(['low', 'medium'], axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(df)

scaled_df = scaler.transform(df)
scaled_df.shape
from sklearn.decomposition import PCA



pca = PCA(n_components=2)

pca.fit(scaled_df)

X_pca = pca.transform(scaled_df)
X_pca.shape
plt.figure(figsize=(8,6))

plt.scatter(X_pca[:,0], X_pca[:,1], s=10, c=left, cmap='plasma')

plt.xlabel('First Principal Component')

plt.ylabel('Second Principal Component');
comp_df = pd.DataFrame(pca.components_, columns=df.columns)

comp_df.head()
sns.heatmap(comp_df, yticklabels=['PC1','PC2']);