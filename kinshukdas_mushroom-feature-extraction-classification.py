# Importing the libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/mushrooms.csv')
df.head()
df.isnull().sum()
df.describe()
from sklearn.preprocessing import LabelEncoder
# Encoding categoricakl variables



encoder = LabelEncoder()

data = pd.DataFrame()



for col in df.columns:

    data[col] = encoder.fit_transform(df[col])



data.head()
sns.countplot(x='class', data=data)

plt.xlabel('Class')

plt.ylabel('Count')
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
X = data.drop(['class'], axis=1)

y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()



X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=500)

model.fit(X_train_scaled, y_train)
importance = model.feature_importances_

indices = np.argsort(importance)[::-1]
feature_labels = list(X_train.columns)
import plotly.offline as pyo

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode



init_notebook_mode(connected=True)
# Plotting the feature importance



trace = go.Bar(

    x = [feature_labels[i] for i in range(X_train_scaled.shape[1])],

    y = importance[indices]

)



layout = go.Layout(

    {'title': 'Random Forest Feature Importance'},

    xaxis = {'title': 'features'},

    yaxis = {'title': 'feature importance'}

)



fig = go.Figure(data=[trace], layout=layout)

pyo.iplot(fig)
from sklearn.decomposition import PCA
pca = PCA(n_components=5)



X_train_reduced = pca.fit_transform(X_train)

X_test_reduced = pca.transform(X_test)
pca.explained_variance_ratio_
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
lr_model = LogisticRegression()

lr_model.fit(X_train_reduced, y_train)
y_pred = lr_model.predict(X_test_reduced)
print('Logistic Regression Accuracy: {}'.format(metrics.accuracy_score(y_test, y_pred)))
print(metrics.classification_report(y_test, y_pred))