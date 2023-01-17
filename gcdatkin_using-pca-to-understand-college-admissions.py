import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
data = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv', index_col='Serial No.')
data
y = data['Chance of Admit ']

X = data.drop('Chance of Admit ', axis=1)
X
pca = PCA(n_components=2)

pca.fit(X)

X_PCA = pca.transform(X)
X_PCA = pd.DataFrame(X_PCA, columns=['PC1', 'PC2'])
X_PCA
plt.figure(figsize=(14, 10))

plt.scatter(X_PCA['PC1'], X_PCA['PC2'])

plt.xlabel('PC1')

plt.ylabel('PC2')
PCA_max = np.argmax(X_PCA['PC1'])

PCA_min = np.argmin(X_PCA['PC1'])



print(PCA_max)

print(PCA_min)
X.iloc[PCA_max, :]
X.iloc[PCA_min, :]
scaler = MinMaxScaler()

X = scaler.fit_transform(X)
pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)
model = LinearRegression()

model.fit(X_train, y_train)
model.score(X_test, y_test)
pred = model.predict(X_test)
plt.figure(figsize=(14, 10))

plt.plot(pred, y_test, 'o')

plt.xlabel('Predicted value')

plt.ylabel('Actual value')