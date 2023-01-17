import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv("../input/iris/Iris.csv")

dataset

dataset.info()

X = dataset.iloc[:,0:13]

y = dataset.iloc[:, 13]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

%matplotlib inline

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

cancer.keys()

df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

df.head()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df)

scaled_data = scaler.transform(df)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)

x_pca.shape

plt.figure(figsize=(8,6))

plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='rainbow')

plt.xlabel('First principal component')

plt.ylabel('Second Principal Component')