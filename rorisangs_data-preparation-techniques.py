import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer
from numpy import set_printoptions
df = pd.read_csv('../input/diabetes/diabetes.csv')
df.head()
df.rename(columns={'Pregnancies': 'Preg', 'BloodPressure':'BP', 'SkinThickness': 'Skin', 'DiabetesPedigreeFunction': 'DPF', 'Outcome': 'Class'}, inplace = True)
df.head()
X = df.values[:, 0:8]
Y = df.values[:, 8]
X.shape, Y.shape
# Standardize data
X = df.values[:, 0:8]
Y = df.values[:, 8]
scaler2 = StandardScaler().fit(X)
rescaledX2 = scaler2.transform(X)
set_printoptions(precision = 3)
print(rescaledX2[0:5, :])
# Rescale data
X = df.values[:, 0:8]
Y = df.values[:, 8]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
set_printoptions(precision = 3)
print(rescaledX[0:5, :])
# Normalize data
scaler3 = Normalizer().fit(X)
normalizedX = scaler3.transform(X)
print(normalizedX[0:5, :])
# Binarizer
binarizer = Binarizer(threshold = 0.0).fit(X)
binaryX = binarizer.transform(X)
print(binaryX[0:5, :])

