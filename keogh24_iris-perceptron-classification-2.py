# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

df = pd.read_csv('../input/Iris.csv')
df.head()
df.describe().transpose()
df.corr()
df['Species'].value_counts()
colors = df['Species'].replace(to_replace=['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], value=['red', 'blue', 'green'])
plt.scatter(df['SepalLengthCm'], df['SepalWidthCm'], c=colors)
plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')
df.head()
df2 = df.drop(['Id', 'PetalLengthCm', 'PetalWidthCm'], axis=1)
df2['Species'].replace(to_replace=['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], value=[-1., 1., 1.], inplace=True)
df2.head()
y = df2[['Species']]
y.head()
X = df2[['SepalLengthCm', 'SepalWidthCm']]
X.head()
inputs = X.as_matrix()
labels = y.as_matrix()
def predict(weights, bias, input):
    return np.sign(np.dot(np.transpose(weights), input) + bias)
def train(inputs, labels):
    n_samples, n_features = inputs.shape
    epochs = 3
    current_epoch = 0
    
    weights = np.zeros(n_features)
    bias = 0.0
    
    while True:
        errors = 0
        for i in range(n_samples):
            if (predict(weights, bias, inputs[i]) != labels[i]):
                errors += 1
                weights += labels[i] * inputs[i]
                bias += labels[i]
        if errors == 0:
            break
    return weights, bias
weights, bias = train(inputs, labels)
slope = -weights[0]/weights[1]
x2cut = -bias[0]/weights[1]

print('weights: ', weights)
print('bias: ', bias)
print('classifier line: x2 = %s*x1%s' % (slope, x2cut))
df.head()
pd.Series.max(df['SepalLengthCm'])
min = pd.Series.min(df['SepalLengthCm'])
max = pd.Series.max(df['SepalLengthCm'])
x1 = [min, max]
x2 = [slope*min+x2cut, slope*max+x2cut]
plt.scatter(df['SepalLengthCm'], df['SepalWidthCm'], c=colors)
plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')
plt.plot(x1, x2)

# 