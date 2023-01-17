import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data=pd.read_csv('/kaggle/input/iris/Iris.csv')

data.head()
#dimensions of data

print("No of rows in dataset is: ",data.shape[0])

print("No of columns in dataset is: ",data.shape[1])
data.info()
data.Species.unique()

#There are three unique classes of Iris present in the dataset.
plt.figure(figsize=(8,8))

sns.pairplot(data=data,hue='Species')
sns.heatmap(data.corr(),annot=True)
data['Species'].value_counts().plot.pie(autopct='%0.2f%%',colors=['blue','green','red'],figsize=(5,5))
sns.scatterplot(x='SepalLengthCm',y='SepalWidthCm',hue='Species',data=data)
sns.scatterplot(x='PetalLengthCm',y='PetalWidthCm',hue='Species',data=data)
plt.figure(figsize=(10,6))

plt.subplot(121)

sns.boxplot(x='Species',y='PetalLengthCm',data=data)

plt.title('Variation of PetalLength with Species \n PLOT 1')

plt.subplot(122)

sns.boxplot(x='Species',y='PetalWidthCm',data=data)

plt.title('Variation of PetalWidth with Species \n PLOT 2')
plt.figure(figsize=(10,6))

plt.subplot(121)

sns.boxplot(x='Species',y='SepalLengthCm',data=data)

plt.title('Variation of SepalLength with Species \n PLOT 1')

plt.subplot(122)

sns.boxplot(x='Species',y='SepalWidthCm',data=data)

plt.title('Variation of SepalWidth with Species \n PLOT 2')