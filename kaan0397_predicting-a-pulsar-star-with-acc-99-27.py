import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





plt.style.use('ggplot')
data = pd.read_csv("../input/pulsar_stars.csv")
data.head()
data.info()
data.isnull().sum()
data.corr()
plt.figure(figsize=(20,3))



plt.subplot(4,4,1)

sns.boxplot(x = data[" Mean of the integrated profile"])



plt.subplot(4,4,2)

sns.boxplot(x = data[" Standard deviation of the integrated profile"])



plt.subplot(4,4,3)

sns.boxplot(x = data[" Excess kurtosis of the integrated profile"])



plt.subplot(4,4,4)

sns.boxplot(x = data[" Skewness of the integrated profile"])



plt.subplot(2,4,5)

sns.boxplot(x = data[" Mean of the DM-SNR curve"])



plt.subplot(2,4,6)

sns.boxplot(x = data[" Standard deviation of the DM-SNR curve"])



plt.subplot(2,4,7)

sns.boxplot(x = data[" Excess kurtosis of the DM-SNR curve"])



plt.subplot(2,4,8)

sns.boxplot(x = data[" Skewness of the DM-SNR curve"])





plt.show()
#Remove the outliers

data = data.drop(data[(data[" Mean of the integrated profile"] > 150)  & (data[" Mean of the integrated profile"] < 80)].index)

data = data.drop(data[(data[" Standard deviation of the integrated profile"] > 65)  & (data[" Standard deviation of the integrated profile"] < 25)].index)

data = data.drop(data[(data[" Excess kurtosis of the integrated profile"] > 0.5)  & (data[" Excess kurtosis of the integrated profile"] < -0.6)].index)

data = data.drop(data[(data[" Skewness of the integrated profile"] > 1)].index)

data = data.drop(data[(data[" Mean of the DM-SNR curve"] > 2)  & (data[" Mean of the DM-SNR curve"] < -1)].index)

data = data.drop(data[(data[" Standard deviation of the DM-SNR curve"] > 32)].index)

data = data.drop(data[(data[" Excess kurtosis of the DM-SNR curve"] > 18)  & (data[" Excess kurtosis of the DM-SNR curve"] < -3)].index)

data = data.drop(data[(data[" Skewness of the DM-SNR curve"] > 250)].index)
data["target_class"].value_counts() # We should use stratify in train_test_split. Because 0 and 1 classes have not same or close value counts.
y = data["target_class"]

x = data.drop(["target_class"] , axis=1)
#Normalization

from sklearn.preprocessing import StandardScaler

x = StandardScaler().fit_transform(x)
#train test split

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , stratify = y , test_size = 0.20 , random_state = 42)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=20 , random_state=42)

rf.fit(x_train , y_train)

print(rf.score(x_test , y_test))