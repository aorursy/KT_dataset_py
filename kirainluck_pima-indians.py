import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns

from sklearn.model_selection import train_test_split

from matplotlib.colors import ListedColormap

from scipy.special import expit

%matplotlib inline
data = pd.read_csv('../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv', names=['Pregnancies', 'Glucose', 

                                                       'BloodPressure', 'SkinThickness', 

                                                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Class'])

#data = data.dropna()

data.head()
data.isna().sum()
data.isnull().values.any()
data.describe()
data.corr()
sns.catplot(x="Class", y="Glucose", data=data[data.Glucose >= 150], height=8.27, aspect=11.7/8.27);
sns.catplot(x="Class", y="BMI", data=data[data.BMI >= 10], height=8.27, aspect=11.7/8.27);
sns.catplot(x="Class", y="Age", data=data[data.Age >= 10], height=8.27, aspect=11.7/8.27);
data['glucosebmiagepregn'] = data.apply(lambda row: row.Glucose+row.BMI+row.Age, axis=1)
sns.catplot(x="Class", y="glucosebmiagepregn", data=data, kind="swarm", hue="Pregnancies", height=8.27, aspect=11.7/8.27);
plt.figure(figsize=(20, 15))

data2 = data[(data.Insulin != 0) & (data.Glucose >=50)]

plt.xlabel('Глюкоза')

plt.ylabel('Инсулин')

plt.plot(data2.Glucose, data2.Insulin, 'o', color="r")

plt.show()
plt.figure(figsize=(20, 15))

sns.countplot(y="Age", data=data, color="blue");
plt.figure(figsize=(20, 15))

sns.countplot(y="Pregnancies", data=data, color="green");
plt.figure(figsize=(20, 15))

data2 = data[(data.SkinThickness != 0) & (data.BMI >=15) & (data.BMI <=60)]

plt.xlabel('BMI')

plt.ylabel('Толщина кожи')

plt.plot(data2.BMI, data2.SkinThickness, 'o', color="g")

plt.show()