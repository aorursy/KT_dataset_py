import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy.stats as st

from sklearn import ensemble, tree, linear_model

import missingno as msno
data = pd.read_csv('../input/Dataset Diabetes Type1 (Total).csv')
data.head()
data.tail()
data.describe()
data.shape
data.info()
numeric_features = data.select_dtypes(include=[np.number])



numeric_features.columns
categorical_features = data.select_dtypes(include=[np.object])

categorical_features.columns
msno.heatmap(data)
msno.bar(data.sample(305))
msno.dendrogram(data)
data_cat = data.select_dtypes(include=['object']).copy()
data_cat.head()
print(data_cat.isnull().values.sum())
print(data_cat['Age'].value_counts())
print(data_cat['Sex'].value_counts())
print(data_cat['Area of Residence '].value_counts())
print(data_cat['HbA1c'].value_counts())
print(data_cat['Duration of disease'].value_counts())
print(data_cat['Other diease'].value_counts())
print(data_cat['Adequate Nutrition '].value_counts())
print(data_cat['Education of Mother'].value_counts())
print(data_cat['Standardized growth-rate in infancy'].value_counts())
print(data_cat['Standardized birth weight'].value_counts())
print(data_cat['Autoantibodies'].value_counts())
print(data_cat['Impaired glucose metabolism '].value_counts())
print(data_cat['Insulin taken'].value_counts())
print(data_cat['How Taken'].value_counts())
print(data_cat['Family History affected in Type 1 Diabetes'].value_counts())
print(data_cat['Family History affected in Type 2 Diabetes'].value_counts())
print(data_cat['Hypoglycemis'].value_counts())
print(data_cat['Hypoglycemis'].value_counts())
print(data_cat['pancreatic disease affected in child '].value_counts())
print(data_cat['Affected'].value_counts())