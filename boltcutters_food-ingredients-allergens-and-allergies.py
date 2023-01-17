import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os

import io
data_filepath = '../input/food-allergens-and-allergies/FoodData.csv'

raw_data = pd.read_csv(data_filepath)
raw_data.head()
raw_data.describe()
raw_data['Class'].value_counts().plot(kind = 'bar', title = 'Class Distribution')
raw_data['Class'].value_counts().plot(kind = 'pie', title = 'Class Distribution')
raw_data['Type'].value_counts().plot(kind = 'bar', title = 'Class Distribution')
raw_data['Group'].value_counts().plot(kind = 'bar', title = 'Class Distribution')
raw_data['Allergy'].value_counts().head()
raw_data['Allergy'].value_counts().plot(kind = 'bar', figsize = (15, 5), title = 'Allergy Distribution')