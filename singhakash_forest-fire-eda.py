# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing the necessary libraries

import matplotlib.pyplot as plt
import math
import random



# Loading the dataset
db = pd.read_csv('../input/forest-fires-data-set/forestfires.csv')
#Printing the first 5  rows of the loaded Dataset
db.head()
# Extracting the dataset information
db.info()
# Libraries and configurations for figure plotting
plt.style.use('seaborn')
db.hist(bins=30, figsize=(20,15)) # plotting the histogram
# Coverting the days and months into the integers
db.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
db.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)
#Printing after replacement
db.head(10)
# Correlation analysis of the dataset
db.corr()
db.describe() # Generate descriptive statistics that summarize the central tendency,dispersion and shape of a dataset's distribution
from sklearn.model_selection import train_test_split
# dividing the data into test and training sets
train_set, test_set = train_test_split(db, test_size=0.2, random_state=42)
work_set = train_set.copy() # assigning a copy of train set to work_set
train_set.head()
test_set.head()
work_set.plot(kind='scatter', x='X', y='Y', alpha=0.1, s=300) # scatter plot for the dataset
work_set.plot(kind='scatter', x='X', y='Y', alpha=0.2, s=20*work_set['area']) # plotting the graphs by increasing the size to see the affect of area over the datapoints
# Extracting featuresfrom the dataset

# converting to list
x_values = list(work_set['X'])
y_values = list(work_set['Y'])

loc_values = []

for index in range(0, len(x_values)):
    temp_value = []
    temp_value.append(x_values[index])
    temp_value.append(y_values[index])
    loc_values.append(temp_value)
# counting the instances location in the dataset

def count_points(x_points, y_points, scaling_factor):
    count_array = []
    
    for index in range(0, len(x_points)):
        temp_value = [x_points[index], y_points[index]]
        count = 0
        
        for value in loc_values:
            if(temp_value == value):
                count = count + 1
        count_array.append(count * scaling_factor )

    return count_array
work_set.head()

# Plotting the histogram for the RH attribute
from pandas.plotting import scatter_matrix

attributes = ['RH']
scatter_matrix(work_set[attributes], figsize=(15,10))

# Plotting the histogram for the temp attribute
from pandas.plotting import scatter_matrix

attributes = ['temp']
scatter_matrix(work_set[attributes], figsize=(15,10))
# Plotting the histogram for the DMC attribute
from pandas.plotting import scatter_matrix

attributes = ['DMC']
scatter_matrix(work_set[attributes], figsize=(15,10))
# Plotting the histogram for the area attribute
from pandas.plotting import scatter_matrix

attributes = ['area']
scatter_matrix(work_set[attributes], figsize=(15,10))
db['month'].unique()
db['day'].unique()
db['area'].unique()
# defining the method for plotting the histogram
def histogram_plot(db, title):
    plt.figure(figsize=(8, 6))    
    
    ax = plt.subplot()    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)
    
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left() 
    
    plt.title(title, fontsize = 22)
    plt.hist(db, edgecolor='black', linewidth=1.2)
    plt.show()
# Scattering the plot with the help of the location

plt.figure(figsize=(8, 6))    
    
ax = plt.subplot()    
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)
    
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left() 
    
plt.title("Fire location plot", fontsize = 22)
plt.scatter(x_values, y_values, s = count_points(x_values, y_values, 25), alpha = 0.3)
plt.show()
#Encoding the data using the Label Encoder

from sklearn.preprocessing import LabelEncoder

month_encoder = LabelEncoder()
day_encoder = LabelEncoder()

months = db['month']
days = db['day']

month_1hot = month_encoder.fit_transform(months) # label encoding month
day_1hot = day_encoder.fit_transform(days) # label encoding day
month_1hot

day_1hot

# Standardizing the data (Feature Scaling) so that all the features are of the same scale
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

numerical_features = db.drop(['month', 'day'], axis=1)
scaled_features = scaler.fit_transform(numerical_features)
scaled_features
from sklearn.base import BaseEstimator, TransformerMixin

# defining the methods  for the AttributeSelector
class AttributeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values
from sklearn.preprocessing import MultiLabelBinarizer
# defining the methods  for the CustomBinarizer
class CustomBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, class_labels):
        self.class_labels = class_labels
    def fit(self, X, y=None,**fit_params):
        return self
    def transform(self, X):
        return MultiLabelBinarizer(classes=self.class_labels).fit_transform(X)
from sklearn.pipeline import Pipeline


numerical_attributes = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain'] # Selecting the numerical columns
categorical_attributes = ['month', 'day'] # # Selecting the categorical columns
categorical_classes = np.concatenate((db['month'].unique(), db['day'].unique()), axis=0)

# creating the separate numerical and categorical pipelines
numerical_pipeline = Pipeline([
    ('selector', AttributeSelector(numerical_attributes)),
    ('standardize', StandardScaler()),
])
categorical_pipeline = Pipeline([
    ('selector', AttributeSelector(categorical_attributes)),
    ('encode', CustomBinarizer(categorical_classes)),
])
#FFMC distrubution
#  Creating Histogram based on FFMC attribute
histogram_plot(db['FFMC'], title = "FFMC distribution")
plt.show()
#DC distrubution
#  Creating Histogram based on DC attribute 
histogram_plot(db['DC'], title = "DC distribution")
plt.show()
#  Separating the features and labels into X and Y
X = db.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11]].values
Y = db.iloc[:, 11].values
# Separating the test and training set
train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size=0.3, random_state = 9)
mse_values = []
variance_score = []

train_x
train_y
