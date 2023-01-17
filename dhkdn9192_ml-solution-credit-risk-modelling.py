# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read Training dataset as well as drop the index column

training_data = pd.read_csv('/kaggle/input/give-me-some-credit-dataset/cs-training.csv').drop('Unnamed: 0', axis=1)

print(f'training_data.shape : {training_data.shape}')
training_data.head()
# For each column heading we replace "-" and convert the heading in lowercase 

cleancolumn = []

for i in range(len(training_data.columns)):

    cleancolumn.append(training_data.columns[i].replace('-', '').lower())

training_data.columns = cleancolumn
training_data.head()
print(cleancolumn)
training_data.info()
training_data.describe()
training_data.median()
sns.countplot(training_data['seriousdlqin2yrs'])

plt.show()

print(training_data['seriousdlqin2yrs'].value_counts() / len(training_data['seriousdlqin2yrs']) * 100)
plt.title('missing values')

sns.barplot(y=training_data.columns, x=training_data.isnull().sum().values)

plt.show()

print(training_data.isnull().sum())
# imputing missing values - with mean

training_data_mean_replace = training_data.fillna(training_data.mean())

training_data_mean_replace.isnull().sum()
# imputing missing values - with median

training_data_median_replace = training_data.fillna(training_data.median())

training_data_median_replace.isnull().sum()
plt.figure(figsize=(11,8))

sns.heatmap(training_data_median_replace.corr(), annot=True)

plt.show()
def percentile_based_outlier(data, threshold=95):

    diff = (100 - threshold) / 2.0

    (minval, maxval) = np.percentile(data, [diff, 100 - diff])

    return ((data < minval) | (data > maxval))
def mad_based_outlier(points, threshold=3.5):

    if len(points.shape) == 1:

        points = points[:, None]

    median_y = np.median(points)

    mad = np.median([np.abs(y - meidan_y) for y in points])

    modified_z_scores = np.abs([0.6745 * (y - median_y) / mad for y in points])        

    return modified_z_scores > threshold
def std_div(data, threshold=3):

    std = data.std()

    mean = data.mean()

    isOutlier = []

    for val in data:

        if val/std > threshold:

            isOutlier.append(True)

        else:

            isOutlier.append(False)

    return isOutlier    
def outlierVote(data):

    x = percentile_based_outlier(data)

    y = mad_based_outlier(data)

    z = std_div(data)

    temp = zip(data.index, x, y, z)

    final = []

    for i in range(len(temp)):

        if temp[i].count(False) >= 2:

            final.append(False)

        else:

            final.append(True)

    return final
def plotOutlier(x):

    fig, axes = plt.subplots(nrows=4)

    

    for ax, func in zip(axes, [percentile_based_outlier, mad_based_outlier, std_div, outlierVote]):

        sns.distplot(x, ax=ax, rug=True, hist=False)

        outliers = x[func(x)]

        
np.zeros_like()