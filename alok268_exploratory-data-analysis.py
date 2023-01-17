# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import seaborn as sns # Data visualization
import matplotlib.pyplot as plt # Standard visualization
# Let's load the data sets
data = pd.read_csv('../input/data.csv')
# shape of our data set
print("Our datasets has ", data.shape[0],"x", data.shape[1], " dimensions")
print("Number of rows: ", data.shape[0])
print("Number of columns: ", data.shape[1])
pd.set_option('max_columns', None)
print(data.head(5))
# let's see last 5 rows of the Unnamed: 32 column
print(data['Unnamed: 32'].tail())
# let's find percentage of missing values in each columns
#columns = data.columns
percentage_missing = data.isnull().sum() / data.shape[0] * 100
# missing_perc_df = pd.DataFrame({'column_name': columns,
#                               'percent_missing': percentage_missing})
print(percentage_missing.sort_values(ascending=False))
print("Columns of our dataset:\n", list(data.columns))
print(data.info())
data_label = data['diagnosis']
data_features = data.drop(['id', 'Unnamed: 32', 'diagnosis'], axis=1)
print(data_features.head())
# Let's see a statistiacal summary of our feature data set
print(data_features.describe())
# let's see class balance in our dataset
sns.countplot(data_label, label='count')
B, M = data_label.value_counts()
print('Number of Benign class: ', B)
print("Number of Malignant class:", M)
print("Malignant class differ by:", round((((B - M )/ B) * 100),2), "%")
# Normalization of data
data_std = (data_features - data_features.mean()) / data_features.std()
print(data_std.describe())
df_std = pd.concat([data_label, data_std], axis=1)
# now visualize the box plots for the first 10 columns
plt.figure(figsize=(21,9))
data_y = data_label
data = data_features
data_std = (data - data.mean()) / (data.std())              # standardization
data_10 = pd.concat([data_y,data_std.iloc[:,0:10]],axis=1)
data_10 = pd.melt(data_10,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
sns.boxplot(x="features", y="value", hue="diagnosis", data=data_10)
plt.xticks(rotation=90)
fig, ax = plt.subplots(figsize=(21,9))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data_10, split=True)
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data_10)
plt.xticks(rotation=90)
# visualize next  columns with box plot
plt.figure(figsize=(21,9))
data_20 = pd.concat([data_y,data_std.iloc[:,10:20]],axis=1)
data_20 = pd.melt(data_20,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
sns.boxplot(x="features", y="value", hue="diagnosis", data=data_20)
plt.xticks(rotation=90)
fig, ax = plt.subplots(figsize=(21,9))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data_20, split=True)
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data_20)
plt.xticks(rotation=90)
# visualize next  10 columns with box plot
plt.figure(figsize=(21,9))
data_30 = pd.concat([data_y,data_std.iloc[:,20:]],axis=1)
data_30 = pd.melt(data_30,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
sns.boxplot(x="features", y="value", hue="diagnosis", data=data_30)
plt.xticks(rotation=90)
fig, ax = plt.subplots(figsize=(21,9))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data_30, split=True)
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data_30)
plt.xticks(rotation=90)
# radius_mean: mean of distances from center to points on the perimeter
# let's see the distribuiton of radius_mean for both malignant and benign tumors

# first rearange our dataset in original format
data = pd.concat([data_label, data_features], axis=1)

# now we do have our original dataset, not normalized because we want to see how the distribution of our data column is
plt.hist(data[data['diagnosis'] == 'M'].radius_mean, bins=20, label="malignant")
plt.hist(data[data['diagnosis'] == 'B'].radius_mean, bins=20, label="Benign")
plt.legend()
plt.xlabel('Radius mean')
plt.ylabel('Frequency')
plt.title('Distribution of radius mean')
plt.show()
# let's see the histogram for perimeter_mean
plt.hist(data[data["diagnosis"] == "M"].perimeter_mean, bins=20, edgecolor='k', label="Malignant")
plt.hist(data[data["diagnosis"] == "B"].perimeter_mean, bins=20, edgecolor='k', label="Benign")
plt.xlabel("Perimeter mean")
plt.ylabel("Frequency")
plt.legend()
plt.title("Perimeter mean histogram for both tumors")
plt.show()
# let's plot an Empirical cumulative distribution function graph for both radius_mean and perimeter_mean

def ecdf(data):
    '''Plot the ecdf plot for given x'''
    # number of data points
    n = len(data)
    # x-data for ecdf
    x = np.sort(data)    
    # y-data for ecdf
    y = np.arange(1, n+1) / n
    
    return x, y

# plot ecdf for radius mean
x_radius_mean_m, y_radius_mean_m = ecdf(data[data['diagnosis'] == 'M'].radius_mean)
x_radius_mean_b, y_radius_mean_b = ecdf(data[data['diagnosis'] == 'B'].radius_mean)
plt.plot(x_radius_mean_m, y_radius_mean_m, marker='.', linestyle='none')
plt.plot(x_radius_mean_b, y_radius_mean_b, marker='.', linestyle='none')
plt.legend(('Malignant', 'Benign'), loc='lower right')
plt.xlabel('Radius mean')
plt.ylabel('ECDF')
plt.title('ECDF for Radius mean of tumor')
plt.show()
# plot ecdf for perimeter mean
x_per_mean_m, y_per_mean_m = ecdf(data[data['diagnosis'] == 'M'].perimeter_mean)
x_per_mean_b, y_per_mean_b = ecdf(data[data['diagnosis'] == 'B'].perimeter_mean)
plt.plot(x_per_mean_m, y_per_mean_m, marker='.', linestyle='none')
plt.plot(x_per_mean_b, y_per_mean_b, marker='.', linestyle='none')
plt.legend(('Malignant', 'Benign'), loc='lower right')
plt.xlabel('Perimeter mean')
plt.ylabel('ECDF')
plt.title('ECDF for Perimeter mean of tumor')
plt.show()
# let's see outliers for both of our previously selected columns
def outlier_iqr(arr):
    """returns outliers for a numpy array"""
    quartile_1, quartile_3 = np.percentile(arr, [25, 75])
    # inter quartile range
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    
    return np.where((arr > upper_bound) | (arr < lower_bound))


# outlier for benign radius mean
outlier_radius_mean_m = list(outlier_iqr(data[data['diagnosis'] == 'M'].radius_mean))
print("Numbre of Outliers for Malignant radius mean is: ", len(outlier_radius_mean_m[0]))

# outliers for malignant radius mean
outlier_radius_mean_b = list(outlier_iqr(data[data['diagnosis'] == 'B'].radius_mean))
print("Numbre of Outliers for benign radius mean is: ", len(outlier_radius_mean_b[0]))
# outliers for malignant perimeter mean
outlier_perimeter_mean_m = list(outlier_iqr(data[data['diagnosis'] == 'M'].perimeter_mean))
print("Numbre of Outliers for Malignant perimeter mean is: ", len(outlier_perimeter_mean_m[0]))

# outliers for benign perimeter mean
outlier_perimeter_mean_b = list(outlier_iqr(data[data['diagnosis'] == 'B'].perimeter_mean))
print("Numbre of Outliers for benign perimeter mean is: ", len(outlier_perimeter_mean_b[0]))
# Here we will look for only radius mean
print("Sample minimum for Malignant radius mean:", data[data['diagnosis'] == 'M'].radius_mean.min())
print("Sample minimum for Benign radius mean:", data[data['diagnosis'] == 'B'].radius_mean.min())
print("The lower quartile for Malignant radius mean:", np.percentile(data[data['diagnosis'] == 'M'].radius_mean, [25]))
print("The lower quartile for Benign radius mean:", np.percentile(data[data['diagnosis'] == 'B'].radius_mean, [25]))
print("The median for malignant radius mean:", np.median(data[data['diagnosis']=='M'].radius_mean))
print("The median for Benign radius mean:", np.median(data[data['diagnosis']=='B'].radius_mean))
print("The upper quartile for Malignant radius mean:", np.percentile(data[data['diagnosis'] == 'M'].radius_mean, [75]))
print("The upper quartile for Malignant radius mean:", np.percentile(data[data['diagnosis'] == 'B'].radius_mean, [75]))
print("Sample maximum for Malignant radius mean:", data[data['diagnosis'] == 'M'].radius_mean.max())
print("Sample maximum for Benign radius mean:", data[data['diagnosis'] == 'B'].radius_mean.max())
# Let's analyse the relation between the radius mean and concavity mean
# let's see the ecdf for concavity mean to get a brief insight for the data points

x_con_mean_m, y_con_mean_m = ecdf(data[data['diagnosis'] == 'M'].concavity_mean)
x_con_mean_b, y_con_mean_b = ecdf(data[data['diagnosis'] == 'B'].concavity_mean)
plt.plot(x_con_mean_m, y_con_mean_m, marker='.', linestyle='none')
plt.plot(x_con_mean_b, y_con_mean_b, marker='.', linestyle='none')
plt.legend(('Malignant', 'Benign'), loc='lower right')
plt.xlabel('Concavity mean')
plt.ylabel('ECDF')
plt.title('ECDF for Concavity mean of tumor')
plt.show()
# joint plot to see the relation among two variables
plt.figure()
sns.jointplot(x='radius_mean', y='concavity_mean', data=data, kind='hex')
plt.colorbar()
plt.title("joint plot for radius mean and concavity")
plt.xlabel('Radius mean')
plt.ylabel('Concavity mean')
plt.show()
plt.figure()
sns.jointplot(x='radius_mean', y='concavity_mean', data=data, kind='reg')
# plt.colorbar()
plt.title("regression plot for radius mean and concavity")
plt.xlabel('Radius mean')
plt.ylabel('Concavity mean')
plt.show()
# pairplot for pair wise scatter plots 
desired_data = data.loc[:, ['radius_mean', 'texture_mean', 'area_mean', 'symmetry_mean','diagnosis']]
sns.pairplot(desired_data, hue='diagnosis')
# Now it's time to see correlation heat map, let's see what and how features are correlated to each other.
fig, ax = plt.subplots(figsize=(21,21))
sns.heatmap(data_features.corr(), annot=True, square=True, ax=ax)
