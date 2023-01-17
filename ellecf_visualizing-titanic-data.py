# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#import data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
#explore the data a little bit
print(train_data.columns.values)
print(train_data.describe())
train_data.head()
#find out what the null sitch is
print(train_data.isnull().sum())
#Look at the target, how many survivors?
train_data['Survived'].value_counts()

train_data['Survived'].astype(int).plot.hist();
#let's turn sex into a numerical feature instead of categorical
from sklearn.preprocessing import LabelEncoder
train_data['Sex'] = LabelEncoder().fit_transform(train_data['Sex'])
#handling missing values
#print(train_data.isnull().sum())
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
train_data['Age'] = imp.fit_transform(train_data['Age'].values.reshape(-1,1)).reshape(-1)
print(train_data.isnull().sum())
# Find correlations with the target and sort
correlations = train_data.corr()['Survived'].sort_values()

# Display correlations
print('Correlations: \n', correlations)
#let's look at how the variables correlate with each other
allcorr = train_data.corr()
allcorr
# Heatmap of correlations
plt.figure(figsize = (8, 6))
sns.heatmap(allcorr, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');
plt.figure(figsize = (10, 8))

# KDE plot - smoothed histograms showing distribution of a variable for survived/died outcomes
sns.kdeplot(train_data.loc[train_data['Survived'] == 0, 'Age'], label = 'Survived == 0')
sns.kdeplot(train_data.loc[train_data['Survived'] == 1, 'Age'], label = 'Survived == 1')

# Labeling of plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
plt.figure(figsize = (10, 8))

# KDE plot - smoothed histograms showing distribution of a variable for survived/died outcomes
sns.kdeplot(train_data.loc[train_data['Survived'] == 0, 'Fare'], label = 'Survived == 0')
sns.kdeplot(train_data.loc[train_data['Survived'] == 1, 'Fare'], label = 'Survived == 1')

# Labeling of plot
plt.xlabel('Fare'); plt.ylabel('Density'); plt.title('Distribution of Fare');
plt.subplots(figsize = (15,10))
sns.barplot(x = "Pclass", 
            y = "Survived", 
            data=train_data, 
            linewidth=2)
plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25)
plt.xlabel("Socio-Economic class", fontsize = 15);
plt.ylabel("% of Passenger Survived", fontsize = 15);
labels = ['Upper', 'Middle', 'Lower']
#val = sorted(train.Pclass.unique())
val = [0,1,2] ## this is just a temporary trick to get the label right. 
plt.xticks(val, labels);
