# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/carpriceprediction/data.csv')
data.head()
print(data.values.shape)
print(data.keys())
data = data.dropna()
data = data.reset_index(drop=True)
data = data.drop(columns=['Model', 'Make'])
print(data.values.shape)
data.head()
feature = 'Market Category'
unq_vals = []
new_data = []
for val in data[feature]:
    tmp = val.split(sep=',')
    for i in tmp:
        if i not in unq_vals:
            unq_vals.append(i)
print(unq_vals, len(unq_vals), sep='\n')
count = 0

# for i in range(data.shape[0]):
#     vals = data[feature].values[i].split(',')
#     print(vals, [data[feature][i]])

for val in data[feature].values:
    tmp_data = [0 for i in range(len(unq_vals))]
    tmp = val.split(sep=',')
    
    for i in range(len(tmp)):
        for j in range(len(unq_vals)):
            if tmp[i] == unq_vals[j]:
                tmp_data[j] = 1.0
    new_data.append(tmp_data)
    count += 1
new_data = np.asarray(new_data)
print(new_data.shape)
new_df = pd.DataFrame(new_data, columns=unq_vals)
data = data.join(new_df)
data = data.drop(columns=['Market Category'])
data.head()
feature = 'Year'
bin1 = [1990, 2000]
bin2 = [2000, 2005]
bin3 = [2005, 2010]
bin4 = [2010, 2015]
bin5 = [2015, 2017]
bins = [bin1, bin2, bin3, bin4, bin5]
year_bins = []
for i in range(data.shape[0]):
    tmp = [0 for i in range(len(bins))]
    ye = data['Year'][i]
    for j in range(len(bins)):
        if ye >= bins[j][0] and ye < bins[j][1]:
            tmp[j] = 1.0
    year_bins.append(tmp)
year_bins = np.asarray(year_bins)
new_df = pd.DataFrame(year_bins, columns=['Year_Bin'+str(i+1) for i in range(len(bins))])
data = data.join(new_df)
data = data.drop(columns=['Year'])
data.head()
non_numeric_features = []
numeric_features = []
for i in range(len(data.keys())):
    if isinstance(data[data.keys()[i]].values[1], str):
        non_numeric_features.append(data.keys()[i])
    else:
        numeric_features.append(data.keys()[i])
    
print('String Keys - ', non_numeric_features, sep='\n')
print('Non-String Keys - ', numeric_features, sep='\n')
def print_bar(data, feature):
    print(feature, ' : ', len(data[feature].unique()))
    print(data.values.shape)
    counts = data[feature].value_counts()
    print(data[feature].unique())
    print('Min : {} \t Max : {}'.format(min(data[feature].unique()), max(data[feature].unique())))
    print(counts.values)
    plt.bar((['f'+str(i) for i in range(len(data[feature].unique()))]), counts.values)
    plt.title(feature)
    plt.show()
features = ['Year_Bin'+str(i+1) for i in range(len(bins))]
for feat in features:
    print_bar(data, feat)
features = ['Factory Tuner', 'Luxury', 'High-Performance', 'Performance', 'Flex Fuel', 'Hatchback', 'Hybrid', 'Diesel', 'Exotic', 'Crossover']
for val in features:
    print_bar(data, val)
for feat in non_numeric_features:
    print_bar(data, feat)

norm_features = ['Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg', 'Popularity']
for feat in norm_features:
    data[[feat]] = data[[feat]]/data[[feat]].mean()
mean_y_true = data[['MSRP']].mean()
data[['MSRP']] = data[['MSRP']] / mean_y_true
data[['Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg', 'Popularity', 'MSRP']].describe()

data = pd.get_dummies(data, dummy_na=False, columns=['Engine Fuel Type', 'Transmission Type', 
                                               'Driven_Wheels', 'Vehicle Size', 'Vehicle Style', 'Number of Doors'])
print(data.shape)
for key in data.keys():
    print(key)
data.head()
