# Import libraries

import pandas as pd
import numpy as np

# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
%matplotlib inline
sns.set(style='white', context='notebook', palette='deep') 
import matplotlib.style as style
style.use('fivethirtyeight')
# Import the data

train = pd.read_csv('../input/train.csv', index_col='ID')
#test = pd.read_csv('../input/test.csv', index_col='ID')
print('train shape:', train.shape)
print('test shape:', test.shape)
train.head()
test.head()
# Capture the necessary data
variables = train.columns

count = []

for variable in variables:
    length = train[variable].count()
    count.append(length)
    
count_pct = np.round(100 * pd.Series(count) / len(train), 2)
count = pd.Series(count)

missing = pd.DataFrame()
missing['variables'] = variables
missing['count'] = len(train) - count
missing['count_pct'] = 100 - count_pct
missing = missing[missing['count_pct'] > 0]
missing.sort_values(by=['count_pct'], inplace=True)
missing_train = np.array(missing['variables'])

#Plot number of available data per variable
plt.subplots(figsize=(15,6))

# Plots missing data in percentage
plt.subplot(1,2,1)
plt.barh(missing['variables'], missing['count_pct'])
plt.title('Count of missing training data in percent', fontsize=15)

# Plots total row number of missing data
plt.subplot(1,2,2)
plt.barh(missing['variables'], missing['count'])
plt.title('Count of missing training data as total records', fontsize=15)

plt.show()
# Capture the necessary data
variables = test.columns

count = []

for variable in variables:
    length = test[variable].count()
    count.append(length)
    
count_pct = np.round(100 * pd.Series(count) / len(test), 2)
count = pd.Series(count)

missing = pd.DataFrame()
missing['variables'] = variables
missing['count'] = len(test) - count
missing['count_pct'] = 100 - count_pct
missing = missing[missing['count_pct'] > 0]
missing.sort_values(by=['count_pct'], inplace=True)
missing_test = np.array(missing['variables'])

#Plot number of available data per variable
plt.subplots(figsize=(15,6))

# Plots missing data in percentage
plt.subplot(1,2,1)
plt.barh(missing['variables'], missing['count_pct'])
plt.title('Count of missing test data in percent', fontsize=15)

# Plots total row number of missing data
plt.subplot(1,2,2)
plt.barh(missing['variables'], missing['count'])
plt.title('Count of missing test data as total records', fontsize=15)

plt.show()
unique_df = train.nunique().reset_index()
unique_df.columns = ["col_name", "unique_count"]
constant_df = unique_df[unique_df["unique_count"]==1]
constant_df.shape
# Delete columns with constant values

train = train.drop(constant_df.col_name.tolist(), axis=1)
dtype = train.dtypes.reset_index()
dtype.columns = ["Count", "Column Type"]
dtype.groupby("Column Type").aggregate('count').reset_index()
sns.set()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#plt.figure(figsize=(10,6))
ax = axes[0]
sns.distplot(train['target'], ax=ax)
ax.set_title('Histogram of Target')

ax = axes[1]
sns.boxplot(data=train, x='target', ax=ax)
ax.set_title('Boxplot of Target')

plt.show()
plt.figure(figsize=(14,5))
sns.distplot(np.log1p(train['target']))
plt.title('Histogram of Log of Target')
plt.show()
# Scale the data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
labels = []
values = []
for col in train_scaled.columns:
    if col != 'target':
        labels.append(col)
        values.append(np.corrcoef(train_scaled[col].values, train_scaled['target'].values)[0,1])
corr = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr = corr.sort_values(by='corr_values')

cols_to_use = corr[(corr['corr_values']>0.25) | (corr['corr_values']<-0.25)].col_labels.tolist()

temp_df = train_scaled[cols_to_use]
corrmat = temp_df.corr(method='spearman')

# Generate a mask for the upper triangle
mask = np.zeros_like(corrmat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(14, 14))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrmat, mask=mask, 
            square=True, linewidths=.5, annot=False, cmap=cmap)
plt.yticks(rotation=0)
plt.title("Correlation Matrix of Most Correlated Features", fontsize=15)
plt.show()