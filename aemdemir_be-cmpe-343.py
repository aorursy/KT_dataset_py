# Importing

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Data shape
print(train.shape)
print(test.shape)
# Data size
print(train.size)
print(test.size)
# First n headers
n = 10
list(train.columns.values[0:n])
# First n data types
train.dtypes[0:n]
# Inspecting the count of null values
threshold = 0
null_counts = train.isnull().sum()

for column, count in null_counts.items():
    if count > threshold:
        print(column, count)
train.sample(5)
# Target
train.columns.values[-1]
# Describe target
train['SalePrice'].describe()
sns.set(style="white")

data = train.iloc[:, 1:]

# Compute the correlation matrix
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
_ = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Histogram of SalePrice
_ = sns.distplot(train['SalePrice'])
# Histogram of LotArea
_ = sns.distplot(train['LotArea'])
# Average Price per One Average Square Feet
priceMean = train['SalePrice'].mean()
lotMean = train['LotArea'].mean()

print('Mean of SalePrice: ${}'.format(priceMean), '\nMean of LotArea:', lotMean)
print('Average Price per One Average Square Feet: ${}'.format(priceMean/lotMean))
train['SalePrice'].kurt()
data = pd.concat([train['SalePrice'], train['LotArea']], axis=1)
data.plot.scatter(x='LotArea', y='SalePrice', ylim=(0,800000));
# Delete columns which have null values more than threshold.
threshold = 500

# Preprocessing train dataset
train_null_counts = train.isnull().sum()

print('Train')
print(train.shape)

for column, count in train_null_counts.items():
    if count > threshold:
        del train[column]
        print(column, 'is deleted, null count:', count)
        
print(train.shape)


# Preprocessing test dataset
test_null_counts = test.isnull().sum()

print('\nTest')
print(test.shape)

for column, count in test_null_counts.items():
    if count > threshold:
        del test[column]
        print(column, 'is deleted, null count:', count)
        
print(test.shape)
# Conversion from object to categoric data type just for whole train dataset.

# Train dataset
print('Train')
print(train.dtypes[0:n], '\n')

for column in train.columns.values:
    if train[column].dtype == np.dtype(object):
        train[column] = train[column].astype('category')
        
print(train.dtypes[0:n])


# Test dataset
print('\nTest')
print(test.dtypes[0:n], '\n')

for column in test.columns.values:
    if test[column].dtype == np.dtype(object):
        test[column] = test[column].astype('category')
        
print(test.dtypes[0:n])
# Fill NaN records with their corresponding column's mean for numeric data.

# Train dataset
print('Train')
print(train.shape)

train = train.fillna(train.mean())

# Remove rows that contain missing values for categoric data.
train = train.dropna(axis=0)

print(train.shape)

train.isnull().sum()[0:n]


# Test dataset
print('\nTest')
print(test.shape)

test = test.fillna(test.mean())

# Remove rows that contain missing values for categoric data.
# test = test.dropna(axis=0)

print(test.shape)

test.isnull().sum()[0:n]
# Divide dataset into numeric and object for future use.
df_numeric = train.copy(deep=True)
df_categoric = train.copy(deep=True)

for column in train.columns.values:
    if train[column].dtype.name == 'category':
        del df_numeric[column]
    else:
        del df_categoric[column]

print(df_numeric.shape)
df_numeric.dtypes[0:n]
train = pd.get_dummies(train)
test = pd.get_dummies(test)
# For further ease let's move 'SalePrice' column to the index 1.
print('Before')
print('MSSubClass at index:', train.columns.get_loc('MSSubClass'))
print('SalePrice at index:', train.columns.get_loc('SalePrice'))

cols = list(train)
cols[1], cols[37] = cols[37], cols[1]

train = train.loc[:, cols]

print('\nAfter')
print('MSSubClass at index:', train.columns.get_loc('MSSubClass'))
print('SalePrice at index:', train.columns.get_loc('SalePrice'))
#from sklearn.preprocessing import MinMaxScaler

#mms = MinMaxScaler()
#X_train_norm = mms.fit_transform(X_train)
#X_test_norm = mms.transform(X_test)
#from sklearn.preprocessing import StandardScaler

#stdsc = StandardScaler()
#X_train_std = stdsc.fit_transform(X_train)
#X_test_std = stdsc.transform(X_test)
# k is the starting point for slicing train's columns
start = 2
end = 38

X_f, y_f = train.iloc[:, start:end].values, train['SalePrice'].values

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_f, y_f, test_size=0.3, random_state=1)
from sklearn.ensemble import RandomForestClassifier

feat_labels = train.columns[start:]

forest = RandomForestClassifier(n_estimators=500, random_state=1)

forest.fit(X_train_f, y_train_f)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train_f.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train_f.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X_train_f.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train_f.shape[1]])
plt.tight_layout()
#plt.savefig('images/feature_importance.png', dpi=300)
plt.show()
# A helper function in order to calculate performance of prospective predictions.
def calc_performance(predictions, y_test, test_name):
    errors = abs(predictions - y_test)
    print('{}: mean absolute error: ${}'.format(test_name, round(np.mean(errors), 2)))

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)

    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('{}: accuracy: {}%'.format(test_name, round(accuracy, 2)))
# Indices of important features
indices
# The two most important features
feat_labels[indices[0:2]]
df_train_best = train[feat_labels[indices[0:2]]]
df_train_best.head()
# Dividing feature and target columns. 
X_best, y_best = df_train_best.iloc[:, 0:train.shape[1]].values, train['SalePrice'].values

# Spliting to train and test data.
X_train_best, X_test_best, y_train_best, y_test_best = train_test_split(X_best, y_best, test_size=0.3, random_state=42)

# Create a random forest classifier by setting criterion as gini index.
rf_best = RandomForestClassifier(criterion='gini', 
                            n_estimators=128, 
                            random_state=42, 
                            n_jobs=2)

# Train the model.
_ = rf_best.fit(X_train_best, y_train_best)

# Prediction on X_test data.
predictions_best = rf_best.predict(X_test_best)

# Measure performance
calc_performance(predictions_best, y_test_best, 'Best Dataset')
# Dividing feature and target columns. 
X, y = train.iloc[:, 2:train.shape[1]].values, train['SalePrice'].values

# Spliting to train and test data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a random forest classifier by setting criterion as gini index.
rf = RandomForestClassifier(criterion='gini', 
                            n_estimators=128, 
                            random_state=42, 
                            n_jobs=2)

# Train the model.
_ = rf.fit(X_train, y_train)

# Prediction on X_test data.
predictions = rf.predict(X_test)

# Measure performance
calc_performance(predictions, y_test, 'All Dataset')
from sklearn import linear_model


# Create linear regression object
regr = linear_model.LinearRegression()


# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
linear_predictions = regr.predict(X_test)

linear_predictions

calc_performance(linear_predictions, y_test, 'Linear Prediction: All Dataset')
