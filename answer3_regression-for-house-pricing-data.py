# libraries import

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
# Dataset import



dataset = pd.read_csv('../input/HousePrices_HalfMil.csv')

print('Training data shape: ', dataset.shape)

dataset.head(20)
missing_count = dataset.isnull().sum()

missing_percent = 100 * dataset.isnull().sum() / len(dataset)

mis_val_table = pd.concat([missing_count, missing_percent], axis=1)

        

print("--- Missing values count ---")

print(missing_count)

print("--- Missing values percentage ---")

print(missing_percent)

dataset.describe()
# Histogram Area

plt.hist(dataset['Area'], bins = 5, rwidth=0.8)

plt.ylabel('Count')

plt.xlabel('Area')

plt.show()



dataset['Area'].groupby(pd.cut(dataset['Area'], [0,50,100,150,200,250])).count()
dataset = dataset[dataset.Area > 50]

dataset.head(10)
# Prices histogram

plt.hist(dataset['Prices'], bins = 5, rwidth=0.8)

plt.ylabel('Count')

plt.show()

print(dataset['Prices'].describe())

dataset['Prices'].groupby(pd.cut(dataset['Prices'], list(range(0, 80000+1, (80000)//5)))).count()
# Histograms for rest columns

rest_data = dataset.iloc[:, 1:-1]

hist_per_row = 4 

unique_arr = rest_data.nunique()

n_row, n_col = rest_data.shape

column_names = list(rest_data)

n_hist_row = (n_col + hist_per_row - 1) / hist_per_row

plt.figure(num=None, figsize=(6*hist_per_row, 8*n_hist_row), dpi=80, facecolor='w', edgecolor='k')

for i in range(n_col):

    plt.subplot(n_hist_row, hist_per_row, i+1)

    rest_data.iloc[:,i].hist()

    plt.ylabel('counts')

    plt.xticks(rotation=90)

    plt.title(f'{column_names[i]} (column {i})')

plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plt.show()

print(rest_data['FirePlace'].nunique())

for cname in list(rest_data):

    print(rest_data[cname].groupby(pd.cut(rest_data[cname], bins=rest_data[cname].nunique())).count())
sample_set = dataset.sample(n=3000, random_state=42)
dfDummies = pd.get_dummies(sample_set['City'], prefix = 'City')

sample_set = pd.concat([dfDummies, sample_set], axis=1).drop(['City'], axis=1)

sample_set = sample_set.drop(['City_1'], axis=1)

sample_set.head(20)
import seaborn as sns

corr_matrix = sample_set.corr().abs().astype(float)

plt.figure(figsize=(20,16))

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
sns.pairplot(sample_set, diag_kind='kde', palette='husl')
X = sample_set.iloc[:, 0:16].values

y = sample_set.iloc[:, 16].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Fitting Multiple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



# Predicting the Test set results

y_pred = regressor.predict(X_test)



from sklearn.metrics import mean_squared_error, r2_score

print("Mean squared error: %.2f" % mean_squared_error(y_pred, y_test))

print('R-sqared: %.2f' % r2_score(y_pred, y_test))



fig, ax = plt.subplots()

ax.scatter(y_test, y_pred)

ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

ax.set_xlabel('measured')

ax.set_ylabel('predicted')

plt.show()