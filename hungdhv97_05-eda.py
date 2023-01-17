import numpy as np 

import pandas as pd

from sklearn.impute import SimpleImputer

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
data_file_path = "../input/123456/05-CYP2C9_Inhibition.csv"

data = pd.read_csv(data_file_path, index_col = "No.")

data.head()
# Split train data X and label y

features = [col for col in data.columns if col not in ['NAME', 'CLASS', 'nHBonds']]

X = data[features]

y = data.CLASS
y
X.isnull().sum()
# Fill missing data (NaN)

my_imputer = SimpleImputer(strategy = 'most_frequent')

imputed_X = pd.DataFrame(my_imputer.fit_transform(X))

imputed_X.columns = X.columns

X = imputed_X

X.head()
X.describe()
X.info(verbose = True)
X.isnull().sum()
count_categorical_col = 0

dict_categorical_col = {}

for col in X.columns:

    if len(X[col].unique()) < 50:

        dict_categorical_col[col] = len(X[col].unique())

        count_categorical_col += 1

        

print(X.shape[1] - count_categorical_col, count_categorical_col)

print(dict_categorical_col)
len([X[col].nunique() for col in X.columns if X[col].nunique() == 2])
X.dtypes
VarianceThreshold(threshold = 0.4).fit_transform(X).shape
dict_categorical_col.values()
def count_frequency(dic):

    freq = {}

    for item in dic.values():

        if item in freq:

            freq[item] += 1

        else:

            freq[item] = 1

    return freq
freq_col = count_frequency(dict_categorical_col)
fig, ax0 = plt.subplots(figsize = (10, 5))

unique_value_column = freq_col.keys()

frequency = freq_col.values()

ax0.bar(unique_value_column, frequency)

ax0.set_title("Frequency of the sum of unique value of each column")

ax0.set_xlabel("Sum of unique value of each column")

ax0.set_ylabel("Frequency")
X.iloc[:100,:10].plot.line(title = "C Dataset", figsize = (15, 10))
X.iloc[:100, 0].plot.hist()
X.iloc[:100,:4].plot.hist(subplots = True, figsize = (15, 10), bins = 50, layout = (2, 2))
X.iloc[:100, 3650].value_counts().sort_index().plot.bar()
sns.distplot(X.iloc[:, 0], bins = 20, kde = True)
sns.countplot(X.iloc[:100, 3650])
sns.boxplot('F10[N-N]', 'Se', data = X)
sns.heatmap(X.iloc[:100,:10].corr(), annot = True)
sns.FacetGrid(X.iloc[:100], col = 'F10[N-N]').map(sns.kdeplot, 'Se')
sns.pairplot(X.iloc[:100, :10])
X1 = X

X1.index = y.index

X1['CLASS'] = y

X1
idx_col = list(np.arange(10,20))

idx_col.append(-1)

sns.pairplot(X1.iloc[100:200,idx_col], hue = 'CLASS')