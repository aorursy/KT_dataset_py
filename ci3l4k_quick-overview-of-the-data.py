import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
import random
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from itertools import combinations
from sklearn.metrics import r2_score

%matplotlib inline
df = pd.read_csv("../input/train.csv")
df.head(10)
df.shape
df.describe()
df.isnull().values.any()
df.isnull().sum().sum()
df.dtypes.value_counts().plot.bar()
df.ID.value_counts(ascending=False).head()
df.columns[2:].value_counts(ascending=False).head()
set(df.ID.tolist()).intersection(df.columns[2:].tolist())
y = pd.DataFrame(df["target"])
X = df.drop(["target", "ID"], axis=1, )
plt.figure(figsize=(16, 10))
plt.scatter(y.target.index, y.target)
#plt.title('target - value count')
plt.xlabel('client')
plt.ylabel('target value')
plt.figure(figsize=(16, 10))
plt.scatter(y.target.value_counts().index, y.target.value_counts().values)
plt.title('target - value count')
plt.xlabel('value')
plt.ylabel('count')
plt.figure(figsize=(16, 10))
plt.hist(y.target, )
y_sorted = y.target.sort_values()

plt.figure(figsize=(16, 10))
plt.scatter(np.arange(0, len(y_sorted), 1), y_sorted)
# Log-transform 
y_log_transformed = y.target.apply(lambda x: np.log1p(x))
plt.figure(figsize=(16, 10))
plt.scatter(y_log_transformed.index, y_log_transformed.values)
#plt.title('target - value count')
plt.xlabel('client')
plt.ylabel('target value')
plt.figure(figsize=(16, 10))
plt.scatter(y_log_transformed.value_counts().index, y_log_transformed.value_counts().values)
plt.title('target - value count')
plt.xlabel('value')
plt.ylabel('count')
plt.figure(figsize=(16, 10))
plt.hist(y_log_transformed.values, )
scalar = preprocessing.StandardScaler().fit(pd.DataFrame(y_log_transformed))
y_log_normalized = pd.DataFrame(scalar.transform(pd.DataFrame(y_log_transformed)))
print(y_log_normalized.describe())
fig, axeslist = plt.subplots(ncols=3, nrows=1, figsize=(16, 10))
axeslist[0].scatter(y.index, y.values)
axeslist[0].set_title('Original data')
axeslist[1].scatter(y_log_transformed.index, y_log_transformed.values)
axeslist[1].set_title('Log1p data')
axeslist[2].scatter(y_log_normalized.index, y_log_normalized.values)
axeslist[2].set_title('Normalized log1p data')
#axeslist[2].scatter(y_log_normalized.index, y_log_normalized.values)
fig.tight_layout()
fig, axeslist = plt.subplots(ncols=3, nrows=1, figsize=(16, 10))
axeslist[0].hist(y.values)
axeslist[0].set_title('Original data')
axeslist[1].hist(y_log_transformed.values)
axeslist[1].set_title('Log1p data')
axeslist[2].hist(y_log_normalized.values)
axeslist[2].set_title('Normalized log1p data')

fig.tight_layout()
random_clients_no = list(np.random.choice(X.index, 9))
fig, axeslist = plt.subplots(ncols=3, nrows=3, figsize=(16, 16))
axeslist = axeslist.flatten()

for num, client_id in enumerate(random_clients_no):
    axeslist[num].set_title(str(client_id))
    axeslist[num].scatter(np.arange(0, len(X.columns), 1), X.iloc[client_id])

fig.tight_layout()
fig, axeslist = plt.subplots(ncols=3, nrows=3, figsize=(16, 16))
axeslist = axeslist.flatten()

for num, client_id in enumerate(random_clients_no):
    axeslist[num].set_title(str(client_id))
    axeslist[num].hist(X.iloc[client_id].values)
    
fig.tight_layout()
fig, axeslist = plt.subplots(ncols=3, nrows=3, figsize=(16, 16))
axeslist = axeslist.flatten()

for num, client_id in enumerate(random_clients_no):
    axeslist[num].set_title(str(client_id))
    axeslist[num].scatter(np.arange(0, len(X.iloc[client_id].values), 1), X.iloc[client_id].sort_values())
    
fig.tight_layout()
# getting number of nonzeros in each column is
non_zeros = X.astype(bool).sum(axis=0)
zeros = X.shape[1] - non_zeros
fig, axeslist = plt.subplots(ncols=2, nrows=1, figsize=(16, 10))
axeslist[0].scatter(np.arange(0, len(zeros),1), zeros)
axeslist[0].set_title('Number of columns values equal to zero')
axeslist[1].scatter(np.arange(0, len(non_zeros),1), non_zeros)
axeslist[1].set_title('Number of non zero columns values')
#axeslist[2].scatter(y_log_normalized.index, y_log_normalized.values)
fig.tight_layout()
plt.figure(figsize=(16, 10))
plt.scatter(np.arange(0, len(zeros), 1), zeros.sort_values(), label='Sorted number of values equal to zero')
plt.scatter(np.arange(0, len(non_zeros), 1), non_zeros.sort_values(), label='Sorted number of non zero values')
plt.grid()
plt.legend()
outsiders = non_zeros[non_zeros >= 1000].index
# source: https://seaborn.pydata.org/examples/many_pairwise_correlations.html
start = timeit.default_timer()

corr = X[outsiders].corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap,
            #vmax=.2, center=0,
            square=True, linewidths=.8, cbar_kws={"shrink": .8})

stop = timeit.default_timer()
print( 'Running time: ', np.round(stop-start, 2), ' sec.' )
# Produce a scatter matrix for each pair of features in the data
pd.plotting.scatter_matrix(corr, alpha=0.3, figsize=(16, 10), diagonal = 'kde');
below_one_hundred = non_zeros[non_zeros <= 100].index
middle_earth = non_zeros[(non_zeros > 100) & (non_zeros < 1000)].index
# Log-transform 
X_log_transformed = X.apply(lambda x: np.log1p(x))
fig, axeslist = plt.subplots(ncols=3, nrows=3, figsize=(16, 16))
axeslist = axeslist.flatten()

for num, client_id in enumerate(random_clients_no):
    axeslist[num].set_title(str(client_id))
    axeslist[num].scatter(np.arange(0, len(X_log_transformed.columns), 1), X_log_transformed.iloc[client_id])

fig.tight_layout()
fig, axeslist = plt.subplots(ncols=3, nrows=3, figsize=(16, 16))
axeslist = axeslist.flatten()

for num, client_id in enumerate(random_clients_no):
    axeslist[num].set_title(str(client_id))
    axeslist[num].hist(X_log_transformed.iloc[client_id].values)

fig.tight_layout()
X_log_transformed_stats = X_log_transformed.describe()
X_log_transformed_stats
fig, axeslist = plt.subplots(ncols=4, nrows=2, figsize=(16, 16))
axeslist = axeslist.flatten()

for num, title in enumerate(X_log_transformed_stats.index):
    axeslist[num].set_title(title)
    axeslist[num].scatter(np.arange(0, len(X_log_transformed_stats.columns), 1), X_log_transformed_stats.iloc[num])

fig.tight_layout()
X_max = X_log_transformed_stats.iloc[-1]
print(len(X_max[X_max == 0]))
X_zero_features = X_max[X_max == 0].index
X_log_transformed.drop(labels=X_zero_features, inplace=True, axis=1)
print(X_log_transformed.shape)
X_log_transformed_stats = X_log_transformed.describe()
fig, axeslist = plt.subplots(ncols=4, nrows=2, figsize=(16, 16))
axeslist = axeslist.flatten()

for num, title in enumerate(X_log_transformed_stats.index):
    axeslist[num].set_title(title)
    axeslist[num].scatter(np.arange(0, len(X_log_transformed_stats.columns), 1), X_log_transformed_stats.iloc[num])

fig.tight_layout()
scalar = preprocessing.StandardScaler().fit(pd.DataFrame(X_log_transformed))
X_log_normalized = pd.DataFrame(scalar.transform(pd.DataFrame(X_log_transformed)))
X_train, X_test, y_train, y_test = train_test_split(X_log_normalized, y_log_normalized, test_size=0.11, random_state=42)