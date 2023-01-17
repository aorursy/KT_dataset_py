import pandas as pd
import numpy as np
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
target = train['target'].values
train.info()
test.info()
np.sum(train.iloc[:, 2:].nunique() == 1)
np.sum(test.iloc[:, 1:].nunique() == 1)
import matplotlib.pyplot as plt
%matplotlib inline
RANDOM_CUSTOMERS = [30, 55, 67]
def plot_customers(rand_cust):
    for i in rand_cust:
        plt.plot(train.iloc[i].values)
        plt.title("For {}th customer target value: {}".format(i, target[i]))
        plt.xlabel("Features")
        plt.ylabel("Values")
        plt.show()
plot_customers(RANDOM_CUSTOMERS)
unique_df = train.nunique().reset_index()
unique_df.columns = ['col_name', 'unique_count']
constant_df = unique_df[unique_df["unique_count"]==1]
train = train.drop(constant_df['col_name'].values, axis=1)
plot_customers(RANDOM_CUSTOMERS)
TRESHOLD = 0.98
cols_to_drop = [col for col in train.columns[2:]
                    if [i[1] for i in list(train[col].value_counts().items()) 
                    if i[0] == 0][0] >= train.shape[0] * TRESHOLD]

exclude = ['ID', 'target']
train_features = []
for c in train.columns:
    if c not in cols_to_drop and c not in exclude:
        train_features.append(c)
print("Number of training features after dropping values: {}".format(len(train_features)))
train, test = train[train_features], test[train_features]
print("Train shape: {}\nTest shape: {}".format(train.shape, test.shape))
plot_customers(RANDOM_CUSTOMERS)