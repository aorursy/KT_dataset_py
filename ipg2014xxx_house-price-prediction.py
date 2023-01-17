import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
import seaborn as sns
import operator

# set random seed to get consistent results
np.random.seed(1)

% matplotlib notebook

sns.set_style('darkgrid')
X_train = pd.read_csv("../input/train.csv")
X_test = pd.read_csv("../input/test.csv")
X_train.head(7)
y_train = X_train['SalePrice']
X_train.drop(['SalePrice', 'Id'], axis=1, inplace=True)

test_id = X_test['Id']
X_test.drop(['Id'], axis=1, inplace=True)

print("Training Samples =",X_train.shape[0])
print("Testing Samples =",X_test.shape[0])
all_data = pd.concat([X_train, X_test])
null_values = all_data.isnull().sum()
sorted_null_values = sorted(null_values.items(), reverse=True, key=operator.itemgetter(1))

print("%15s| %5s|%12s" % ("Attribute", "Null_Val", "Data Type"))
print('-'*50)
for (k, v) in sorted_null_values:
    if v > 0:
        print("%15s %5d %12s" % (k, v, all_data[k].dtype))
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
sns.distplot(y_train, ax=ax)
ax.set_title('Distribution of house prices in training data')
plt.tight_layout()
y_train = np.log1p(y_train)
# Numerical Features

num_attr = all_data.dtypes[all_data.dtypes != "object"].index
skewed_attr = X_train[num_attr].apply(lambda x: skew(x.dropna()))
skewed_attr = skewed_attr[skewed_attr > 0.75]
skewed_attr = skewed_attr.index

all_data[skewed_attr] = np.log1p(all_data[skewed_attr])
# categorical features
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
all_data.head(7)
x_rows = X_train.shape[0]
X_train = all_data[:x_rows]
X_test = all_data[x_rows:]
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

model = LassoCV(alphas = [0.0003]).fit(X_train, y_train)
error = mean_squared_error(y_train, model.predict(X_train))
print("Mean Squared Error =", error)
y_test = np.expm1(model.predict(X_test))
solution = pd.DataFrame({"Id": test_id, "SalePrice": y_test})
solution.to_csv("submission.csv", index = False)
solution.head()