import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
pd.set_option('max_columns', None)
import catboost
from catboost import CatBoostClassifier
from catboost import Pool
df_train = pd.read_csv('../input/cat-in-the-dat/train.csv')
df_test = pd.read_csv('../input/cat-in-the-dat/test.csv')
df_train.head()
X = df_train.drop(columns=['id', 'target'])
y = df_train['target']
test = df_test.drop(columns=['id'])
labels = X.columns
IDs = df_test['id']
print("Training set shape: {} \nTest set shape: {}".format(X.shape, test.shape))
bin_cols = [col for col in X.columns.values if col.startswith('bin')]
num_cols = [col for col in X.columns.values if col.startswith('nom')]
ord_cols = [col for col in X.columns.values if col.startswith('ord')]
tim_cols = [col for col in X.columns.values if col.startswith('day') or col.startswith('month')]
bin_cols
X.nunique()
# Count of the dtypes for each column in our training set
X.dtypes.value_counts()
X.dtypes
# Finding and plotting the count of the target variable
counts = y.value_counts()
plt.bar(counts.index, counts)
plt.gca().set_xticks([0,1])
plt.title('Distribution of Target Variable')
plt.show()
counts
# The dataset is imbalanced
def logistic(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LogisticRegression(solver='liblinear')
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    print('Accuracy : ' , accuracy_score(y_test, pred))
# Using label encoding to convert categorical variables in the training set to numerical variables
encoder = LabelEncoder()
train = pd.DataFrame()
for col in X.columns:
    if (X[col].dtype == "object"):
        train[col] = encoder.fit_transform(X[col])
    else:
        train[col] = X[col]
train.head()
# All dtypes are now int64
train.dtypes.value_counts()
logistic(train, y)
# Using one hot encoding to convert categorical variables in the training set to numerical variables
one = OneHotEncoder(handle_unknown="ignore")
one.fit(X)
df_train = one.transform(X)
df_test = one.transform(test)
# df_train is a sparse matrix, which is the default type returned with OneHotEncoding
type(df_train)
# Using one hot encoding added a lot of features to our training set, as would be expected
print('train data set has {} rows and {} columns'.format(df_train.shape[0],df_train.shape[1]))
logistic(df_train, y)
# Accuracy of logistic regression improves with one hot encoding compared to label encoding
X_train_hash = X.copy()
for c in X.columns:
    X_train_hash[c]=X[c].astype('str')
hashing=FeatureHasher(input_type='string')
train = hashing.transform(X_train_hash.to_numpy())
print("Train shape: {}".format(train.shape))
logistic(train, y)
# Accuracy is better than with label encoding, but not quite as good as one hot encoding
cat_features = list([])
#cat_features.append(X.columns.get_loc(c)) for c in labels if c.dtypes == object
for column in labels:
    if X[column].dtype == 'object':
        cat_features.append(labels.get_loc(column))
cat_features
def cross_entropy(known, predicted):
    ce_array = np.average(-known *np.log(predicted) - (1-known) * np.log(1-predicted))
    return np.average(ce_array)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=1)
train_pool = Pool(
    data = X_train,
    label = y_train,
    cat_features = cat_features
    )

validation_pool = Pool(
    data = X_validation,
    label = y_validation,
    cat_features = cat_features
    )

all_pool = Pool(
    data = X,
    label = y,
    cat_features = cat_features
    )

test_pool = Pool(
    data = test,
    cat_features = cat_features)
cat_model = CatBoostClassifier(iterations=500, learning_rate = .01, l2_leaf_reg=1, loss_function='CrossEntropy')
cat_model.fit(train_pool, eval_set = validation_pool, verbose=100)
pred = cat_model.predict(test_pool)
submission = pd.DataFrame(IDs, columns = ['id'])
submission['target'] = pred
submission
submission.to_csv('submission.csv', index = False)
reg = LogisticRegression(solver = 'liblinear')
reg.fit(df_train, y)
# Fitting the final classifier using the one hot encoding dataframe
pred = reg.predict(df_test)
submission = pd.DataFrame(IDs, columns=['id'])
submission['target'] = pred
submission
submission.to_csv('submission.csv', index=False)