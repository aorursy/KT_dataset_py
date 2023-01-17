# Import necessary libraries

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import emoji

from sklearn.model_selection import train_test_split, GridSearchCV

from catboost import CatBoostClassifier, Pool

from sklearn.metrics import confusion_matrix

from sklearn import preprocessing

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier



import warnings

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

warnings.filterwarnings(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
data = pd.read_csv('/kaggle/input/feb-2020-us-flight-delay/feb-20-us-flight-delay.csv')
data.head()
data = data.drop(['Unnamed: 9'], axis=1)
data['DEP_DEL15'].value_counts()
# Split the data into positive and negative

positive_rows = data.DEP_DEL15 == 1.0

data_pos = data.loc[positive_rows]

data_neg = data.loc[~positive_rows]



# Merge the balanced data

data = pd.concat([data_pos, data_neg.sample(n = len(data_pos))], axis = 0)



# Shuffle the order of data

data = data.sample(n = len(data)).reset_index(drop = True)
data.isna().sum()
data = data.dropna(axis=0)
data.info()
data['DEP_DEL15'] = data['DEP_DEL15'].astype(int)
print(f"There are {data.shape[0]} rows and {data.shape[1]} columns in our dataset.")
data.describe()
plt.figure(figsize=(15,5))

sns.distplot(data['DISTANCE'], hist=False, color="b", kde_kws={"shade": True})

plt.xlabel("Distance")

plt.ylabel("Frequency")

plt.title("Distribution of distance")

plt.show()
print(emoji.emojize("Let's find it out :fire:"))
print(f"Average distance if there is a delay {data[data['DEP_DEL15'] == 1]['DISTANCE'].values.mean()} miles")

print(f"Average distance if there is no delay {data[data['DEP_DEL15'] == 0]['DISTANCE'].values.mean()} miles")
plt.figure(figsize=(15,5))

sns.countplot(x=data['OP_UNIQUE_CARRIER'], data=data)

plt.xlabel("Carriers")

plt.ylabel("Count")

plt.title("Count of unique carrier")

plt.show()
plt.figure(figsize=(10,70))

sns.countplot(y=data['ORIGIN'], data=data, orient="h")

plt.xlabel("Airport")

plt.ylabel("Count")

plt.title("Count of Unique Origin Airports")

plt.show()
plt.figure(figsize=(10,70))

sns.countplot(y=data['DEST'], data=data, orient="h")

plt.xlabel("Airport")

plt.ylabel("Count")

plt.title("Count of Unique Destination Airports")

plt.show()
data = data.rename(columns={'DEP_DEL15':'TARGET'})
def label_encoding(categories):

    """

    To perform mapping of categorical features

    """

    categories = list(set(list(categories.values)))

    mapping = {}

    for idx in range(len(categories)):

        mapping[categories[idx]] = idx

    return mapping
data['OP_UNIQUE_CARRIER'] = data['OP_UNIQUE_CARRIER'].map(label_encoding(data['OP_UNIQUE_CARRIER']))
data['ORIGIN'] = data['ORIGIN'].map(label_encoding(data['ORIGIN']))
data['DEST'] = data['DEST'].map(label_encoding(data['DEST']))
data.head()
data['TARGET'].value_counts()
X = data[['DAY_OF_MONTH', 'DAY_OF_WEEK', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'DEP_TIME', 'DISTANCE']].values

y = data[['TARGET']].values
# Splitting Train-set and Test-set

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=41)



# Splitting Train-set and Validation-set

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=41)
# Formula to get accuracy

def get_accuracy(y_true, y_preds):

    # Getting score of confusion matrix

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_true, y_preds).ravel()

    # Calculating accuracy

    accuracy = (true_positive + true_negative)/(true_negative + false_positive + false_negative + true_positive)

    return accuracy
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0).fit(X_train, y_train)
# Initialize CatBoostClassifier

catboost = CatBoostClassifier(random_state=0)

catboost.fit(X_train, y_train, verbose=False)
gnb = GaussianNB()

gnb.fit(X_train, y_train)
rf = RandomForestClassifier(random_state=0)

rf.fit(X_train, y_train)
knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train, y_train)
xgb = XGBClassifier()

xgb.fit(X_train, y_train)
models = [lr, catboost, gnb, rf, knn, xgb]

acc = []

for model in models:

    preds_val = model.predict(X_val)

    accuracy = get_accuracy(y_val, preds_val)

    acc.append(accuracy)
model_name = ['Logistic Regression', 'Catboost', 'Naive Bayes', 'Random Forest', 'KNN', 'XGBoost']

accuracy = dict(zip(model_name, acc))
plt.figure(figsize=(15,5))

ax = sns.barplot(x = list(accuracy.keys()), y = list(accuracy.values()))

for p, value in zip(ax.patches, list(accuracy.values())):

    _x = p.get_x() + p.get_width() / 2

    _y = p.get_y() + p.get_height() + 0.008

    ax.text(_x, _y, round(value, 3), ha="center") 

plt.xlabel("Models")

plt.ylabel("Accuracy")

plt.title("Model vs. Accuracy")

plt.show()
test_preds = knn.predict(X_test)

get_accuracy(y_test, test_preds)
leaf_size = list(range(1,5))

n_neighbors = list(range(1,3))

p=[1,2]



hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)



knn_2 = KNeighborsClassifier()



clf = GridSearchCV(knn_2, hyperparameters, cv=2)



best_model = clf.fit(X_train,y_train)



print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])

print('Best p:', best_model.best_estimator_.get_params()['p'])

print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
knn_best = KNeighborsClassifier(leaf_size=3, p=1, n_neighbors=1)
knn_best.fit(X_train, y_train)

test_preds_1 = knn_best.predict(X_test)
get_accuracy(y_test, test_preds_1)