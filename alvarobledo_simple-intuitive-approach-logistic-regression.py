%matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv('../input/mushrooms.csv')
df.head()
df.describe()
df.drop(['veil-type'], axis=1, inplace=True)
for col in df.columns:
    if len(df[col].value_counts()) == 2:
        le = LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
df.head()
df = pd.get_dummies(df)
df.head()
y = df['class'].to_frame()
X = df.drop('class', axis=1)
y.head()
X.head()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=19)
logreg = LogisticRegression()
logreg.fit(X_train, y_train.values.ravel())
y_pred_test = logreg.predict(X_test)
print('Accuracy of Logistic Regression classifier on the test set: {:.2f}'.format(accuracy_score(y_test, y_pred_test)))
scores = cross_val_score(logreg, X_train, y_train.values.ravel(), cv=StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=19), scoring='accuracy')
print('Accuracy of Logistic Regression classifier using 10-fold cross-validation: {}'.format(scores.mean()))
features_coeffs = pd.DataFrame(logreg.coef_, columns=X.columns, index=['coefficients'])
features_coeffs.sort_values('coefficients', axis=1, ascending=False, inplace=True)
features_coeffs.T.head()
features_coeffs.T.tail()
def plot_features_containing(feature_name):
    categories = X.columns[X.columns.str.contains(feature_name)]
    edible_num = []
    poisonous_num = []
    for cat in categories:
        y[X[cat]==0]
        edible_count = sum((y[X[cat]==1]==0).values[:,0])
        poisonous_count = sum(X[cat]==1) - edible_count
        edible_num.append(edible_count)
        poisonous_num.append(poisonous_count)
    odor_df = pd.DataFrame(index=categories, columns=['edible', 'poisonous'])
    odor_df.edible = edible_num
    odor_df.poisonous = poisonous_num
    odor_df.plot(kind='bar')
plot_features_containing('odor')
plot_features_containing('spore-print-color')
plot_features_containing('cap-color')