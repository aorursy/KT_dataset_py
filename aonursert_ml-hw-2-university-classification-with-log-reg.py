# Classification of Universities into two groups, Private and Public.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("../input/us-news-and-world-reports-college-data/College.csv", index_col=0)
df.head()
df.info()
df["Private"] = pd.get_dummies(df["Private"], drop_first=True) # Make private column numerical
df.head()
# It's looking good for classification
X = df.drop("Private", axis=1) # Features that I am going to use as labels
y = df["Private"] # The feature that I am going to predict
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # I split my data into train and test datas
from sklearn.linear_model import LogisticRegression
logm = LogisticRegression()
logm.solver = "liblinear" # Sci-kit learn changes, I have to specify
logm.fit(X_train, y_train) # I train/fit my data
predictions = logm.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
# This is a really good result actually.
from sklearn.model_selection import cross_val_score # 10 fold cross validation scroe
print(cross_val_score(logm, X, y, cv=10))
# It's a good result too