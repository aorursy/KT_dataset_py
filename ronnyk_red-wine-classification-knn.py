# Import useful libaries
import pandas as pd
import numpy as np
# Import dataset from csv.
dataframe = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009",engine='python')
# show first rows
dataframe.head(5)
# Count rows and columns
dataframe.shape
# Count values
dataframe.count()
# Use some statistics for better understanding
dataframe.describe()
# Checking for NULL
dataframe.isnull().sum()
# Understand correlation between these features
import seaborn as sns
corrmat = dataframe.corr()
sns.heatmap(corrmat, annot=True)
# Separate label and features
X = dataframe.drop("quality",axis=1)
y = dataframe["quality"]
# Check selection
X.head()
# Split test- and traindata
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
# Scale the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Import KNN and train classifier with one neighbor
from sklearn.neighbors import KNeighborsClassifier as KNN
clf = KNN(1)
clf.fit(X_train, y_train)
# Evaluate classification
clf.score(X_train, y_train)
# Predict testset
y_pred = clf.predict(X_test)
# Evaluate classification with accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))