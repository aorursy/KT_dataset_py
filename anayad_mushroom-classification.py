# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import classification_report, recall_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

sns.set(style="darkgrid")

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load Data

mushroom_data = pd.read_csv("../input/mushroom-classification/mushrooms.csv")
mushroom_data.info()
# First Look at the data

mushroom_data.head(10)
# Check for mising values

mushroom_data.isna().sum()

# No missing data
# Distribution of classes

plt.figure(figsize=(7, 6))

sns.countplot(x="class", data=mushroom_data)
# Attribute Count for each feature in dataset

plt.figure(figsize=(25, 15)).tight_layout(pad=3.0)

cols = list(mushroom_data.columns)[1:]

for i in range(len(mushroom_data.columns)-1):

  plt.subplot(4, 6, i+1)

  sns.countplot(x=cols[i], data=mushroom_data)

#Attribute count w.r.t Class for each feature in the dataset

plt.figure(figsize=(25, 15)).tight_layout(pad=3.0)

cols = list(mushroom_data.columns)[1:]

for i in range(len(mushroom_data.columns)-1):

  plt.subplot(4, 6, i+1)

  sns.countplot(x=cols[i], hue="class", data=mushroom_data)
mushroom_processed = mushroom_data.copy(deep=True)
lb = LabelEncoder()

target = lb.fit_transform(mushroom_processed["class"].tolist())

print(lb.classes_)

# Poisonous - 1, Edible - 0

mushroom_processed.drop("class", axis=1, inplace=True)
for col in list(mushroom_processed.columns):

  mushroom_processed[col] = lb.fit_transform(mushroom_processed[col].tolist())

  print("Label Num for "+col+": ", lb.classes_)

mushroom_processed.head()
# Get Dummy Variables

mushroom_all_dummy = pd.get_dummies(mushroom_processed, columns=list(mushroom_processed.columns))

mushroom_all_dummy.head()
# Train - Test Split

trainx, testx, trainy, testy = train_test_split(mushroom_all_dummy.to_numpy(), target, test_size=0.25, random_state=4199)
lr = LogisticRegression()

lr.fit(trainx, trainy)
print("Accuracy for Logistic Regression Classifier: {}".format(lr.score(testx, testy)))
# Classification Report 

preds = lr.predict(testx)

print(classification_report(testy, preds))
lr_coeffs = lr.coef_

all_cols = list(mushroom_all_dummy.columns)

sort_args = np.argsort(lr_coeffs)
# Visualizing LR Coeffs



plt.figure(figsize=(20, 10))

plt.bar(x=list(range(lr.coef_.shape[1])), height=lr.coef_[0, sort_args[0]])

plt.xticks(ticks = list(range(lr.coef_.shape[1])), labels=[all_cols[i] for i in sort_args[0]], rotation=90)

plt.xlabel("Features")

plt.ylabel("Coefficient")

plt.title("LR Coefficients")
rf = RandomForestClassifier(n_estimators=10)

rf.fit(trainx, trainy)
print("Accuracy for Random Forest Classifier: {}".format(rf.score(testx, testy)))
rf_fi = rf.feature_importances_

rf_sort = np.argsort(rf_fi)
# Visualizing Feature Importances for Random Forest Classifier



plt.figure(figsize=(10, 20))

plt.barh(y=list(range(rf.feature_importances_.shape[0])), width=rf.feature_importances_[rf_sort])

plt.yticks(ticks = list(range(rf.feature_importances_.shape[0])), labels=[all_cols[i] for i in rf_sort], rotation=0)

plt.ylabel("Features")

plt.xlabel("Importance")

plt.title("Feature Importances for Random Forest Classifier")
dt = DecisionTreeClassifier()

dt.fit(trainx, trainy)
print("Accuracy for Decision Tree Classifier: {}".format(dt.score(testx, testy)))
# Visualizing Feature Importances for Decision Tree Classifier



plt.figure(figsize=(20, 10))

plt.bar(x=list(range(dt.feature_importances_.shape[0])), height=dt.feature_importances_[:])

plt.xticks(ticks = list(range(dt.feature_importances_.shape[0])), labels=list(mushroom_all_dummy.columns), rotation=90)

plt.xlabel("Features")

plt.ylabel("Importance")

plt.title("Feature Importances for Decision Tree Classifier")