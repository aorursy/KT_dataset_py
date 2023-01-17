!pip install scikit-learn==0.21.3

!pip install imbalanced-learn==0.4.3
import matplotlib.pyplot as plt

import seaborn as sns



import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, classification_report

from sklearn.datasets import make_classification



from imblearn.over_sampling import RandomOverSampler, SMOTE

from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours

from imblearn.combine import SMOTETomek, SMOTEENN

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,

                           n_redundant=0, n_repeated=0, n_classes=3,

                           n_clusters_per_class=1,

                           weights=[0.06, 0.02, 0.92],

                           class_sep=0.8, random_state=0)



colors = ['#4E6B8A' if v == 0 else '#F26419' if v == 1 else '#F6AE2D' for v in y]

fig = plt.Figure(figsize=(12,8))

plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='grey', linewidths=0.5)

sns.despine()
ros = RandomOverSampler(random_state=0, sampling_strategy={0: 300, 1: 300})

ros.fit(X, y)

X_resampled, y_resampled = ros.fit_resample(X, y)

colors = ['#4E6B8A' if v == 0 else '#F26419' if v == 1 else '#F6AE2D' for v in y_resampled]

plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=colors, linewidth=0.5, edgecolor='grey')

sns.despine()

plt.title("Random Oversampling")
smote = SMOTE(random_state=0,  sampling_strategy={0: 300, 1: 500})

smote.fit(X, y)

X_resampled, y_resampled = smote.fit_resample(X, y)

colors = ['#4E6B8A' if v == 0 else '#F26419' if v == 1 else '#F6AE2D' for v in y_resampled]

plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=colors, linewidth=0.5, edgecolor='grey')

sns.despine()

plt.title("SMOTE")
smote = SMOTEENN(random_state=0)

smote.fit(X, y)

X_resampled, y_resampled = smote.fit_resample(X, y)

colors = ['#4E6B8A' if v == 0 else '#F26419' if v == 1 else '#F6AE2D' for v in y_resampled]

plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=colors, linewidth=0.5, edgecolor='grey')

sns.despine()

plt.title("SMOTEEEN")
rus = RandomUnderSampler(random_state=0)

rus.fit(X, y)

X_resampled, y_resampled = rus.fit_resample(X, y)

colors = ['#4E6B8A' if v == 0 else '#F26419' if v == 1 else '#F6AE2D' for v in y_resampled]

plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=colors, linewidth=0.5, edgecolor='grey')

sns.despine()

plt.title("Random Undersampling")
rus = NearMiss()

rus.fit(X, y)

X_resampled, y_resampled = rus.fit_resample(X, y)

colors = ['#4E6B8A' if v == 0 else '#F26419' if v == 1 else '#F6AE2D' for v in y_resampled]

plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=colors, linewidth=0.5, edgecolor='grey')

sns.despine()

plt.title("Near Miiss")
bank = pd.read_csv('../input/bank-marketing/bank.csv')
bank.shape
sns.countplot(x="y", data=bank)
bank.y.value_counts()
bank.head()
bank["default"] = bank["default"].map({"no":0,"yes":1})



bank["housing"] = bank["housing"].map({"no":0,"yes":1})



bank["loan"] = bank["loan"].map({"no":0,"yes":1})



bank["y"] = bank["y"].map({"no":0,"yes":1})



bank.education = bank.education.map({"primary": 0, "secondary":1, "tertiary":2})



bank.month = pd.to_datetime(bank.month, format = "%b").dt.month
# Let's remove a few features that are not really relevant for the purposes of our task

bank.drop(["poutcome", "contact"], axis = 1, inplace = True)

bank.dropna(inplace = True)
bank = pd.get_dummies(bank, drop_first = True)
# The final features

bank.columns
# Separate the target variable from the rest of the feautures



X = bank.drop("y", axis = 1)

y = bank.y
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state = 1, stratify=y)
# Sanity check for class distribution in train and test.

fig, axs = plt.subplots(1,2, figsize=(12,4))

sns.countplot(x='y', data=pd.DataFrame(y_train), ax=axs[0])

sns.countplot(x='y', data=pd.DataFrame(y_test), ax=axs[1])

axs[0].title.set_text('Train')

axs[1].title.set_text('Test')
y_test.value_counts()
# a helper function to draw confusion matrices

def draw_cm(y_test, y_pred):

  cm = confusion_matrix(y_test, y_pred)

  cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  df_cm = pd.DataFrame(cm_norm)

  plt.figure(figsize = (6,4))

  sns.heatmap(df_cm, annot=True, cmap="Blues")

  plt.xlabel("Predicted class")

  plt.ylabel("True class")

  plt.show()

  print("Accuracy: {0:.3f}".format(accuracy_score(y_test, y_pred)))

  print("Recall: {0:.3f}".format(recall_score(y_test, y_pred)))
lr = LogisticRegression(solver='liblinear',random_state=1)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("Accuracy: {0:.3f}".format(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

draw_cm(y_test, y_pred)
ros = RandomOverSampler(sampling_strategy='minority', random_state=1)



X_train_ros, y_train_ros = ros.fit_sample(X_train, y_train)

np.bincount(y_train_ros)
lr = LogisticRegression(solver='liblinear',random_state=1)

lr.fit(X_train_ros, y_train_ros)



y_pred = lr.predict(X_test)
draw_cm(y_test, y_pred)
from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=1)



X_train_sm, y_train_sm = smt.fit_sample(X_train, y_train)

np.bincount(y_train_sm)
lr = LogisticRegression(solver='liblinear',random_state=1)

lr.fit(X_train_sm, y_train_sm)



y_pred = lr.predict(X_test)
draw_cm(y_test, y_pred)
rus = RandomUnderSampler(sampling_strategy='majority', random_state=1)



X_train_rus, y_train_rus = rus.fit_sample(X_train, y_train)

np.bincount(y_train_rus)
lr = LogisticRegression(solver='liblinear',random_state=1)

lr.fit(X_train_rus, y_train_rus)



y_pred = lr.predict(X_test)
draw_cm(y_test, y_pred)
st = SMOTETomek(random_state=1)

X_train_st, y_train_st = st.fit_sample(X_train, y_train)

np.bincount(y_train_st)
lr = LogisticRegression(solver='liblinear',random_state=1)

lr.fit(X_train_st, y_train_st)



y_pred = lr.predict(X_test)
draw_cm(y_test, y_pred)