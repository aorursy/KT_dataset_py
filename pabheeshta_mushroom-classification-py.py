import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
style.use(['fivethirtyeight'])
shroom_DF = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
shroom_DF.head(15)
shroom_DF.shape
list(shroom_DF.columns)
shroom_DF.isnull().sum()
f, ax1 = plt.subplots(4, 2, figsize = (13,17))
sns.countplot(data = shroom_DF, x = "class", ax = ax1[0,0])
sns.countplot(data = shroom_DF, x = "cap-shape", ax = ax1[0,1])
sns.countplot(data = shroom_DF, x = "cap-surface", ax = ax1[1,0])
sns.countplot(data = shroom_DF, x = "cap-color", ax = ax1[1,1])
sns.countplot(data = shroom_DF, x = "bruises", ax = ax1[2,0])
sns.countplot(data = shroom_DF, x = "odor", ax = ax1[2,1])
sns.countplot(data = shroom_DF, x = "gill-attachment", ax = ax1[3,0])
sns.countplot(data = shroom_DF, x = "gill-spacing", ax = ax1[3,1])
plt.show()
f, ax2 = plt.subplots(4, 2, figsize = (13,17))
sns.countplot(data = shroom_DF, x = "gill-size", ax = ax2[0,0])
sns.countplot(data = shroom_DF, x = "gill-color", ax = ax2[0,1])
sns.countplot(data = shroom_DF, x = "stalk-shape", ax = ax2[1,0])
sns.countplot(data = shroom_DF, x = "stalk-root", ax = ax2[1,1])
sns.countplot(data = shroom_DF, x = "stalk-surface-above-ring", ax = ax2[2,0])
sns.countplot(data = shroom_DF, x = "stalk-surface-below-ring", ax = ax2[2,1])
sns.countplot(data = shroom_DF, x = "stalk-color-above-ring", ax = ax2[3,0])
sns.countplot(data = shroom_DF, x = "stalk-color-below-ring", ax = ax2[3,1])
plt.show()
f, ax3 = plt.subplots(4, 2, figsize = (13,17))
sns.countplot(data = shroom_DF, x = "veil-type", ax = ax3[0,0])
sns.countplot(data = shroom_DF, x = "veil-color", ax = ax3[0,1])
sns.countplot(data = shroom_DF, x = "ring-number", ax = ax3[1,0])
sns.countplot(data = shroom_DF, x = "ring-type", ax = ax3[1,1])
sns.countplot(data = shroom_DF, x = "spore-print-color", ax = ax3[2,0])
sns.countplot(data = shroom_DF, x = "population", ax = ax3[2,1])
sns.countplot(data = shroom_DF, x = "habitat", ax = ax3[3,0])
f.delaxes(ax = ax3[3,1])
plt.show()
lbe = LabelEncoder()
shroom_DF_1 = shroom_DF.copy()
for col in shroom_DF_1.columns:
    shroom_DF_1[col] = lbe.fit_transform(shroom_DF_1[col])
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
X = shroom_DF_1.drop(['class', 'veil-type', 'stalk-root'], axis = 1)
Y = shroom_DF_1['class']
X_norm = X.copy()
for col in X_norm.columns:
    X_norm[col] = NormalizeData(X_norm[col])
X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y, test_size = 0.3, random_state = 42)
shroom_log = LogisticRegression().fit(X_train, Y_train)
shroom_log_pred = shroom_log.predict(X_test)
print("Accuracy:\t", metrics.accuracy_score(Y_test, shroom_log_pred))
shroom_NB = CategoricalNB().fit(X_train, Y_train)
shroom_NB_pred = shroom_NB.predict(X_test)
print("Accuracy:\t", metrics.accuracy_score(Y_test, shroom_NB_pred))
shroom_SGD = SGDClassifier(loss = 'hinge').fit(X_train, Y_train)
shroom_SGD_pred = shroom_SGD.predict(X_test)
print("Accuracy:\t", metrics.accuracy_score(Y_test, shroom_SGD_pred))