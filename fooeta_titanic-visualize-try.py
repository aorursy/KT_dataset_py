import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
train_path = os.path.join("../input/", "train.csv")
raw_data = pd.read_csv(train_path)

raw_data.head(5)
raw_data.info()
passengers = raw_data.drop(['PassengerId','Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

X = passengers.drop('Survived', axis=1)
y = passengers['Survived'].copy()

X_train, X_validate, y_train, y_validate = train_test_split(X, y, random_state=0)
X_train.head(5)
X_train.describe()
X_train_dummies = pd.get_dummies(X_train)
X_train_dummies = X_train_dummies.drop('Sex_male', axis=1)
X_train_dummies.head(5)
passengers.info()
X_train_dummies.info()
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
imputer.fit(X_train_dummies)
X_train_filled = pd.DataFrame(imputer.transform(X_train_dummies), columns=X_train_dummies.columns)
X_train_filled.info()
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_train_filled)
X_train_pca = pca.transform(X_train_filled)
Surviver_f = np.zeros((len(y_train), 2))
Surviver_m = np.zeros((len(y_train), 2))
Not_Surviver_f = np.zeros((len(X_train_filled), 2))
Not_Surviver_m = np.zeros((len(X_train_filled), 2))

i, j, l, m = 0, 0, 0, 0
for k, data, f in zip(y_train, X_train_pca, X_train_filled['Sex_female']):
    if k == 1:
        if f == 1:
            Surviver_f[i] = data
            i = i + 1
        else:
            Surviver_m[l] = data
            l = l + 1
    else:
        if f == 1:
            Not_Surviver_f[j] = data
            j = j + 1
        else:    
            Not_Surviver_m[m] = data
            m = m + 1
            

plt.scatter(Surviver_f[:, 0], Surviver_f[:, 1], color='green', alpha=.4)
plt.scatter(Surviver_m[:, 0], Surviver_m[:, 1], color='blue', alpha=.4)

plt.scatter(Not_Surviver_m[:, 0], Not_Surviver_m[:, 1], color='red', alpha=.4)
plt.scatter(Not_Surviver_f[:, 0], Not_Surviver_f[:, 1], color='orange', alpha=.4)
plt.matshow(pca.components_, cmap='viridis')
plt.xticks(range(len(X_train_filled.columns)), X_train_filled.columns, rotation=60, ha='left')
Sample = X_train_filled[['Age', 'Sex_female', 'Pclass', 'Fare']]
from mpl_toolkits.mplot3d import Axes3D
import math

Tmp1 = np.zeros((len(Sample), 4))
Tmp2 = np.zeros((len(Sample), 4))
Tmp3 = np.zeros((len(Sample), 4))
Tmp4 = np.zeros((len(Sample), 4))
i, j, k, l = 0, 0, 0, 0
for data, s, f in zip(Sample.values, y_train, Sample['Sex_female']):
    if s == 1:
        if f == 1:
            Tmp1[i] = data
            i += 1
        else:
            Tmp2[j] = data
            j += 1
    else:
        if f == 1:
            Tmp3[k] = data
            k += 1
        else:
            Tmp4[l] = data
            l += 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Age')
ax.set_ylabel('Pclass')
ax.set_zlabel('Fare')
ax.set_yticks([0, 1, 2, 3])

ax.scatter(Tmp1[:,0], Tmp1[:,2], Tmp1[:,3], color='blue', alpha=.4)
ax.scatter(Tmp2[:,0], Tmp2[:,2], Tmp2[:,3], color='green', alpha=.4)
ax.scatter(Tmp3[:,0], Tmp3[:,2], Tmp3[:,3], color='red', alpha=.4)
ax.scatter(Tmp4[:,0], Tmp4[:,2], Tmp4[:,3], color='orange', alpha=.4)
from sklearn.linear_model import LogisticRegression

X_val_dummies = pd.get_dummies(X_validate).drop('Sex_male', axis=1)
X_val_filled = pd.DataFrame(imputer.transform(X_val_dummies), columns=X_val_dummies.columns)
X_val = X_val_filled[['Age', 'Sex_female', 'Pclass', 'Fare']]

clf = LogisticRegression().fit(Sample, y_train)
print("Test set score:{}".format(clf.score(Sample, y_train)))
print("Validation set score:{}".format(clf.score(X_val, y_validate)))
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)
tree.fit(Sample, y_train)
print("Test set score:{}".format(tree.score(Sample, y_train)))
print("Validation set score:{}".format(tree.score(X_val, y_validate)))
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(Sample, y_train)
print("Test set score:{}".format(forest.score(Sample, y_train)))
print("Validation set score:{}".format(forest.score(X_val, y_validate)))