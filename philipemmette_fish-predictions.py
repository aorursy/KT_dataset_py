# Data manipulation
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import xgboost

# Models and metrics. Since the data is small, we will test many models
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, KFold


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('../input/fish-market/Fish.csv')
df.head()
df.describe(include='all')
df['Species'].value_counts()
# Split the data into features and targets
X = df[['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, shuffle=True)
plt.figure(figsize=(10,5))
plt.hist(y, bins = [0,1,2,3,4,5,6,7], align='left', rwidth=0.8, color = 'green')
plt.title('Distribution of Fish Species')
plt.xlabel('Fish Species')
plt.ylabel('Amount of Fish with Given Species')
plt.grid(b=True, axis='y')
# Here we are creating all of our models that we want to test
# This code is modified from [1]

classifiers = []
model1 = xgboost.XGBClassifier()
classifiers.append(model1)
model2 = svm.SVC()
classifiers.append(model2)
model3 = tree.DecisionTreeClassifier(class_weight = 'balanced')
classifiers.append(model3)
model4 = RandomForestClassifier(class_weight = 'balanced')
classifiers.append(model4)
model5 = LogisticRegression(class_weight = 'balanced')
classifiers.append(model5)
model6 = KNeighborsClassifier( n_neighbors=1)
classifiers.append(model6)
model7 = KNeighborsClassifier( n_neighbors=2)
classifiers.append(model7)
model8 = KNeighborsClassifier( n_neighbors=3)
classifiers.append(model8)
model9 = KNeighborsClassifier( n_neighbors=4)
classifiers.append(model9)
model10 = KNeighborsClassifier( n_neighbors=5)
classifiers.append(model10)
model11 = GaussianNB()
classifiers.append(model11)
# This code is modified from [1]
maxAccuracy = 0
maxCV = 0
for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred= clf.predict(X_test)
    acc = balanced_accuracy_score(y_test, y_pred)
    print("Accuracy of %s is %s"%(clf, acc))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix of %s is %s"%(clf, cm))
    
    # Get CV Score and max single score. Check if it is max for all models
    cvScore = cross_val_score(clf, X, y, cv = 6)
    cvMean = np.mean(cvScore)
    maxScore = np.max(cvScore)
    if maxAccuracy < acc:
        maxAccuracy = acc
        model = clf
    if maxCV < cvMean:
        maxCV = cvMean
        modelCV = clf
    print(cvScore)
print("Our best model was", model, "with a balanced accuracy of", maxAccuracy, ".")
print("Our best cross-validation model was", modelCV, "with the score,", maxCV, ".")