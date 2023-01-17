import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import shuffle
from scipy.stats import mode
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.ensemble import IsolationForest
from sklearn.ensemble import IsolationForest
from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier

# =========================================================

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "Logistic Regression", "Bagging", "GradBoost", "ExtraTree"]#, "Gaussian Process"]

classifiers = [
    KNeighborsClassifier(n_neighbors=5),
    SVC(kernel="linear", C = 0.6), #C=0.025
    SVC(C = 0.6),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200, max_depth=2),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(n_estimators=200),
    GaussianNB(),
    LogisticRegression(random_state=1),
    BaggingClassifier(n_estimators=200),
    GradientBoostingClassifier(n_estimators=350, learning_rate=.1,max_depth=2),
    ExtraTreeClassifier()]#,
    #GaussianProcessClassifier(1.0 * RBF(1.0))]

# =========================================================

# Read DataSet and put in X and y
dataset = pd.read_csv('../input/train.csv')
dataset_test = pd.read_csv("../input/test.csv")
to_drop = ['PlayerID', 'Name']
dataset.drop(to_drop, inplace=True, axis=1)
dataset_test.drop(to_drop, inplace=True, axis=1)

#dataset = dataset.interpolate(method='values')
dataset = dataset.fillna(dataset.mean())
dataset_test = dataset_test.fillna(dataset_test.mean())

dataset = shuffle(dataset)

#--------------------------------------

X = dataset.iloc[:, 0:-1].values
X[:, 8] = 1
y = dataset.iloc[:, 19].values

X_value = dataset_test.iloc[:, 0:].values
X_value[:, 8] = 1
# =========================================================

# #preprocess dataset, split into training and test part
# sc = StandardScaler()
# X = sc.fit_transform(X)
# X_value = sc.fit_transform(X_value)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

# =========================================================

# iterate over classifiers
y_pred_matrix = []
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    #score = clf.score(X_test, y_test)
    #print(name, score)
    y_pred = clf.predict(X_test)
    y_pred[y_pred==-1] = 0
    y_pred_matrix.append(y_pred)
    f1_scr = f1_score(y_test, y_pred, average='binary')
    print(name, f1_scr)
    
y_pred_matrix = np.array(y_pred_matrix)
final_pred = mode(y_pred_matrix, axis=0)[0]
final_pred = final_pred.flatten()
final_score = f1_score(y_test, final_pred, average='binary')
print (final_score)

# =========================================================

#Voting by all classifiers
ZipList = list(zip(names, classifiers))
clf_Vot = VotingClassifier(estimators = ZipList, voting='hard')
clf_Vot.fit(X_train, y_train)
#score = clf_Vot.score(X_test, y_test)
#print(score)
y_pred = clf_Vot.predict(X_test)
f1_scr = f1_score(y_test, y_pred, average='binary')
print(f1_scr)

# =========================================================

clf_Vot.fit(X, y)
y_pre = clf_Vot.predict(X_value)
#print(y_pre)

cols = { 'PlayerID': [i+901 for i in range(440)] , 'TARGET_5Yrs': y_pre }
result = pd.DataFrame(cols)
result.to_csv("Sub_Vot.csv", index=False)
#result

# =========================================================

clf_GB = GradientBoostingClassifier(n_estimators=350, learning_rate=.1, max_depth=1)
clf_GB.fit(X, y)
y_pre = clf_GB.predict(X_value)
#print(y_pre)

cols = { 'PlayerID': [i+901 for i in range(440)] , 'TARGET_5Yrs': y_pre }
result = pd.DataFrame(cols)
result.to_csv("Sub_GB.csv", index=False)
#resultclf_GB = GradientBoostingClassifier(n_estimators=350, learning_rate=.1, max_depth=1)
clf_GB.fit(X, y)
y_pre = clf_GB.predict(X_value)
#print(y_pre)

cols = { 'PlayerID': [i+901 for i in range(440)] , 'TARGET_5Yrs': y_pre }
result = pd.DataFrame(cols)
result.to_csv("Sub_GB.csv", index=False)
#result

print("Complete Runing!")