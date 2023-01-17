import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/training.csv')
test = pd.read_csv('../input/test.csv')
df.isnull().sum().sort_values(ascending=False)
df.drop(['description', 'caption', 'name', 'id'], axis= 1, inplace=True)
df.head()
X = df.iloc[:, [0,1,2,3,4]].values
Y = df.iloc[:, 5].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
Y_sc = StandardScaler()
X_train = X_sc.fit_transform(X_train)
X_test = X_sc.transform(X_test)
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]
for clf in classifiers:
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    
    cm = confusion_matrix(Y_test, Y_pred)
    
    print('#####################')
    print((cm[0][0]+cm[1][1])/8053)
    print('#####################')
print("="*30)
