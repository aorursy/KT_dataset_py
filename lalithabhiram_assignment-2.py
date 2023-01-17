import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import pandas as pd
Abhi = pd.read_csv("../input/User_Data.csv")
Abhi.head()
X = Abhi.iloc[:,[2]].values
y = Abhi.iloc[:, 4].values
print("The Column of Age is",X)
print("The Column of Purchasement is",y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)
print(X_test)
from sklearn import svm
Dandi=svm.LinearSVC()
Dandi=Dandi.fit(X_train,y_train)
y_pred=Dandi.predict(X_test)
print("The Predicted values in the model are: \n",y_pred)
print(Dandi)
print("\nAccuracy score: %f" %(accuracy_score(y_test,y_pred) * 100))
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import StratifiedShuffleSplit
C_range=[0.1, 1, 10, 100]
gamma_range= [1, 0.1, 0.01, 0.001]
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
xx.ravel()