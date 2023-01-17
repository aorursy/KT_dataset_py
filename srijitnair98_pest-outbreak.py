import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

pest_train = pd.read_csv("../input/pestoutbreaknew1/pesttrain1.csv", sep=',',header=0)
pest_test = pd.read_csv("../input/pestoutbreaknew1/pesttrain1.csv", sep=',',header=0)

y_tr = pest_train.iloc[:,1]
X_tr = pest_train.iloc[:,1:]

y_test = pest_test.iloc[:,1]
X_test = pest_test.iloc[:,1:]


RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_tr, y_tr)
l=RF.predict(X_test)
print(l)
#round(RF.score(X_test, y_test), 4)


import pandas as pd
pesttrain1 = pd.read_csv("../input/pesttrain1.csv")