import numpy as np 

import pandas as pd 



from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/creditcard.csv')

X = data.drop('Class', 1).values

y = data['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.ensemble import RandomForestClassifier
w = 50 # The weight for the positive class



RF = RandomForestClassifier(class_weight={0: 1, 1: w})
RF.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix
y_pred = RF.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

recall = tp / (tp + fn)

prec = tp / (tp + fp)

F1 = 2 * recall * prec / (recall + prec)

print(recall, prec, F1)
# Some results for different weights (bad implementation, 

# these weights should be chosen agains a validation set)



#w=1 : 0.735632183908 0.888888888889 0.805031446541

#w=10 : 0.701149425287 0.938461538462 0.802631578947

#w=100 : 0.724137931034 0.940298507463 0.818181818182

#w=1000 : 0.701149425287 0.953125 0.807947019868