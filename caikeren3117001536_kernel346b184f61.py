import numpy as np 

import pandas as pd 

from tqdm import tqdm_notebook

from sklearn.metrics import accuracy_score, mean_squared_error

import matplotlib.pyplot as plt
train = pd.read_csv("../input/inputa/forest-cover-type-kernels-only/train.csv")

test = pd.read_csv("../input/inputa/forest-cover-type-kernels-only/test.csv")

sample = pd.read_csv("../input/inputa/forest-cover-type-kernels-only/sample_submission.csv")

print(train.tail())

print(sample.head())
Id = test.iloc[:,0]

Y_train = train.iloc[:,-1]

X_train = train.iloc[:,1:-1]

X_test = test.iloc[:,1:]
X_test.describe()

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 0)



X_train_scaling = X_train.iloc[:,0:10]

X_val_scaling = X_val.iloc[:,0:10]

X_test_scaling = X_test.iloc[:,0:10]
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sc_X = StandardScaler()

X_train_scaling = sc_X.fit_transform(X_train_scaling)

X_val_scaling = sc_X.transform(X_val_scaling)

X_test_scaling = sc_X.transform(X_test_scaling)
X_train = np.concatenate((X_train_scaling, X_train.values[:,10:]), axis=1)

X_val = np.concatenate((X_val_scaling, X_val.values[:,10:]), axis=1)

X_test = np.concatenate((X_test_scaling, X_test.values[:,10:]), axis=1)
from sklearn.ensemble import RandomForestClassifier



classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, Y_train)
Y_pred_train = classifier.predict(X_train)

Y_pred_val = classifier.predict(X_val)

accuracy_train = accuracy_score(Y_pred_train, Y_train)

accuracy_val = accuracy_score(Y_pred_val, Y_val)
print("Training accuracy", round(accuracy_train, 5))

print("Validation accuracy", round(accuracy_val, 5))
preds = classifier.predict(X_test)
sub = pd.DataFrame({"Id": Id,"Cover_Type": preds})

sub.to_csv("submission.csv", index=False)