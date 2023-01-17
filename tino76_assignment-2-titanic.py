import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook
%matplotlib inline
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
unused_col = ["Name", "Ticket", "Cabin"]
train = train.drop(unused_col, axis=1)
test = test.drop(unused_col, axis=1)
numeric_feats = train.dtypes[train.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 4.0]
skewed_feats = skewed_feats.index
train[skewed_feats] = np.log1p(train[skewed_feats])
test[skewed_feats] = np.log1p(test[skewed_feats])

train = pd.get_dummies(train)
test = pd.get_dummies(test)
train.describe()
train_x = train.drop(["PassengerId", "Survived"], axis=1)
train_y = train["Survived"]
train = (train_x - train_x.mean()) / (train_x.max() - train_x.min())
train["Survived"] = train_y

test_id = test["PassengerId"]
test = test.drop("PassengerId", axis=1)
test = (test - train_x.mean()) / (train_x.max() - train_x.min())
test["PassengerId"] = test_id

#filling NA's with the mean of the column:
train = train.fillna(train.mean())
test = test.fillna(train.mean())
train_x = train.drop("Survived", axis=1)
train_y = train["Survived"]
X_tr, X_val, y_tr, y_val = train_test_split(train_x, train_y, test_size=0.25)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score

c = list(np.power(10.0, np.arange(-3, 3)))
model_log = LogisticRegressionCV(cv=10, solver='lbfgs', scoring='accuracy', penalty='l2', Cs=c)
model_log.fit(X_tr, y_tr)

predict = model_log.predict(X_tr)
score = accuracy_score(y_tr, predict)
print("Training accuracy: %.2f%%" % (score * 100))

predict = model_log.predict(X_val)
score = accuracy_score(y_val, predict)
print("Validation accuracy: %.2f%%" % (score * 100))
coef = pd.Series(model_log.coef_[0,:], index = X_tr.columns)
print("LogisticRegression picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values()])
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the LogisticRegression")
predict = model_log.predict(test.drop("PassengerId", axis=1))
solution = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived":predict})
solution.to_csv("log_sol.csv", index = False)
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xgb = XGBClassifier(booster='gbtree')
xgb.fit(X_tr, y_tr)

predict = xgb.predict(X_tr)
score = accuracy_score(y_tr, predict)
print("Training accuracy: %.2f%%" % (score * 100))

predict = xgb.predict(X_val)
score = accuracy_score(y_val, predict)
print("Validation accuracy: %.2f%%" % (score * 100))
predict = xgb.predict(test.drop("PassengerId", axis=1))
solution = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived":predict})
solution.to_csv("xgb_sol.csv", index = False)
from keras.layers import Dense
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
BatchNormalization
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
BatchNormalization
model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
BatchNormalization
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
BatchNormalization
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_tr, y_tr, batch_size = 32, epochs = 200)
predict = model.predict(X_tr)
predict = (predict > 0.5).astype(int).reshape(X_tr.shape[0])
score = accuracy_score(y_tr, predict)
print("Training accuracy: %.2f%%" % (score * 100))

predict = model.predict(X_val)
predict = (predict > 0.5).astype(int).reshape(X_val.shape[0])
score = accuracy_score(y_val, predict)
print("Validation accuracy: %.2f%%" % (score * 100))
predict = model.predict(test.drop("PassengerId", axis=1))
predict = (predict > 0.5).astype(int).reshape(test.shape[0])
solution = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived":predict})
solution.to_csv("nn_sol.csv", index = False)