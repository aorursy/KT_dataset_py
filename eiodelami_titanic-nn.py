from keras.models import Sequential

from keras.layers import Dense

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

from sklearn import metrics 

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

from sklearn import model_selection

print(os.listdir("../input"))

#fix random seed for reproducibility

np.random.seed(7)
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
train.describe()
test.describe()
train.dtypes.sample(10)

test.dtypes.sample(10)
non_categorical = train.select_dtypes(include='int64')

non_categorical2 = test.select_dtypes(include='int64')
non_categorical.head()
X = non_categorical.iloc[:, 2:]

x= non_categorical2.iloc[:, 1:]

Y = non_categorical.Survived
X.head()
#create model

model = Sequential()

model.add(Dense(12, input_dim=3, activation='relu'))

model.add(Dense(3, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.summary()
#compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mean_absolute_error'])
#fit the model

model.fit(X, Y, epochs=832, batch_size=50)
#evaluate the model

scores = model.evaluate(X,Y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#calculate prediction

predictions = model.predict(x)
print(predictions)
#round predictions

rounded_predictions = [round(X[0]) for X in predictions]

print(rounded_predictions)
#evaluate the model

scores = model.evaluate(X,Y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
scoring='roc_auc'

LRmodel = LogisticRegression()

kfold= model_selection.KFold(n_splits=10)

results = model_selection.cross_val_score(LRmodel, X, Y, cv=kfold, scoring=scoring)

print("ROC: %.3f (%.3f)" % (results.mean(), results.std()))
scoring='neg_log_loss'

kfold= model_selection.KFold(n_splits=10)

results = model_selection.cross_val_score(LRmodel, X, Y, cv=kfold, scoring=scoring)

print("LogLoss: %.3f (%.3f)" % (results.mean(), results.std()))
scoring='accuracy'

kfold= model_selection.KFold(n_splits=10)

results = model_selection.cross_val_score(LRmodel, X, Y, cv=kfold, scoring=scoring)

print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))
# print the linear regression and display datapoints

from sklearn.linear_model import LinearRegression  

regressor = LinearRegression()  

regressor.fit(x, predictions)  

y_fit = regressor.predict(x) 
reg_intercept = round(regressor.intercept_[0],4)

reg_coef = round(regressor.coef_.flatten()[0],4)

reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)
plt.scatter(x, x, color='blue', label= 'data')

plt.plot(x, predictions, color='red', linewidth=2, label = 'Linear regression\n'+reg_label) 

plt.title('Linear Regression')

plt.legend()

plt.xlabel('observed')

plt.ylabel('predicted')

plt.show()