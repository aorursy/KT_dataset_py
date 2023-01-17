import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-pastel")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
all_vars = np.array(df.columns)
all_vars
# features: columns the classifier will use to predict

features = np.array(all_vars[0:8])
features
# target: column we want to predict

target = np.array(all_vars[8])
target
# split dataset using arrays as filters
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size = 0.2,
                                                      stratify = df[target], random_state = 0)
# Creating variables to store the results
all_models = np.array([])
all_scores = np.array([])
all_models
from sklearn.svm import LinearSVC
def svm_test(X_train, y_train, cv = 10):
  np.random.seed(0)
  svc = LinearSVC()
  cv_scores = cross_val_score(svc, X_train, y_train, scoring = 'accuracy', cv = cv, n_jobs = -1)
  print('Average of ', cv, 'tests: ', cv_scores.mean())
  return cv_scores.mean()
res = svm_test(X_train, y_train)
# updating results
all_models = np.append(all_models, "SVM")
all_scores = np.append(all_scores, res)
all_models, all_scores
from sklearn.ensemble import ExtraTreesClassifier
def ext_test(X_train, y_train, n_estimators = 100, cv = 10):
  np.random.seed(0)
  ext = ExtraTreesClassifier(n_estimators = n_estimators, criterion = 'entropy', random_state = 0, n_jobs = -1)
  cv_scores = cross_val_score(ext, X_train, y_train, scoring = 'accuracy', cv = cv, n_jobs = -1)
  print('Average of ', cv, 'tests: ', cv_scores.mean())
  return cv_scores.mean()
res = ext_test(X_train, y_train)
# updating results
all_models = np.append(all_models, "ETC")
all_scores = np.append(all_scores, res)
from sklearn.ensemble import RandomForestClassifier
def rfc_test(X_train, y_train, n_estimators = 100, cv = 10):
  np.random.seed(0)
  rfc = RandomForestClassifier(n_estimators = n_estimators, random_state = 0, n_jobs = -1)
  cv_scores = cross_val_score(rfc, X_train, y_train, scoring = 'accuracy', cv = cv, n_jobs = -1)
  print('Average of ', cv, 'tests: ', cv_scores.mean())
  return cv_scores.mean()
res = rfc_test(X_train, y_train)
# updating results
all_models = np.append(all_models, "RFC")
all_scores = np.append(all_scores, res)
from xgboost import XGBClassifier
def xgbc_test(X_train, y_train, n_estimators = 100, cv = 10):
  np.random.seed(0)
  xgb = XGBClassifier()
  cv_scores = cross_val_score(xgb, X_train, y_train, scoring = 'accuracy', cv = cv, n_jobs = -1)
  print('Average of ', cv, 'tests: ', cv_scores.mean())
  return cv_scores.mean()
res = xgbc_test(X_train, y_train)
# updating results
all_models = np.append(all_models, "XGB")
all_scores = np.append(all_scores, res)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
def mlp_test(X_train, y_train, cv = 10):
  np.random.seed(0)

  mlp = MLPClassifier()
  scaler = StandardScaler()

  pipe = Pipeline([('scaler', scaler), ('mlp', mlp)])

  cv_scores = cross_val_score(pipe, X_train, y_train, scoring = 'accuracy', cv = cv, n_jobs = -1)
  print('Average of ', cv,  'tests: ', cv_scores.mean())
  return cv_scores.mean()
res = mlp_test(X_train, y_train)
# updating results
all_models = np.append(all_models, "MLP")
all_scores = np.append(all_scores, res)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
# define the keras model
model = Sequential()

# 8 input features (input_dim)
model.add(Dense(12, input_dim=8, activation='relu'))

model.add(Dense(8, activation='relu'))

# last layer must be activated with sigmoid or softmax since we want results in (0, 1) range (probabilities)
model.add(Dense(1, activation='sigmoid'))

# compile the keras model, choose metrics
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=200, batch_size=10, verbose = 0)
# evaluate model
test_loss, res = model.evaluate(X_test, y_test)
round(test_loss, 4), round(res, 4)
# updating results
all_models = np.append(all_models, "Keras trained")
all_scores = np.append(all_scores, res)
# predict classes with the model
# class 0 : no diabetes 
# class 1 : diabetes predicted :(
predict_class = model.predict_classes(X_test)
predict_class[10:15]
# predict on test data
y_pred = model.predict(X_test)
y_pred[0:5]
# we will need this to calculate confusion matrix
rounded = [round(x[0]) for x in y_pred]
rounded[0:20]
# summarize the first n cases
n = 5
for i in range(n):
	print('%s => \n %d (expected %d)\n\n' % (X_test.iloc[i, ].tolist(), rounded[i], y_test.iloc[i]))
# input to confusion_matrix must be an array of int (rounded)
# obviously, we can only call confusion_matrix once we already called the fit method on the model
matrix = confusion_matrix(y_test, rounded)
matrix
# check models and scores arrays
all_models, all_scores
# plot model results

fig, ax = plt.subplots()
ax.barh(all_models, all_scores)
plt.xlim(0, 1)
for index, value in enumerate(all_scores):
    plt.text(value, index, str(round(value, 4)), fontsize = 12)
best_model = all_models[all_scores.argmax()]
# this is just a string, it doesn't contain the model parameters
best_model
# Defining model
mlp = MLPClassifier()

# using a scaler, since it is a neural network
scaler = StandardScaler()

# creating the pipeline with scaler and then MLP
pipe = Pipeline([('scaler', scaler), ('mlp', mlp)])
# fit/train the algorithm on the train data
pipe.fit(X_train, y_train)
# predict classes with the model
# class 0 : no diabetes 
# class 1 : diabetes predicted :(
y_pred = pipe.predict(X_test)
y_pred
pipe.predict_proba(X_train)
res = pipe.score(X_test, y_test)
res
# now that we trained (fit) the model, we can calculate the confusion matrix
cm = confusion_matrix(y_pred, y_test)
cm
# updating results (appending trained model)
all_models = np.append(all_models, "MLP trained")
all_scores = np.append(all_scores, res)
all_models, all_scores
# plot model results with trained model

fig, ax = plt.subplots()
ax.barh(all_models, all_scores)
plt.xlim(0, 1)
plt.title("Diabetes prediction: Model vs Accuracy")
for index, value in enumerate(all_scores):
    plt.text(value, index, str(round(value, 4)), fontsize = 12)