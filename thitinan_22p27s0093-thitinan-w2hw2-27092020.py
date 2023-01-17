import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from keras import backend as K
file_path = "../input/titanic/"
data = pd.read_csv(os.path.join(file_path,'train.csv'))
#print(data.head())
#print(data.describe())
#print(data.corr())
print(data.isnull().sum())
# replace missing data (Age)
# convert to array
age = data['Age'].values
age = np.reshape(age,(-1,1))
imp = SimpleImputer(missing_values = np.nan , strategy='most_frequent')
imp.fit(age)
data['Age'] = imp.transform(age)
print(data.isnull().sum())
#convert label to int
data.Sex=data.Sex.astype('category').cat.codes
print(data.head())
# input and output data
features = data[["Pclass", "Fare", "Age"]]
target = data.Survived
#features scaling
scale = StandardScaler()
features = scale.fit_transform(features)
#with grid search you can find an optimal parameter "parameter tuning"
param_grid = {'max_depth': np.arange(1, 10)}
#initializes the tree randomly: thats why you get different results !!!
model = GridSearchCV(DecisionTreeClassifier(), param_grid)

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=.2)

model.fit(feature_train, target_train)

print("Best parameter with Grid Search: ", model.best_params_)
model = DecisionTreeClassifier(criterion='entropy', max_depth=8)
scoring = ['precision_macro', 'recall_macro', 'f1_macro']
scores = cross_validate(model, features, target, cv=5, scoring=scoring)
print(scores.keys())
print("precision = ", scores['test_precision_macro'])
print("recall = ", scores['test_recall_macro'])
print("f1-measure = ", scores['test_f1_macro'])
print("average f1-measure = ", scores['test_f1_macro'].mean())
model = GaussianNB()
scoring = ['precision_macro', 'recall_macro', 'f1_macro']
scores = cross_validate(model, features, target, cv=5, scoring=scoring)
print(scores.keys())
print("precision = ", scores['test_precision_macro'])
print("recall = ", scores['test_recall_macro'])
print("f1-measure = ", scores['test_f1_macro'])
print("average f1-measure = ", scores['test_f1_macro'].mean())
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# Define per-fold score containers
f1_per_fold = []
precision_per_fold = []
recall_per_fold = []

no_epochs = 25
batch_size = 28
verbosity = 1
num_folds = 5
# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(features, target):

  # Define the model architecture
  model = Sequential()
  model.add(Dense(10, input_dim=3, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])


  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(features[train], target[train],
              batch_size=batch_size,
              epochs=no_epochs,
              verbose=verbosity)

  # Generate generalization metrics
  scores = model.evaluate(features[test], target[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  f1_per_fold.append(scores[2])
  precision_per_fold.append(scores[3])
  recall_per_fold.append(scores[4])

  # Increase fold number
  fold_no = fold_no + 1

# == Provide scores ==
print('------------------------------------------------------------------------')
print('Scores for all folds:')
print(f'> Recall: {recall_per_fold}')
print(f'> Precision: {precision_per_fold}')
print(f'> F1-measure: {f1_per_fold}')
print(f'> Average F1-measure: {np.mean(f1_per_fold)}')
print('------------------------------------------------------------------------')
