import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
#Importing Preprocessing liberaries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, Binarizer, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
#Importing model liberaries 
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
%matplotlib inline
seed = 1
scoring='accuracy'
num_folds = 10
test_size = 0.3
kfold = KFold(n_splits=num_folds, random_state=seed)
filepath = '../input/train.csv'
labeled_images = pd.read_csv(filepath)
images = labeled_images.iloc[:5000, 1:].values.astype('float32')
labels = labeled_images.iloc[:5000, :1].values.astype('int32')

train_images,test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size, random_state=seed)
sns.countplot(labeled_images['label'])
print(labeled_images.isnull().any().any())
Counter(labeled_images['label'])
train_images /= 255.0
test_images /= 255.0
plt.figure(figsize=(15,15))
for i in range(5):
  plt.subplot(5, 5, i+1)
  img = train_images[i].reshape(28,28)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap='gray')
  plt.colorbar()
  plt.title(train_labels[i])
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('GB', GaussianNB()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('ExtraTrees', ExtraTreesClassifier()))
models.append(('GradientBoosting', GradientBoostingClassifier()))
models.append(('RandomForest', RandomForestClassifier()))
results = []
names = []
# for name, model in models:
#   cv_results = cross_val_score(model, train_images, train_labels.ravel(), cv=kfold)
#   results.append(cv_results)
#   names.append(name)
#   print(name, cv_results.mean(), cv_results.std())
# k_range = list(range(1,10))
# weight_options = ['uniform', 'distance']
# algorithm = ['auto', 'brute']
# knn_params_grid = dict(n_neighbors=k_range, weights=weight_options)

# def knn_param_selection(X, y, params_grid, num_folds):
#     knn = KNeighborsClassifier()
#     knnGrid = GridSearchCV(knn, params_grid, cv=num_folds, n_jobs=-1, scoring='accuracy')
#     knnGrid.fit(train_images, train_labels.ravel())
#     print(knnGrid.best_estimator_, knnGrid.best_score_, knnGrid.best_params_)
    
# knn_param_selection(train_images, train_labels, knn_params_grid, num_folds)
#Fine Tuning KNN Hyperparameters
model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=6, p=2,
           weights='distance')
final_results = cross_val_score(model, train_images, train_labels, cv=kfold)
print("%s %f %f" % ('KNN - FineTuned Grid', final_results.mean(), final_results.std()))
# kernels = ['rbf']
# #first iteration params
# #kernels = ['linear', 'rbf']
# # Cs = [0.001, 0.01, 0.1, 1, 10]         
# # gammas = [0.001, 0.01, 0.1, 1]
# #Second iteration params
# # Cs = [9, 10, 11, 12, 13, 14, 15]
# # gammas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
# #third iteration params
# Cs = [13, 14, 15, 16, 17]
# gammas = [0.018, 0.019, 0.020, 0.021, 0.022 ]
# svcParams_grid = dict(C=Cs, gamma=gammas, kernel=kernels)
# def svc_param_selection(X, y, params_grid, num_folds):
#     svc = SVC()
#     svcRand = RandomizedSearchCV(svc, svcParams_grid, cv=num_folds, n_jobs=-1, scoring='accuracy')
#     svcRand.fit(train_images, train_labels.ravel())
#     print(svcRand.best_estimator_, svcRand.best_score_, svcRand.best_params_)

# svc_param_selection(train_images, train_labels, svcParams_grid, num_folds)
# #First iteration: {'kernel': 'rbf', 'gamma': 0.01, 'C': 10}  Score: 0.950857
# #Second iteration: {'kernel': 'rbf', 'gamma': 0.02, 'C': 15}  Score: 0.9588571428571429
# #Third iteration: {'kernel': 'rbf', 'gamma': 0.022, 'C': 17} Score: 0.9588571428571429 
#Fine Tuning SVM Hyperparameters
svc = SVC(C=15, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.022, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
final_results = cross_val_score(svc, train_images, train_labels, cv=kfold)
print("%s %f %f" % ('SVM - FineTuned Random', final_results.mean(), final_results.std()))
#Predictions after tuning KNN Hyperparameter
model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=6, p=2,
           weights='distance')
model.fit(train_images, train_labels)
predictions = model.predict(test_images)
print("%s %f" % ('KNN - Fine Tuned', accuracy_score(test_labels, predictions)))
#Predictions after tuning SVM Hyperparameter
svc = SVC(C=15, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.022, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
svc.fit(train_images, train_labels)
predictions = svc.predict(test_images)
print("%s %f" % ('SVM - Fine Tuned', accuracy_score(test_labels, predictions)))
filepath = '../input/test.csv'
test_data = pd.read_csv(filepath)
test_data_images = test_data.iloc[:, :].values.astype('float32')
test_data_images /= 255.0
test_data_images.shape
images = labeled_images.iloc[:, 1:].values.astype('float32')
labels = labeled_images.iloc[:, :1].values.astype('int32')
images /= 255.0
# svc = SVC(C=15, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape='ovr', degree=3, gamma=0.022, kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False)
# svc.fit(images, labels)
# results = svc.predict(test_data_images)
# df = pd.DataFrame(results)
# df.index.name='ImageId'
# df.index+=1
# df.columns=['Label']
# df.to_csv('results.csv', header=True)
#Trying to create an ensemble of few best performing 
from sklearn.ensemble import VotingClassifier
lr_clf  = LogisticRegression()
knn_clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=6, p=2,
           weights='distance')
svm_clf = SVC(C=15, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.022, kernel='rbf',
  max_iter=-1, random_state=None, shrinking=True,
  tol=0.001, verbose=False, probability=True)
extree_clf =  ExtraTreesClassifier()


voting_clf = VotingClassifier(
              estimators= [('lr', lr_clf), ('knn', knn_clf), ('svm', svm_clf), ('extree', extree_clf)],
              voting='soft')
voting_clf.fit(images, labels)
results = voting_clf.predict(test_data_images)
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)
plt.figure(figsize=(15,15))
for i in range(25):
  plt.subplot(5, 5, i+1)
  img = test_data_images[i].reshape(28,28)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap='gray')
  plt.colorbar()
  plt.title(results[i])
sns.countplot(df['Label'])
Counter(df['Label'])