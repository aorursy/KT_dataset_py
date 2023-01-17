#import library
import pandas as pd
import pandas_profiling
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('whitegrid')
#import data
data = pd.read_csv('../input/breast-cancer.csv')
data
#Check the null data
data.isnull().sum()
#Check data types and memory usage
data.info()
# Change diagnosis to numerical data M --> 1; B--> 0
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
data['diagnosis'] = lb.fit_transform(data['diagnosis'])
# Delete unnecessary features
del data['Unnamed: 32']
del data['id']
# Check data shape after removal
data.shape
pandas_profiling.ProfileReport(data)
# Correlation test of group 1
group_1 = data.loc[:, ["radius_mean", "perimeter_mean","area_mean","radius_se","perimeter_se", "area_se",
"radius_worst","perimeter_worst", "area_worst"]].copy()
sns.heatmap(group_1.corr(),annot=True)
# Correlation test of group 1
group_2 = data.loc[:, ["concave points_mean", "concavity_mean","texture_mean","concave points_se","concavity_se", "texture_se",
"concave points_worst","concavity_worst", "texture_worst"]].copy()
sns.heatmap(group_2.corr(),annot=True)
data = data.drop(['perimeter_mean','area_mean','perimeter_se','area_se','radius_worst','perimeter_worst', 'area_worst',
                 'concavity_mean','concave points_worst','texture_worst'],1)
# See the correlation again after removing unwanted features
data.corr()
#Check the current features
data.columns
# Check summary statistics
data.describe()
# Data transformation using Standard Scaler
from sklearn.preprocessing import StandardScaler
numeric_data = data.iloc[:,1:22]
sc = StandardScaler()
input = pd.DataFrame(sc.fit_transform(numeric_data))
input.columns = ['radius_mean', 'texture_mean', 'smoothness_mean',
       'compactness_mean', 'concave points_mean', 'symmetry_mean',
       'fractal_dimension_mean', 'radius_se', 'texture_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'smoothness_worst', 'compactness_worst',
       'concavity_worst', 'symmetry_worst', 'fractal_dimension_worst']
# Preview the result of transformation
input
# Train test split
from sklearn.model_selection import train_test_split
X = input
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
#Import random forest calassifier
from sklearn.ensemble import RandomForestClassifier
# Create random forest model 
rf_model = RandomForestClassifier(random_state=0)
# Apply the model
rf_model.fit(X_train, y_train)
# Predicted value
y_pred1 = rf_model.predict(X_test)
#Create model evaluation function
def evaluate(model, test_features, test_labels):
    from sklearn.metrics import f1_score
    predictions = model.predict(test_features)
    F1 = np.mean(f1_score(test_labels, predictions))
    print('Model Performance')
    print('F1 score = %.3f' % F1)
    
    return f1_score
#f1 score before optimization
f1_before_rf= evaluate(rf_model, X_test, y_test)
#confusion matrix before optimization
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred1)
# Random forest optimization parameters
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 778, stop = 784, num = 7)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt',5,6,7,8]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(start = 8, stop = 14, num = 7)]
# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in np.linspace(start = 10, stop = 14, num = 5)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in np.linspace(start = 1, stop = 6, num = 5)]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Method of selecting xriterion
criterion = ['gini', 'entropy']
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion':criterion}
print(random_grid)
#Create new model using the parameters
rf_random = RandomizedSearchCV(estimator = rf_model, param_distributions = random_grid, n_iter = 15,
                               cv = 5, verbose=2, random_state=0, n_jobs = -1)
#Apply the model
rf_random.fit(X_train, y_train)
#View the best parameters
rf_random.best_params_
# Predicted value
y_pred1_ = rf_random.best_estimator_.predict(X_test)
#f1 score after optimization
best_random = rf_random.best_estimator_
f1_after_rf= evaluate(best_random, X_test, y_test)
#confusion matrix after optimization
confusion_matrix(y_test, y_pred1_)
#Import KNN calassifier
from sklearn.neighbors import KNeighborsClassifier
# Create KNN model
kn_model = KNeighborsClassifier(n_neighbors=5)
# Apply the model
kn_model.fit(X_train, y_train)
# Predicted value
y_pred2 = kn_model.predict(X_test)
#f1 score before optimization
f1_before_kn= evaluate(kn_model, X_test, y_test)
#confusion matrix before optimization
confusion_matrix(y_test, y_pred2)
# KNN optimization parameters
n_neighbors = [5,6,7,8,9,10]
leaf_size = [1,2,3,5]
weights = ['uniform', 'distance']
algorithm = ['auto', 'ball_tree','kd_tree','brute']

random_grid_kn = {'n_neighbors':n_neighbors,
                  'leaf_size':leaf_size,
                  'weights':weights,
                  'algorithm':algorithm}
print(random_grid_kn)
#Create new model using the parameters
kn_random = RandomizedSearchCV(estimator = kn_model, param_distributions = random_grid_kn, n_iter = 15,
                           cv = 5, verbose=2, random_state=123, n_jobs = -1)
#Apply the model
kn_random.fit(X_train, y_train)
#View the best parameters
kn_random.best_params_
# Predicted value
y_pred2_ = kn_random.best_estimator_.predict(X_test)
#f1 score after optimization
best_random_kn = kn_random.best_estimator_
f1_after_kn= evaluate(best_random_kn, X_test, y_test)
#confusion matrix after optimization
confusion_matrix(y_test, y_pred2_)
#Import SVM calassifier
from sklearn.svm import SVC
# Create SVM model
svc_model = SVC(random_state=123)
# Apply the model
svc_model.fit(X_train, y_train)
# Predicted value
y_pred3 = svc_model.predict(X_test)
#f1 score before optimization
f1_before_svc= evaluate(svc_model, X_test, y_test)
#confusion matrix score optimization
confusion_matrix(y_test, y_pred3)
# SVM optimization parameters
C= [0.123,0.124, 0.125, 0.126, 0.127]
kernel = ['linear','rbf','poly']
gamma = [0, 0.0000000000001, 0.000000000001, 0.00000000001]

random_grid_svm = {'C': C,
                   'kernel': kernel,
                   'gamma': gamma}
print(random_grid_svm)
#Create new model using the parameters
svc_random = RandomizedSearchCV(estimator = svc_model, param_distributions = random_grid_svm, n_iter = 15,
                           cv = 5, verbose=2, random_state=123, n_jobs = -1)
#Apply the model
svc_random.fit(X_train, y_train)
#View the best parameters
svc_random.best_params_
# Predicted value
y_pred3_ = svc_random.best_estimator_.predict(X_test)
#f1 score after optimization
best_random_svc = svc_random.best_estimator_
f1_after_svc= evaluate(best_random_svc, X_test, y_test)
#confusion matrix after optimization
confusion_matrix(y_test, y_pred3_)
