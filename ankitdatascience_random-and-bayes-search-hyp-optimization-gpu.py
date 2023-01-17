import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!pip install scikit-optimize==0.8.1
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xg
from collections import Counter
!pip install kneed
# kneed is not installed in kaggle. uncomment the above line.
from kneed import KneeLocator
import matplotlib.pyplot as plt
from functools import partial
from skopt import space, gp_minimize
# Reading train dataset in the environment.
dataset_pd = pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/train.csv", index_col = 0)
print(dataset_pd.shape)
# Reading test dataset in the environment.
dataset_pd2 = pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/test.csv", index_col = 0)
print(dataset_pd2.shape)
# Creating a predictor matrix (removing the response variable column)
dataset_train = dataset_pd.values
X = dataset_train[:,0:93] # Predictors
y = dataset_train[:,93] # Response 

# XGBoost do not take a categorical variable as input. We can use LabelEncoder to assign labels to categorical variables.
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y)
label_encoder_y = label_encoder.transform(y)
# Train and test split of the data
X_train, X_test, y_train, y_test = train_test_split(X, label_encoder_y, test_size = 0.33, random_state = 7)
# Running a XGBoost with default settings.
model = xg.XGBClassifier(nthreads = -1)
model.fit(X_train, y_train)

# Check the accuracy of the model on train and test dataset.
accuracy_train = accuracy_score(y_train, model.predict(X_train))
print("Accuracy on train dataset %.2f%%" % (accuracy_train * 100))

accuracy_test = accuracy_score(y_test, model.predict(X_test))
print("Accuracy on test dataset %.2f%%" % (accuracy_test * 100))
classifier = xg.XGBClassifier(n_thread = 6, tree_method='gpu_hist')
# Defining the parameter grid for the Random Search.
param_grid = {
    "n_estimators" : np.arange(100, 1000, 100),
    "max_depth" : np.arange(1, 20, 2),
    "colsample_bytree": np.arange(0.5,1, 0.1),
    "learning_rate" : [0.0001, 0.001, 0.01, 0.1],
    "criterion": ["gini",'entropy']
}
model = RandomizedSearchCV(estimator = classifier,
                          param_distributions = param_grid,
                          n_iter = 10,
                          scoring = "accuracy",
                          verbose = 10,
                          n_jobs = -1,
                          cv = 5)
model.fit(X, label_encoder_y)
model.best_score_
print(model.best_estimator_.get_params())
# Train and test split of the data
X_train, X_test, y_train, y_test = train_test_split(X, label_encoder_y, test_size = 0.33, random_state = 7)

classifier = xg.XGBClassifier(n_thread = 6, tree_method='gpu_hist', 
                              n_estimators = 600, 
                              max_depth = 5, 
                              colsample_bytree = 0.8,
                              learning_rate = 0.1,
                              criterion = "entropy")
classifier.fit(X_train, y_train)

# Check the accuracy of the model on train and test dataset.
accuracy_train = accuracy_score(y_train, classifier.predict(X_train))
print("Accuracy on train dataset %.2f%%" % (accuracy_train * 100))

accuracy_test = accuracy_score(y_test, classifier.predict(X_test))
print("Accuracy on test dataset %.2f%%" % (accuracy_test * 100))
dataset_test = dataset_pd2.values

classifier = xg.XGBClassifier(n_thread = 6, tree_method='gpu_hist', 
                              n_estimators = 600, 
                              max_depth = 5, 
                              colsample_bytree = 0.8,
                              learning_rate = 0.1,
                              criterion = "entropy")
classifier.fit(X, label_encoder_y)

prediction_sub = classifier.predict(dataset_test)

#dataset_pd2["prediction"] = prediction_sub
X_sub = np.array(prediction_sub).reshape(-1,1)
onehot_encoder = OneHotEncoder(sparse = False)
submission_file = onehot_encoder.fit_transform(X_sub)

submission_file_df = pd.DataFrame(submission_file, 
                                  columns = ['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6',
                                            'Class_7','Class_8','Class_9'], index = dataset_pd2.index)


submission_file_df.to_csv("submission_otto_ver2.csv")
# optimize function for gp_minimize
def optimize(params, param_names, x, y):
    params = dict(zip(param_names, params))
    model = xg.XGBClassifier(**params)
    kf = StratifiedKFold(n_splits = 5)
    accuracies = []
    for idx in kf.split(X = x, y = y):
        train_idx , test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]
        
        xtest = x[test_idx]
        ytest = y[test_idx]
        
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc = accuracy_score(ytest, preds)
        accuracies.append(fold_acc)
    
    return -1.0 * np.mean(accuracies)
# Parameter Space for XGBoost
param_space = [
    space.Integer(3,15, name = 'max_depth'),
    space.Integer(100, 600, name = 'n_estimators'),
    space.Categorical(['gini', 'entropy'], name = 'criterion'),
    space.Real(0.01,1, prior = 'uniform', name = 'colsample_bytree'),
    space.Real(0.001,1, prior = 'uniform', name = 'learning_rate') 
]
param_names = [
    "max_depth",
    "n_estimators",
    "criterion",
    "colsample_bytree",
    "learning_rate"
]

# Optimization Function
optimization_function = partial(
    optimize,
    param_names = param_names,
    x = X,
    y = label_encoder_y
)
result = gp_minimize(optimization_function,
                    dimensions = param_space,
                    n_calls = 10,
                    n_random_starts = 10,
                    verbose = 10, 
                    n_jobs = -1
)
print(dict(zip(param_names, result.x)))
# Train and test split of the data
X_train, X_test, y_train, y_test = train_test_split(X, label_encoder_y, test_size = 0.33, random_state = 7)

classifier = xg.XGBClassifier(n_thread = 6, tree_method='gpu_hist', 
                              n_estimators = 171, 
                              max_depth = 12, 
                              colsample_bytree = 0.9444262241947871,
                              learning_rate = 0.253008978,
                              criterion = "entropy")
classifier.fit(X_train, y_train)

# Check the accuracy of the model on train and test dataset.
accuracy_train = accuracy_score(y_train, classifier.predict(X_train))
print("Accuracy on train dataset %.2f%%" % (accuracy_train * 100))

accuracy_test = accuracy_score(y_test, classifier.predict(X_test))
print("Accuracy on test dataset %.2f%%" % (accuracy_test * 100))
dataset_test = dataset_pd2.values

classifier = xg.XGBClassifier(n_thread = 6, tree_method='gpu_hist', 
                              n_estimators = 171, 
                              max_depth = 12, 
                              colsample_bytree = 0.9444262241947871,
                              learning_rate = 0.253008978,
                              criterion = "entropy")
classifier.fit(X, label_encoder_y)

prediction_sub = classifier.predict(dataset_test)

#dataset_pd2["prediction"] = prediction_sub
X_sub = np.array(prediction_sub).reshape(-1,1)
onehot_encoder = OneHotEncoder(sparse = False)
submission_file = onehot_encoder.fit_transform(X_sub)

submission_file_df = pd.DataFrame(submission_file, 
                                  columns = ['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6',
                                            'Class_7','Class_8','Class_9'], index = dataset_pd2.index)


submission_file_df.to_csv("submission_otto_ver2.csv")
