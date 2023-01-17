import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!pip install scikit-optimize==0.8.1
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xg
from functools import partial
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
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
# optimize function 
def optimize(params, x, y):

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
param_space = {
    'max_depth' : scope.int(hp.quniform('max_depth', 3,15, 1)),
    'n_estimators' : scope.int(hp.quniform('n_estimators', 100, 600, 1)),
    'criterion' : hp.choice('criterion', ['gini', 'entropy']),
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.01,1),
    'learning_rate' : hp.uniform('learning_rate', 0.001,1) 
}
# Optimization Function
optimization_function = partial(
    optimize,
    x = X,
    y = label_encoder_y
)
trials = Trials()
result = fmin(fn = optimization_function,
                    space = param_space,
                    algo = tpe.suggest,
                    max_evals = 15,
                    trials = trials
)
print(result)
# Train and test split of the data
X_train, X_test, y_train, y_test = train_test_split(X, label_encoder_y, test_size = 0.33, random_state = 7)

classifier = xg.XGBClassifier(n_thread = 6, 
                              n_estimators = 396, 
                              max_depth = 6, 
                              colsample_bytree = 0.9292372781188178,
                              learning_rate = 0.28725052863307404,
                              criterion = "gini")
classifier.fit(X_train, y_train)

# Check the accuracy of the model on train and test dataset.
accuracy_train = accuracy_score(y_train, classifier.predict(X_train))
print("Accuracy on train dataset %.2f%%" % (accuracy_train * 100))

accuracy_test = accuracy_score(y_test, classifier.predict(X_test))
print("Accuracy on test dataset %.2f%%" % (accuracy_test * 100))
# code for submission file.
dataset_test = dataset_pd2.values

classifier = xg.XGBClassifier(n_thread = 6, 
                              n_estimators = 396, 
                              max_depth = 6, 
                              colsample_bytree = 0.9292372781188178,
                              learning_rate = 0.28725052863307404,
                              criterion = "gini")
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
