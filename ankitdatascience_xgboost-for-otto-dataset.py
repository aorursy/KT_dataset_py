import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xg
from collections import Counter
!pip install kneed
# kneed is not installed in kaggle. uncomment the above line.
from kneed import KneeLocator
import matplotlib.pyplot as plt
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
model = xg.XGBClassifier()
model.fit(X_train, y_train)
# Check the accuracy of the model on train and test dataset.
accuracy_train = accuracy_score(y_train, model.predict(X_train))
print("Accuracy on train dataset %.2f%%" % (accuracy_train * 100))

accuracy_test = accuracy_score(y_test, model.predict(X_test))
print("Accuracy on test dataset %.2f%%" % (accuracy_test * 100))
# Creating a confusion matrix 
print(confusion_matrix(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))
# Running a XGBoost with less column sample.
model = xg.XGBClassifier(colsample_bytree = 0.5)
model.fit(X_train, y_train)
# Check the accuracy of the model on train and test dataset.
accuracy_train = accuracy_score(y_train, model.predict(X_train))
print("Accuracy on train dataset %.2f%%" % (accuracy_train * 100))

accuracy_test = accuracy_score(y_test, model.predict(X_test))
print("Accuracy on test dataset %.2f%%" % (accuracy_test * 100))
Counter(y_test)
# Storing the feature importance matrix
feature_imp = pd.DataFrame(model.feature_importances_, 
                           index = dataset_pd.drop('target', axis = 1).columns, columns = ['imp'])
feature_imp.sort_values(by = 'imp', ascending = False, inplace = True)
# Calculating accuracy considering different threshold for feature importance.
num = []
score = []
for thresh in model.feature_importances_:
    selection = SelectFromModel(model, threshold = thresh, prefit = True)
    Select_X_train = selection.transform(X_train)
    selection_model = xg.XGBClassifier()
    selection_model.fit(Select_X_train, y_train)
    Select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(Select_X_test)
    num.append(Select_X_train.shape[1])
    score.append(accuracy_score(y_test, y_pred)* 100)
    print("Thresh: %.3f, n = %d, Accuracy: %.2f%%" % (thresh, Select_X_train.shape[1], accuracy_score(y_test, y_pred)* 100))
# Storing the accuracy table for different threshold and then plotting it.
accuracy_table = pd.DataFrame({'params' : num, 'accuracy' : score})
accuracy_table.sort_values(by = 'accuracy', ascending = False, inplace = True)
plt.plot(range(93), accuracy_table['accuracy'])
plt.show()
# Storing the accuracy table for different threshold and then plotting it.
accuracy_table.sort_values(by = 'params', inplace = True)
plt.plot(range(93), accuracy_table['accuracy'])
plt.show()
# We can find the elbow using KneeLocator.
kl = KneeLocator(range(1, 94), accuracy_table['accuracy'], curve="concave", direction="increasing")
kl.elbow
# We can select 25 top variables and then fit the model again.
feature_imp[:25].index
# Selecting the top 25 variables.
data_top25 = dataset_pd[feature_imp[:25].index]
X_top25 = data_top25.values
# Train and test split of the data
X_train, X_test, y_train, y_test = train_test_split(X_top25, label_encoder_y, test_size = 0.33, random_state = 7)
# Running a XGBoost with default settings with only top 25 variables.
model = xg.XGBClassifier()
model.fit(X_train, y_train)

# Check the accuracy of the model on train and test dataset.
accuracy_train = accuracy_score(y_train, model.predict(X_train))
print("Accuracy on train dataset %.2f%%" % (accuracy_train * 100))

accuracy_test = accuracy_score(y_test, model.predict(X_test))
print("Accuracy on test dataset %.2f%%" % (accuracy_test * 100))
# Grid Search for number of trees
model = xg.XGBClassifier(n_thread = -1)
n_estimators = range(100, 500, 50)
#max_depth = [2,4,6,8]
param_grid = dict(n_estimators = n_estimators)
kfold = StratifiedKFold(n_splits = 8, shuffle = True, random_state = 7)
grid_search = GridSearchCV(model, param_grid, scoring = "neg_log_loss", n_jobs = -1, cv = kfold, verbose = 3)
result = grid_search.fit(X_top25, label_encoder_y)
print("Best paramter is %s " % result.best_params_)
# Mean score for all the paramters tested
pd.DataFrame({"params": result.cv_results_['params'], "mean_score": result.cv_results_['mean_test_score'],
             "std_score": result.cv_results_['std_test_score']})
plt.errorbar(n_estimators, result.cv_results_['mean_test_score'], yerr = result.cv_results_['std_test_score'])
plt.xlabel("n_estimators")
plt.ylabel("Log Loss")
plt.show()
# Grid Search for learning rate
model = xg.XGBClassifier()
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(learning_rate = learning_rate, n_estimators = [150])
kfold = StratifiedKFold(n_splits = 8, shuffle = True, random_state = 7)
grid_search = GridSearchCV(model, param_grid, scoring = "neg_log_loss", n_jobs = -1, cv = kfold, verbose = 1)
result = grid_search.fit(X, label_encoder_y)
print("Best paramter is %s " % result.best_params_)
print("Best score is %f" % result.best_score_)
pd.DataFrame({"params": result.cv_results_['params'], "mean_score": result.cv_results_['mean_test_score'],
             "std_score": result.cv_results_['std_test_score']})
plt.errorbar(learning_rate, result.cv_results_['mean_test_score'], yerr = result.cv_results_['std_test_score'])
plt.xlabel("Learning_rate")
plt.ylabel("Log Loss")
plt.show()
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

# Running a XGBoost with less column sample.
model = xg.XGBClassifier(n_estimators = 150, learning_rate = 0.2)
model.fit(X_train, y_train)
# Check the accuracy of the model on train and test dataset.
accuracy_train = accuracy_score(y_train, model.predict(X_train))
print("Accuracy on train dataset %.2f%%" % (accuracy_train * 100))

accuracy_test = accuracy_score(y_test, model.predict(X_test))
print("Accuracy on test dataset %.2f%%" % (accuracy_test * 100))
# Train and test split of the data
X_train, X_test, y_train, y_test = train_test_split(X_top25, label_encoder_y, test_size = 0.33, random_state = 7)

# Running a XGBoost with less column sample.
model = xg.XGBClassifier(n_estimators = 150, learning_rate = 0.2)
model.fit(X_train, y_train)
# Check the accuracy of the model on train and test dataset.
accuracy_train = accuracy_score(y_train, model.predict(X_train))
print("Accuracy on train dataset %.2f%%" % (accuracy_train * 100))

accuracy_test = accuracy_score(y_test, model.predict(X_test))
print("Accuracy on test dataset %.2f%%" % (accuracy_test * 100))
# Train and test split of the data
X_train, X_test, y_train, y_test = train_test_split(X_top25, label_encoder_y, test_size = 0.33, random_state = 7)

# Running a XGBoost with less column sample.
model = xg.XGBClassifier(n_estimators = 150, learning_rate = 0.2, colsample_bytree = 0.7)
model.fit(X_train, y_train)
# Check the accuracy of the model on train and test dataset.
accuracy_train = accuracy_score(y_train, model.predict(X_train))
print("Accuracy on train dataset %.2f%%" % (accuracy_train * 100))

accuracy_test = accuracy_score(y_test, model.predict(X_test))
print("Accuracy on test dataset %.2f%%" % (accuracy_test * 100))
dataset_test = dataset_pd2.values
# Selecting the top 25 variables.
data_top25_test = dataset_pd2[feature_imp[:25].index]
dataset_test = data_top25_test.values
prediction_sub = model.predict(dataset_test)

#dataset_pd2["prediction"] = prediction_sub
X_sub = np.array(prediction_sub).reshape(-1,1)
onehot_encoder = OneHotEncoder(sparse = False)
submission_file = onehot_encoder.fit_transform(X_sub)

submission_file_df = pd.DataFrame(submission_file, 
                                  columns = ['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6',
                                            'Class_7','Class_8','Class_9'], index = dataset_pd2.index)

submission_file_df.to_csv("submission_otto_ver2.csv")