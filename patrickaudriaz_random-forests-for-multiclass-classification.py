import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt 
%matplotlib inline
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


import os
print(os.listdir("../input"))
train = shuffle(pd.read_csv("../input/train.csv"))
test = shuffle(pd.read_csv("../input/test.csv"))
#Frequency distribution of classes"
train_outcome = pd.crosstab(index=train["Activity"],  # Make a crosstab
                              columns="count")      # Name the count column

train_outcome

temp = train["Activity"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values})


labels = df['labels']
sizes = df['values']

x_pos = [i for i, _ in enumerate(labels)]

plt.figure(1, [14, 6])
plt.bar(x_pos, sizes,width=0.6)

plt.xticks(x_pos, labels)
# Seperating Predictors and Outcome values from train and test sets
X_train = pd.DataFrame(train.drop(['Activity','subject'],axis=1))
Y_train_label = train.Activity.values.astype(object)

X_test = pd.DataFrame(test.drop(['Activity','subject'],axis=1))
Y_test_label = test.Activity.values.astype(object)

# Dimension of Train and Test set 
print("Dimension of Train set",X_train.shape)
print("Dimension of Test set",X_test.shape,"\n")

# Transforming non numerical labels into numerical labels
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()

# encoding train labels 
encoder.fit(Y_train_label)
Y_train = encoder.transform(Y_train_label)

# encoding test labels 
encoder.fit(Y_test_label)
Y_test = encoder.transform(Y_test_label)

#Total Number of Continous and Categorical features in the training set
num_cols = X_train._get_numeric_data().columns
print("Number of numeric features:",num_cols.size)
#list(set(X_train.columns) - set(num_cols))


names_of_predictors = list(X_train.columns.values)

# Scaling the Train and Test feature set 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_estimators = [25]
max_depth = [25]
min_samples_leaf = [2]
bootstrap = [True, False]

param_grid = {
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "min_samples_leaf": min_samples_leaf,
    "bootstrap": bootstrap,
}

rf = RandomForestRegressor(random_state=42)

rf_model = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, verbose=10, n_jobs=-1)
rf_model.fit(X_train_scaled, Y_train)

print("Using hyperparameters --> \n", rf_model.best_params_)
rf = RandomForestRegressor(random_state = 42)

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, n_iter=50, cv =2, verbose = 10, random_state=42, n_jobs = 10)
rf_random.fit(X_train_scaled, Y_train)

# svm_model = GridSearchCV(estimator = rf, param_grid =param_grid, cv = 3, verbose = 10, n_jobs = 6)
# svm_model.fit(X_train_scaled, Y_train)
rf_model.best_params_
rf_model.best_score_
rf_model.best_estimator_
print(rf_model.best_params_)
Y_pred = rf_model.predict(X_test_scaled)
Y_pred_label = list(encoder.inverse_transform(Y_pred))

print("Training set score for SVM: %f" % rf_random.score(X_train_scaled , Y_train))
print("Testing  set score for SVM: %f" % rf_random.score(X_test_scaled  , Y_test ))
# View the accuracy score
print('Best score for training data:', rf_random.best_score_,"\n") 

'''
# View the best parameters for the model found using grid search
print('Best C:',rf_random.best_estimator_.C,"\n") 
print('Best Kernel:',rf_random.best_estimator_.kernel,"\n")
print('Best Gamma:',rf_random.best_estimator_.gamma,"\n")
'''

final_model = rf_random.best_estimator_
Y_pred = final_model.predict(X_test_scaled)
Y_pred_label = list(encoder.inverse_transform(Y_pred))
# Making the Confusion Matrix
print(confusion_matrix(Y_test_label,Y_pred_label))
print("\n")
print(classification_report(Y_test_label,Y_pred_label))

print("Training set score for SVM: %f" % final_model.score(X_train_scaled , Y_train))
print("Testing  set score for SVM: %f" % final_model.score(X_test_scaled  , Y_test ))

rf_random.score
