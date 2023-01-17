#importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_style("darkgrid")
%matplotlib inline
#reading dataset
dataset = pd.read_csv('../input/train.csv', index_col='id')
#Data Analysis
dataset.describe(include = 'all')
sns.heatmap(dataset.corr(), annot = True, fmt = ".2f")
fig, ax = plt.subplots(figsize= (15, 8))
sns.countplot(x='color', hue='type', data=dataset, ax= ax)
#g = sns.FacetGrid(data = dataset, col="type")
#g = g.map(plt.hist, "color", color="m")
g = sns.factorplot(x = "color", col="type", kind="count", data=dataset)
sns.pairplot(dataset, hue="type")
sns.lmplot(x = "hair_length", y = "has_soul", hue="type", data=dataset,
           fit_reg=False)
df_goblin = dataset[dataset['type'] == 'Goblin']
df_ghoul = dataset[dataset['type'] == 'Ghoul']
df_ghost = dataset[dataset['type'] == 'Ghost']

#PDF of types
ax = sns.kdeplot(df_goblin['has_soul'], shade=True, label = "Goblin")
ax = sns.kdeplot(df_ghoul['has_soul'], shade=True, label = "Ghoul")
ax = sns.kdeplot(df_ghost['has_soul'], shade=True, label = "Ghost")
#Data Preprocessing
X = dataset.iloc[:, :5].values
y = dataset.iloc[:, 5:].values
#Encoding Categorical Data(Independent)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_X = LabelEncoder()
X[:, 4] = LabelEncoder_X.fit_transform(X[:, 4])
OneHotEncoder_X = OneHotEncoder(categorical_features = [4])
X = OneHotEncoder_X.fit_transform(X).toarray()

#Encoding Categorical Data(Dependent)
LabelEncoder_y = LabelEncoder()
y = LabelEncoder_y.fit_transform(y)

#Drop one columns of X for dummy variable trap
X = X[:, 1:]
#Splitting data to test and traning sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 0)
#Data Modelling 
#Applying Logistic Regression to training set
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression()
classifier_LR.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifier_LR.predict(X_test)

#Getting accuracy score
from sklearn.metrics import accuracy_score
LR_Score = accuracy_score(y_test, y_pred)
print("Logistic Reg. Score: " + str(LR_Score))
#applying kernel SVM to training set
from sklearn.svm import SVC
classifier_SVC = SVC(kernel = 'rbf', random_state = 0)
classifier_SVC.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifier_SVC.predict(X_test)

#Getting accuracy score
SVM_Score = accuracy_score(y_test, y_pred)
print("Kernel SVM Score: " + str(SVM_Score))
#applying random forest to training set
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 100, criterion = "entropy",
                                       random_state = 0)
classifier_RF.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifier_RF.predict(X_test)

#Getting accuracy score
RF_Score = accuracy_score(y_test, y_pred)
print("Random Forest Score : " + str(RF_Score))
#applying xgboost to training set
from xgboost import XGBClassifier
classifier_XGB = XGBClassifier()
classifier_XGB.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifier_XGB.predict(X_test)

#Getting accuracy score
XG_Score = accuracy_score(y_test, y_pred)
print("XGBoost Score : " + str(XG_Score))
Scores_dict = {'Logistic Reg.': LR_Score, 'Kernel SVM': SVM_Score, 
               'Random Forest': RF_Score, 'XGBoost': XG_Score}
df_scores = pd.DataFrame.from_dict(data = Scores_dict, orient = "index")
df_scores.plot.bar(legend = False)
importances = classifier_RF.feature_importances_
std = np.std([tree.feature_importances_ for tree in classifier_RF.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
# Plot the feature importances of the forest
plt.figure(figsize = (15, 5))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
#dropping color columns
X_train = X_train[:, 5:]
X_test = X_test[:, 5:]
#fit again random forest
classifier_RF = RandomForestClassifier(n_estimators = 1000, criterion = "entropy",
                                       max_depth=7,random_state = 0)
classifier_RF.fit(X_train, y_train)
#Predicting the test set results
y_pred = classifier_RF.predict(X_test)
#Getting accuracy score
RF_Score = accuracy_score(y_test, y_pred)
print("accuracy score without color : " + str(RF_Score))
#model selection
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_RF, X = X_train, y = y_train, cv = 10)
print("accurasies mean: " + str(accuracies.mean()))
print("accurasies std : " + str(accuracies.std()))
# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = {"n_estimators" : [1, 10, 100, 256, 512, 1024],
              "min_samples_leaf": [1,2,3,4,5],
              "min_samples_split" : [2,3,4,5],
              "max_depth" : [1, 2, 4, 8, 16]}
grid_search = GridSearchCV(estimator = classifier_RF, param_grid = parameters,
                           scoring = "accuracy", n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print("best score : " + str(best_accuracy))
print("best parameters : " + str(best_parameters))