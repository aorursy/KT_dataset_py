import pandas as pd               # For manipulation of data
import numpy as np                # For mathematical computation of data
import matplotlib.pyplot as plt   # For visualization
import seaborn as sns             # Better visualization as built on top of matplolib
# Read the dataset stored in github repo
pharma_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/pharma_data/Training_set_begs.csv')
dataset = pharma_data.copy()
# let's observe first five rows of dataset
dataset.head()
# Basic info
dataset.info()
# Let's have a look at columns and their datatypes
from tabulate import tabulate

def get_col_names(dataset):
    col_list = list(dataset.columns)
    dtype_list = list(dataset.dtypes)
    table = dict()
    table["cols"] = col_list
    table["dtype"] = dtype_list
    return print(tabulate(table, showindex="always", tablefmt="grid"))
        
get_col_names(dataset)
# Let's look at missing values
import missingno
missingno.matrix(dataset);
# Percentage of missing values of dataset
missing = dataset.isna().sum()
percent = (missing.sum() / len(dataset)) * 100
percent
# Deleting records of people whose age is greater than 100
new_dataset = dataset[dataset["Patient_Age"] < 100]
len(new_dataset[new_dataset["Patient_Age"] > 100])
# Filling the missing values is other columns with mode
cols = ["A", "B", "C", "D", "E", "F", "Z", "Number_of_prev_cond"]
new_dataset[cols] = new_dataset[cols].fillna(new_dataset.mode().iloc[0])
# Dropping columns which are not needed for model
cols_to_drop = ["ID_Patient_Care_Situation", "Patient_ID", "Patient_mental_condition"]
new_dataset = new_dataset.drop(columns=cols_to_drop, axis=1)
# Label encoding the Treated_with_drugs column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
new_dataset["Treated_with_drugs"] = le.fit_transform(new_dataset["Treated_with_drugs"])
# One hot encoding the Patient_Smoker and Patient_Rural_Urban columns and dropping the originals
dummy = pd.get_dummies(new_dataset["Patient_Smoker"], prefix="smoker_")
new_dataset = pd.concat([new_dataset, dummy], axis=1)

dummy_2 = pd.get_dummies(new_dataset["Patient_Rural_Urban"], prefix="patient_")
new_dataset = pd.concat([new_dataset, dummy_2], axis=1)

new_dataset = new_dataset.drop(columns=["Patient_Smoker", "Patient_Rural_Urban"], axis=1)
# Splitting the dataset
X = new_dataset.drop(columns="Survived_1_year")
y = new_dataset["Survived_1_year"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7, shuffle=True)
print(f"X_train={X_train.shape}  X_test={X_test.shape}")
# Initializing the model
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
model = LGBMClassifier()
model.fit(X_train, y_train)
# Performing cross-validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1, error_score='raise')
np.mean(n_scores)
from sklearn.metrics import f1_score

y_preds_train = model.predict(X_train)
print(f1_score(y_train, y_preds_train))

y_preds_test = model.predict(X_test)
print(f1_score(y_test, y_preds_test))
# Visualizing the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_preds_train)
sns.heatmap(cm, annot=True, fmt="d");
cm = confusion_matrix(y_test, y_preds_test)
sns.heatmap(cm, annot=True, fmt="d");
from sklearn.metrics import plot_roc_curve
plot_roc_curve(model, X_test, y_test)
plt.show()
# Tuning the model
from sklearn.model_selection import GridSearchCV

parameter_space = {
    'boosting_type': ["gbdt", "dart"],
    'num_leaves': [31, 62],
    'max_depth': [3, 5, 10, 15],
    'n_estimators': [200, 300, 500]
}

from sklearn.model_selection import GridSearchCV
lgbm_gscv = GridSearchCV(model, parameter_space, scoring = 'f1')
lgbm_gscv.fit(X_train, y_train)
lgbm_gscv.best_params_
# Fitting the tuned model, making predictions & calculating f1 score
lgbm_grid = LGBMClassifier(boosting_type="gbdt", max_depth=3, n_estimators=500, num_leaves=31)
lgbm_grid.fit(X_train, y_train)

y_train_pred_2 = lgbm_grid.predict(X_train)
y_test_pred_2 = lgbm_grid.predict(X_test)

f1_tuned_train = f1_score(y_train, y_train_pred_2)
f1_tuned_test = f1_score(y_test, y_test_pred_2)
print(f"F1 of tuned model on train data = {f1_tuned_train}")
print(f"F1 of tuned model on validation data = {f1_tuned_test}")
cm = confusion_matrix(y_train, y_train_pred_2)
sns.heatmap(cm, annot=True, fmt="d");
cm = confusion_matrix(y_test, y_test_pred_2)
sns.heatmap(cm, annot=True, fmt="d");
from sklearn.metrics import plot_roc_curve
plot_roc_curve(lgbm_grid, X_test, y_test)
plt.show()
# Getting the test dataset
test_new = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/pharma_data/Testing_set_begs.csv')
test_new.head()
# Dropping irrelevant columns
cols_to_drop = ["ID_Patient_Care_Situation", "Patient_ID", "Patient_mental_condition"]
test_new = test_new.drop(columns=cols_to_drop, axis=1)
test_new.head()
# Label encoding and one hot encoding the columns
le = LabelEncoder()
test_new["Treated_with_drugs"] = le.fit_transform(test_new["Treated_with_drugs"])

dummy = pd.get_dummies(test_new["Patient_Smoker"], prefix="smoker_")
test_new = pd.concat([test_new, dummy], axis=1)

dummy_2 = pd.get_dummies(test_new["Patient_Rural_Urban"], prefix="patient_")
test_new = pd.concat([test_new, dummy_2], axis=1)
# Dropping the original columns
test_new = test_new.drop(columns=["Patient_Smoker", "Patient_Rural_Urban"], axis=1)
test_new.shape
# Making predictions on test data and saving
predictions = lgbm_grid.predict(test_new)
res = pd.DataFrame(predictions) 
res.index = test_new.index
res.columns = ["prediction"]
res.to_csv("prediction_results_HP.csv")
