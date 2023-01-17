# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# import pandas as pd
import pandas as pd

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# file stored in variable
main_file_path = '../input/train.csv'
# file import as dataframe
data = pd.read_csv(main_file_path)
# print columns of dataframe
print(data.columns)
# select survival column from dataframe and store it in variable
survival_data = data.Survived
# print top 5 values of Survived column
print(survival_data.head())
print(data.info())

# Creating the model
# select your target variable and store it as y
y = survival_data
# create the list of prediction columns named predictors
predictors = ['Pclass','SibSp','Parch','Fare']
# select predictors list data and store it as X
X = data[predictors]
# Define model
Survival_model = DecisionTreeClassifier()
# fit model
Survival_model.fit(X,y)
# predict using model
prediction = Survival_model.predict(X.head())
# print predictions
print("The predictions for first 5 passengers in training dataset")
print(prediction)
# test file stored in variable
test_file_path = '../input/test.csv'
# test file import as dataframe
test_data = pd.read_csv(test_file_path)
# print columns of dataframe
print(test_data.columns)
# extract predictors list data from test_data
test_data_predictors = test_data[predictors]
test_predictions = Survival_model.predict(test_data_predictors.head())
print("predictions for first five test data")
print(test_predictions)
# import train_test_split to split the data into train and test data

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
# Define model
Survival_model = DecisionTreeClassifier()
# Fit model
Survival_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = Survival_model.predict(val_X)
print(val_predictions)
forest_survival_model = RandomForestClassifier()
forest_survival_model.fit(train_X, train_y)

# predict first 5 test data samples using Random forest classifier
test_predictions = forest_survival_model.predict(test_data_predictors.head())
print("predictions for first five test data")
print(test_predictions)
# considering all columns for prediction
survival_predictors = data.drop((['Survived','Name']), axis=1)
test_data_predictors = test_data.drop((['Name']), axis=1)


# For the sake of keeping the example simple, we'll use only numeric predictors first
survival_numeric_predictors = survival_predictors.select_dtypes(exclude=['object'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(survival_numeric_predictors, 
                                                    y,
                                                    train_size=0.8, 
                                                    test_size=0.2, 
                                                    random_state=0)
# get one-hot encodings for categorical data for test_data
one_hot_encoded_test_data = pd.get_dummies(test_data_predictors)
# checking the data types for one-hot encoded data
one_hot_encoded_test_data.dtypes.sample(10)
# get one-hot encodings for categorical data in train data
one_hot_survival_predictors = pd.get_dummies(survival_predictors)
# checking the data types for one-hot encoded train data
one_hot_survival_predictors.dtypes.sample(10)
from sklearn.preprocessing import Imputer

my_imputer = Imputer()
# splitting of data after one hot encoding i.e. change of categorical data to numericals
one_hot_X_train, one_hot_X_test, one_hot_y_train, one_hot_y_test = train_test_split(one_hot_survival_predictors, 
                                                    y,
                                                    train_size=0.8, 
                                                    test_size=0.2, 
                                                    random_state=0)

imputed_one_hot_X_train_plus = one_hot_X_train.copy()
imputed_one_hot_X_test_plus = one_hot_X_test.copy()


cols_with_missing = (col for col in one_hot_X_train.columns 
                                 if one_hot_X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_one_hot_X_train_plus[col + '_was_missing'] = imputed_one_hot_X_train_plus[col].isnull()
    imputed_one_hot_X_test_plus[col + '_was_missing'] = imputed_one_hot_X_test_plus[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_one_hot_X_train_plus = my_imputer.fit_transform(imputed_one_hot_X_train_plus)
imputed_one_hot_X_test_plus = my_imputer.transform(imputed_one_hot_X_test_plus)
# Ensure the test data is encoded in the same manner as the training data with the align command
one_hot_encoded_training_predictors = pd.get_dummies(survival_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_data_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)

#splitting of data after alignment of test data with training data
one_hot_X_train, one_hot_X_test, one_hot_y_train, one_hot_y_test = train_test_split(final_train, 
                                                    y,
                                                    train_size=0.8, 
                                                    test_size=0.2, 
                                                    random_state=0)

imputed_one_hot_X_train_plus = one_hot_X_train.copy()
imputed_one_hot_X_test_plus = one_hot_X_test.copy()
imputed_one_hot_test_plus = final_test.copy()

cols_with_missing = (col for col in one_hot_X_train.columns 
                                 if one_hot_X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_one_hot_X_train_plus[col + '_was_missing'] = imputed_one_hot_X_train_plus[col].isnull()
    imputed_one_hot_X_test_plus[col + '_was_missing'] = imputed_one_hot_X_test_plus[col].isnull()
    imputed_one_hot_test_plus[col + '_was_missing'] = imputed_one_hot_test_plus[col].isnull()
# Imputation along with imputation of test data
my_imputer = Imputer()
imputed_one_hot_X_train_plus = my_imputer.fit_transform(imputed_one_hot_X_train_plus)
imputed_one_hot_X_test_plus = my_imputer.transform(imputed_one_hot_X_test_plus)
imputed_one_hot_test_plus = my_imputer.transform(imputed_one_hot_test_plus)
# running random forest model on one hot encoded plus impututed data
model = RandomForestClassifier()
model.fit(imputed_one_hot_X_train_plus, y_train)
# prediction on test data
preds = model.predict(imputed_one_hot_test_plus)

acc_RF = round(model.score(imputed_one_hot_X_train_plus, y_train) * 100, 2)
acc_RF
# create csv output file for submission
my_submission = pd.DataFrame({'PassengerId': final_test.PassengerId, 'Survived': preds})
# you could use any filename. We choose submission_6 here
my_submission.to_csv('submission.csv', index=False)
# support vector machine
svc = SVC()
svc.fit(imputed_one_hot_X_train_plus, y_train)
preds = svc.predict(imputed_one_hot_test_plus)
acc_svc = round(svc.score(imputed_one_hot_X_train_plus, y_train) * 100, 2)
acc_svc
# Decision tree classifier 

decision_tree = DecisionTreeClassifier()
decision_tree.fit(imputed_one_hot_X_train_plus, y_train)
preds = decision_tree.predict(imputed_one_hot_test_plus)
acc_decision_tree = round(decision_tree.score(imputed_one_hot_X_train_plus, y_train) * 100, 2)
acc_decision_tree
# create csv output file for submission
my_submission = pd.DataFrame({'PassengerId': final_test.PassengerId, 'Survived': preds})
# you could use any filename. We choose submission_6 here
my_submission.to_csv('submission_2.csv', index=False)
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(imputed_one_hot_X_train_plus, y_train)
preds = sgd.predict(imputed_one_hot_test_plus)
acc_sgd = round(sgd.score(imputed_one_hot_X_train_plus, y_train) * 100, 2)
acc_sgd
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(imputed_one_hot_X_train_plus, y_train)
preds = logreg.predict(imputed_one_hot_test_plus)
acc_log = round(logreg.score(imputed_one_hot_X_train_plus, y_train) * 100, 2)
acc_log
# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(imputed_one_hot_X_train_plus, y_train)
preds = knn.predict(imputed_one_hot_test_plus)
acc_knn = round(knn.score(imputed_one_hot_X_train_plus, y_train) * 100, 2)
acc_knn
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(imputed_one_hot_X_train_plus, y_train)
preds = gaussian.predict(imputed_one_hot_test_plus)
acc_gaussian = round(gaussian.score(imputed_one_hot_X_train_plus, y_train) * 100, 2)
acc_gaussian
# Perceptron

perceptron = Perceptron()
perceptron.fit(imputed_one_hot_X_train_plus, y_train)
preds = perceptron.predict(imputed_one_hot_test_plus)
acc_perceptron = round(perceptron.score(imputed_one_hot_X_train_plus, y_train) * 100, 2)
acc_perceptron
# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(imputed_one_hot_X_train_plus, y_train)
preds = linear_svc.predict(imputed_one_hot_test_plus)
acc_linear_svc = round(linear_svc.score(imputed_one_hot_X_train_plus, y_train) * 100, 2)
acc_linear_svc
# running XGboost with categorical data
from xgboost import XGBClassifier
model = XGBClassifier(learning_rate=0.05 )
model.fit(imputed_one_hot_X_train_plus, y_train)
# prediction of house price on test data
preds = model.predict(imputed_one_hot_test_plus)
acc_xgboost = round(model.score(imputed_one_hot_X_train_plus, y_train) * 100, 2)
acc_xgboost
from catboost import CatBoostClassifier
model = CatBoostClassifier(learning_rate=0.05)
model.fit(imputed_one_hot_X_train_plus, y_train)
# prediction of house price on test data
preds = model.predict(imputed_one_hot_test_plus)
acc_catboost = round(model.score(imputed_one_hot_X_train_plus, y_train) * 100, 2)
acc_catboost
# create csv output file for submission
my_submission = pd.DataFrame({'PassengerId': final_test.PassengerId, 'Survived': preds})
# you could use any filename. We choose submission_3 here
my_submission.to_csv('submission_3.csv', index=False)
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree','XGBoost Classifier','CatBoost Classifier'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_RF, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree,acc_xgboost,acc_catboost]})
models.sort_values(by='Score', ascending=False)