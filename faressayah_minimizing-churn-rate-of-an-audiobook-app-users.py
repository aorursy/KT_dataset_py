import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set_style("whitegrid")



pd.set_option("display.max_columns", 80)

pd.set_option("display.max_rows", 80)

pd.set_option("display.float_format", "{:.2f}".format)
data = pd.read_csv("/kaggle/input/audiobook-app-data/audiobook_data_2.csv", index_col=0)

data.head()
data.describe()
data.isnull().sum()
data.info()
data['Book_length(mins)_overall'].value_counts()
def book_length(length):

    if length > 1200:

        return 1

    else:

        return 0

    

data['purchases_hour_>3h'] = data['Book_length(mins)_overall'].apply(book_length)
data['Book_length(mins)_avg'].apply(book_length).value_counts()
data['purchases_hour_>3h'].value_counts()
columns = ['purchases_hour_>3h', 'Book_length(mins)_overall', 'Book_length(mins)_avg']

plt.figure(figsize=(12, 7))



for i, column in enumerate(columns, 1):

    plt.subplot(2, 2, i)

    data[data["Target"] == 0][column].hist(bins=35, color='blue', label='Bought Again = NO', alpha=0.6)

    data[data["Target"] == 1][column].hist(bins=35, color='red', label='Bought Again = YES', alpha=0.6)

    plt.legend()

    plt.xlabel(column)
columns = ["Price_overall", "Price_avg"]

plt.figure(figsize=(12, 7))

df = data[(data.Price_overall < 20) & (data.Price_avg < 20)]



for i, column in enumerate(columns, 1):

    plt.subplot(2, 2, i)

    df[df["Target"] == 0][column].hist(bins=35, color='blue', label='Bought Again = NO', alpha=0.6)

    df[df["Target"] == 1][column].hist(bins=35, color='red', label='Bought Again = YES', alpha=0.6)

    plt.legend()

    plt.xlabel(column)
print(data[data['Review'] == 0].Target.value_counts(normalize=True))

print(data[data['Review'] == 1].Target.value_counts(normalize=True))
data['Review10/10'].value_counts()
columns = ["Review", "Review10/10"]

plt.figure(figsize=(12, 7))



for i, column in enumerate(columns, 1):

    plt.subplot(2, 2, i)

    data[data["Target"] == 0][column].hist(bins=35, color='blue', label='Bought Again = NO', alpha=0.6)

    data[data["Target"] == 1][column].hist(bins=35, color='red', label='Bought Again = YES', alpha=0.6)

    plt.legend()

    plt.xlabel(column)
def listened_to_books(minutes):

    if minutes > 0.0:

        return 0

    else:

        return 1

data['listened_to_books'] = data.Minutes_listened.apply(listened_to_books)
def completion_state(minutes):

    if minutes > 0.5:

        return 1

    else:

        return 0

data['completion_state'] = data.Completion.apply(completion_state)
columns = ["Minutes_listened", "Completion", "listened_to_books", "completion_state"]

plt.figure(figsize=(12, 7))



for i, column in enumerate(columns, 1):

    plt.subplot(2, 2, i)

    data[data["Target"] == 0][column].hist(bins=35, color='blue', label='Bought Again = NO', alpha=0.6)

    data[data["Target"] == 1][column].hist(bins=35, color='red', label='Bought Again = YES', alpha=0.6)

    plt.legend()

    plt.xlabel(column)
data.drop('Minutes_listened', axis=1, inplace=True)
def asked_for_request(request):

    if request == 0:

        return 0

    else:

        return 1

    

data["asked_for_request"] = data.Support_Request.apply(asked_for_request)
def acc_purchases(purchase):

    if purchase == 0:

        return 0

    else:

        return 1

data['acc_purchases'] = data.Last_Visited_mins_Purchase_date.apply(acc_purchases)
data.Last_Visited_mins_Purchase_date.value_counts()
columns = ["Support_Request", "Last_Visited_mins_Purchase_date", "asked_for_request", "acc_purchases"]

plt.figure(figsize=(12, 7))



for i, column in enumerate(columns, 1):

    plt.subplot(2, 2, i)

    data[data["Target"] == 0][column].hist(bins=35, color='blue', label='Bought Again = NO', alpha=0.6)

    data[data["Target"] == 1][column].hist(bins=35, color='red', label='Bought Again = YES', alpha=0.6)

    plt.legend()

    plt.xlabel(column)
data.drop('Support_Request', axis=1, inplace=True)
print(f"{data.Target.value_counts()}")

print(f"{data.Target.value_counts()[0] / data.Target.value_counts()[1]}")
dummies = [column for column in data.drop('Target', axis=1).columns if data[column].nunique() < 10]
data_1 = pd.get_dummies(data, columns=dummies, drop_first=True)

data_1.head()
data_1.info()
# print(data_1.shape)



# # Remove duplicate Features

# data_1 = data_1.T.drop_duplicates()

# data_1 = data_1.T



# # Remove Duplicate Rows

# data_1.drop_duplicates(inplace=True)



# print(data_1.shape)
data_1.Target.value_counts()
from sklearn.model_selection import train_test_split



X = data_1.drop('Target', axis=1)

y = data_1.Target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score



def print_score(clf, X_train, y_train, X_test, y_test, train=True):

    if train:

        pred = clf.predict(X_train)

        print("Train Result:\n================================================")

        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")

        print("_______________________________________________")

        print("Classification Report:", end='')

        print(f"\tPrecision Score: {precision_score(y_train, pred) * 100:.2f}%")

        print(f"\t\t\tRecall Score: {recall_score(y_train, pred) * 100:.2f}%")

        print(f"\t\t\tF1 score: {f1_score(y_train, pred) * 100:.2f}%")

        print("_______________________________________________")

        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")

        

    elif train==False:

        pred = clf.predict(X_test)

        print("Test Result:\n================================================")        

        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")

        print("_______________________________________________")

        print("Classification Report:", end='')

        print(f"\tPrecision Score: {precision_score(y_test, pred) * 100:.2f}%")

        print(f"\t\t\tRecall Score: {recall_score(y_test, pred) * 100:.2f}%")

        print(f"\t\t\tF1 score: {f1_score(y_test, pred) * 100:.2f}%")

        print("_______________________________________________")

        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
from sklearn.linear_model import LogisticRegression



lr_classifier = LogisticRegression(solver='liblinear', penalty='l2')

lr_classifier.fit(X_train, y_train)



print_score(lr_classifier, X_train, y_train, X_test, y_test, train=True)

print_score(lr_classifier, X_train, y_train, X_test, y_test, train=False)
from sklearn.model_selection import cross_val_score



scores = cross_val_score(lr_classifier, X, y, cv=5)

print(f"Logistic Accuracy: {scores.mean() * 100:.2f}% +/- ({scores.std() * 100:.2f})")
zeros = (y_train.value_counts()[0] / y_train.shape)[0]

ones = (y_train.value_counts()[1] / y_train.shape)[0]



print(f"Doesn't purchase again users Rate: {zeros * 100:.2f}%")

print(f"Purchase again users Rate: {ones * 100 :.2f}%")
from sklearn.ensemble import RandomForestClassifier



rand_forest = RandomForestClassifier()

rand_forest.fit(X_train, y_train)



print_score(rand_forest, X_train, y_train, X_test, y_test, train=True)

print_score(rand_forest, X_train, y_train, X_test, y_test, train=False)
from sklearn.model_selection import GridSearchCV



rf_clf = RandomForestClassifier(n_estimators=100, oob_score=True, class_weight={0:zeros, 1:ones})



param_grid = {'n_estimators':[100, 500, 1000, 1500],

              'max_depth':[3, 5, 7, 10, 15, None], 

              'min_samples_split':[2, 3, 10], 

              'min_samples_leaf':[1, 3, 5, 7, 10], 

              'criterion':["gini", "entropy"]}



rf_grid_cv = GridSearchCV(rf_clf, param_grid, scoring="f1", n_jobs=-1, verbose=1, cv=3)

rf_grid_cv.fit(X_train, y_train)
rf_grid_cv.best_estimator_
rf_clf = RandomForestClassifier(criterion='entropy',

                                max_depth=15,

                                min_samples_leaf=1, 

                                min_samples_split=10,

                                n_estimators=500, 

                                oob_score=True, 

                                class_weight={0:zeros, 1:ones})



rf_clf.fit(X_train, y_train)

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)

print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)
scores = cross_val_score(rf_clf, X, y, cv=5, scoring='f1')

print(scores)

print(f"Random Forest F1_score: {scores.mean() * 100:.2f}% +/- ({scores.std() * 100:.2f})")
from xgboost import XGBClassifier



xgb_clf = XGBClassifier(learning_rate=0.5, 

                        n_estimators=150, 

                        base_score=0.3)

xgb_clf.fit(X_train, y_train)



print_score(xgb_clf, X_train, y_train, X_test, y_test, train=True)

print_score(xgb_clf, X_train, y_train, X_test, y_test, train=False)
from sklearn.model_selection import RandomizedSearchCV



xgb_clf = XGBClassifier(learning_rate=0.5, 

                        n_estimators=150, 

                        base_score=0.3)



hyperparameter_grid = {'colsample_bytree': [ 0.5, 0.75, 0.85, 0.9, 1], 

                       'colsample_bylevel': [ 0.5, 0.75, 0.85, 0.9, 1],

                       'colsample_bynode': [ 0.5, 0.75, 0.85, 0.9, 1],

#                        'learning_rate' : [0.01, 0.5, 0.1], 

#                        'n_estimators': [100, 350, 500],

                       'min_child_weight' : [2, 3, 5, 10],

                       'max_depth': [3, 5, 10, 15], 

#                        'base_score' : [0.1, 0.5, 0.9]

                      }



xgb_grid_cv = GridSearchCV(xgb_clf, hyperparameter_grid, scoring="f1", 

                           n_jobs=-1, verbose=1, cv=3)

# xgb_grid_cv.fit(X_train, y_train)
# xgb_grid_cv.best_estimator_
xgb_clf = XGBClassifier(base_score=0.3, 

                        min_child_weight=2,

                        max_depth=3,

                        colsample_bytree=0.85,

                        colsample_bylevel=0.5,

                        colsample_bynode=0.5,

                        learning_rate=0.5, 

                        n_estimators=150)



xgb_clf.fit(X_train, y_train)

print_score(xgb_clf, X_train, y_train, X_test, y_test, train=True)

print_score(xgb_clf, X_train, y_train, X_test, y_test, train=False)
scores = cross_val_score(xgb_clf, X, y, cv=5, scoring='f1')

print(scores)

print(f"XGBoost F1_score: {scores.mean() * 100:.2f}% +/- ({scores.std() * 100:.2f})")