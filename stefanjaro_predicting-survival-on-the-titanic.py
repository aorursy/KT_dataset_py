# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# configure matplotlib
plt.rcParams["figure.figsize"] = 12,8
# import data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# info
train.info()
# describe
train.describe()
# head
train.head()
# print out the unique values of certain variables
for col in ["Pclass", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]:
    print(f"Variable: {col}")
    print(train[col].unique())
    print("\n")
# are there as many tickets as there are passengers?
print(len(train["PassengerId"].unique()) == len(train["Ticket"].unique()))
# so it seems that multiple people boarded with one ticket then
# but are there enough shared tickets to constitute using the column as a variable?
len(train["PassengerId"].unique()) - len(train["Ticket"].unique())
# are there cases where one record is assigned to multiple cabins?
train[train["Cabin"] == "C23 C25 C27"]
# who are the people with no embarked value?
train[pd.isnull(train["Embarked"])]
# distribution of fares and ages
f,ax = plt.subplots(1,2)
ax = ax.ravel()

ax[0].hist(train.Fare, bins=100)
ax[1].hist(train[pd.notnull(train["Age"])]["Age"], bins=100) # excluding null ages
plt.show()
# counts of pclass, sex, sibsp, parch, embarked, and survived
f, ax = plt.subplots(3,2)
f.tight_layout(h_pad=4, w_pad=2)
ax = ax.ravel()
for pos, col in enumerate(["Pclass", "Sex", "SibSp","Parch", "Embarked", "Survived"]):
    vis1 = sns.countplot(data=train, x=col, ax=ax[pos])
    vis1.set_title(col)
    vis1.set_xlabel("")
# get the median value for age from the training data, this will be used on the test data as well
median_age = train["Age"].median()
# get the median fare from the training data since the test dataset has one fare value missing
median_fare = train["Fare"].median()
# cleaning the training dataset

# take a copy of the data so we won't be changing the original dataframe
traincp = train.copy()
# impute nan values for age using the median we calculated before
traincp["Age"] = traincp["Age"].fillna(value=median_age)
# replace nan values for cabin with "Shared"
traincp["Cabin"] = traincp["Cabin"].fillna(value="Shared")
# remove name and passenger ID column from the training data
traincp = traincp.drop(columns=["Name", "PassengerId"])
# turn PClass, Ticket, Cabin, Embarked and Sex into dummies
traincp = pd.get_dummies(data=traincp, columns=["Pclass", "Sex", "Cabin", "Embarked", "Ticket"])
# divide train_clean into the dependent and independent variables
y_interim = traincp["Survived"]
X_interim = traincp.iloc[:, 1:]
# clean the test dataset

# take a copy of the data so we won't be changing the original dataframe
testcp = test.copy()
# impute nan values for age using the median we calculated before
testcp["Age"] = testcp["Age"].fillna(value=median_age)
# impute nan values for fare using the median we calculated before
testcp["Fare"] = testcp["Fare"].fillna(value=median_fare)
# replace nan values for cabin with "Shared"
testcp["Cabin"] = testcp["Cabin"].fillna(value="Shared")
#remove name and passenger ID column from the training data
testcp = testcp.drop(columns=["Name", "PassengerId"])
# turn PClass, Ticket, Cabin, Embarked and Sex into dummies
testcp = pd.get_dummies(data=testcp, columns=["Pclass", "Sex", "Cabin", "Embarked", "Ticket"])
# assign testcp to X_test
X_test = testcp.copy()
# get a list of the columns in the training dataset that aren't in the test dataset since we need to add them to the test dataset
# likewise get a list of the columns in the test dataset that aren't in the training dataet
cols_not_in_test = []
cols_not_in_train = []

for col in X_interim.columns:
    if col not in X_test.columns:
        cols_not_in_test.append(col)
        
for col in X_test.columns:
    if col not in X_interim.columns:
        cols_not_in_train.append(col)
# add columns not in the test/train dataset to it with a value of 0 for every record
for col in cols_not_in_test:
    X_test[col] = np.zeros((len(X_test)))

for col in cols_not_in_train:
    X_interim[col] = np.zeros((len(X_interim)))
# check if both the training data and test data have the same number of columns
len(X_test.columns) == len(X_interim.columns)
# to avoid the dummy variable trap we need to drop one column from the training and test datasets
# since if Pclass_1 and Pclass_2 are equal to 0, Pclass_3 will automatically be 1, we can drop Pclass_3
X_test = X_test.drop(columns=["Pclass_3"])
X_interim = X_interim.drop(columns=["Pclass_3"])
# divide the training data into training data and cross validation data
from sklearn.model_selection import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(X_interim, y_interim, test_size=0.25, random_state=0)
# import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# train a logistic regression
def train_log_reg(X_train, y_train, X_cv):
    # train
    log_reg = LogisticRegression(solver="lbfgs", max_iter=1000)
    log_reg.fit(X_train, y_train)
    # predict using X_cv
    y_pred = log_reg.predict(X_cv)
    return y_pred

# train a decision tree model
def train_dec_tree(X_train, y_train, X_cv):
    # train
    dec_tree = DecisionTreeClassifier()
    dec_tree.fit(X_train, y_train)
    # predict
    y_pred = dec_tree.predict(X_cv)
    return y_pred

# train a random forest model
def train_rand_for(X_train, y_train, X_cv, max_depth=None, random_state=None):
    # train
    rand_for = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=random_state)
    rand_for.fit(X_train, y_train)
    # predict
    y_pred = rand_for.predict(X_cv)
    return y_pred
# import metric calculators
from sklearn.metrics import accuracy_score, precision_score, recall_score

# function to calculate a classifier model's metrics
def evaluate_model(y_true, y_pred):
    # calculate the scores
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = ((precision * recall) / (precision + recall)) * 2
    return accuracy, f1
# function to train a model a given number of times and plot the accuracy and f1 metrics
def train_and_plot(model, X_train, y_train, X_cv, y_cv, training_times=10):
    # lists to store y axis data
    accuracy_y_axis, f1_y_axis = [], []
    for i in range(0, training_times):
        # train the model
        y_pred = model(X_train, y_train, X_cv)
        # get the evaluation metrics
        accuracy, f1 = evaluate_model(y_cv, y_pred)
        # append to y_axis lists
        accuracy_y_axis.append(accuracy)
        f1_y_axis.append(f1)
    # plot the graphs
    plt.plot(range(0, training_times), accuracy_y_axis, label="Accuracy")
    plt.plot(range(0, training_times), f1_y_axis, label="F1")
    plt.legend(loc="upper right")
    plt.show()
# check the logistic regression model
train_and_plot(train_log_reg, X_train, y_train, X_cv, y_cv, training_times=10)
# check the decision tree model
train_and_plot(train_dec_tree, X_train, y_train, X_cv, y_cv, training_times=10)
# check the random forest model
train_and_plot(train_rand_for, X_train, y_train, X_cv, y_cv, training_times=10)
# determine the best max_depth for the random forest model
accuracy_y_axis, f1_y_axis = [], [] # to store the y_axis_data
num_of_depths = 500

# test out various depths keeping the random state static
for depth in range(1, num_of_depths):
    y_pred = train_rand_for(X_train, y_train, X_cv, max_depth=depth, random_state=9631)
    accuracy, f1 = evaluate_model(y_cv, y_pred)
    accuracy_y_axis.append(accuracy)
    f1_y_axis.append(f1)
# plot results
plt.plot(range(0, num_of_depths-1), accuracy_y_axis, label="Accuracy")
plt.plot(range(0, num_of_depths-1), f1_y_axis, label="F1")
plt.legend(loc="upper left")
plt.show()
# train the random forest model
rand_for = RandomForestClassifier(max_depth=120)
rand_for.fit(X_train, y_train)
# predict on X_test
y_pred_final = rand_for.predict(X_test)
# create the submission document
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": y_pred_final})
submission.to_csv("submission.csv", index=False)
# # REMOVED CODE

# # get only the letter of the cabin number and assign it to a new column
# traincp = traincp.assign(cabin_new = traincp["Cabin"].apply(lambda x: x[0] if pd.isnull(x) == False else np.nan))
# traincp["cabin_new"] = traincp["cabin_new"].fillna(value="Shared")
# # assign a value of group to non-unique tickets and individual to unique tickets
# traincp = traincp.assign(ticket_new = traincp["Ticket"].apply(lambda x: x if train_tick_freq[x] > 1 else "Individual"))
# traincp = traincp.drop(columns=["Name", "PassengerId", "Cabin", "Ticket"])
# traincp = pd.get_dummies(data=traincp, columns=["Pclass", "Sex", "cabin_new", "Embarked", "ticket_new"])
# # get only the letter of the cabin number and assign it to a new column
# testcp = testcp.assign(cabin_new = testcp["Cabin"].apply(lambda x: x[0] if pd.isnull(x) == False else np.nan))
# testcp["cabin_new"] = testcp["cabin_new"].fillna(value="Shared")
# # assign a value of group to non-unique tickets and individual to unique tickets
# testcp = testcp.assign(ticket_new = testcp["Ticket"].apply(lambda x: x if test_tick_freq[x] > 1 else "Individual"))
# testcp = testcp.drop(columns=["Name", "PassengerId", "Cabin", "Ticket"])
# testcp = pd.get_dummies(data=testcp, columns=["Pclass", "Sex", "cabin_new", "Embarked", "ticket_new"])