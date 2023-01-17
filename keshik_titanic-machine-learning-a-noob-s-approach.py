# Load the relevant libraries
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.linear_model import LogisticRegression 
# Load data
X_data = pd.read_csv("../input/train.csv")

# Define the training and validation sets
X_train = X_data.sample(frac=0.8, random_state=100)
X_valid = X_data.drop(X_train.index)

# Define the labeled data for training and validation sets
Y_data = X_data["Survived"]
Y_train = X_train["Survived"]
Y_valid = X_valid["Survived"]

# Test data
X_test = pd.read_csv("../input/test.csv")
ID_test = X_test["PassengerId"]

# Display sample data
display(X_train.head(5))
display(Y_train.head(5))

# Errors
errors_list = dict()
# View summary of training data
display(X_train.describe())
display(Y_train.describe())
# Check if any columns has NaN or empty values
def sanity_check_NaN(df):
    output_dict = {}
    for column in df:
        if df[column].isnull().any():
            output_dict[column] = df[column].isnull().sum()
    
    return output_dict

sanity_check_results = sanity_check_NaN(X_data)
display(sanity_check_results)

# Also check for any duplicate data
display(X_data.duplicated().sum())
# Visualize the Age

fig, axs = plt.subplots(1, 2)

# Visualize age
age_non_na = X_data["Age"].dropna(inplace=False)
axs[0].hist(age_non_na.tolist())
axs[1].boxplot(age_non_na.tolist())
display(age_non_na.describe())
age_non_na.skew()

# Define a function to preprocess the data
def preprocess(df):
    
    # fill the missing ages
    df["Name"] = df["Name"].str.lower()
    #df.loc[(df["Age"].isnull()) & (df["Name"].str.contains("miss")), "Age"] = random.randrange(20, 28)
    #df.loc[(df["Age"].isnull()) & (df["Name"].str.contains("master")), "Age"] = random.randrange(1, 16)
    #df.loc[df["Age"].isnull(), "Age"] = random.randrange(14, 43)

    # fill missing Embarked values
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    
    # fill missing fare values
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    
    # Convert categorical variables to indicator variables
    df = df.join(pd.get_dummies(df["Embarked"]))
    df = df.join(pd.get_dummies(df["Sex"]))
    df = df.join(pd.get_dummies(df["Pclass"]))
    
    # Add the two binary features
    # Family feature
    df["hasFamily"] = 0
    df.loc[(df["SibSp"] != 0) | (df["Parch"] != 0), "hasFamily"] = 1 
    
    # Child feature
    df["isChild"] = 0
    df.loc[df["Age"] < 16, "isChild"] = 1
    
    # hasCabin feature
    df["hasCabin"] = 1
    df.loc[df["Cabin"].isnull(), "hasCabin"] = 0
    
    # drop the columns
    # ["Embarked", "Sex", "Pclass", "Ticket", "Cabin", "Name", "PassengerId", "C", "Q", "S", 1, 2, 3, "SibSp", "Parch", "hasFamily", "isChild", "hasCabin"]
    df.drop(["Embarked", "Sex", "Ticket", "Cabin", "Name", "PassengerId"], inplace=True, axis=1)
    
    return df

X_data = preprocess(X_data)
X_train = preprocess(X_train)
X_valid = preprocess(X_valid)

display(X_data.head())
# Plot correlation heat map
original_data = X_data.copy(deep=True)
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(original_data.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
# Drop the labels in dataframes
X_data.drop(["Survived"], axis = 1, inplace=True)
X_train.drop(["Survived"], axis = 1, inplace=True)
X_valid.drop(["Survived"], axis = 1, inplace=True)

# Initialize the Classifier
clf_lg_r = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')

# Fit the training data which is a subset of X_data
clf_lg_r.fit(X_train, Y_train)

# Evaluate the testing error on validation set
score_valid_lg_r = clf_lg_r.score(X_valid, Y_valid)
print("Testing Accuracy = ", score_valid_lg_r)

# Fit the classifier on entire training data
clf_lg_r.fit(X_data, Y_data)

# Evaluate the training error
score_train_lg_r = clf_lg_r.score(X_data, Y_data)
print("Training Accuracy = ", score_train_lg_r)

# Generate the output
X_test = preprocess(X_test)
Y_test = clf_lg_r.predict(X_test)
ans = pd.DataFrame({"PassengerId": ID_test, "Survived": Y_test})
ans.to_csv("submit_lg_r.csv", index = False)

errors_list["Logistic Classifier"] = (score_train_lg_r, score_valid_lg_r)
from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=100, random_state=0)
clf_rf.fit(X_train, Y_train)

score_valid_rf = clf_rf.score(X_valid, Y_valid)
print("Testing accuracy = ", score_valid_rf)

# fit the data
clf_rf.fit(X_data, Y_data)

# Evaluate the training error
score_train_rf = clf_rf.score(X_data, Y_data)
print("Training accuracy = ", score_train_rf)

# Generate the output
Y_test = clf_rf.predict(X_test)
ans = pd.DataFrame({"PassengerId": ID_test, "Survived": Y_test})
ans.to_csv("submit_rf.csv", index = False)

errors_list["Random Forest"] = (score_train_rf, score_valid_rf)
from sklearn.ensemble import AdaBoostClassifier

clf_adb = AdaBoostClassifier( n_estimators=10, learning_rate=1.1, algorithm='SAMME.R', random_state=100)
clf_adb.fit(X_train, Y_train)

score_valid_adb = clf_adb.score(X_valid, Y_valid)
print("Testing accuracy = ", score_valid_adb)

# fit the data
clf_adb.fit(X_data, Y_data)

# Evaluate the training error
score_train_adb = clf_adb.score(X_data, Y_data)
print("Training accuracy = ", score_train_adb)

# Generate the output
Y_test = clf_adb.predict(X_test)
ans = pd.DataFrame({"PassengerId": ID_test, "Survived": Y_test})
ans.to_csv("submit_adb.csv", index = False)

errors_list["Adaptive Boosting"] = (score_train_adb, score_valid_adb)
from sklearn.naive_bayes import BernoulliNB

clf_bnb = BernoulliNB(alpha=1.0, binarize=None, fit_prior=True, class_prior=None)

# create binary feature matrix
X_data_bin = X_data.copy(deep=True)
X_train_bin = X_train.copy(deep=True)
X_valid_bin = X_valid.copy(deep=True)

X_data_bin.drop(["Pclass", "SibSp", "Parch", "Age", "Fare"], axis=1, inplace=True)
X_train_bin.drop(["Pclass", "SibSp", "Parch", "Age", "Fare"], axis=1, inplace=True)
X_valid_bin.drop(["Pclass", "SibSp", "Parch", "Age", "Fare"], axis=1, inplace=True)

# fit train data
clf_bnb.fit(X_train_bin, Y_train)

# get validation error
score_valid_bnb = clf_bnb.score(X_valid_bin, Y_valid)
print("Testing accuracy = ", score_valid_bnb)

# fit the data
clf_bnb.fit(X_data_bin, Y_data)

# Evaluate the training error
score_train_bnb = clf_bnb.score(X_data_bin, Y_data)
print("Training accuracy = ", score_train_bnb)

# Generate the output
X_test_bin = X_test.copy(deep=True)
X_test_bin.drop(["Pclass", "SibSp", "Parch", "Age", "Fare"], axis=1, inplace=True)
Y_test = clf_bnb.predict(X_test_bin)
ans = pd.DataFrame({"PassengerId": ID_test, "Survived": Y_test})
ans.to_csv("submit_bnb.csv", index = False)

errors_list["Bernouli Naive Bayes"] = (score_train_bnb, score_valid_bnb)
from sklearn.naive_bayes import MultinomialNB

clf_mnb = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)

# fit train data
clf_mnb.fit(X_train, Y_train)

# get validation error
score_valid_mnb = clf_mnb.score(X_valid, Y_valid)
print("Testing accuracy = ", score_valid_mnb)

# fit the data
clf_mnb.fit(X_data, Y_data)

# Evaluate the training error
score_train_mnb = clf_mnb.score(X_data, Y_data)
print("Training accuracy = ", score_train_mnb)

# Generate the output
Y_test = clf_mnb.predict(X_test)
ans = pd.DataFrame({"PassengerId": ID_test, "Survived": Y_test})
ans.to_csv("submit_mnb.csv", index = False)

errors_list["Mulitnomial Naive Bayes"] = (score_train_mnb, score_valid_mnb)
from sklearn.svm import SVC

clf_sv = SVC(kernel ='poly', C = 1.0, degree = 2, gamma = 'auto')

# fit train data
clf_sv.fit(X_train, Y_train)

# get validation error
score_valid_sv = clf_sv.score(X_valid, Y_valid)
print("Testing accuracy = ", score_valid_sv)

# fit the data
clf_mnb.fit(X_data, Y_data)

# Evaluate the training error
score_train_sv = clf_sv.score(X_data, Y_data)
print("Training accuracy = ", score_train_sv)

# Generate the output
Y_test = clf_sv.predict(X_test)
ans = pd.DataFrame({"PassengerId": ID_test, "Survived": Y_test})
ans.to_csv("submit_svc.csv", index = False)

# Best score in Kaggle: 0.78468 with C = 1.0 and linear kernel
# Best score in Kaggle: 0.78947 with C = 1.0 and polynomial kernel with degree 2

errors_list["Support Vector Machine"] = (score_train_sv, score_valid_sv)
from sklearn.tree import DecisionTreeClassifier

clf_dt = DecisionTreeClassifier(max_depth = 8)
clf_dt.fit(X_train_bin, Y_train)

# get validation error
score_valid_dt = clf_dt.score(X_valid_bin, Y_valid)
print("Testing accuracy = ", score_valid_dt)

# fit the data
clf_dt.fit(X_data_bin, Y_data)

# Evaluate the training error
score_train_dt = clf_dt.score(X_data_bin, Y_data)
print("Training accuracy = ", score_train_dt)

# Generate the output
Y_test = clf_dt.predict(X_test_bin)
ans = pd.DataFrame({"PassengerId": ID_test, "Survived": Y_test})
ans.to_csv("submit_dt.csv", index = False)

errors_list["Decision Tree Classifier"] = (score_train_dt, score_valid_dt)
# Plot the errors

clfs = errors_list.keys()
errors = list(errors_list.values())
train_errors = [x[0] for x in errors]
valid_erros = [x[1] for x in errors]

# create plot
fig, ax = plt.subplots()
index = np.arange(len(clfs))
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.barh(index, train_errors, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Training Error')
 
rects2 = plt.barh(index + bar_width, valid_erros, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Testing Error')
 
plt.ylabel('Classifier')
plt.xlabel('Error Score')
plt.title('Model Performance')
plt.yticks(index + bar_width, clfs)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2))
plt.show()
