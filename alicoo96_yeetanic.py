import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-darkgrid") # Plot style

import seaborn as sns

from collections import Counter



import warnings # Warnings are turned off

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId = test_df["PassengerId"]
def cat_variable_plot(variable): # Input = Categorical variable, Output = Plot + Value count

    var = train_df[variable] # Get categorical variable

    var_value = var.value_counts() # Number of categorical variables

    

    # Visualize

    plt.figure(figsize = (12,3)) # Figure size

    plt.bar(var_value.index, var_value) # Bar plot

    plt.xticks(var_value.index, var_value.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    

    print("{}:\n{}".format(variable, var_value))
categorical_variables1 = ["Survived", "Pclass", "Sex", "SibSp", "Parch", "Embarked"]



for c in categorical_variables1:

    cat_variable_plot(c)
categorical_variables2 = ["Name", "Ticket", "Cabin"]



for c in categorical_variables2:

    print("{} \n".format(train_df[c].value_counts))
def num_variable_plot(variable):

    plt.figure(figsize = (12,3)) # Figure size

    plt.hist(train_df[variable], bins = 50)

    

    # Visualize

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("Histogram of {}".format(variable))

    plt.show()
numerical_variables = ["PassengerId", "Age", "Fare"]



for n in numerical_variables:

    num_variable_plot(n)
variables = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]

i = 0

for i in range(len(variables)):

    surv = train_df[[variables[i], "Survived"]].groupby(variables[i], as_index = False).mean().sort_values(by = "Survived", ascending = False)

    print("-----------------------\n",surv)

    i+=1
def outlier_detection(dataframe,features):

    outlier_index = []

    

    for c in features:

        # Q1 calculation

        Q1 = np.percentile(dataframe[c], 25)



        # Q3 calculation

        Q3 = np.percentile(dataframe[c], 75)

        

        # IQR calculation

        IQR = Q3 - Q1

        

        # ODS

        ODS = IQR * 1.5

        

        # Outliers and their index

        outliers = dataframe[(dataframe[c] < Q1-ODS) | (dataframe[c] > Q3+ODS)].index

        

        # Append outliers to the list

        outlier_index.extend(outliers)

    

    outlier_index = Counter(outlier_index)

    

    multiple_outliers = list(i for i, v in outlier_index.items() if v > 2) # If feature has more than 2 outliers it will be removed.

    

    return multiple_outliers

# Detect outliers of "Age", "SibSp", "Parch", "Fare"

train_df.loc[outlier_detection(train_df,["Age", "SibSp", "Parch", "Fare"])]
# Drop outliers

train_df = train_df.drop(outlier_detection(train_df,["Age", "SibSp", "Parch", "Fare"]), axis = 0).reset_index(drop = True)
# Step 1

train_df_len = len(train_df)

train_df = pd.concat([train_df, test_df], axis = 0).reset_index(drop = True)
# Step 2

train_df.isnull().sum()
# Step 3

train_df[train_df["Embarked"].isnull()]
# Step 3

train_df.boxplot(column = "Fare", by = "Embarked")

plt.show()
# Step 3

train_df["Embarked"] = train_df["Embarked"].fillna("C")

train_df[train_df["Embarked"].isnull()]
# Step 4

train_df[train_df["Fare"].isnull()]
# Step 4

train_df[train_df["Pclass"] == 3]
# Step 4

train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))

train_df[train_df["Fare"].isnull()]
# Step 1

# Correlation matrix

list1 = ["Survived", "Pclass", "Name", "Sex", "SibSp", "Parch", "Ticket", "Cabin", "Embarked", "PassengerId", "Age", "Fare"]

sns.heatmap(train_df[list1].corr(), annot = True, fmt = ".2f")

plt.show()
# Step 2

# We can consider extracting new features according to these relations between given features.

# For ex. if SibSp == (0,1,2), SibSp = 1 else 0 because sibsp(0,1,2) have higher chances of survival then rest.

# Black lines on the plots are standart deviations.

variables = ["Pclass", "Embarked","Sex", "SibSp", "Parch"]

i = 0

for i in range(len(variables)):

    g = sns.factorplot(x = variables[i], y = "Survived", data = train_df, kind = "bar", size = 5)

    plt.show()

    i+=1
# Step 2

# Survived-Age relation

# We can see from the plot that children are mostly survived.

g = sns.FacetGrid(train_df, col = "Survived", size=5)

g.map(sns.distplot, "Age", bins = 10)

plt.show()
# Step 2

# Embarked-Sex-Pclass-Survived relation

g = sns.FacetGrid(train_df, row = "Embarked", size=4)

g.map(sns.pointplot, "Pclass", "Survived", "Sex" )

g.add_legend()

plt.show()
# Step 2

# Embarked-Sex-Fare-Survived relation

g = sns.FacetGrid(train_df, row = "Embarked", col = "Survived", size=3)

g.map(sns.barplot, "Sex", "Fare" )

g.add_legend()

plt.show()
# Step 1

train_df[train_df["Age"].isnull()]
# Step 2

sns.factorplot(x = "Sex", y = "Age", hue = "Pclass", data = train_df, kind = "box", size = 7) 

plt.show()
# Step 2

sns.factorplot(x = "SibSp", y = "Age", data = train_df, kind = "box", size = 7) 

sns.factorplot(x = "Parch", y = "Age", data = train_df, kind = "box", size = 7) 

plt.show()

# Step 3

train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]
# Step 4

# Correlation matrix

# As seen on matrix there is corr. with Sibsp-Parch-Pclass-Age and no corr. with Sex-Age

list2 = ["Pclass","Sex", "SibSp", "Parch", "Age"]

sns.heatmap(train_df[list2].corr(), annot = True, fmt = ".3f")

plt.show()
# Step 5

# 1st we get the indices of the "nan" age values as a list

# In this list we make a prediction for age based on Sibsp-Parch-Pclass features

# If any of Sibsp-Parch-Pclass is null then we predict age value depending on the median age values

nan_age_index = list(train_df[train_df["Age"].isnull()].index)

for i in nan_age_index:

    age_prediction = train_df["Age"][(train_df["SibSp"] == train_df.iloc[i]["SibSp"]) & (train_df["Parch"] == train_df.iloc[i]["Parch"]) & (train_df["Pclass"] == train_df.iloc[i]["Pclass"])].median()

    age_median = train_df["Age"].median()

   

    if not np.isnan(age_prediction):

        train_df["Age"].iloc[i] = age_prediction

    else:

        train_df["Age"].iloc[i] = age_median    
# Step 1

name = train_df["Name"]

name
# Step 2-3-4

train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]

train_df["Title"]
# Step 5

sns.countplot(x = "Title", data = train_df)

plt.xticks(rotation = 70)

plt.show()
# Step 6

train_df["Title"] = train_df["Title"].replace(["Don","Rev","Dr","Mme","Major","Lady","Sir","Col","Capt","the Countess","Jonkheer","Dona"],"Other")

train_df["Title"] = train_df["Title"].replace(["Mrs","Miss","Ms","Mlle"],"Female")



train_df["Title"] = [0 if i == "Master" else 1 if i == "Mr" else 2 if i == "Female" else 3 for i in train_df["Title"]]
# Step 6 visualizaiton

sns.countplot(x = "Title", data = train_df)

plt.xticks(rotation = 70)

plt.show()
# Step 7

g = sns.factorplot(x = "Title", y = "Survived", data = train_df, kind = "bar", size = 5)

plt.show()
# Step 8

train_df = pd.get_dummies(train_df, columns = ["Title"], prefix = "T")

train_df.head()
# Step 9

train_df.drop(labels = ["Name"], axis = 1, inplace = True)

train_df.head()
# Step 1

# If parch and sibsp are 0 familysize will be 0 means that we aren't counting the person. That's why we add +1.

train_df["FS"] = train_df["Parch"] + train_df["SibSp"] + 1
# Step 2

g = sns.factorplot(x = "FS", y = "Survived", data = train_df, kind = "bar", size = 5)

plt.show()
# Step 3

train_df["FamilySize"] = [2 if i == 2 or i == 3 or i == 4 else 1 if i == 1 or i == 5 or i == 7 else 0 for i in train_df["FS"]]
# Step 3

g = sns.countplot(x = "FamilySize", data = train_df)

plt.show()
# Step 4

g = sns.factorplot(x = "FamilySize", y = "Survived", data = train_df, kind = "bar", size = 5)

plt.show()
# Step 5

train_df = pd.get_dummies(train_df, columns = ["FamilySize"], prefix = "FS")

train_df.head()
train_df = pd.get_dummies(train_df, columns = ["Embarked"], prefix = "Emb")

train_df.head()
# Step 1

tickets = []



for i in list(train_df.Ticket):

    if not i.isdigit():

        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])

    else:

        tickets.append("x")

        

train_df["Ticket"] = tickets

train_df["Ticket"].head(20) 
# Step 2

train_df = pd.get_dummies(train_df, columns = ["Ticket"], prefix = "Tck")

train_df.head()
train_df["Pclass"] = train_df["Pclass"].astype("category")

train_df = pd.get_dummies(train_df, columns = ["Pclass"], prefix = "PC")

train_df.head()
train_df["Sex"] = train_df["Sex"].astype("category")

train_df = pd.get_dummies(train_df, columns = ["Sex"])

train_df.head()
train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)

train_df.columns
# Step 1

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
# Step 2

test = train_df[train_df_len:]

test.drop(labels = ["Survived"], axis = 1, inplace = True)



train = train_df[:train_df_len]



X_train = train.drop(labels = ["Survived"], axis = 1)

Y_train = train["Survived"]



x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size = 0.33, random_state = 42)



print("x_train size:", len(x_train))

print("x_test size:", len(x_test))

print("y_train size:", len(y_train))

print("y_test size:", len(y_test))



print("test size:",len(test)) # For testing model. Validation data
# Step 3

logreg = LogisticRegression()

logreg.fit(x_train, y_train)



acc_logreg_train = logreg.score(x_train, y_train) 

acc_logreg_test = logreg.score(x_test, y_test)



print("Train acc:",acc_logreg_train)

print("Test acc:",acc_logreg_test)
# Step 4

random_state = 42

classifier = [DecisionTreeClassifier(random_state = random_state),

              SVC(random_state = random_state),

              RandomForestClassifier(random_state = random_state),

              KNeighborsClassifier(),

              LogisticRegression(random_state = random_state),]
# Step 4

# Grids

dt_param_grid = {"min_samples_split": range(10, 500, 20),

                 "max_depth": range(1, 20, 2)}



svc_param_grid = {"kernel": ["rbf"],

                  "gamma": [0.001, 0.01, 0.1, 1],

                  "C": [1, 10, 50, 100, 200 ,300, 1000]}



rf_param_grid = {"max_features": [1, 3, 10],

                 "min_samples_split": [2, 3, 10],

                 "min_samples_leaf": [1, 3, 10],

                 "bootstrap": [False],

                 "n_estimators": [100, 300],

                 "criterion": ["gini"]}



knn_param_grid = {"n_neighbors": np.linspace(1, 19,10, dtype = int).tolist(),

                  "weights": ["uniform", "distance"],

                  "metric": ["euclidean", "manhattan"]}



logreg_param_grid = {"C": np.logspace(-3, 3, 7),

                     "penalty": ["l1", "l2"]}



classifier_param = [dt_param_grid,

                    svc_param_grid,

                    rf_param_grid,

                    knn_param_grid,

                    logreg_param_grid]
# Step 5

# Finding best model and hyperparameters

cross_val_result = []

best_estimators = []



for i in range(len(classifier)):

    

    c = GridSearchCV(classifier[i],

                        param_grid = classifier_param[i],

                        cv = StratifiedKFold(n_splits = 10),

                        scoring = "accuracy",

                        n_jobs = -1,

                        verbose = 1)

    c.fit(x_train, y_train)

    cross_val_result.append(c.best_score_)

    best_estimators.append(c.best_estimator_)

    print(cross_val_result[i])
cv_results = pd.DataFrame({"Cross Val. Means":cross_val_result, "Models": ["DecisionTreeClassifier", "SVC", "RandomForestClassifier", "KNeighborsClassifier", "LogisticRegression"]})
g = sns.barplot("Cross Val. Means", "Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")

plt.show()
# Step 6

# We take models with accuracies above 0.8

votingC = VotingClassifier(estimators = [("dt", best_estimators[0]),

                                         ("rf", best_estimators[2]),

                                         ("lr", best_estimators[4])],

                                         voting = "soft", n_jobs = -1)



votingC.fit(x_train, y_train)

print(accuracy_score(votingC.predict(x_test), y_test))
test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)

results = pd.concat([test_PassengerId, test_survived], axis = 1)

results.to_csv("yeetanic.csv", index = False)