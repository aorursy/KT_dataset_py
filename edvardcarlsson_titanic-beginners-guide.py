import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder

from tabulate import tabulate



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



import warnings

warnings.filterwarnings("ignore")



sns.set(style="darkgrid")

sns.set_palette(sns.cubehelix_palette(8))



data = pd.read_csv("../input/titanic/train.csv")

print(data.describe())

data.head()

table = pd.DataFrame(data.nunique())

table["null"]=data.isnull().sum()

table["dtype"] = data.dtypes

header = ["Column", "Unique Values","Null Values","Datatype"]

print(tabulate(table,headers=header,tablefmt="fancy_grid"))



test = pd.read_csv("../input/titanic/test.csv")

table = pd.DataFrame(test.nunique())

table["null"]=test.isnull().sum()

table["dtype"] = test.dtypes

header = ["Column", "Unique Values","Null Values","Datatype"]

print(tabulate(table,headers=header,tablefmt="fancy_grid"))

# Training

data.loc[data["Age"] < 1, "Age"] = 0

data["Embarked"] = data["Embarked"].replace({"C":"Cherbourg","Q":"Queenstown", "S":"Southampton"})



data = data.drop("Cabin", axis =1) 

data = data.dropna(subset=["Embarked"])

data["index"] = range(len(data))

data = data.set_index("index")



# Test 

test.loc[test["Age"] < 1, "Age"] = 0

test = test.drop("Cabin", axis =1) 

test["Embarked"] = test["Embarked"].replace({"C":"Cherbourg","Q":"Queenstown", "S":"Southampton"})



test.sort_values("Fare").tail() 

test["Fare"] = test["Fare"].replace({np.nan:test.groupby("Pclass")["Fare"].mean()[3]})

print(data.corr()["Age"].sort_values(ascending = False))
# As seen Pclass and SibSp are the strongest correlating features

proc_data = [data.copy(), test.copy()]





# I then divide SibSp into 3 groups (0,1,1+) and matches these with the the groups of Pclass. Createing 9 groups in total, I then assign the median age value of each group as the missing age.



for k in range(2):

    proc_data[k]["group"] = 0

    proc_data[k]["naAge"] = 0

    for i in range(len(proc_data[k])):

        if pd.isna(proc_data[k]["Age"][i]):

            proc_data[k]["naAge"][i] = 1

        if proc_data[k]["SibSp"][i] == 0:

            if proc_data[k]["Pclass"][i] == 1:

                proc_data[k]["group"][i] = 1

            elif proc_data[k]["Pclass"][i] == 2:

                proc_data[k]["group"][i] = 2

            elif proc_data[k]["Pclass"][i] == 3:

                proc_data[k]["group"][i] = 3

        if proc_data[k]["SibSp"][i] == 1:

            if proc_data[k]["Pclass"][i] == 1:

                proc_data[k]["group"][i] = 4

            elif proc_data[k]["Pclass"][i] == 2:

                proc_data[k]["group"][i] = 5

            elif proc_data[k]["Pclass"][i] == 3:

                proc_data[k]["group"][i] = 6

        if proc_data[k]["SibSp"][i] > 1:

            if proc_data[k]["Pclass"][i] == 1:

                proc_data[k]["group"][i] = 7

            elif proc_data[k]["Pclass"][i] == 2:

                proc_data[k]["group"][i] = 8

            elif proc_data[k]["Pclass"][i] == 3:

                proc_data[k]["group"][i] = 9

    

    med1 = np.median(proc_data[k]["Age"][(proc_data[k]["group"]==1) & (proc_data[k]["naAge"]==0)])

    med2 = np.median(proc_data[k]["Age"][(proc_data[k]["group"]==2) & (proc_data[k]["naAge"]==0)])

    med3 = np.median(proc_data[k]["Age"][(proc_data[k]["group"]==3) & (proc_data[k]["naAge"]==0)])

    med4 = np.median(proc_data[k]["Age"][(proc_data[k]["group"]==4) & (proc_data[k]["naAge"]==0)])

    med5 = np.median(proc_data[k]["Age"][(proc_data[k]["group"]==5) & (proc_data[k]["naAge"]==0)])

    med6 = np.median(proc_data[k]["Age"][(proc_data[k]["group"]==6) & (proc_data[k]["naAge"]==0)])

    med7 = np.median(proc_data[k]["Age"][(proc_data[k]["group"]==7) & (proc_data[k]["naAge"]==0)])

    med8 = np.median(proc_data[k]["Age"][(proc_data[k]["group"]==8) & (proc_data[k]["naAge"]==0)])

    med9 = np.median(proc_data[k]["Age"][(proc_data[k]["group"]==9) & (proc_data[k]["naAge"]==0)])

    

    for i in range(len(proc_data[k])):

        if np.isnan(proc_data[k]["Age"][i]): 

            if proc_data[k]["group"][i] == 1:

                proc_data[k]["Age"][i] = med1

            if proc_data[k]["group"][i] == 2:

                proc_data[k]["Age"][i] = med2

            if proc_data[k]["group"][i] == 3:

                proc_data[k]["Age"][i] = med3

            if proc_data[k]["group"][i] == 4:

                proc_data[k]["Age"][i] = med4

            if proc_data[k]["group"][i] == 5:

                proc_data[k]["Age"][i] = med5

            if proc_data[k]["group"][i] == 6:

                proc_data[k]["Age"][i] = med6

            if proc_data[k]["group"][i] == 7:

                proc_data[k]["Age"][i] = med7

            if proc_data[k]["group"][i] == 8:

                proc_data[k]["Age"][i] = med8

            if proc_data[k]["group"][i] == 9:

                proc_data[k]["Age"][i] = med9

                

    proc_data[k] = proc_data[k].drop(["group","naAge"], axis=1) 

fig, ax = plt.subplots(1,3,figsize=(20,5))

sns.countplot(data["Survived"],ax=ax[0],palette="Set2").set_title("Survived")

sns.countplot(data["Embarked"],ax=ax[1],palette="Set2").set_title("Embarked")

sns.countplot(data["Pclass"],ax=ax[2],palette="Set2").set_title("Ticket Class")

plt.show()





fig, ax = plt.subplots(1,3,figsize=(20,5))

sns.countplot(data["Sex"],ax=ax[0],palette="Set2").set_title("Sex")

sns.countplot(data["SibSp"],ax=ax[1],palette="Set2").set_title("Siblings & Spouses")

sns.countplot(data["Parch"],ax=ax[2],palette="Set2").set_title("Parents & Children")

plt.show()





fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.distplot(data["Fare"], color="g", ax=ax[0], kde=False).set_title("Fare")

sns.distplot(proc_data[0]["Age"], color="g", ax=ax[1], kde=False).set_title("Age")

plt.show()
width = 1      

groupgap=1



y1=[np.average(data["Survived"][data["Sex"] == "male"])*100,

     np.average(data["Survived"][data["Sex"] == "female"])*100]



y2=[np.average(proc_data[0]["Survived"][proc_data[0]["Age"] < 15])*100,

          np.average(proc_data[0]["Survived"][proc_data[0]["Age"] >= 15])*100]



y3=[np.average(data["Survived"][data["Pclass"] == 1])*100,

     np.average(data["Survived"][data["Pclass"] == 2])*100,

     np.average(data["Survived"][data["Pclass"] == 3])*100]





x1 = np.arange(len(y1))

x2 = np.arange(len(y2))+groupgap+len(y1)

x3 = np.arange(len(y3))+groupgap+len(y1)+groupgap+len(y2)





ind = np.concatenate((x1,x2,x3))

fig, ax = plt.subplots()

rects1 = ax.bar(x1, y1, width, color="r",  edgecolor= "black")

rects2 = ax.bar(x2, y2, width, color="b",  edgecolor= "black")

rects3 = ax.bar(x3, y3, width, color="g",  edgecolor= "black")



ax.set_ylabel("Survived (%)",fontsize=14)

ax.set_xticks(ind)

plt.xticks(rotation=45)



ax.set_xticklabels(("Male", "Female","Child","Adult","First Class","Second Class","Third Class"),fontsize=14)

ax.set_title("Groups Rate of Survival")



plt.show()

for k in range(2):

    proc_data[k]["male"] = proc_data[k]["Sex"].replace({"male":1,"female":0})

    

    proc_data[k] = proc_data[k].drop(["Sex","Name","Ticket","PassengerId"], axis=1) 

    

    L_E = LabelEncoder()

    proc_data[1]["Embarked"] = proc_data[1]["Embarked"].astype(str) # seems to be something strange here so this one is needed

    proc_data[k]["Embarked"] = L_E.fit_transform(proc_data[k]["Embarked"])



    proc_data[k]["Fare"] = pd.qcut(proc_data[k]["Fare"], 4, labels=[0, 1, 2, 3]).astype(int)

    proc_data[k]["Age"] = pd.qcut(proc_data[k]["Age"], 4, labels=[0, 1, 2, 3]).astype(int)







for i in range(2):

    proc_data[i]["family"] = proc_data[i]["SibSp"] + proc_data[i]["Parch"]

    proc_data[i] = proc_data[i].drop(["SibSp", "Parch"], axis = 1)

    sns.countplot(proc_data[i]["family"],palette="Set2").set_title("Family Size")
for i in range(2):    

    proc_data[i]["alone"] = 0

    proc_data[i].loc[proc_data[i]["family"] == 0, "alone"] = 1

    proc_data[i] = proc_data[i].drop("family", axis =1)

proc_data[0]
train = proc_data[0]



table = pd.DataFrame(train.corr()["Survived"].sort_values(ascending = False))

table["Description"] = ["0: Died 1: Survived",

     "0:(0,8.05] 1:(8.05, 15.646] 2:(15.646, 33] 3:(33, 512.329]\n 25% of the sample in each group",

     "0:(0,21] 1:(21, 26] 2:(26, 35] 3:(35, 80]\n 25% of the sample in each group",

     "0:Cherbourg 1:Queenstown 2:Southampton",

     "0:With family 1:Alone",

     "1:First Class 2:Second 3:Third",

     "0:Female 1:Male"]

header = ["Feature", "Correlation", "Description"]

print(tabulate(table,headers=header,tablefmt="fancy_grid"))





from sklearn.preprocessing import MinMaxScaler 



scaler = MinMaxScaler()



train = pd.DataFrame(scaler.fit_transform(train),columns = train.columns)



X_train = train.drop(["Survived"],axis=1)

Y_train = train["Survived"]





X_test = pd.DataFrame(scaler.fit_transform(proc_data[1]),columns = proc_data[1].columns)



from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, test_size = 0.30, random_state = 42)



rand_forr = RandomForestClassifier(n_estimators = 100)



rand_forr.fit(x_train, y_train)

Y_pred = rand_forr.predict(x_valid)



from sklearn.metrics import accuracy_score ,roc_auc_score, precision_score, confusion_matrix

print('Random Forest Model Validation set\n')



print('Accuracy Score: {}\n\nPrecision Score: {}\n\nConfusion Matrix:\n {}\n\nAUC Score: {}'

      .format(accuracy_score(y_valid,Y_pred), precision_score(y_valid,Y_pred),confusion_matrix(y_valid,Y_pred), roc_auc_score(y_valid,Y_pred)))

# Hyperparameter tuning for RandomForest

# Create a parameter grid to sample from during fitting



# Parameters:



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]



# Number of features to consider at every split

max_features = ['auto', 'sqrt']



# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)



# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]



# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]



# Method of selecting samples for training each tree

bootstrap = [True, False]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV



# Use the random grid to narrow down the possible best hyperparameters

rf = RandomForestClassifier()



# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)



# Fit the random search model

rf_random.fit(x_train, y_train)
# Find best parameters

rf_random.best_params_
# Find the specific best parameters by searching every combination close to our earlier findings



# Create a new parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [10,20, 30],

    'max_features': [2, 3],

    'min_samples_leaf': [1, 2],

    'min_samples_split': [1, 2, 3],

    'n_estimators': [800, 1000, 1200]

}



# Create a based model

rf = RandomForestClassifier()

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)



grid_search.fit(x_train, y_train)



grid_search.best_params_
# Validation of RF 

# Using F1, accuracy, and AUC



rf_best = RandomForestClassifier(n_estimators = 1000, min_samples_split = 2,

                                 min_samples_leaf = 1, max_features = 2,

                                 max_depth = 10, bootstrap = True)



rf_best.fit(x_train, y_train)

y_pred_best = rf_best.predict(x_valid)



print('Accuracy Score: {}\n\nPrecision Score: {}\n\nConfusion Matrix:\n {}\n\nAUC Score: {}'

      .format(accuracy_score(y_valid,y_pred_best), precision_score(y_valid,y_pred_best),confusion_matrix(y_valid,y_pred_best), roc_auc_score(y_valid,y_pred_best)))

submission_pred = rf_best.predict(X_test)

submission = pd.DataFrame(test["PassengerId"])

submission["Survived"] = submission_pred.astype(int)

submission.tail(15)

submission.to_csv("../working/submit.csv", index=False)
