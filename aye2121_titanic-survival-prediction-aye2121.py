import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

train = pd.read_csv("train_titanic.csv")

test = pd.read_csv("test_titanic.csv")
len(train), len(test)
# we need to clean the data, convert strings to numbers, and manage null values



train.isna().sum()
train["Age"].fillna(train["Age"].mean(),inplace = True)

# test["Age"].fillna(test["Age"].mean(), inplace = True)
test.isna().sum(), train.isna().sum()
train.Embarked.fillna(value ="S", inplace =True)

# test.Embarked.fillna(value="S", inplace = True)
test.isna().sum()
train.Fare.fillna(train["Fare"].mean(), inplace = True)

# test.Fare.fillna(test["Fare"].mean(), inplace = True)
train.isna().sum()
train["Cabin_No"] = train["Cabin"].notnull().astype(int)

# test["Cabin_No"] = test["Cabin"].notnull().astype(int)



train.drop(["Name","Ticket", "Cabin"], axis = 1, inplace = True)

# test.drop(["Name","Ticket", "Cabin"], axis = 1, inplace = True)
train["Fare"] = train['Fare'].round(2)

# test["Fare"] = test["Fare"].round(2)

train.head()
sns.barplot(x= train["Sex"], y = train["Survived"]);
sns.violinplot(x= "Survived", y= "Sex", data = train)
sns.barplot(x= "Pclass", y = "Survived", data = train)
sns.boxplot(x= train["Survived"], y =train["Age"], hue= train["Sex"]);
sns.barplot(x = "SibSp", y = "Survived", data = train)
sns.barplot(x = "SibSp", y = "Survived", data = train)
sns.lmplot(x = "Survived", y = "Age", hue = "Sex", data = train)
train.head()
sns.lmplot(x = "Survived", y = "Age", hue = "Sex", col = "Pclass", data = train);
sns.lmplot(x = "Survived", y = "Age", hue = "Sex", col = "Embarked", data = train);
sns.pairplot(train, hue= "Survived")
sns.jointplot(x = "Age", y = "Survived", data = train, kind = "reg")
sns.jointplot(x = "Fare", y = "Survived", data = train, kind = "reg")
sns.jointplot(x = "Pclass", y = "Survived", data = train, kind = "reg")
def age_buckets(x): 

    if x < 1: return 'Infants' 

    elif x < 11: return 'Children' 

    elif x < 17: return 'Teen' 

    elif x < 30: return 'Young' 

    elif x < 65: return 'Adult' 

    elif x >=65: return 'Elderly' 

    



train["Age_Group"] =train.Age.apply(age_buckets)

# test["Age_Group"] =test.Age.apply(age_buckets)



from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()



train["Sex"] = number.fit_transform(train["Sex"].astype(str))

train["Embarked"] = number.fit_transform(train["Embarked"].astype(str))

train["Age_Group"] = number.fit_transform(train["Age_Group"].astype(str))



# test["Sex"] = number.fit_transform(test["Sex"].astype(str))

# test["Embarked"] = number.fit_transform(test["Embarked"].astype(str))

# test["Age_Group"] = number.fit_transform(test["Age_Group"].astype(str))

train = train.drop(["Age"], axis = 1)
train
train.info()
plt.figure(figsize=(15,10))

sns.heatmap(train.corr(), annot=True, cmap="coolwarm")
train.head()
x = train.drop(["PassengerId", "Survived"], axis = 1)

y= train["Survived"]
len(x), len(y)
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)



lr = LogisticRegression()

lr.fit(X_train, y_train)

lr.score(X_test, y_test)
rf_classifier = RandomForestClassifier()

rf_classifier.fit(X_train, y_train)

rf_classifier.score(X_test, y_test)


knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

knn.score(X_test, y_test)
Accuracy_scores = {"Logistic":lr.score(X_test, y_test), "KNN": knn.score(X_test, y_test), "Random_forest": rf_classifier.score(X_test, y_test)}

Accuracy_scores
Accuracy_scores=pd.DataFrame(Accuracy_scores, index = ["Accuracy"])

Accuracy_scores.T.plot.bar() 
#Logistic regression



lr_random_grid = {"C": np.arange(4,48,4), "solver": ["liblinear"]} 



lr_randomsearchCV = RandomizedSearchCV(LogisticRegression(), param_distributions= lr_random_grid, cv=5, n_iter=20, verbose=True )

lr_randomsearchCV.fit(X_train, y_train);

lr_randomsearchCV.best_params_

lr = LogisticRegression(C=lr_randomsearchCV.best_params_["C"], solver=lr_randomsearchCV.best_params_["solver"] )

lr_tuned = lr.fit(X_train, y_train)

lr_tuned.score(X_test, y_test)
# KNN



knn_grid = {"n_neighbors": list(range(1,50))}



knn_randomsearchCV = RandomizedSearchCV(KNeighborsClassifier(), param_distributions=knn_grid,cv=5,  verbose=True  )

knn_randomsearchCV.fit(X_train, y_train);



knn_randomsearchCV.best_params_

knn_tuned = KNeighborsClassifier(n_neighbors=knn_randomsearchCV.best_params_["n_neighbors"])



knn_tuned.fit(X_train, y_train)

knn_tuned.score(X_test, y_test)
#Random Forest



rf_grid = {"n_estimators": np.arange(10,1000,50), "max_depth": [None, 3,5,10], 

            "min_samples_split":np.arange(2,20,2), "min_samples_leaf": np.arange(1,20,2)}



rf_randomsearchCV = RandomizedSearchCV(RandomForestClassifier(), param_distributions=rf_grid, cv=15, n_iter=2, verbose=True)

rf_randomsearchCV.fit(X_train, y_train)



rf_randomsearchCV.best_params_



rf_tuned = RandomForestClassifier(max_depth=rf_randomsearchCV.best_params_["max_depth"], 

                                  min_samples_leaf = rf_randomsearchCV.best_params_["min_samples_leaf"],

                                 min_samples_split = rf_randomsearchCV.best_params_["min_samples_split"],

                                 n_estimators =rf_randomsearchCV.best_params_["n_estimators"])



rf_tuned.fit(X_train, y_train)

rf_tuned.score(X_test, y_test)
Tuned_scores = {"Logistic":lr_tuned.score(X_test, y_test), 

                "Random Forest": rf_tuned.score(X_test, y_test), 

                "KNN": knn_tuned.score(X_test, y_test)}
Tuned_score = pd.DataFrame(Tuned_scores, index = ["Accuracy Scores"])

Tuned_score
Accuracy_scores.T.plot.bar()
# logistic regression tunning parameters



np.random.seed (21)



lr_grid_params = {"C": np.logspace(-4,4,30), "solver": ["liblinear"]} 



lr_gridsearchcv = GridSearchCV(LogisticRegression(), param_grid=lr_grid_params, cv=5, verbose= True)

lr_gridsearchcv.fit(X_train, y_train)

lr_gridsearchcv.best_params_

# do model based on best parameters



LR_tuned = LogisticRegression(C= lr_gridsearchcv.best_params_["C"], solver=lr_gridsearchcv.best_params_["solver"])



LR_tuned.fit(X_train, y_train)

LR_Accuracy = LR_tuned.score(X_test, y_test)

LR_Accuracy
# random forest tunning parameters



RF_grid = {"n_estimators": np.arange(10,50,10), "max_depth": [None, 3,5,10], 

            "min_samples_split":np.arange(2,20,2), "min_samples_leaf": np.arange(1,4,2)}



RF_gridsearch = GridSearchCV(RandomForestClassifier(), param_grid=RF_grid, cv=2, verbose=True)



RF_gridsearch.fit(X_train, y_train)

RF_gridsearch.best_params_



RF_tuned = RandomForestClassifier(max_depth= RF_gridsearch.best_params_["max_depth"], min_samples_leaf=RF_gridsearch.best_params_["min_samples_leaf"], min_samples_split= RF_gridsearch.best_params_["min_samples_split"], n_estimators=RF_gridsearch.best_params_["n_estimators"])

RF_tuned

RF_tuned.fit(X_train, y_train)

RF_tuned.score(X_test, y_test)
# KNN tuning parameters



KNN_grid = {"n_neighbors": list(range(1,50)), "leaf_size": np.arange(2,10,2)}

KNN_gridsearch = GridSearchCV(KNeighborsClassifier(), param_grid= KNN_grid, cv=5,  verbose=True  )

KNN_gridsearch.fit(X_train, y_train);



KNN_gridsearch.best_params_

KNN_tuned = KNeighborsClassifier(n_neighbors=knn_randomsearchCV.best_params_["n_neighbors"], leaf_size=KNN_gridsearch.best_params_["leaf_size"])



KNN_tuned.fit(X_train, y_train)

KNN_tuned.score(X_test, y_test)
Tuned_scores_Grid = {"Logistic":LR_tuned.score(X_test, y_test), 

                "Random Forest": RF_tuned.score(X_test, y_test), 

                "KNN": KNN_tuned.score(X_test, y_test)}



Tuned_scores_Grid = pd.DataFrame(Tuned_scores_Grid, index=["Accuracy Score"])

Tuned_scores_Grid
Tuned_scores_Grid.T.plot.bar()


from sklearn.model_selection import cross_val_score



#check best param



rf_randomsearchCV.best_params_



#cross validated auracy

rf_tuned = RandomForestClassifier(max_depth=rf_randomsearchCV.best_params_["max_depth"], 

                                  min_samples_leaf = rf_randomsearchCV.best_params_["min_samples_leaf"],

                                 min_samples_split = rf_randomsearchCV.best_params_["min_samples_split"],

                                 n_estimators =rf_randomsearchCV.best_params_["n_estimators"])





cv_acc = cross_val_score(rf_tuned, x, y, cv=5, verbose=True, scoring="accuracy")

cv_acc = np.mean(cv_acc)

cv_acc
#cross validated Precision



cv_precision = cross_val_score(rf_tuned, x, y, cv=5, verbose=True, scoring="precision")

cv_precision = np.mean(cv_precision)

cv_precision



#cross validated recall



cv_recall = cross_val_score(rf_tuned, x, y, cv=5, verbose=True, scoring="recall")

cv_recall = np.mean(cv_recall)

cv_recall

#cross validated f1



cv_f1 = cross_val_score(rf_tuned, x, y, cv=5, verbose=True, scoring="f1")

cv_f1 = np.mean(cv_f1)

cv_f1



#visualize as metrics



metrics_compare = pd.DataFrame({"Cross validated Accuracy":cv_acc,

                  "Cross validated Precision": cv_precision,

                  "Cross validated Recall": cv_recall,

                  "Cross validated F1": cv_f1}, index =[1])

metrics_compare
metrics_compare.T.plot.bar(title = "Evaluation of Matrices using cross validated data")
 # Attach a text label above each bar displaying its height

    

for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,

                '%d' % int(height),

                ha='center', va='bottom')



autolabel(rects1)

autolabel(rects2)



plt.show()
X_test
test = pd.read_csv("test_titanic.csv")



test["Age"].fillna(test["Age"].mean(), inplace = True)

test.Embarked.fillna(value="S", inplace = True)

test.Fare.fillna(test["Fare"].mean(), inplace = True)

test["Cabin_No"] = test["Cabin"].notnull().astype(int)

test.drop(["Name","Ticket", "Cabin"], axis = 1, inplace = True)

test["Fare"] = test["Fare"].round(2)

test["Age"] = test["Age"].round(2)





def age_buckets(x): 

    if x < 1: return 'Infants' 

    elif x < 11: return 'Children' 

    elif x < 17: return 'Teen' 

    elif x < 30: return 'Young' 

    elif x < 65: return 'Adult' 

    elif x >=65: return 'Elderly' 

    



test["Age_Group"] =test.Age.apply(age_buckets)



test = test.drop(["Age"], axis =1)





number = LabelEncoder()



test["Sex"] = number.fit_transform(test["Sex"].astype(str))

test["Embarked"] = number.fit_transform(test["Embarked"].astype(str))

test["Age_Group"] = number.fit_transform(test["Age_Group"].astype(str))
test
PassengerId = test["PassengerId"]

test.drop(["PassengerId"], axis = 1, inplace = True)
# Logistic Regression



y_pred_LR = LR_tuned.predict(test)

y_pred_LR
# Random_Forest



y_pred_RF = RF_tuned.predict(test)

y_pred_RF
# KNN



y_pred_KNN = KNN_tuned.predict(test)

y_pred_KNN
Pred_Output = pd.DataFrame({"Passenger ID": PassengerId, 

                            "Logistic Regression":y_pred_LR, 

                            "Random_forest": y_pred_RF, "KNN": y_pred_KNN })

Pred_Output
Pred_Output.to_csv("Submission_titanic_Ayu.csv")
len(Pred_Output)
y_pred_rf = rf_tuned.predict(test)

y_pred_rf
submission_Ayu = pd.DataFrame({"PassengerId": PassengerId, "Survived": y_pred_rf})

submission_Ayu
submission_Ayu.to_csv("submssion_Ayu.csv", index = False)
submission_Ayu.head()
#Random Forest





#with best hyperparameter result: rf_randomsearchCV.best_params_



rf_tuned = RandomForestClassifier(max_depth=rf_randomsearchCV.best_params_["max_depth"], 

                                  min_samples_leaf = rf_randomsearchCV.best_params_["min_samples_leaf"],

                                 min_samples_split = rf_randomsearchCV.best_params_["min_samples_split"],

                                 n_estimators =rf_randomsearchCV.best_params_["n_estimators"])



rf_tuned.fit(X_train, y_train)

rf_tuned.score(X_test, y_test)
rf_tuned.feature_importances_
Feature_importance = dict(zip(test.columns, list(rf_tuned.feature_importances_)))

Feature_importance
Feature = pd.DataFrame(Feature_importance, index =[0])



Feature.T.plot.bar()


