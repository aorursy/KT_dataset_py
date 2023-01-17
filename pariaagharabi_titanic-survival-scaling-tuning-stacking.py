from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Import the necessary packages

import numpy as np

import pandas as pd



import warnings

warnings.simplefilter(action ="ignore")



from collections import Counter



# Data visualization

import matplotlib.pyplot as plt

import scikitplot as skplt

from scikitplot.plotters import plot_learning_curve

from mlxtend.plotting import plot_learning_curves

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

import seaborn as sns



# Algorithms

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn import metrics

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import VotingClassifier

from mlxtend.classifier import StackingCVClassifier

from sklearn.model_selection import GridSearchCV

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn import svm

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import mean_squared_error
train = pd.read_csv('../input/titanic/train.csv')

test  = pd.read_csv('../input/titanic/test.csv')
train.head()
train.columns.values
test.head()
test.columns.values
# Analyse statically insight of train data

train.describe()
# Analyse statically insight of test data

test.describe()
train.info()
test.info()
print(f"The train data size: {train.shape}")

print(f"The test data size: {test.shape}")
diff_train_test = set(train.columns) - set(test.columns)

diff_train_test
train["Survived"].describe()
total_survived= train["Survived"].sum()

total_no_survived = 891 - total_survived



plt.figure(figsize = (10,5))

plt.subplot(121)

sns.countplot(x="Survived", data=train)

plt.title("Survival count")



plt.subplot(122)

plt.pie([total_no_survived, total_survived], labels=["No Survived","Survived"], autopct="%1.0f%%")

plt.title("Survival rate") 



plt.show()
numeric_data=train.select_dtypes(exclude="object")

numeric_corr=numeric_data.corr()

f,ax=plt.subplots(figsize=(17,1))

sns.heatmap(numeric_corr.sort_values(by=["Survived"], ascending=False).head(1), cmap="Greens")

plt.title(" Numerical features correlation with the survival", weight='bold', fontsize=18, color="green")

plt.xticks(weight="bold")

plt.yticks(weight="bold", color="green", rotation=0)





plt.show()
Num_feature=numeric_corr["Survived"].sort_values(ascending=False).head(10).to_frame()



cm = sns.light_palette("green", as_cmap=True)



style = Num_feature.style.background_gradient(cmap=cm)

style
ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train["Survived"].to_frame()



#Combine train and test sets

concat_data = pd.concat((train, test), sort=False).reset_index(drop=True)

#Drop the target "Survived" and Id columns

concat_data.drop(["Survived"], axis=1, inplace=True)

concat_data.drop(["PassengerId"], axis=1, inplace=True)

print("Total size is :",concat_data.shape)
concat_data.head()
concat_data.tail()
concat_data.info()
# Count the null columns

null_columns = concat_data.columns[concat_data.isnull().any()]

concat_data[null_columns].isnull().sum()
# Outlier detection 

def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        

        # 1st quartile (25%)

        Q1 = np.percentile(concat_data[col], 25)

        

        # 3rd quartile (75%)

        Q3 = np.percentile(concat_data[col],75)

        

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = concat_data[(concat_data[col] < Q1 - outlier_step) | 

                              (concat_data[col] > Q3 + outlier_step )].index

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

   

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)  



    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    return multiple_outliers   



# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(concat_data,2,["Age","SibSp","Parch","Fare"])
# Drop outliers

df = concat_data.drop(Outliers_to_drop)
print(f"The full data size: {df.shape}")
# Count the null columns

null_columns = df.columns[df.isnull().any()]

print(f"Number of null columns in concatenated data:\n{df[null_columns].isnull().sum()}")
numeric_features = df.select_dtypes(include=[np.number])

categorical_features = df.select_dtypes(exclude=[np.number])



print(f"Numerical features size: {df.shape}")

print(f"Categorical features size: {categorical_features.shape}")
categ =  ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]

numic = ["Fare", "Age"]



#Distribution

fig = plt.figure(figsize=(30, 20))

for i in range (0,len(categ)):

    fig.add_subplot(3,3,i+1)

    sns.countplot(x=categ[i], data=train);  



for col in numic:

    fig.add_subplot(3,3,i + 2)

    sns.distplot(df[col].dropna());

    i += 1

    

plt.show()

fig.clear()
numeric_features = df.select_dtypes(include=[np.number])

corr_numeric_features = numeric_features.corr()

corr_numeric_features
print(f"Numerical features: {numeric_features.shape}")
unique_list_numeric_features = [(item, np.count_nonzero(df[item].unique())) for item in numeric_features]

unique_list_numeric_features
# Corralation between Numeric features 

corr_numeric_features = numeric_features.corr()



#Using Pearson Correlation

sns.heatmap(corr_numeric_features, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap="Greens")



plt.show()
# Count the null columns' numeric_features in data set

null_columns_numeric_features = numeric_features.columns[numeric_features.isnull().any()]

print(f"Missing values in numerical features: \n{numeric_features[null_columns_numeric_features].isnull().sum()}")
highly_correlated_visualization = sns.heatmap(train[["Pclass", "Age", "SibSp", "Parch", "Fare", "Survived"]].corr(), annot=True, fmt = ".2f", cmap = "Greens")
unique_list_numeric_features = [(item, np.count_nonzero(df[item].unique())) for item in numeric_features]

unique_list_categorical_features = [(item, np.count_nonzero(df[item].unique())) for item in                                                  categorical_features]



print(f"List of unique numerical features:\n{unique_list_numeric_features}")

print(f"List of unique categorical features:\n{unique_list_categorical_features}")
# Corralation between Numeric features 

corr_numeric_features = numeric_features.corr()



#Using Pearson Correlation



sns.heatmap(corr_numeric_features, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap="Greens")



plt.show()
df[["Name"]]
# Count the null in "Name" column

null_name_columns = df["Name"].isnull().sum()

null_name_columns
def GetTitle(name):

    first_name_with_title = name.split(",")[1]

    title = first_name_with_title.split(".")[0]

    title = title.strip().lower()

    return title
df["Name"].map(lambda x : GetTitle(x))
df["Name"].map(lambda x : GetTitle(x)).unique()
def GetTitle(name):

    title_group = {

        'mr': "Mr"

        , 'mrs': "Mrs"

        , 'miss': "Miss"

        , 'master': "master"

        , 'don': "Sir"

        , 'rev': "Sir"

        , 'dr': "Officer"

        , 'mme': "Mrs"

        , 'ms': "Mrs"

        ,'major': "Officer"

        , 'lady': "Lady"

        , 'sir': "Sir"

        , 'mlle': "Miss"

        , 'col': "Officer"

        , 'capt': "Officer"

        , 'the countess': "Lady"

        ,'jonkheer': "Sir"

        , 'dona': "Lady"

    }

    first_name_with_title = name.split(",")[1]

    title = first_name_with_title.split(".")[0]

    title = title.strip().lower()

    return title_group[title]
df["Name"].map(lambda x: GetTitle(x))
df["Title"] = df["Name"].map(lambda x: GetTitle(x))

df["Title"]
title_age_median = df.groupby(df["Title"]).Age.transform("median")

title_age_median
df[["Sex"]]
Survived = ["Survived"]

Not_Survived = ["Not Survived"]



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

women = train[train["Sex"] == "female"]

men = train[train["Sex"] == "male"]



ax = sns.distplot(women[women["Survived"] == 1].Age.dropna(), bins=18, label=Survived, ax=axes[0], kde=False, color="#006666")

ax = sns.distplot(women[women["Survived"] == 0].Age.dropna(), bins=40, label=Not_Survived, ax=axes[0], kde=False, color="#00b3b3")

ax.legend()

ax.set_title("Female")



ax = sns.distplot(men[men["Survived"] == 1].Age.dropna(), bins=18, label=Survived, ax=axes[1], kde=False, color="#006666")

ax = sns.distplot(men[men["Survived"] == 0].Age.dropna(), bins=40, label=Not_Survived, ax=axes[1], kde=False, color="#00b3b3")

ax.legend()

_ = ax.set_title("Male")



plt.show()
# Count the null in "Sex" column

null_sex_columns = df["Sex"].isnull().sum()

null_sex_columns
def getGender(gender):

    if (gender == "male"):

        return 0

    else:

        return 1
df["Sex"] = df["Sex"].map(lambda x: getGender(x))
by_age_gender = df.groupby(["Age", "Sex"])

age_gen_sz = by_age_gender.size().unstack()

age_gen_sz.plot(kind="bar", figsize=(30,12), color=["#80CBC4", "#00897B"]);
df[["Sex"]]
df[["Age"]]
# Count the null in "Age" column

null_age_columns = df["Age"].isnull().sum()

null_age_columns
df["Age"].fillna(title_age_median, inplace = True)
df[["Age"]]
df["AgeState"] = np.where(df["Age"] >= 18, "Adult","Child")

df["AgeState"]
df["AgeState"].value_counts()
df.assign(AgeState = lambda x: np.where(x.Age >= 18, "Adult", "Child"))
plt.figure(figsize= (10 ,5))

sns.barplot(data=train, x="Pclass", y="Survived",ci=None, color="#00897B")



plt.show()
df[["Parch"]]
# Count the null in "Parch" column

null_parch_columns = df["Parch"].isnull().sum()

null_parch_columns
df[["Ticket"]]
# Count the null in "Ticket" column

null_ticket_columns = df["Parch"].isnull().sum()

null_ticket_columns
df[["Fare"]]
sns.distplot(df["Fare"], color="#00897B")
# Count the null in "Fare" column

null_fare_columns = df["Fare"].isnull().sum()

null_fare_columns
median_fare = df.loc[(df["Pclass"] == 3) & (df["Embarked"] == "S"), ["Fare"]].median()

print(median_fare)

print(type(median_fare))

print(median_fare[0])
df["Fare"].fillna(median_fare[0] , inplace = True )
pd.qcut(df["Fare"], q=4)
pd.qcut(df["Fare"], q = 4, labels = ["very_low", "low", "high", "very_high"])
df["Fare_Bin"] = pd.qcut(df["Fare"], q = 4, labels = ["very_low", "low", "high", "very_high"])
df[["Cabin"]]
# Count the null in "Cabin" column

null_cabin_columns = df["Cabin"].isnull().sum()

null_cabin_columns
df.Cabin.value_counts(normalize=True)
df["Cabin"].unique()
df.loc[df["Cabin"] == "T"]
df.loc[df["Cabin"] == "T", "Cabin"] = np.NaN
df["Cabin"].unique()
def Get_Deck(cabin):

    return np.where(pd.notnull(cabin), str(cabin)[0].upper(), "Z")
df["Deck"] = df["Cabin"].map(lambda x : Get_Deck(x))
df["Deck"].value_counts()
df[["Embarked"]]
FaceGrid = sns.FacetGrid(train, row="Embarked", size=4.5, aspect=1.6)

FaceGrid.map(sns.pointplot, "Pclass", "Survived", palette=None, order=None, hue_order=None, color="#00897B")

FaceGrid.add_legend()

plt.show()
# Count the null in "Embarked" column

null_embarked_columns = df["Embarked"].isnull().sum()

null_embarked_columns
df.groupby(["Pclass", "Embarked"]).agg({"Fare": ["median"]}).unstack()
df["Embarked"].fillna("C", inplace = True )
# Check the null in "Embarked" column

null_embarked_columns = df["Embarked"].isnull().sum()

null_embarked_columns
df.drop(["Cabin", "Name", "Ticket", "SibSp", "Parch", "Sex"], axis=1, inplace=True) 
# Check the null columns in concat_data

null_columns = df.columns[df.isnull().any()]

df[null_columns].isnull().sum()
print(f"The Shape of all data: {df.shape}")
# One hot encoding:

final_df = pd.get_dummies(df, columns = ["Deck", "Pclass", "Title", "Fare_Bin", "Embarked", "AgeState"])
final_df.columns
print(f"The shape of the original dataset: {df.shape}")

print(f"Encoded dataset shape: {final_df.shape}")

print(f"We have: {final_df.shape[1] - df.shape[1]} new encoded features")
TrainData = final_df[:ntrain] 

TestData = final_df[ntrain:]
TrainData.shape, TestData.shape
TrainData.info()
TestData.info()
print(f"Encoded dataset shape: {final_df.shape}")
target = train[["Survived"]]
print("We make sure that both train and target sets have the same row number:")

print(f"Train: {TrainData.shape[0]} rows")

print(f"Target: {target.shape[0]} rows")
# Remove any duplicated column names

final_df = final_df.loc[:,~final_df.columns.duplicated()]
x = TrainData

y = np.array(target)
from sklearn.model_selection import train_test_split

# Split the data set into train and test sets 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
scaler = MinMaxScaler()



# transform "x_train"

x_train = scaler.fit_transform(x_train)

# transform "x_test"

x_test = scaler.transform(x_test)

#Transform the test set

X_test= scaler.transform(TestData)
# Baseline model of gradient boosting classifier with default parameters:

gbc = GradientBoostingClassifier()

gbc_mod = gbc.fit(x_train, y_train)

print(f"Baseline gradient boosting classifier: {round(gbc_mod.score(x_test, y_test), 3)}")



pred_gbc = gbc_mod.predict(x_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred_gbc)

roc_auc = auc(false_positive_rate, true_positive_rate)

print(f"Area Under Curve for Gradient Boosting Classifier: {round(roc_auc, 3)}")
cv_method = StratifiedKFold(n_splits=3, 

                            random_state=42

                            )
# Cross validate Gradient Boosting Classifier model

scores_GBC = cross_val_score(gbc, x_train, y_train, cv = cv_method, n_jobs = 2, scoring = "accuracy")



print(f"Scores(Cross validate) for Gradient Boosting Classifier model:\n{scores_GBC}")

print(f"CrossValMeans: {round(scores_GBC.mean(), 3)}")

print(f"CrossValStandard Deviation: {round(scores_GBC.std(), 3)}")
params_GBC = {"loss": ["deviance"],

              "learning_rate": [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1], 

              "n_estimators": [200],

              "max_depth": [3, 5, 8],

              "min_samples_split": np.linspace(0.1, 1.0, 10, endpoint=True),

              "min_samples_leaf": np.linspace(0.1, 0.5, 5, endpoint=True),

              "max_features": ["log2","sqrt"],

              "criterion": ["friedman_mse", "mae"]

              }
GridSearchCV_GBC = GridSearchCV(estimator=GradientBoostingClassifier(), 

                                param_grid=params_GBC, 

                                cv=cv_method,

                                verbose=1, 

                                n_jobs=2,

                                scoring="accuracy", 

                                return_train_score=True

                                )
# Fit model with train data

GridSearchCV_GBC.fit(x_train, y_train);
# Get the best estimator values.

best_estimator_GBC = GridSearchCV_GBC.best_estimator_

print(f"Best estimator values for GBC model:\n{best_estimator_GBC}")
# Get the best parameter values.

best_params_GBC = GridSearchCV_GBC.best_params_

print(f"Best parameter values for GBC model:\n{best_params_GBC}")
# Best score for GBC by using the best_score attribute.

best_score_GBC = GridSearchCV_GBC.best_score_

print(f"Best score value foe GBC model: {round(best_score_GBC, 3)}")
# Test with new parameter for GBC model

gbc = GradientBoostingClassifier(criterion="friedman_mse", learning_rate=1, loss="deviance", max_depth=5, max_features="log2", min_samples_leaf=0.2, min_samples_split=0.5, n_estimators=200, random_state=42)

gbc_mod = gbc.fit(x_train, y_train)

pred_gbc = gbc_mod.predict(x_test)



mse_gbc = mean_squared_error(y_test, pred_gbc)

rmse_gbc = np.sqrt(mean_squared_error(y_test, pred_gbc))

score_gbc_train = gbc_mod.score(x_train, y_train)

score_gbc_test = gbc_mod.score(x_test, y_test)
print(f"Mean Square Error for Gradient Boosting Classifier = {round(mse_gbc, 3)}")

print(f"Root Mean Square Error for Gradient Boosting Classifier = {round(rmse_gbc, 3)}")

print(f"R^2(coefficient of determination) on training set = {round(score_gbc_train, 3)}")

print(f"R^2(coefficient of determination) on testing set = {round(score_gbc_test, 3)}")
print("Classification Report")

print(classification_report(y_test, pred_gbc))
print("Confusion Matrix:")

print(confusion_matrix(y_test, pred_gbc))
ax= plt.subplot()

sns.heatmap(confusion_matrix(y_test, pred_gbc), annot=True, ax = ax, cmap = "BuGn");



# labels, title and ticks

ax.set_xlabel("Predicted labels");

ax.set_ylabel("True labels"); 

ax.set_title("Confusion Matrix"); 

ax.xaxis.set_ticklabels(["Survived", "No Survived"]);
# Baseline model of XGBoost classifier with default parameters:

XGBoost_classifier = XGBClassifier()

XGBoost_classifier_mod = XGBoost_classifier.fit(x_train, y_train)

print(f"Baseline XGBoost classifier: {round(XGBoost_classifier_mod.score(x_test, y_test), 3)}")



pred_XGBoost_classifier = XGBoost_classifier_mod.predict(x_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred_XGBoost_classifier)

roc_auc = auc(false_positive_rate, true_positive_rate)

print(f"Area Under Curve for XGBoost Classifier: {round(roc_auc, 3)}")
# Cross validate  XGBoost Classifier model

scores_XGBoost = cross_val_score(XGBoost_classifier, x_train, y_train, cv = cv_method, n_jobs = 2, scoring = "accuracy")



print(f"Scores(Cross validate) for XGBoost Classifier model:\n{scores_XGBoost}")

print(f"CrossValMeans: {round(scores_XGBoost.mean(), 3)}")

print(f"CrossValStandard Deviation: {round(scores_XGBoost.std(), 3)}")
params_XGBoost = {"learning_rate": [0.01, 0.02, 0.05, 0.1], 

              "max_depth": [1, 3, 5],

              "min_child_weight": [1, 3, 5],

              "subsample": [0.6, 0.7, 1],

              "colsample_bytree": [1]

              }
GridSearchCV_XGBoost = GridSearchCV(estimator=XGBClassifier(), 

                                param_grid=params_XGBoost, 

                                cv=cv_method,

                                verbose=1, 

                                n_jobs=2,

                                scoring="accuracy", 

                                return_train_score=True

                                )
# Fit model with train data

GridSearchCV_XGBoost.fit(x_train, y_train);
best_estimator_XGBoost = GridSearchCV_XGBoost.best_estimator_

print(f"Best estimator value for XGboost model:\n{best_estimator_XGBoost}")
best_params_XGBoost = GridSearchCV_XGBoost.best_params_

print(f"Best parameter values for XGBoost model:\n{best_params_XGBoost}")
best_score_XGBoost = GridSearchCV_XGBoost.best_score_

print(f"Best score for XGBoost model: {round(best_score_XGBoost, 3)}")
# The grid search returns the following as the best parameter set

XGBoost_classifier = XGBClassifier(colsample_bytree=1, learning_rate=0.05, max_depth=5, min_child_weight=1, subsample=0.6, random_state=42)

XGBoost_classifier_mod = XGBoost_classifier.fit(x_train, y_train)

pred_XGBoost_classifier = XGBoost_classifier_mod.predict(x_test)



mse_gbc = mean_squared_error(y_test, pred_XGBoost_classifier)

rmse_gbc = np.sqrt(mean_squared_error(y_test, pred_XGBoost_classifier))

score_XGBoost_classifier_train = XGBoost_classifier_mod.score(x_train, y_train)

score_XGBoost_classifier_test = XGBoost_classifier_mod.score(x_test, y_test)
print(f"Mean Square Error for XGBoost Classifier = {round(mse_gbc, 3)}")

print(f"Root Mean Square Error for XGBoost Classifier = {round(rmse_gbc, 3)}")

print(f"R^2(coefficient of determination) on training set = {round(score_XGBoost_classifier_train, 3)}")

print(f"R^2(coefficient of determination) on testing set = {round(score_XGBoost_classifier_test, 3)}")
print("Classification Report")

print(classification_report(y_test, pred_XGBoost_classifier))
print("Confusion Matrix:")

print(confusion_matrix(y_test, pred_XGBoost_classifier))
ax= plt.subplot()

sns.heatmap(confusion_matrix(y_test, pred_XGBoost_classifier), annot=True, ax = ax, cmap = "BuGn");



# labels, title and ticks

ax.set_xlabel("Predicted labels");

ax.set_ylabel("True labels"); 

ax.set_title("Confusion Matrix"); 

ax.xaxis.set_ticklabels(["Survived", "No Survived"]);
# Baseline model of Adaptive Boosting with default parameters:

adaboost = AdaBoostClassifier()

adaboost_mod = adaboost.fit(x_train, y_train)

print(f"Baseline Adaptive Boosting: {round(adaboost_mod.score(x_test, y_test), 3)}")



pred_adaboost = adaboost_mod.predict(x_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred_adaboost)

roc_auc = auc(false_positive_rate, true_positive_rate)

print(f"Area Under Curve for Adaptive Boosting: {roc_auc}")
# Cross validate Adaptive Boosting model

scores_Adaptive = cross_val_score(adaboost, x_train, y_train, cv = 3, n_jobs = 2, scoring = "accuracy")



print(f"Scores(Cross validate) for Adaptive Boosting model:\n{scores_Adaptive}")

print(f"CrossValMeans: {round(scores_Adaptive.mean(), 3)}")

print(f"CrossValStandard Deviation: {round(scores_Adaptive.std(), 3)}")
params_adaboost = {"learning_rate": [0.001, 0.01, 0.5, 0.1], 

              "n_estimators": [750],

              "algorithm": ["SAMME.R"]

              }
GridSearchCV_adaboost = GridSearchCV(estimator=AdaBoostClassifier(), 

                                param_grid=params_adaboost, 

                                cv=cv_method,

                                verbose=1, 

                                n_jobs=2,

                                scoring="accuracy", 

                                return_train_score=True

                                )
# Fit model with train data

GridSearchCV_adaboost.fit(x_train, y_train);
best_estimator_adaboost = GridSearchCV_adaboost.best_estimator_

print(f"Best estimatore for Adaboost model:\n{best_estimator_adaboost}")
best_params_adaboost = GridSearchCV_adaboost.best_params_

print(f"Best parameter values for Adaboost model:\n{best_params_adaboost}")
best_score_adaboost = GridSearchCV_adaboost.best_score_

print(f"Best score for Adaboost model: {round(best_score_adaboost, 3)}")
# The grid search returns the following as the best parameter set

adaboost = AdaBoostClassifier(algorithm="SAMME.R", learning_rate=0.5, n_estimators=750, random_state=42)

adaboost_mod = adaboost.fit(x_train, y_train)

pred_adaboost = adaboost_mod.predict(x_test)



mse_adaboost = mean_squared_error(y_test, pred_adaboost)

rmse_adaboost = np.sqrt(mean_squared_error(y_test, pred_adaboost))

score_adaboost_train = adaboost_mod.score(x_train, y_train)

score_adaboost_train = adaboost_mod.score(x_test, y_test)
print(f"Mean Square Error for Adaptive Boosting = {round(mse_adaboost, 3)}")

print(f"Root Mean Square Error for Adaptive Boosting = {round(rmse_adaboost, 3)}")

print(f"R^2(coefficient of determination) on training set = {round(score_adaboost_train, 3)}")

print(f"R^2(coefficient of determination) on testing set = {round(score_adaboost_train, 3)}")
print("Classification Report")

print(classification_report(y_test, pred_adaboost))
print("Confusion Matrix:")

print(confusion_matrix(y_test, pred_adaboost))
ax= plt.subplot()

sns.heatmap(confusion_matrix(y_test, pred_adaboost), annot=True, ax = ax, cmap = "BuGn");



# labels, title and ticks

ax.set_xlabel("Predicted labels");

ax.set_ylabel("True labels"); 

ax.set_title("Confusion Matrix"); 

ax.xaxis.set_ticklabels(["Survived", "No Survived"]);
# Baseline model of Logistic Regression with default parameters:



logistic_regression = linear_model.LogisticRegression()

logistic_regression_mod = logistic_regression.fit(x_train, y_train)

print(f"Baseline Logistic Regression: {round(logistic_regression_mod.score(x_test, y_test), 3)}")



pred_logistic_regression = logistic_regression_mod.predict(x_test)
y_pred_proba = logistic_regression.predict_proba(x_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr, tpr, color="#00897B", label="data 1, auc="+str(auc))

plt.legend(loc=4)



plt.show()
# Cross validate Logistic Regression model

scores_Logistic = cross_val_score(logistic_regression, x_train, y_train, cv =cv_method, n_jobs = 2, scoring = "accuracy")



print(f"Scores(Cross validate) for Logistic Regression model:\n{scores_Logistic}")

print(f"CrossValMeans: {round(scores_Logistic.mean(), 3)}")

print(f"CrossValStandard Deviation: {round(scores_Logistic.std(), 3)}")
params_LR = {"tol": [0.0001,0.0002,0.0003],

            "C": [0.01, 0.1, 1, 10, 100],

            "intercept_scaling": [1, 2, 3, 4]

              }
GridSearchCV_LR = GridSearchCV(estimator=linear_model.LogisticRegression(), 

                                param_grid=params_LR, 

                                cv=cv_method,

                                verbose=1, 

                                n_jobs=2,

                                scoring="accuracy", 

                                return_train_score=True

                                )
# Fit model with train data

GridSearchCV_LR.fit(x_train, y_train);
best_estimator_LR = GridSearchCV_LR.best_estimator_

print(f"Best estimator for LR model:\n{best_estimator_LR}")
best_params_LR = GridSearchCV_LR.best_params_

print(f"Best parameter values for LR model:\n{best_params_LR}")
print(f"Best score for LR model: {round(GridSearchCV_LR.best_score_, 3)}")
# The grid search returns the following as the best parameter set

logistic_regression = linear_model.LogisticRegression(C=1, intercept_scaling=1, tol=0.0001, penalty="l2", solver="liblinear", random_state=42)

logistic_regression_mod = logistic_regression.fit(x_train, y_train)

pred_logistic_regression = logistic_regression_mod.predict(x_test)



mse_logistic_regression = mean_squared_error(y_test, pred_logistic_regression)

rmse_logistic_regression = np.sqrt(mean_squared_error(y_test, pred_logistic_regression))

score_logistic_regression_train = logistic_regression_mod.score(x_train, y_train)

score_logistic_regression_test = logistic_regression_mod.score(x_test, y_test)
print(f"Mean Square Error for Logistic Regression = {round(mse_logistic_regression, 3)}")

print(f"Root Mean Square Error for Logistic Regression = {round(rmse_logistic_regression, 3)}")

print(f"R^2(coefficient of determination) on training set = {round(score_logistic_regression_train, 3)}")

print(f"R^2(coefficient of determination) on testing set = {round(score_logistic_regression_test, 3)}")
print("Classification Report")

print(classification_report(y_test, pred_logistic_regression))
print("Confusion Matrix:")

print(confusion_matrix(y_test, pred_logistic_regression))
ax= plt.subplot()

sns.heatmap(confusion_matrix(y_test, pred_logistic_regression), annot=True, ax = ax, cmap = "BuGn");



# labels, title and ticks

ax.set_xlabel("Predicted labels");

ax.set_ylabel("True labels"); 

ax.set_title("Confusion Matrix"); 

ax.xaxis.set_ticklabels(["Survived", "No Survived"]);
# Baseline model of Stochastic Gradient Descent with default parameters:



sgd = linear_model.SGDClassifier()

sgd_mod = sgd.fit(x_train, y_train)

print(f"Baseline Stochastic Gradient Descent: {round(sgd_mod.score(x_test, y_test), 3)}")



pred_sgd = sgd_mod.predict(x_test)
# Cross validate Stochastic Gradient Descent model

scores_SGD = cross_val_score(sgd, x_train, y_train, cv = 3, n_jobs = 2, scoring = "accuracy")



print(f"Scores(Cross validate) for Stochastic Gradient Descent model:\n{scores_SGD}")

print(f"CrossValMeans: {round(scores_SGD.mean(), 3)}")

print(f"CrossValStandard Deviation: {round(scores_SGD.std(), 3)}")
params_SGD = {"alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 1], 

             "max_iter": [1, 3, 5, 7], 

             "tol": [0.0001,0.0002,0.0003]

              }
GridSearchCV_SGD = GridSearchCV(estimator=linear_model.SGDClassifier(), 

                                param_grid=params_SGD, 

                                cv=cv_method,

                                verbose=1, 

                                n_jobs=2,

                                scoring="accuracy", 

                                return_train_score=True

                                )
# Fit model with train data

GridSearchCV_SGD.fit(x_train, y_train);
best_estimator_SGD = GridSearchCV_SGD.best_estimator_

print(f"Best estimator for SGD model:\n{best_estimator_SGD}")
best_params_SGD = GridSearchCV_SGD.best_params_

print(f"Best parameter values for SGD model:\n{best_params_SGD}")
best_score_SGD = GridSearchCV_SGD.best_score_

print(f"Best score for SGD model: {round(best_score_SGD, 3)}")
sgd = linear_model.SGDClassifier(alpha=0.005, max_iter=7, tol=0.0003, random_state=42)

sgd_mod = sgd.fit(x_train, y_train)

pred_sgd = sgd_mod.predict(x_test)



mse_sgd = mean_squared_error(y_test, pred_sgd)

rmse_sgd = np.sqrt(mean_squared_error(y_test, pred_sgd))

score_sgd_train = sgd_mod.score(x_train, y_train)

score_sgd_test = sgd_mod.score(x_test, y_test)
print(f"Mean Square Error for Stochastic Gradient Descent = {round(mse_sgd, 3)}")

print(f"Root Mean Square Error for Stochastic Gradient Descent = {round(rmse_sgd, 3)}")

print(f"R^2(coefficient of determination) on training set = {round(score_sgd_train, 3)}")

print(f"R^2(coefficient of determination) on testing set = {round(score_sgd_test, 3)}")
print("Classification Report")

print(classification_report(y_test, pred_sgd))
print("Confusion Matrix:")

print(confusion_matrix(y_test, pred_sgd))
ax= plt.subplot()

sns.heatmap(confusion_matrix(y_test, pred_sgd), annot=True, ax = ax, cmap = "BuGn");



# labels, title and ticks

ax.set_xlabel("Predicted labels");

ax.set_ylabel("True labels"); 

ax.set_title("Confusion Matrix"); 

ax.xaxis.set_ticklabels(["Survived", "No Survived"]);
random_forest = RandomForestClassifier()

random_forest_mod = random_forest.fit(x_train, y_train)

print(f"Baseline Random Forest: {round(random_forest_mod.score(x_test, y_test), 3)}")



pred_random_forest = random_forest_mod.predict(x_test)
# Cross validate Random forest model

scores_RF = cross_val_score(random_forest, x_train, y_train, cv = cv_method, n_jobs = 2, scoring = "accuracy")



print(f"Scores(Cross validate) for Random forest model:\n{scores_RF}")

print(f"CrossValMeans: {round(scores_RF.mean(), 3)}")

print(f"CrossValStandard Deviation: {round(scores_RF.std(), 3)}")
params_RF = {"min_samples_split": [2, 6, 20],

              "min_samples_leaf": [1, 4, 16],

              "n_estimators" :[100,200,300,400],

              "criterion": ["gini"]             

              }
GridSearchCV_RF = GridSearchCV(estimator=RandomForestClassifier(), 

                                param_grid=params_RF, 

                                cv=cv_method,

                                verbose=1, 

                                n_jobs=2,

                                scoring="accuracy", 

                                return_train_score=True

                                )
# Fit model with train data

GridSearchCV_RF.fit(x_train, y_train);
best_estimator_RF = GridSearchCV_RF.best_estimator_

print(f"Best estimator for RF model:\n{best_estimator_RF}")
best_params_RF = GridSearchCV_RF.best_params_

print(f"Best parameter values for RF model:\n{best_params_RF}")
best_score_RF = GridSearchCV_RF.best_score_

print(f"Best score for RF model: {round(best_score_RF, 3)}")
random_forest = RandomForestClassifier(criterion="gini", n_estimators=200, min_samples_leaf=1, min_samples_split=6, random_state=42)

random_forest_mod = random_forest.fit(x_train, y_train)

pred_random_forest = random_forest_mod.predict(x_test)



mse_random_forest = mean_squared_error(y_test, pred_random_forest)

rmse_random_forest = np.sqrt(mean_squared_error(y_test, pred_random_forest))

score_random_forest_train = random_forest_mod.score(x_train, y_train)

score_random_forest_test = random_forest_mod.score(x_test, y_test)
print(f"Mean Square Error for Random Forest = {round(mse_random_forest, 3)}")

print(f"Root Mean Square Error for Random Forest = {round(rmse_random_forest, 3)}")

print(f"R^2(coefficient of determination) on training set = {round(score_random_forest_train, 3)}")

print(f"R^2(coefficient of determination) on testing set = {round(score_random_forest_test, 3)}")
print("Classification Report")

print(classification_report(y_test, pred_random_forest))
print("Confusion Matrix:")

print(confusion_matrix(y_test, pred_random_forest))
ax= plt.subplot()

sns.heatmap(confusion_matrix(y_test, pred_random_forest), annot=True, ax = ax, cmap = "BuGn");



# labels, title and ticks

ax.set_xlabel("Predicted labels");

ax.set_ylabel("True labels"); 

ax.set_title("Confusion Matrix"); 

ax.xaxis.set_ticklabels(["Survived", "No Survived"]);
svc = SVC()

svc_mod = svc.fit(x_train, y_train)

print(f"Baseline Support Vector Machine: {round(svc_mod.score(x_test, y_test), 3)}")



pred_svc = svc_mod.predict(x_test)
# Cross validate SVC model

scores_SVC = cross_val_score(svc, x_train, y_train, cv = cv_method, n_jobs = 2, scoring = "accuracy")



print(f"Scores(Cross validate) for SVC model:\n{scores_SVC}")

print(f"CrossValMeans: {round(scores_SVC.mean(), 3)}")

print(f"CrossValStandard Deviation: {round(scores_SVC.std(), 3)}")
params_SVC = {"C": [0.1, 1, 10, 100, 1000],  

              "gamma": [1, 0.1, 0.01, 0.001, 0.0001], 

              "kernel": ["rbf"]

              }
GridSearchCV_SVC = GridSearchCV(estimator=SVC(), 

                                param_grid=params_SVC, 

                                cv=cv_method,

                                verbose=1, 

                                n_jobs=-1,

                                refit = True,

                                scoring="accuracy", 

                                return_train_score=True

                                )
# Fit model with train data

GridSearchCV_SVC.fit(x_train, y_train);
best_estimator_SVC = GridSearchCV_SVC.best_estimator_

print(f"Best estimator for SVC model:\n{best_estimator_SVC}")
best_params_SVC = GridSearchCV_SVC.best_params_

print(f"Best parameter values:\n{best_params_SVC}")
best_score_SVC = GridSearchCV_SVC.best_score_

print(f"Best score for SVC model: {round(best_score_SVC, 3)}")
svc = SVC(C=1000, gamma=0.01, kernel="rbf" , random_state=42)

svc_mod = svc.fit(x_train, y_train)

pred_svc = svc_mod.predict(x_test)



mse_svc = mean_squared_error(y_test, pred_svc)

rmse_svc = np.sqrt(mean_squared_error(y_test, pred_svc))

score_svc_train = svc_mod.score(x_train, y_train)

score_svc_test = svc_mod.score(x_test, y_test)
print(f"Mean Square Error for Linear Support Vector Machine = {round(mse_svc, 3)}")

print(f"Root Mean Square Error for Linear Support Vector Machine = {round(rmse_svc, 3)}")

print(f"R^2(coefficient of determination) on training set = {round(score_svc_train, 3)}")

print(f"R^2(coefficient of determination) on testing set = {round(score_svc_test, 3)}")
print("Classification Report")

print(classification_report(y_test, pred_svc))
print("Confusion Matrix:")

print(confusion_matrix(y_test, pred_svc))
ax= plt.subplot()

sns.heatmap(confusion_matrix(y_test, pred_svc), annot=True, ax = ax, cmap = "BuGn");



# labels, title and ticks

ax.set_xlabel("Predicted labels");

ax.set_ylabel("True labels"); 

ax.set_title("Confusion Matrix"); 

ax.xaxis.set_ticklabels(["Survived", "No Survived"]);
decision_tree = DecisionTreeClassifier(random_state= 42)

decision_tree_mod = decision_tree.fit(x_train, y_train)

print(f"Baseline Decision Tree: {round(decision_tree_mod.score(x_test, y_test), 3)}")



pred_decision_tree = decision_tree_mod.predict(x_test)
# Cross validate Decision Tree model

scores_DT = cross_val_score(decision_tree, x_train, y_train, cv = cv_method, n_jobs = 2, scoring = "accuracy")



print(f"Scores(Cross validate) for Decision Tree model:\n{scores_DT}")

print(f"CrossValMeans: {round(scores_DT.mean(), 3)}")

print(f"CrossValStandard Deviation: {round(scores_DT.std(), 3)}")
params_DT = {"criterion": ["gini", "entropy"],

             "max_depth": [1, 2, 3, 4, 5, 6, 7, 8],

             "min_samples_split": [2, 3]}
GridSearchCV_DT = GridSearchCV(estimator=DecisionTreeClassifier(), 

                                param_grid=params_DT, 

                                cv=cv_method,

                                verbose=1, 

                                n_jobs=-1,

                                scoring="accuracy", 

                                return_train_score=True

                                )
# Fit model with train data

GridSearchCV_DT.fit(x_train, y_train);
best_estimator_DT = GridSearchCV_DT.best_estimator_

print(f"Best estimator for DT model:\n{best_estimator_DT}")
best_params_DT = GridSearchCV_DT.best_params_

print(f"Best parameter values:\n{best_params_DT}")
best_score_DT = GridSearchCV_DT.best_score_

print(f"Best score for DT model: {round(best_score_DT, 3)}")
decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_impurity_split=2, min_samples_leaf=0.4, random_state=42)

decision_tree_mod = decision_tree.fit(x_train, y_train)

pred_decision_tree = decision_tree_mod.predict(x_test)



mse_decision_tree = mean_squared_error(y_test, pred_decision_tree)

rmse_decision_tree = np.sqrt(mean_squared_error(y_test, pred_decision_tree))

score_decision_tree_train = decision_tree_mod.score(x_train, y_train)

score_decision_tree_test = decision_tree_mod.score(x_test, y_test)
print(f"Mean Square Error for Decision Tree = {round(mse_decision_tree, 3)}")

print(f"Root Mean Square Error for Decision Tree = {round(rmse_decision_tree, 3)}")

print(f"R^2(coefficient of determination) on training set = {round(score_decision_tree_train, 3)}")

print(f"R^2(coefficient of determination) on testing set = {round(score_decision_tree_test, 3)}")
print("Classification Report")

print(classification_report(y_test, pred_decision_tree))
print("Confusion Matrix:")

print(confusion_matrix(y_test, pred_decision_tree))
ax= plt.subplot()

sns.heatmap(confusion_matrix(y_test, pred_decision_tree), annot=True, ax = ax, cmap = "BuGn");



# labels, title and ticks

ax.set_xlabel("Predicted labels");

ax.set_ylabel("True labels"); 

ax.set_title("Confusion Matrix"); 

ax.xaxis.set_ticklabels(["Survived", "No Survived"]);
gaussianNB = GaussianNB()

gaussianNB_mod = gaussianNB.fit(x_train, y_train)

print(f"Baseline Gaussin Navie Bayes: {round(gaussianNB_mod.score(x_test, y_test), 3)}")



pred_gaussianNB = random_forest_mod.predict(x_test)
# Cross validate Gaussian Naive Bayes model

scores_GNB = cross_val_score(gaussianNB, x_train, y_train, cv = cv_method, n_jobs = 2, scoring = "accuracy")



print(f"Scores(Cross validate) for Gaussian Naive Bayes model:\n{scores_GNB}")

print(f"CrossValMeans: {round(scores_GNB.mean(), 3)}")

print(f"CrossValStandard Deviation: {round(scores_GNB.std(), 3)}")
params_GNB = {"C": [0.1,0.25,0.5,1],

              "gamma": [0.1,0.5,0.8,1.0],

              "kernel": ["rbf","linear"]}
GridSearchCV_GNB = GridSearchCV(estimator=svm.SVC(), 

                                param_grid=params_GNB, 

                                cv=cv_method,

                                verbose=1, 

                                n_jobs=-1,

                                scoring="accuracy", 

                                return_train_score=True

                                )
# Fit model with train data

GridSearchCV_GNB.fit(x_train, y_train);
best_estimator_GNB = GridSearchCV_GNB.best_estimator_

print(f"Best estimator for DT model:\n{best_estimator_GNB}")
best_params_GNB = GridSearchCV_GNB.best_params_

print(f"Best parameter values:\n{best_params_GNB}")
best_score_GNB = GridSearchCV_GNB.best_score_

print(f"Best score for GNB model: {round(best_score_GNB, 3)}")
mse_gaussianNB = mean_squared_error(y_test, pred_gaussianNB)

rmse_gaussianNB = np.sqrt(mean_squared_error(y_test, pred_gaussianNB))

score_gaussianNB_train = gaussianNB_mod.score(x_train, y_train)

score_gaussianNB_test = gaussianNB_mod.score(x_test, y_test)
print(f"Mean Square Error for Gaussian Naive Bayes = {round(mse_gaussianNB, 3)}")

print(f"Root Mean Square Error for Gaussian Naive Bayes = {round(rmse_gaussianNB, 3)}")

print(f"R^2(coefficient of determination) on training set = {round(score_gaussianNB_train, 3)}")

print(f"R^2(coefficient of determination) on testing set = {round(score_gaussianNB_test, 3)}")
print("Classification Report")

print(classification_report(y_test, pred_gaussianNB))
print("Confusion Matrix:")

print(confusion_matrix(y_test, pred_gaussianNB))
ax= plt.subplot()

sns.heatmap(confusion_matrix(y_test, pred_gaussianNB), annot=True, ax = ax, cmap = "BuGn");



# labels, title and ticks

ax.set_xlabel("Predicted labels");

ax.set_ylabel("True labels"); 

ax.set_title("Confusion Matrix"); 

ax.xaxis.set_ticklabels(["Survived", "No Survived"]);
feat_labels = ["Deck", "Pclass", "Title", "Fare_Bin", "Embarked", "AgeState"]



# Print the name and gini importance of each feature

for feature in zip(feat_labels, XGBoost_classifier_mod.feature_importances_):

    print(f"Feature importances:\n{feature}")
# plot feature importance

from xgboost import plot_importance



plot_importance(XGBoost_classifier_mod, color="#00897B")

plt.title("Feature importances in XGBoost classifier model")

plt.show()
# Plot learning curve

def plot_learning_curve(estimator, title, x, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

        

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="#80CBC4",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="#00897B",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
# Gradient Boosting Classifier

plot_learning_curve(GridSearchCV_GBC.best_estimator_,title = "Gradient Boosting Classifier learning curve", x = x_train, y = y_train, cv = cv_method);
# XGBoost Classifier

plot_learning_curve(GridSearchCV_XGBoost.best_estimator_,title = "XGBoost Classifier learning curve", x = x_train, y = y_train, cv = cv_method);
# Adaptive Boosting

plot_learning_curve(GridSearchCV_adaboost.best_estimator_,title = "Adaptive Boosting learning curve", x = x_train, y = y_train, cv = cv_method);
# Logistic Regression

plot_learning_curve(GridSearchCV_LR.best_estimator_,title = "Logistict Regression learning curve", x = x_train, y = y_train, cv = cv_method);
# Stochastic Gradient Descent

plot_learning_curve(GridSearchCV_SGD.best_estimator_,title = "Stochastic Gradient Descent learning curve", x = x_train, y = y_train, cv = cv_method);
# Random forest

plot_learning_curve(GridSearchCV_RF.best_estimator_,title = "Random Forest learning curve", x = x_train, y = y_train, cv = cv_method);
# Support Vector Machine

plot_learning_curve(GridSearchCV_SVC.best_estimator_,title = "Support Vector Machine learning curve", x = x_train, y = y_train, cv = cv_method);
# Decision Tree

plot_learning_curve(GridSearchCV_DT.best_estimator_,title = "Decision Tree learning curve", x = x_train, y = y_train, cv = cv_method);
# Gaussian Naive Bayes

plot_learning_curve(GridSearchCV_LR.best_estimator_,title = "Gaussian Naive Bayes learning curve", x = x_train, y = y_train, cv = cv_method);
results = pd.DataFrame({

                        "Model": ["Gradient Boosting Classifier",

                                    "XGBoost Classifier",

                                    "Adaptive Boosting",

                                    "Logistic Regression",

                                    "Stochastic Gradient Descent",

                                    "Random Forest",

                                    "Support Vector Machine",

                                    "Decision Tree",

                                    "Gaussian Naive Bayes"],

                        "Score": [gbc_mod.score(x_train, y_train),

                                    XGBoost_classifier_mod.score(x_train, y_train),

                                    adaboost_mod.score(x_train, y_train),

                                    logistic_regression_mod.score(x_train, y_train),

                                    sgd_mod.score(x_train, y_train),

                                    random_forest_mod.score(x_train, y_train),

                                    svc_mod.score(x_train, y_train),

                                    decision_tree_mod.score(x_train, y_train),

                                    gaussianNB_mod.score(x_train, y_train)]

                        })

result_df = results.sort_values(by="Score", ascending=False)

result_df = result_df.set_index("Score")

result_df.head(10)
stack = StackingCVClassifier(classifiers=[random_forest, adaboost, svc, logistic_regression, gbc],

                            meta_classifier=XGBoost_classifier,

                            random_state=42)



stack_mod = stack.fit(x_train, y_train.ravel())

stack_pred = stack_mod.predict(x_test)



print(f"Root Mean Square Error test for STACKING CV Classifier: {round(np.sqrt(mean_squared_error(y_test, stack_pred)), 3)}")
test["PassengerId"].value_counts()
# final = (final1 + final2 + final3)



Final_Submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": stack_mod.predict(X_test)})



Final_Submission.to_csv("Final_Submission.csv", index=False)

Final_Submission.head(10)