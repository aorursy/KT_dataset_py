import numpy as np

import pandas as pd

import seaborn as sbn

import xgboost as xgb

from xgboost import plot_importance



from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import accuracy_score



import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test_X = pd.read_csv("/kaggle/input/titanic/test.csv")

train.head()
train.drop("PassengerId", axis=1, inplace=True)

train.head()
train_X = train.loc[:,train.columns != "Survived"].copy()

train_Y = train.loc[:,["Survived"]].copy()



train_X.info()

train_Y.info()
sum(train_Y.values) / len(train_Y)
corr_matrix = train_X.corr()

sbn.heatmap(corr_matrix, annot=True);
train_X["Ticket"].nunique()
train_X.drop("Ticket", axis=1, inplace=True)

test_X.drop("Ticket", axis=1, inplace=True)
train_X.drop("Cabin", axis=1, inplace=True)

test_X.drop("Cabin", axis=1, inplace=True)
train_X.loc[train_X["Sex"] == "female", "Sex"] = 1

train_X.loc[train_X["Sex"] == "male", "Sex"] = 0

train_X["Sex"] = train_X["Sex"].astype(np.int64)



test_X.loc[test_X["Sex"] == "female", "Sex"] = 1

test_X.loc[test_X["Sex"] == "male", "Sex"] = 0

test_X["Sex"] = test_X["Sex"].astype(np.int64)
train_X.head()
test_X.head()
train_X.loc[train_X["Name"].str.contains("Miss. "), "Name"] = "Miss"

train_X.loc[train_X["Name"].str.contains("Mrs. "), "Name"] = "Mrs"

train_X.loc[train_X["Name"].str.contains("Ms. "), "Name"] = "Ms"

train_X.loc[train_X["Name"].str.contains("Master. "), "Name"] = "Master"

train_X.loc[train_X["Name"].str.contains("Mr. |Mr "), "Name"] = "Mr"



print(train_X["Name"].value_counts())
train_X.loc[train_X["Name"]. \

            str.contains("Ms"),"Name"] = "Miss"

train_X.loc[~train_X["Name"]. \

            str.contains("Miss|Mrs|Master|Mr"), "Name"] = "NaN"

print(train_X["Name"].value_counts())
train_X.loc[(train_X["Name"] == "NaN") &

            (train_X["Sex"] == 1), "Name"] = "Miss"

train_X.loc[(train_X["Name"] == "NaN") &

            (train_X["Sex"] == 0), "Name"] = "Mr"

print(train_X["Name"].value_counts())
n_encoder = OneHotEncoder(sparse=False)



# fit encoder

# If you run below line as "n_encoder.fit(train_X["Name"])"

# with one square parantheses, you will get an error.

n_encoder.fit(train_X[["Name"]])



# transform Name column of train_X

name_one_hot_np = n_encoder.transform(train_X[["Name"]])

# convert datatype to int64

name_one_hot_np = name_one_hot_np.astype(np.int64)



# output of transform is a numpy array, convert it to Pandas dataframe

name_one_hot_df = pd.DataFrame(name_one_hot_np,

                    columns=["Name_1", "Name_2", "Name_3", "Name_4"])



# concatenate name_one_hot_df to train_X

train_X = pd.concat([train_X, name_one_hot_df], axis=1)



# drop categorical Name column

train_X.drop("Name", axis=1, inplace=True)



train_X.head()
test_X.loc[test_X["Name"].str.contains("Miss. "), "Name"] = "Miss"

test_X.loc[test_X["Name"].str.contains("Mrs. "), "Name"] = "Mrs"

test_X.loc[test_X["Name"].str.contains("Ms. "), "Name"] = "Ms"

test_X.loc[test_X["Name"].str.contains("Master. "), "Name"] = "Master"

test_X.loc[test_X["Name"].str.contains("Mr. "), "Name"] = "Mr"



print(test_X["Name"].value_counts())
test_X.loc[test_X["Name"]. \

           str.contains("Ms"), "Name"] = "Miss"

test_X.loc[~test_X["Name"]. \

           str.contains("Miss|Mrs|Master|Mr"), "Name"] = "NaN"

print(test_X["Name"].value_counts())
test_X.loc[(test_X["Name"] == "NaN") &

           (test_X["Sex"] == 1), "Name"] = "Miss"

test_X.loc[(test_X["Name"] == "NaN") &

           (test_X["Sex"] == 0), "Name"] = "Mr"



# transform Name column of test_X

name_one_hot_np = n_encoder.transform(test_X[["Name"]])

# convert datatype to int64

name_one_hot_np = name_one_hot_np.astype(np.int64)



# output of transform is a numpy array, convert it to Pandas dataframe

name_one_hot_df = pd.DataFrame(name_one_hot_np,

                    columns=["Name_1", "Name_2", "Name_3", "Name_4"])



# concatenate name_one_hot_df to test_X

test_X = pd.concat([test_X, name_one_hot_df], axis=1)



# drop categorical Name column

test_X.drop("Name", axis=1, inplace=True)



test_X.head()
train_X.info()

test_X.info()
print(train_X["Embarked"].value_counts())
train_X["Embarked"] = train_X["Embarked"].fillna("S")
e_encoder = OneHotEncoder(sparse=False)



# fit encoder

e_encoder.fit(train_X[["Embarked"]])



# transform Embarked column of train_X

e_one_hot_np = e_encoder.transform(train_X[["Embarked"]])

# convert datatype to int64

e_one_hot_np = e_one_hot_np.astype(np.int64)



# output of transform is a numpy array, convert it to Pandas dataframe

e_one_hot_df = pd.DataFrame(e_one_hot_np,

                            columns=["Emb_1", "Emb_2", "Emb_3"])



# concatenate e_one_hot_df to train_X

train_X = pd.concat([train_X, e_one_hot_df], axis=1)



# drop categorical Embarked column

train_X.drop("Embarked", axis=1, inplace=True)
# transform Embarked column of test_X

e_one_hot_np = e_encoder.transform(test_X[["Embarked"]])

# convert datatype to int64

e_one_hot_np = e_one_hot_np.astype(np.int64)



# output of transform is a numpy array, convert it to Pandas dataframe

e_one_hot_df = pd.DataFrame(e_one_hot_np,

                            columns=["Emb_1", "Emb_2", "Emb_3"])



# concatenate e_one_hot_df to test_X

test_X = pd.concat([test_X, e_one_hot_df], axis=1)



# drop categorical Embarked column

test_X.drop("Embarked", axis=1, inplace=True)
train_X.info()

test_X.info()
reg_age = train_X.copy()



# get only rows with non-null age values

reg_age_train_X = reg_age[reg_age["Age"].notnull()].copy()

reg_age_train_Y = reg_age_train_X[["Age"]]

reg_age_train_X.drop("Age", axis=1, inplace=True)
reg = xgb.XGBRegressor(colsample_bylevel=0.7,

                       colsample_bytree=0.5,

                       learning_rate=0.3,

                       max_depth=5,

                       min_child_weight=1.5,

                       n_estimators=18,

                       subsample=0.9)



reg.fit(reg_age_train_X, reg_age_train_Y);
age_all_predicted = reg.predict(train_X.loc[:, train_X.columns != "Age"])
train_X.loc[train_X["Age"].isnull(),

                    "Age"] = age_all_predicted[train_X["Age"].isnull()]
age_all_predicted_t = reg.predict(test_X.loc[:, (test_X.columns != "Age")&

                                    (test_X.columns != "PassengerId")])

test_X.loc[test_X["Age"].isnull(),

                       "Age"] = age_all_predicted_t[test_X["Age"].isnull()]
cls = xgb.XGBClassifier()



parameters = {

    "colsample_bylevel": np.arange(0.4, 1.0, 0.1),

    "colsample_bytree": np.arange(0.7, 1.0, 0.1),

    "learning_rate": [0.4, 0.45, 0.5, 0.55],

    "max_depth": np.arange(4, 6, 1),

    "min_child_weight": np.arange(1, 2, 0.5),

    "n_estimators": [8, 10, 12],

    "subsample": np.arange(0.6, 1.0, 0.1)

}



skf_cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)



gscv = GridSearchCV(

    estimator=cls,

    param_grid=parameters,

    scoring="roc_auc",

    n_jobs=-1,

    cv=skf_cv,

    verbose=True,

    refit=True,

)



gscv.fit(train_X, train_Y.values.ravel());

prm = gscv.cv_results_["params"] 

print(len(prm))
mean_test = gscv.cv_results_["mean_test_score"]

std_test = gscv.cv_results_["std_test_score"]

print(len(mean_test))

print(len(std_test))
#for r in range(len(prm)):

for r in range(5):

    print("Parameter set {}".format(prm[r]))

    print("Test score (mean {:.4f}, std {:.4f}))". \

                  format(mean_test[r], std_test[r]))

    print("\n")
print("Best parameters {}".format(gscv.best_params_))

print("Mean cross-validated score of best estimator {}". \

                              format(gscv.best_score_))
cls2 = xgb.XGBClassifier(**gscv.best_params_)



train_fold_X, valid_fold_X, train_fold_Y, valid_fold_Y = train_test_split(train_X, train_Y,

                                                            test_size=0.2, random_state=0) 



# Fit classifier to new split

cls2.fit(train_fold_X, train_fold_Y.values.ravel())



# Predict probabilities on valid_fold_X

split_pred_proba = cls2.predict_proba(valid_fold_X)
# Compute ROC AUC score

split_score = roc_auc_score(valid_fold_Y, split_pred_proba[:,1])

print("ROC AUC is {:.4f}".format(split_score))



# Draw ROC curve

fpr, tpr, thr = roc_curve(valid_fold_Y, split_pred_proba[:,1])

plt.plot(fpr, tpr, linestyle="-");

plt.title("ROC Curve");

plt.xlabel("False Positive Rate");

plt.ylabel("True Positive Rate");
# Compute optimal operating point and threshold

ind = np.argmin(list(map(lambda x, y: x**2 + (y - 1.0)**2, fpr,tpr)))



# Optimal threshold for our classifier cls2

opt_thr = thr[ind]



print("Optimal operating point on ROC curve (fpr,tpr) ({:.4f}, {:.4f})"

                                              "  with threshold {:.4f}".

                                    format(fpr[ind], tpr[ind], opt_thr))
y_pred = [1 if x > opt_thr else 0 for x in split_pred_proba[:,1]]

acc = accuracy_score(valid_fold_Y, y_pred)

print("Accuracy is {:.4f}".format(acc))
plot_importance(cls2);
train_fold_X3 = train_fold_X.copy()

valid_fold_X3 = valid_fold_X.copy()



train_fold_X3.drop("Emb_1", axis=1, inplace=True)

train_fold_X3.drop("Emb_2", axis=1, inplace=True)



valid_fold_X3.drop("Emb_1", axis=1, inplace=True)

valid_fold_X3.drop("Emb_2", axis=1, inplace=True)



cls3 = xgb.XGBClassifier(**gscv.best_params_)



# Fit classifier to new split

cls3.fit(train_fold_X3, train_fold_Y.values.ravel())



# Predict probabilities on valid_fold_X

split_pred_proba = cls3.predict_proba(valid_fold_X3)



# Compute ROC AUC score

split_score = roc_auc_score(valid_fold_Y, split_pred_proba[:,1])

print("ROC AUC is {:.4f}".format(split_score))



# Draw ROC curve

fpr, tpr, thr = roc_curve(valid_fold_Y, split_pred_proba[:,1])

plt.plot(fpr, tpr, linestyle="-");

plt.title("ROC Curve");

plt.xlabel("False Positive Rate");

plt.ylabel("True Positive Rate");



# Compute optimal operating point and threshold

ind = np.argmin(list(map(lambda x, y: x**2 + (y - 1.0)**2, fpr,tpr)))



# Optimal threshold for our classifier cls3

opt_thr = thr[ind]



print("Optimal operating point on ROC curve (fpr,tpr) ({:.4f}, {:.4f})"

                                              " with threshold {:.4f}".

                                   format(fpr[ind], tpr[ind], opt_thr))



y_pred = [1 if x > opt_thr else 0 for x in split_pred_proba[:,1]]

acc = accuracy_score(valid_fold_Y, y_pred)

print("Accuracy is {:.4f}".format(acc))
test_X.drop("Emb_1", axis=1, inplace=True)

test_X.drop("Emb_2", axis=1, inplace=True)



test_predict_proba = cls3.predict_proba(test_X.loc[:,

                            test_X.columns != "PassengerId"])



# Optimal threshold we found previously is opt_thr

test_pred = [1 if x > opt_thr else 0 for x in test_predict_proba[:,1]]



output = pd.DataFrame({"PassengerId": test_X["PassengerId"],

                                   "Survived": test_pred})



#output.to_csv("my_submission.csv", index=False)



output.head(20)