import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.utils import shuffle

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, log_loss



import lightgbm as lgb
%matplotlib inline



sns.set(color_codes="whitegrid")
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
all_data = pd.concat([train_data, test_data], axis=0, sort=False, ignore_index=True)



all_data.head()
sns.countplot(data=all_data, x="Pclass", hue="Survived")



pclass_maping = {1: 0, 2: 1, 3: 2}

all_data["pclass_code"] = all_data["Pclass"].map(pclass_maping)
sns.countplot(data=all_data, x="Sex", hue="Survived")



sex_maping = {"male": 0, "female": 1}

all_data["sex_code"] = all_data["Sex"].map(sex_maping)
all_data["family_size"] = all_data["SibSp"] + all_data["Parch"]



sns.countplot(data=all_data, x="family_size", hue="Survived")



def family_size_encode(x):

    if x == 0:

        return 0

    elif 0< x <= 3:

        return 1

    elif x >= 4:

        return 2

all_data["family_size_code"] = all_data["family_size"].apply(family_size_encode)
all_data["cabin_initials"] = all_data["Cabin"].apply(lambda x: "U" if pd.isnull(x) else x[0])

sns.countplot(data=all_data, x="cabin_initials", hue="Survived")



cabin_initials_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "T": 7, "U": 8}

all_data["cabin_initials_code"] = all_data["cabin_initials"].map(cabin_initials_map)
all_data["Embarked"] = all_data["Embarked"].fillna(all_data["Embarked"].mode()[0])

sns.countplot(data=all_data, x="Embarked", hue="Survived")



embarked_map = {"S": 0, "C": 1, "Q": 2}

all_data["embarked_code"] = all_data["Embarked"].map(embarked_map)
all_data["title"] = all_data["Name"].apply(lambda x: x.split(",")[1].split(".")[0].replace(" ", ""))

title_map = {"Mlle": "Miss", "Major": "Mr", "Col": "Mr", "Sir": "Mr", "Don": "Mr", "Mme": "Miss", "Jonkheer": "Mr",

             "Lady": "Mrs", "Capt": "Mr", "theCountess": "Mrs", "Ms": "Miss", "Dona": "Mrs", "Miss": "Miss", "Mr": "Mr",

             "Mrs": "Mrs", "Master": "Master", "Dr": "Dr", "Rev": "Rev"}

all_data["title"] = all_data["title"].map(title_map)

sns.countplot(data=all_data, x="title", hue="Survived")





all_data["title_code"] = pd.factorize(all_data["title"])[0]
age_median = all_data.groupby(["title"])["Age"].median()

title_all = list(age_median.index)

for i in range(0, len(title_all)):

    all_data.loc[(all_data["Age"].isnull()) & (all_data["title"] == title_all[i]), "Age"] = age_median[i]

    

all_data["age_code"] = pd.qcut(all_data["Age"], 4, labels=[i for i in range(0, 4)])

sns.countplot(data=all_data, x="age_code", hue="Survived")
all_data["Fare"] = all_data["Fare"].fillna(all_data["Fare"].median())





all_data["fare_code"] = pd.qcut(all_data["Fare"], 8, labels=[i for i in range(0, 8)])

sns.countplot(data=all_data, x="fare_code", hue="Survived")
all_data["last_name"] = all_data["Name"].apply(lambda x: x.split(",")[0])



all_data["family_survival_code"] = 2



for grp, grp_data in all_data.groupby(["last_name", "Fare"]):

    if len(grp_data) > 1:

        for index, row, in grp_data.iterrows():

            grp_data_drop = grp_data.drop(index)

            smax = grp_data_drop["Survived"].max()

            smin = grp_data_drop["Survived"].min()

            if smax == 1.0:

                all_data.loc[all_data["PassengerId"] == row["PassengerId"], "family_survival_code"] = 1

            elif smin == 0.0:

                all_data.loc[all_data["PassengerId"] == row["PassengerId"], "family_survival_code"] = 0



sns.countplot(data=all_data, x="family_survival_code", hue="Survived")
for grp, grp_data in all_data.groupby(["Ticket"]):

    if len(grp_data) > 1:

        for index, row, in grp_data.iterrows():

            if row["family_survival_code"] != 1:

                grp_data_drop = grp_data.drop(index)

                smax = grp_data_drop["Survived"].max()

                smin = grp_data_drop["Survived"].min()

                if smax == 1.0:

                    all_data.loc[all_data["PassengerId"] == row["PassengerId"], "family_survival_code"] = 1

                elif smin == 0.0:

                    all_data.loc[all_data["PassengerId"] == row["PassengerId"], "family_survival_code"] = 0

                    

sns.countplot(data=all_data, x="family_survival_code", hue="Survived")
feature = all_data.drop(["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked", 

                         "family_size", "cabin_initials", "title", "last_name"], axis=1)

feature.head()
x = feature.loc[feature["PassengerId"].isin(train_data["PassengerId"])].iloc[:, 2:].astype("int32")

y = feature.loc[feature["PassengerId"].isin(train_data["PassengerId"])].iloc[:, 1: 2].astype("int32")



x_test = feature.loc[feature["PassengerId"].isin(test_data["PassengerId"])].iloc[:, 2:].astype("int32")



print(x.head())

print(y.head())

print(x_test.head())
class LgbModel(object):



    def __init__(self):



        self.categorical_cols = list(x.columns)



        self.lgb_params = {

                        "task": "train",

                        "objective": "binary",

                        "metric": ["binary_logloss", "binary_error"],

                        "boosting": "gbdt",

                        "num_iterations": 1000,

                        "learning": 0.1,

                        "num_leaves": 80,

                        "max_depth": 7,

                        "min_data_in_leaf": 60,

                        "min_sum_hessian_in_leaf": 0.001,

                        "feature_fraction": 1.0,

                        "bagging_fraction": 1.0,

                        "max_bin": 255,

                        "lambda l1": 0.01,

                        "lambda l2": 0.01,

                        "is_unbalance": "false"}

        

        self.k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

        

        self.models = []

    

    def train(self, x, y):



        self.models.clear()

    

        count = 0



        val_loss, val_accuracy = [], []

        

        for train_index, val_index in self.k_fold.split(x, y):



            print("{} fold:".format(count + 1))

            count += 1

            



            lgb_train = lgb.Dataset(x.iloc[train_index], label=y.iloc[train_index], categorical_feature=self.categorical_cols)

            lgb_val = lgb.Dataset(x.iloc[val_index], label=y.iloc[val_index], reference=lgb_train, 

                                  categorical_feature=self.categorical_cols)

            



            lgb_model = lgb.train(self.lgb_params, lgb_train, valid_sets=lgb_val, 

                                  categorical_feature=self.categorical_cols, early_stopping_rounds=50)

            



            self.models.append(lgb_model)

            



            val_predict = lgb_model.predict(x.iloc[val_index], num_iteration=lgb_model.best_iteration)

            val_loss.append(log_loss(y.iloc[val_index], val_predict))

            

            val_predict[val_predict >= 0.5]=1

            val_predict[val_predict < 0.5]=0

            val_accuracy.append(accuracy_score(y.iloc[val_index], val_predict))

            

        val_loss, val_accuracy = np.array(val_loss), np.array(val_accuracy)

        for i in range(0, len(val_loss)):

            print("Model{}->loss:{}. accuracy:{}.".format(i + 1, val_loss[i], val_accuracy[i]))

        print("Average loss:{}. average accuracy:{}.".format(np.average(val_loss), np.average(val_accuracy)))

        

    def predict(self, x):



        if len(self.models) == 0:

            return None

        

        res = []

        for m in self.models:

            res.append(m.predict(x, num_iteration=m.best_iteration))



        return np.average(np.array(res), axis=0)
model_lgb = LgbModel()



model_lgb.train(x, y)
y_test = model_lgb.predict(x_test)

y_test[y_test >= 0.5] = 1

y_test[y_test < 0.5] = 0



pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": y_test.reshape([-1, ])}, 

             dtype="int32").to_csv("submission.csv", index=0)