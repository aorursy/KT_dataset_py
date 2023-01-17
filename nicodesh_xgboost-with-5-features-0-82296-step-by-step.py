# Standard library

import pickle

import warnings



# Data analysis

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Modelization

import xgboost as xgb

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
# Files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
training_set = pd.read_csv("/kaggle/input/titanic/train.csv")

test_set = pd.read_csv("/kaggle/input/titanic/test.csv")
sns.set(style="whitegrid")

pd.options.display.max_rows = 100

pd.options.display.max_columns = 100

warnings.filterwarnings('ignore')
def cat_analysis(df, feature, x, y, figsize=(8,5), rotation="45", palette=None, order=None):

    

    data = (pd.DataFrame(df[feature].value_counts())

                .reset_index()

                .rename(columns={'index': x, feature: y}))

    

    fig, ax = plt.subplots(figsize=figsize)

    ax = sns.barplot(x=x, y=y, data=data, palette=palette, order=order);

    plt.xticks(rotation=rotation)

    plt.title(f"Distribution of {feature}")

    plt.show()

    

def dummify(data, feature):

    

    # Get dummy variables

    temp_df = pd.get_dummies(data[feature])

    

    # Add prefix to prevent duplicated feature names

    temp_df = temp_df.add_prefix(feature + "_")

    

    # Concatenante the new features with the main dataframe

    data = pd.concat([data, temp_df], axis=1)

    

    # Drop the original feature

    data.drop(feature, axis=1, inplace=True)

    

    # Return the new dataframe

    return data



def prepare_data(data_original):

    """ Prepare the data for the Titanic Competition. """

    

    data = data_original.copy()

    

    ################ Name => Boy / surname ######################

    

    # From the name, extract "is a boy?" and "surname"

    data["boy"] = data["Name"].apply(lambda x: 1 if ("Master." in x.split(" ")[1:-1]) else 0)

    data["surname"] = data["Name"].apply(lambda x: x.split(",")[0])

    data.drop("Name", axis=1, inplace=True)

    

    # Sex

    data = dummify(data, "Sex")

    # Since "Sex" it's a binary variable, we don't need to keep a feature for male AND for female

    data.drop("Sex_male", axis=1, inplace=True) # Just one is enough

    

    ################## Ticket ##########################

    data["wcg_ticket"] = data["Ticket"].copy()

    mask = (data["Sex_female"] == 0) & (data["boy"] == 0)

    data.loc[mask, "wcg_ticket"] = "no_group"

    data.drop("Ticket", axis=1, inplace=True)

    

    ################## Cabin ######################

    data.drop("Cabin", axis=1, inplace=True) # And finally, drop the original column

    

    ################ Embarked ###################

    data.drop("Embarked", axis=1, inplace=True)

    

    ############### Family Size ################

    data["family_size"] = 1 + data["SibSp"] + data["Parch"]

    data.drop(["SibSp", "Parch"], axis=1, inplace=True)

    

    ############## PassengerId #################

    data.drop("PassengerId", axis=1, inplace=True)

    

    ############## Fare ########################

    data.drop("Fare", axis=1, inplace=True)

    

    ############## Age #########################

    data.drop("Age", axis=1, inplace=True)

    

    ############## PClass ######################

    data.drop("Pclass", axis=1, inplace=True)

    

    ############# Woman Child Group ############

    

    # Copy family name to create a "woman child group" based on the surname

    data["wcg_surname"] = data["surname"].copy()



    # Remove men from groups

    mask = (data["Sex_female"] == 0) & (data["boy"] == 0)

    data.loc[mask, "wcg_surname"] = "no_group"

    

    data.drop("surname", axis=1, inplace=True)

    

    return data
def my_xgb(data, target, params):

    

    # Create X and y

    X = data.drop(target, axis=1)

    y = data[target]

    

    # Scale X

    scaler = StandardScaler()

    scaler.fit(X)

    X_scaled = scaler.transform(X)

    

    # Create a XGBoost classifier (scikit-learn API wrapper)

    xgb_clf = xgb.XGBClassifier()

    

    # Perform a gridsearch with sklearn

    kf = KFold(n_splits=10, random_state=42, shuffle=True)

    gridsearch = GridSearchCV(xgb_clf, param_grid=params, scoring="accuracy", cv=kf, return_train_score=True)

    gridsearch.fit(X_scaled, y)

    

    # Return the gridsearch results plus the scaler

    return gridsearch, scaler
print(training_set.shape)

display(training_set.head())
print(test_set.shape)

display(test_set.head())
nanbyfeature = pd.DataFrame(training_set.isna().sum()).sort_values(by=0, ascending=False)

nanbyfeature["percent"] = np.round(nanbyfeature[0] / len(training_set) * 100,2)

nanbyfeature
survived_passengers = training_set["Survived"].sum() / len(training_set)

died_passengers = 1 - survived_passengers

print(f"Survived passengers: {survived_passengers:.2%}")

print(f"Died passengers: {died_passengers:.2%}")
fig, ax = plt.subplots(figsize=(15,5))

ax = sns.barplot(x="Sex", y="Survived", data=training_set);

plt.xticks(rotation=0)

plt.title("Percentage of survived by gender")

plt.show()
fig, ax = plt.subplots(figsize=(15,5))

ax = sns.barplot(x="Sex", y="Survived", data=training_set, hue="Pclass");

plt.xticks(rotation=0)

plt.title("Percentage of survived by gender")

plt.show()
training_set["family_size"] = 1 + training_set["SibSp"] + training_set["Parch"]



# Distribution

cat_analysis(training_set, "family_size", "family_size", "total_people", rotation=90, figsize=(16,5))



# Survived by family_size

fig, ax = plt.subplots(figsize=(15,5))

ax = sns.barplot(x="family_size", y="Survived", data=training_set);

plt.xticks(rotation=0)

plt.title("Percentage of survived by family_size")

plt.show()



# Let's drop it for now

training_set.drop(["family_size"], axis=1, inplace=True)
def name_title(title):

    """ For each passenger, the function parses the name, from the second word to the second-last one.

    The title is still among those positions.

    According to the value, the function assigns a group.

    

    """

    

    for word in title.split(" ")[1:-1]:

        if (word in ["Mme.", "Ms.", "Mrs."]):

            return "woman"



        elif (word in ["Mr."]):

            return "man"



        elif (word in ["Master."]):

            return "boy"



        elif (word in ["Miss.", "Mlle."]):

            return "miss"



        elif (word in ["Capt.", "Col.", "Major.", "Rev.", "Dr."]):

            return "army"



        elif (word in ["Jonkheer.", "Don.", "Sir.", "Countess.", "Dona.", "Lady."]):

            return "gentry"

    

    else:

        return "other"



training_set["title"] = training_set["Name"].apply(name_title)



# Distribution

cat_analysis(training_set, "title", "title", "total_people", rotation=90, figsize=(16,5))



# Survived by family_size

fig, ax = plt.subplots(figsize=(15,5))

ax = sns.barplot(x="title", y="Survived", data=training_set);

plt.xticks(rotation=0)

plt.title("Percentage of survived by title")

plt.show()



# Remove temporary feature

training_set.drop("title", axis=1, inplace=True)
train_prep = prepare_data(training_set)

test_prep = prepare_data(test_set)
# Create a new dataframe

data_wcg_name = train_prep.copy()[["Sex_female", "boy", "wcg_surname", "Survived"]]

data_wcg_name["wcg_surname_size"] = 1



# Remove passengers labelled by "no group"

mask = data_wcg_name["wcg_surname"] != "no_group"

data_wcg_name = data_wcg_name[mask]



# Group by "woman child group" and count the number of members and the number of survivers for each group

columns = {'Survived': 'survived_number'}

data_wcg_name = data_wcg_name.groupby("wcg_surname").agg({'Survived':'sum', 'wcg_surname_size':'count'}).reset_index().rename(columns=columns)



# Create new feature <All died>

data_wcg_name["wcg_name_all_died"] = data_wcg_name["survived_number"].apply(lambda x: 1 if x == 0 else 0)



# Create feature <All survived>

data_wcg_name["wcg_name_all_survived"] = data_wcg_name["survived_number"] == data_wcg_name["wcg_surname_size"]

data_wcg_name["wcg_name_all_survived"] = data_wcg_name["wcg_name_all_survived"] * 1



# Import Test dataset

wcg_name_test = test_prep.copy()[["wcg_surname"]]

wcg_name_test["wcg_surname_size"] = 1

wcg_name_test["survived_number"] = 0

wcg_name_test["wcg_name_all_died"] = 0

wcg_name_test["wcg_name_all_survived"] = 0

mask = wcg_name_test["wcg_surname"] != "no_group"

wcg_name_test = wcg_name_test[mask]



# Merge train and test

data_wcg_name = pd.concat([data_wcg_name, wcg_name_test])

data_wcg_name = data_wcg_name.groupby("wcg_surname").sum().reset_index()



# Keep "woman child groups" composed by more than one people

mask = data_wcg_name["wcg_surname_size"] > 1

data_wcg_name = data_wcg_name[mask]



# Remove useless columns

data_wcg_name.drop(["survived_number", "wcg_surname_size"], axis=1, inplace=True)



# Merge with the training dataset

train_prep = train_prep.merge(data_wcg_name, how="left", on="wcg_surname")
display(data_wcg_name.head())

print(f"Total groups: {data_wcg_name.shape[0]}")

print(f"All died: {data_wcg_name['wcg_name_all_died'].sum()}")

print(f"All survived: {data_wcg_name['wcg_name_all_survived'].sum()}")
# Create a new dataframe

data_wcg_ticket = train_prep.copy()[["Sex_female", "boy", "wcg_ticket", "Survived"]]

data_wcg_ticket["wcg_ticket_size"] = 1



# Remove passengers labelled by "no group"

mask = data_wcg_ticket["wcg_ticket"] != "no_group"

data_wcg_ticket = data_wcg_ticket[mask]



# Group by "woman child group" and count the number of members and the number of survivers for each group

columns = {'Survived': 'survived_number'}

data_wcg_ticket = data_wcg_ticket.groupby("wcg_ticket").agg({'Survived':'sum', 'wcg_ticket_size':'count'}).reset_index().rename(columns=columns)



# Create new feature <All died>

data_wcg_ticket["wcg_ticket_all_died"] = data_wcg_ticket["survived_number"].apply(lambda x: 1 if x == 0 else 0)



# Create feature <All survived>

data_wcg_ticket["wcg_ticket_all_survived"] = data_wcg_ticket["survived_number"] == data_wcg_ticket["wcg_ticket_size"]

data_wcg_ticket["wcg_ticket_all_survived"] = data_wcg_ticket["wcg_ticket_all_survived"] * 1



# Import Test dataset

wcg_ticket_test = test_prep.copy()[["wcg_ticket"]]

wcg_ticket_test["wcg_ticket_size"] = 1

wcg_ticket_test["survived_number"] = 0

wcg_ticket_test["wcg_ticket_all_died"] = 0

wcg_ticket_test["wcg_ticket_all_survived"] = 0

mask = wcg_ticket_test["wcg_ticket"] != "no_group"

wcg_ticket_test = wcg_ticket_test[mask]



# Merge train and test

data_wcg_ticket = pd.concat([data_wcg_ticket, wcg_ticket_test])

data_wcg_ticket = data_wcg_ticket.groupby("wcg_ticket").sum().reset_index()



# Keep "woman child groups" composed by more than one people

mask = data_wcg_ticket["wcg_ticket_size"] > 1

data_wcg_ticket = data_wcg_ticket[mask]



# Remove useless columns

data_wcg_ticket.drop(["survived_number", "wcg_ticket_size"], axis=1, inplace=True)



# Merge with the training dataset

train_prep = train_prep.merge(data_wcg_ticket, how="left", on="wcg_ticket")
display(data_wcg_ticket.head())

print(f"Total groups: {data_wcg_ticket.shape[0]}")

print(f"All died: {data_wcg_ticket['wcg_ticket_all_died'].sum()}")

print(f"All survived: {data_wcg_ticket['wcg_ticket_all_survived'].sum()}")
for i, row in train_prep.iterrows():

    

    # All died processing

    if ((train_prep.loc[i, "wcg_name_all_died"] == 1) or (train_prep.loc[i, "wcg_ticket_all_died"] == 1)):

        train_prep.loc[i, "all_died"] = 1

    elif ((train_prep.loc[i, "wcg_name_all_died"] == 0) or (train_prep.loc[i, "wcg_ticket_all_died"] == 0)):

        train_prep.loc[i, "all_died"] = 0

    else:

        train_prep.loc[i, "all_died"] = np.nan

          

    # All survived processing

    if ((train_prep.loc[i, "wcg_name_all_survived"] == 1) or (train_prep.loc[i, "wcg_ticket_all_survived"] == 1)):

        train_prep.loc[i, "all_survived"] = 1

    elif ((train_prep.loc[i, "wcg_name_all_survived"] == 0) or (train_prep.loc[i, "wcg_ticket_all_survived"] == 0)):

        train_prep.loc[i, "all_survived"] = 0

    else:

        train_prep.loc[i, "all_survived"] = np.nan
# Copy data

wcgdf = train_prep.copy()

wcgdf["class"] = training_set["Pclass"]





#  Only passengers with "all survived" information

mask1 = wcgdf["all_survived"] == 1

mask2 = wcgdf["all_survived"] == 0

wcgdf1 = wcgdf[mask1 | mask2]

fig, ax = plt.subplots(figsize=(15,5))

ax = sns.barplot(x="class", y="all_survived", data=wcgdf);

plt.xticks(rotation=0)

plt.title("Percentage of all survived by class")

plt.show()



mask3 = wcgdf["all_died"] == 1

mask4 = wcgdf["all_died"] == 0

wcgdf2 = wcgdf[mask3 | mask4]

#  Only passengers with "all died" information

fig, ax = plt.subplots(figsize=(15,5))

ax = sns.barplot(x="class", y="all_died", data=wcgdf);

plt.xticks(rotation=0)

plt.title("Percentage of all died by class")

plt.show()
train_prep.drop(["wcg_surname", "wcg_ticket", "wcg_name_all_died", "wcg_name_all_survived", "wcg_ticket_all_died", "wcg_ticket_all_survived"], axis=1, inplace=True)
print(train_prep.shape)

display(train_prep.head())
nanbyfeature2 = pd.DataFrame(train_prep.isna().sum()).sort_values(by=0, ascending=False)

nanbyfeature2["percent"] = np.round(nanbyfeature2[0] / len(train_prep) * 100,2)

nanbyfeature2
params = {

 'learning_rate': [0.01, 0.05, 0.1],

 'subsample': [1],

 'colsample_bylevel': [1],

 'colsample_bynode': [1],

 'colsample_bytree': [0.5],

 'gamma': [0, 1, 2],

 'max_delta_step': [0],

 'max_depth': [2, 3],

 'min_child_weight': [1.6], # Owen Zhang's rule of thumb: mcw = 3/sqrt(event_rate) -> 1.6 (Thanks Tae Hyon Whang)

 'n_estimators': [100],

 'random_state': [42],

 'scale_pos_weight': [1],

 'seed': [42],

 'n_jobs': [-1],

 'reg_lambda': [1, 2, 4, 16]

}
gridsearch, scaler = my_xgb(train_prep, "Survived", params)
results = pd.DataFrame(gridsearch.cv_results_).sort_values(by="rank_test_score")

fig, ax = plt.subplots(figsize=(16,5))

plt.plot(np.arange(len(results)), results["mean_train_score"], label="Train")

plt.plot(np.arange(len(results)), results["mean_test_score"], label="Test")

plt.legend()

plt.show()
display(gridsearch.best_params_)
feature_importances = gridsearch.best_estimator_.feature_importances_

feature_names = train_prep.drop(["Survived"], axis=1).columns



temp_df1 = {

    'feature_name': feature_names,

    'feature_importance': feature_importances

}



temp_df1 = pd.DataFrame(temp_df1).sort_values(by="feature_importance", ascending=False)



display(temp_df1.reset_index(drop=True))
X_training_error = train_prep.copy()

y_training_error_true = X_training_error.pop("Survived")



X_training_error_scaled = scaler.transform(X_training_error) 

y_training_error_pred = gridsearch.predict(X_training_error_scaled)



training_set["true"] = y_training_error_true

training_set["pred"] = y_training_error_pred



true_pos = training_set["true"] == 1

true_false = training_set["true"] == 0

pred_pos = training_set["pred"] == 1

pred_false = training_set["pred"] == 0
mask_results = training_set["Sex"] != -1

true_positive = len(training_set[true_pos & pred_pos & mask_results])

false_positive = len(training_set[true_false & pred_pos & mask_results])

true_negative = len(training_set[true_false & pred_false & mask_results])

false_negative = len(training_set[true_pos & pred_false & mask_results])



print(f"Accuracy: {(training_set['true'] == training_set['pred']).sum() / len(training_set):.2%}")

print(f"True positives: {true_positive} ({true_positive / (true_positive + false_negative):.0%})")

print(f"False positives: {false_positive}")

print(f"True negatives: {true_negative} ({true_negative / (true_negative + false_positive):.0%})")

print(f"False negatives: {false_negative}")
mask_results = training_set["Sex"] == "female"

true_positive = len(training_set[true_pos & pred_pos & mask_results])

false_positive = len(training_set[true_false & pred_pos & mask_results])

true_negative = len(training_set[true_false & pred_false & mask_results])

false_negative = len(training_set[true_pos & pred_false & mask_results])



print(f"Accuracy: {(training_set['true'] == training_set['pred']).sum() / len(training_set):.2%}")

print(f"True positives: {true_positive} ({true_positive / (true_positive + false_negative):.0%})")

print(f"False positives: {false_positive}")

print(f"True negatives: {true_negative} ({true_negative / (true_negative + false_positive):.0%})")

print(f"False negatives: {false_negative}")
mask_results = training_set["Sex"] == "male"

true_positive = len(training_set[true_pos & pred_pos & mask_results])

false_positive = len(training_set[true_false & pred_pos & mask_results])

true_negative = len(training_set[true_false & pred_false & mask_results])

false_negative = len(training_set[true_pos & pred_false & mask_results])



print(f"Accuracy: {(training_set['true'] == training_set['pred']).sum() / len(training_set):.2%}")

print(f"True positives: {true_positive} ({true_positive / (true_positive + false_negative):.0%})")

print(f"False positives: {false_positive}")

print(f"True negatives: {true_negative} ({true_negative / (true_negative + false_positive):.0%})")

print(f"False negatives: {false_negative}")
mask_results = training_set["Sex"] == "male"

mask_results2 = training_set["Name"].str.contains("Master.")

true_positive = len(training_set[true_pos & pred_pos & mask_results & ~mask_results2])

false_positive = len(training_set[true_false & pred_pos & mask_results & ~mask_results2])

true_negative = len(training_set[true_false & pred_false & mask_results & ~mask_results2])

false_negative = len(training_set[true_pos & pred_false & mask_results & ~mask_results2])



print(f"Accuracy: {(training_set['true'] == training_set['pred']).sum() / len(training_set):.2%}")

print(f"True positives: {true_positive} ({true_positive / (true_positive + false_negative):.0%})")

print(f"False positives: {false_positive}")

print(f"True negatives: {true_negative} ({true_negative / (true_negative + false_positive):.0%})")

print(f"False negatives: {false_negative}")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data_prepared = prepare_data(test_data)



# Merge "name wcg"

test_data_prepared = test_data_prepared.merge(data_wcg_name, how="left", on="wcg_surname")

test_data_prepared["wcg_name_all_died"]

test_data_prepared["wcg_name_all_survived"]



# Merge "ticket wcg"

test_data_prepared = test_data_prepared.merge(data_wcg_ticket, how="left", on="wcg_ticket")

test_data_prepared["wcg_ticket_all_died"]

test_data_prepared["wcg_ticket_all_survived"]



for i, row in test_data_prepared.iterrows():

    

    # All died processing

    if ((test_data_prepared.loc[i, "wcg_name_all_died"] == 1) or (test_data_prepared.loc[i, "wcg_ticket_all_died"] == 1)):

        test_data_prepared.loc[i, "all_died"] = 1

    elif ((test_data_prepared.loc[i, "wcg_name_all_died"] == 0) or (test_data_prepared.loc[i, "wcg_ticket_all_died"] == 0)):

        test_data_prepared.loc[i, "all_died"] = 0

    else:

        test_data_prepared.loc[i, "all_died"] = np.nan

          

    # All survived processing

    if ((test_data_prepared.loc[i, "wcg_name_all_survived"] == 1) or (test_data_prepared.loc[i, "wcg_ticket_all_survived"] == 1)):

        test_data_prepared.loc[i, "all_survived"] = 1

    elif ((test_data_prepared.loc[i, "wcg_name_all_survived"] == 0) or (test_data_prepared.loc[i, "wcg_ticket_all_survived"] == 0)):

        test_data_prepared.loc[i, "all_survived"] = 0

    else:

        test_data_prepared.loc[i, "all_survived"] = np.nan



test_data_prepared.drop(["wcg_surname", "wcg_ticket", "wcg_name_all_died", "wcg_name_all_survived", "wcg_ticket_all_died", "wcg_ticket_all_survived"], axis=1, inplace=True)
test_data_prepared_scaled = scaler.transform(test_data_prepared)

y_test = gridsearch.predict(test_data_prepared_scaled)
submission = pd.DataFrame(y_test.copy())

submission['PassengerId'] = pd.read_csv("/kaggle/input/titanic/test.csv")['PassengerId'].copy()

submission.rename(columns={0:'Survived'}, inplace=True)

submission = submission[['PassengerId', 'Survived']]

submission.to_csv("submission.csv", index=False)