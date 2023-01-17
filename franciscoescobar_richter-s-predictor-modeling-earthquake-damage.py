%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats.mstats import winsorize

from sklearn.metrics import f1_score as score #Scoring metric for the competition

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeClassifier

import time

import warnings



warnings.filterwarnings("ignore")

pd.set_option("display.max_columns",None)

pd.set_option("display.max_rows",None)

sns.set_palette("Set2")

sns.set_style("ticks")
train_labels = pd.read_csv("../input/train_labels.csv")

train_values = pd.read_csv("../input/train_values.csv")

test_values = pd.read_csv("../input/test_values.csv")
print("# Train Values: {}".format(train_values.shape))

print("# Train Labels: {}".format(train_labels.shape))

print("# Test Values: {}".format(test_values.shape))
train_labels.head()
train_values.head()
test_values.head()
train_values.isnull().sum() * 100 / len(train_values)
train_labels.isnull().sum() * 100 / len(train_values)
test_values.isnull().sum() * 100 / len(train_values)
train_values.dtypes
train_labels.dtypes
test_values.dtypes
sns.countplot(x="damage_grade", data=train_labels)

plt.title("Damage Grade Distribution")

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(train_values["count_floors_pre_eq"], hue=train_labels["damage_grade"])

plt.ylabel("Frequency")

plt.xlabel("# of Floors Before Earthquake")

plt.xticks(rotation=90)

plt.title("# of Floors Before Earthquake Histograms")

plt.legend(["damage_grade = 1","damage_grade = 2","damage_grade = 3"])

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x=train_values["age"],hue=train_labels["damage_grade"])

plt.ylabel("Frequency")

plt.xlabel("Building's Age")

plt.xticks(rotation=90)

plt.title("Age Histograms")

plt.legend(["damage_grade = 1","damage_grade = 2","damage_grade = 3"])

plt.show()
plt.figure(figsize=(18,9))

sns.countplot(x=train_values["area_percentage"],hue=train_labels["damage_grade"])

plt.ylabel("Frequency")

plt.xlabel("Area Percentage")

plt.xticks(rotation=90)

plt.title("Area Percentage Histograms")

plt.legend(["damage_grade = 1","damage_grade = 2","damage_grade = 3"])

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x=train_values["height_percentage"],hue=train_labels["damage_grade"])

plt.ylabel("Frequency")

plt.xlabel("Height Percentage")

plt.xticks(rotation=90)

plt.title("Height Percentage Histograms")

plt.legend(["damage_grade = 1","damage_grade = 2","damage_grade = 3"])

plt.show()
sns.countplot(x = train_values["land_surface_condition"], hue = train_labels["damage_grade"])

plt.title("Land Surface Condition Distribution")

plt.show()
sns.countplot(x = train_values["foundation_type"], hue = train_labels["damage_grade"])

plt.title("Foundation Type Distribution")

plt.show()
sns.countplot(x = train_values["roof_type"], hue = train_labels["damage_grade"])

plt.title("Roof Type Distribution")

plt.show()
sns.countplot(x = train_values["ground_floor_type"], hue = train_labels["damage_grade"])

plt.title("Ground Floor Type Distribution")

plt.show()
sns.countplot(x = train_values["other_floor_type"], hue = train_labels["damage_grade"])

plt.title("Other Floor Type Distribution")

plt.show()
sns.countplot(x = train_values["position"], hue = train_labels["damage_grade"])

plt.title("Position Distribution")

plt.show()
sns.countplot(x = train_values["plan_configuration"], hue = train_labels["damage_grade"])

plt.title("Plan Configuration Distribution")

plt.show()
sns.countplot(x = train_values["has_superstructure_adobe_mud"], hue = train_labels["damage_grade"])

plt.title("Has Superstructure Adobe/Mud Distribution")

plt.show()
sns.countplot(x = train_values["has_superstructure_mud_mortar_stone"], hue = train_labels["damage_grade"])

plt.title("Has Superstructure Mortar/Stone Distribution")

plt.show()
sns.countplot(x = train_values["has_superstructure_mud_mortar_brick"], hue = train_labels["damage_grade"])

plt.title("Has Superstructure Mortar/Brick Distribution")

plt.show()
sns.countplot(x = train_values["has_superstructure_cement_mortar_brick"], hue = train_labels["damage_grade"])

plt.title("Has Superstructure Cement/Mortar/Brick Distribution")

plt.show()
sns.countplot(x = train_values["has_superstructure_timber"], hue = train_labels["damage_grade"])

plt.title("Has Superstructure Timber Distribution")

plt.show()
sns.countplot(x = train_values["has_superstructure_bamboo"], hue = train_labels["damage_grade"])

plt.title("Has Superstructure Bamboo Distribution")

plt.show()
sns.countplot(x = train_values["has_superstructure_rc_non_engineered"], hue = train_labels["damage_grade"])

plt.title("Has Superstructure Non-engineered Reinforced Concrete Distribution")

plt.show()
sns.countplot(x = train_values["has_superstructure_rc_engineered"], hue = train_labels["damage_grade"])

plt.title("Has Superstructure Engineered Reinforced Concrete Distribution")

plt.show()
sns.countplot(x = train_values["has_superstructure_other"], hue = train_labels["damage_grade"])

plt.title("Has Superstructure Other Distribution")

plt.show()
sns.countplot(x = train_values["legal_ownership_status"], hue = train_labels["damage_grade"])

plt.title("Legal Ownership Status Distribution")

plt.show()
plt.figure(figsize=(18,9))

sns.countplot(x=train_values["count_families"],hue=train_labels["damage_grade"])

plt.ylabel("Frequency")

plt.xlabel("# of Families")

plt.xticks(rotation=90)

plt.title("# of Families Histograms")

plt.legend(["damage_grade = 1","damage_grade = 2","damage_grade = 3"])

plt.show()
sns.countplot(x = train_values["has_secondary_use"], hue = train_labels["damage_grade"])

plt.title("Has Secondary Use Distribution")

plt.show()
sns.countplot(x = train_values["has_secondary_use_agriculture"], hue = train_labels["damage_grade"])

plt.title("Has Secondary Use Agricultural Distribution")

plt.show()
sns.countplot(x = train_values["has_secondary_use_hotel"], hue = train_labels["damage_grade"])

plt.title("Has Secondary Use Hotel Distribution")

plt.show()
sns.countplot(x = train_values["has_secondary_use_rental"], hue = train_labels["damage_grade"])

plt.title("Has Secondary Use Rental Distribution")

plt.show()
sns.countplot(x = train_values["has_secondary_use_institution"], hue = train_labels["damage_grade"])

plt.title("Has Secondary Use Institution Distribution")

plt.show()
sns.countplot(x = train_values["has_secondary_use_school"], hue = train_labels["damage_grade"])

plt.title("Has Secondary Use School Distribution")

plt.show()
sns.countplot(x = train_values["has_secondary_use_industry"], hue = train_labels["damage_grade"])

plt.title("Has Secondary Use Industry Distribution")

plt.show()
sns.countplot(x = train_values["has_secondary_use_health_post"], hue = train_labels["damage_grade"])

plt.title("Has Secondary Use Health Post Distribution")

plt.show()
sns.countplot(x = train_values["has_secondary_use_gov_office"], hue = train_labels["damage_grade"])

plt.title("Has Secondary Use Government Office Distribution")

plt.show()
sns.countplot(x = train_values["has_secondary_use_use_police"], hue = train_labels["damage_grade"])

plt.title("Has Secondary Use Police Distribution")

plt.show()
sns.countplot(x = train_values["has_secondary_use_other"], hue = train_labels["damage_grade"])

plt.title("Has Secondary Use Other")

plt.show()
#Data source

name = "age"

data = train_values[name]



#Before winsorization

sns.boxplot(data)

plt.title("{} Before Winsorization".format(name))

plt.show()



#Winsorization

winsorized_data = winsorize(data,(0, 0.05))



#After winsorization

sns.boxplot(winsorized_data)

plt.title("{} After Winsorization".format(name))

plt.show()



#Replace data in dataset

train_values[name] = winsorized_data
#Data source

name = "area_percentage"

data = train_values[name]



#Before winsorization

sns.boxplot(data)

plt.title("{} Before Winsorization".format(name))

plt.show()



#Winsorization

winsorized_data = winsorize(data,(0, 0.055))



#After winsorization

sns.boxplot(winsorized_data)

plt.title("{} After Winsorization".format(name))

plt.show()



#Replace data in dataset

train_values[name] = winsorized_data
#Data source

name = "height_percentage"

data = train_values[name]



#Before winsorization

sns.boxplot(data)

plt.title("{} Before Winsorization".format(name))

plt.show()



#Winsorization

winsorized_data = winsorize(data,(0, 0.04))



#After winsorization

sns.boxplot(winsorized_data)

plt.title("{} After Winsorization".format(name))

plt.show()



#Replace data in dataset

train_values[name] = winsorized_data
train_values.drop(columns=["building_id"], inplace=True)

train_labels.drop(columns=["building_id"], inplace=True)
train_values = pd.get_dummies(train_values, drop_first = True)
plt.figure(figsize=(10,10))

correlations = train_values.corrwith(train_labels["damage_grade"])

sns.heatmap(pd.DataFrame(correlations), annot=True)

plt.show()
X = train_values

y = train_labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



#Decision Tree

dt = DecisionTreeClassifier(max_features = None,

                            max_depth = 45,

                            min_samples_split = 3,

                            min_samples_leaf = 30,

                            random_state=42)

start_time = time.time()

model = dt.fit(X_train, y_train)

dt_time_fit = time.time() - start_time



#Predictions - Decision Tree

start_time = time.time()

model.predict(X_test)

dt_time_pred = time.time() - start_time

print("Decision Tree")

print("Fit Time: {} seconds".format(dt_time_fit))

print("Prediction Time: {} seconds".format(dt_time_pred))

print("Training Score: {}".format(dt.score(X_train, y_train)))

print("Test Score: {}".format(dt.score(X_test, y_test)))

print("----------------------------------------")



#Random Forest

rf = RandomForestClassifier(max_features = None,

                            max_depth = 45,

                            min_samples_split = 3,

                            min_samples_leaf = 30,

                            random_state=42)

start_time = time.time()

model = rf.fit(X_train, y_train)

rf_time_fit = time.time() - start_time



#Predictions - Decision Tree

start_time = time.time()

model.predict(X_test)

rf_time_pred = time.time() - start_time

print("Random Forest")

print("Fit Time: {} seconds".format(rf_time_fit))

print("Prediction Time: {} seconds".format(rf_time_pred))

print("Training Score: {}".format(rf.score(X_train, y_train)))

print("Test Score: {}".format(rf.score(X_test, y_test)))



y_pred = rf.predict(X_test)



data = confusion_matrix(y_test, y_pred)

cm = pd.DataFrame(data, columns=np.unique(y_test), index = np.unique(y_test))

cm.index.name = 'Actual'

cm.columns.name = 'Predicted'

plt.figure(figsize = (10,7))

sns.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 12})

plt.title("Confusion Matrix")

plt.show()
results = list(zip(X, rf.feature_importances_))

importance = pd.DataFrame(results, columns = ["Feature", "Importance"])

importance = importance.sort_values(by="Importance", ascending=False)

importance
importance_10 = importance.head(10)

plot = sns.barplot(x=importance_10["Feature"], y=importance_10["Importance"])

plot.set_xticklabels(plot.get_xticklabels(), rotation=90)

plt.title("10 Most Important Features")

plt.show()
test_building_id = test_values["building_id"]

test_values.drop(columns = ["building_id"], inplace=True)



#Outliers

test_values["age"] = winsorize(test_values["age"],(0, 0.05))

test_values["area_percentage"] = winsorize(test_values["area_percentage"],(0, 0.055))

test_values["height_percentage"] = winsorize(test_values["height_percentage"],(0, 0.04))



#Dummies

test_values = pd.get_dummies(test_values, drop_first = True)



#Predictions

predictions = rf.predict(test_values)



#Create Submission File

submission = pd.DataFrame()

submission["building_id"] = test_building_id

submission["damage_grade"] = predictions

submission.head()

submission.to_csv("submission.csv", index=False)
predictions = rf.predict(test_values)
submission = pd.DataFrame()

submission["building_id"] = test_building_id

submission["damage_grade"] = predictions

submission.head()

submission.to_csv("submission.csv", index=False)