import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

from sklearn.preprocessing import OneHotEncoder

from sklearn.utils import resample

from imblearn.over_sampling import SMOTE

from sklearn.impute import SimpleImputer

from statsmodels.graphics.gofplots import qqplot

from scipy.stats import shapiro

from scipy.stats import boxcox

from pandas_profiling import ProfileReport

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC, LinearSVC
train_data = pd.read_csv("../input/titanic/train.csv")

train_data.head()
train_data.columns
def delete_unique_obs(data: pd.DataFrame) -> pd.DataFrame:

    valid_cols = []

    for col in data:

        if data[col].nunique()/len(data) <= 0.7:

            valid_cols.append(col)

        else:

            print("[INFO] Deleting "+ col)

    return data[valid_cols]
train_data = delete_unique_obs(train_data)

train_data.head()
msno.matrix(train_data,

            figsize=(16,7),

            width_ratios=(15,1)

           )
def handle_missing_data(data):

    for col in data:

        missing_proportion = data[col].isna().sum()/len(data) 

        if missing_proportion > 0.6:

            print("[INFO] Deleting " + col + " since it is having " +str(missing_proportion) + " missing proportion")

            data.drop(col, axis=1, inplace=True)

        if missing_proportion < 0.02 and missing_proportion != 0.0:

            print("[INFO] Deleting the nan rows from " + col + " with missing proportion " +str(missing_proportion))

            data = data[data[col].notna()]

            

    return data
train_data = handle_missing_data(train_data)

train_data.head()
msno.matrix(train_data,

            figsize=(16,7),

            width_ratios=(15,1)

           )
# Impute missing data in Age

train_data.describe()
# impute the missing data with the mean

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

imputer.fit(train_data.Age.to_numpy().reshape(-1,1))

train_data["Age"] = imputer.transform(train_data.Age.to_numpy().reshape(-1,1))
msno.matrix(train_data,

            figsize=(16,7),

            width_ratios=(15,1)

           )
train_data.info()
ohe = OneHotEncoder(categories='auto')

feature_arr = ohe.fit_transform(train_data[['Sex','Embarked']]).toarray()

feature_labels = ohe.categories_
feature_labels = np.concatenate(feature_labels)
features = pd.DataFrame(feature_arr, columns=feature_labels)

features.head()
features.info()
train_data = pd.concat([train_data, features], axis=1).drop(["Sex","Embarked"], axis=1)

train_data.Pclass.unique()
train_data.Survived.value_counts()
# separate the survival classes

survived = train_data[train_data.Survived == 1]

not_survived = train_data[train_data.Survived == 0]
not_survived.Survived.unique()
# resample the minority class ie., survived

survived_resampled = resample(survived, replace=True, n_samples=len(not_survived), random_state=2)

len(survived_resampled)
resampled_train_data = pd.concat([survived_resampled,not_survived])

resampled_train_data.head()
# # SMOTE

# train_data.dropna(inplace=True)

# y = train_data.Survived 

# x = train_data.drop("Survived", axis=1)

# sm = SMOTE(random_state=2)

# smote_x, smote_y = sm.fit_sample(x,y)
# train_smote_data = smote_x.merge(smote_y.to_frame(), left_index=True, right_index=True)

# train_smote_data.head()
# train_smote_data.Survived.value_counts()
# train_smote_data.head()
resampled_train_data.Pclass.unique()
sns.countplot(x="Pclass", data=resampled_train_data)
sns.distplot(resampled_train_data.Age)
plt.hist(resampled_train_data.Age)

plt.show()
# Quantile Quantile plot

qqplot(resampled_train_data.Age, line='s')

plt.show()
stat, p = shapiro(resampled_train_data.Age)

print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret

alpha = 0.05

if p > alpha:

    print('Sample looks Gaussian (fail to reject H0)')  # Is a Gaussian distribution

else:

    print('Sample does not look Gaussian (reject H0)')  #Not a Gaussian distribution
# Find the optimal value for lambda of boxcox

time_duration_trans, lmbda = boxcox(resampled_train_data.Age)

lmbda
qqplot(boxcox(resampled_train_data.Age,lmbda), line='s')

plt.show()
resampled_train_data["Age"] = boxcox(resampled_train_data.Age,lmbda)

resampled_train_data.head()
plt.hist(resampled_train_data.Age)

plt.show()
resampled_train_data.describe()
# Observations with Fare as 0

len(resampled_train_data[resampled_train_data.Fare == 0])
resampled_train_data = resampled_train_data[resampled_train_data.Fare != 0]

len(resampled_train_data[resampled_train_data.Fare == 0])
plt.hist(resampled_train_data.Fare)

plt.show()
resampled_train_data = resampled_train_data[resampled_train_data.Fare < 150]

plt.hist(resampled_train_data.Fare)

plt.show()
# Find the optimal value for lambda of boxcox

time_duration_trans, lmbda = boxcox(resampled_train_data.Fare)

lmbda
qqplot(boxcox(resampled_train_data.Fare,lmbda), line='s')

plt.show()
resampled_train_data["Fare"] = boxcox(resampled_train_data.Fare,lmbda)

plt.hist(resampled_train_data.Fare)

plt.show()
resampled_train_data.head()
#Which gender survived more

sns.countplot(x="Survived", hue="female", data=resampled_train_data)
#Age group of better survival

sns.boxplot(x="Survived", y="Age", data=resampled_train_data)
# Family Size

resampled_train_data["FamilySize"] = resampled_train_data["SibSp"] + resampled_train_data["Parch"]

resampled_train_data.drop(["SibSp","Parch"], axis=1, inplace=True)

resampled_train_data.head()
# Assuming that if the family size is 0, then he/she is travelling alone. 

print(len(resampled_train_data[resampled_train_data.FamilySize == 0]))

resampled_train_data['FamilySize'] = np.where(resampled_train_data['FamilySize'] == 0, 1, resampled_train_data['FamilySize'])

print(len(resampled_train_data[resampled_train_data.FamilySize == 0]))
sns.countplot(x="FamilySize", data=resampled_train_data)
corr = resampled_train_data.corr()

# mask for upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
profile = ProfileReport(resampled_train_data)

profile
# train-valid split

data = resampled_train_data.copy()

data = data.dropna()

y = data.Survived

x = data.drop("Survived", axis=1)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20, random_state=2)

# sc = StandardScaler()

# train_x = sc.fit_transform(train_x)

# test_x = sc.transform(test_x)
log_reg = LogisticRegression()

log_reg.fit(train_x, train_y)

preds = log_reg.predict(test_x)

print(classification_report(test_y, preds))
knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(train_x, train_y)

preds = knn.predict(test_x)

print(classification_report(test_y, preds))
gnb = GaussianNB()

gnb.fit(train_x, train_y)

preds = gnb.predict(test_x)

print(classification_report(test_y, preds))
svc = SVC(gamma=0.01)

svc.fit(train_x, train_y)

preds = svc.predict(test_x)

print(classification_report(test_y, preds))
tree = DecisionTreeClassifier()

tree.fit(train_x, train_y)

preds = tree.predict(test_x)

print(classification_report(test_y, preds))
forest = RandomForestClassifier(max_depth=3, random_state=0)

forest.fit(train_x, train_y)

preds = forest.predict(test_x)

print(classification_report(test_y, preds))
linear_SVC = LinearSVC(C=0.01)

linear_SVC.fit(train_x, train_y)

preds = linear_SVC.predict(test_x)

print(classification_report(test_y, preds))
from xgboost import XGBClassifier
xgboost = XGBClassifier(n_estimators=1000,reg_alpha = 0.1, gamma=0.001)

xgboost.fit(train_x, train_y)

preds = xgboost.predict(test_x)

print(classification_report(test_y, preds))
test_data = pd.read_csv("../input/titanic/test.csv")

test = pd.DataFrame(test_data.PassengerId)

test_data.head()
test_data.drop(["PassengerId","Name", "Cabin", "Ticket"], axis=1, inplace=True)

test_data.head()
# Encode Sex, Embarcked

ohe = OneHotEncoder(categories='auto')

feature_arr = ohe.fit_transform(test_data[['Sex','Embarked']]).toarray()

feature_labels = ohe.categories_
feature_labels = np.concatenate(feature_labels)
features = pd.DataFrame(feature_arr, columns=feature_labels)

features.head()
test_data = pd.concat([test_data, features], axis=1).drop(["Sex","Embarked"], axis=1)

test_data.head()
test_data["FamilySize"] = test_data["SibSp"] + test_data["Parch"]

print(len(test_data[test_data.FamilySize == 0]))

test_data['FamilySize'] = np.where(test_data['FamilySize'] == 0, 1, test_data['FamilySize'])

print(len(test_data[test_data.FamilySize == 0]))
test_data.drop(["SibSp","Parch"], axis=1, inplace=True)

test_data.head()
# Check for missing data

msno.matrix(test_data,

            figsize=(16,7),

            width_ratios=(15,1)

           )
# Impute age with mean

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

imputer.fit(test_data.Age.to_numpy().reshape(-1,1))

test_data["Age"] = imputer.transform(test_data.Age.to_numpy().reshape(-1,1))
predictions = xgboost.predict(test_data).astype(int)
test["Survived"] = predictions
test.head()
test.to_csv("my_preds.csv", index=False)
test.Survived.value_counts()