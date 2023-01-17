%matplotlib inline



import pandas as pd

import sklearn as sk

import matplotlib.pyplot as plt

import numpy as np



train_data = pd.read_csv("../input/adult-pmr3508/train_data.csv", sep=r'\s*,\s*', engine='python', na_values="?")
train_data.isna().any()
train_data["workclass"].value_counts()
train_data["occupation"].value_counts()
train_data["native.country"].value_counts()
# train_data = train_data.dropna()

train_data["workclass"] = train_data["workclass"].fillna('Private')

train_data["occupation"] = train_data["occupation"].fillna('Prof-specialty')

train_data["native.country"] = train_data["native.country"].fillna('United-States')

train_data.isna().any()
train_data_lt = train_data.loc[train_data['income'] == '<=50K']

train_data_gt = train_data.loc[train_data['income'] == '>50K']



pd.concat([train_data_lt["age"].value_counts(), train_data_gt["age"].value_counts()], keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='age', ylabel='num', figsize=(20, 10))
pd.concat([train_data_lt["workclass"].value_counts(), train_data_gt["workclass"].value_counts()], keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='workclass', ylabel='num')
pd.concat([train_data_lt["marital.status"].value_counts(), train_data_gt["marital.status"].value_counts()], keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='marital.status', ylabel='num')
pd.concat([train_data_lt["relationship"].value_counts(), train_data_gt["relationship"].value_counts()], keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='relationship', ylabel='num')
pd.concat([train_data_lt["education.num"].value_counts(), train_data_gt["education.num"].value_counts()], keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='education.num', ylabel='num')
pd.concat([train_data_lt["occupation"].value_counts(), train_data_gt["occupation"].value_counts()], keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='occupation', ylabel='num')
pd.concat([train_data_lt["race"].value_counts(), train_data_gt["race"].value_counts()], keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='race', ylabel='num')
pd.concat([train_data_lt["sex"].value_counts(), train_data_gt["sex"].value_counts()], keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='sex', ylabel='num')
pd.concat([train_data_lt["hours.per.week"].value_counts(), train_data_gt["hours.per.week"].value_counts()], keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='hours.per.week', ylabel='num', figsize=(20, 10))
pd.concat([train_data_lt["capital.gain"].value_counts(), train_data_gt["capital.gain"].value_counts()], keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='capital.gain', ylabel='num', figsize=(20, 10))
pd.concat([train_data_lt["capital.loss"].value_counts(), train_data_gt["capital.loss"].value_counts()], keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='capital.loss', ylabel='num', figsize=(20, 10))
pd.concat([train_data_lt["native.country"].value_counts(), train_data_gt["native.country"].value_counts()], keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='native.country', ylabel='num', figsize=(20, 10))
val = train_data_lt["native.country"].value_counts().index[:1]

train_data_lt["country"] = np.where(train_data_lt["native.country"].isin(val), "United States", "others")

train_data_gt["country"] = np.where(train_data_gt["native.country"].isin(val), "United States", "others")



pd.concat([train_data_lt["country"].value_counts(), train_data_gt["country"].value_counts()], keys=["<=50K", ">50K"], axis=1).plot(kind='bar', xlabel='country', ylabel='num')
val = train_data["native.country"].value_counts().index[:1]

train_data["native.country"] = np.where(train_data["native.country"].isin(val), "US", "others")

train_data = train_data.drop(columns=["Id", "fnlwgt", "education"])
from sklearn.preprocessing import OrdinalEncoder



features_selection = ["sex", "native.country", "workclass", "marital.status", "income"]

train_data_hm = train_data.copy()

enc = OrdinalEncoder()

train_data_hm[features_selection] = enc.fit_transform(train_data_hm[features_selection])



import seaborn as sns

plt.figure(figsize=(20, 20))

sns.heatmap(train_data_hm.corr(), annot=True, cmap="RdYlBu", vmin=-1)
categorical_features = ["occupation", "relationship", "sex", "race", "native.country", "workclass", "marital.status"]

fs = train_data.columns.tolist()

train_data = pd.get_dummies(train_data, columns=categorical_features, prefix=categorical_features)



train_data.head(10)
train_data.columns.tolist()
dense_num_features = ["age", "education.num", "hours.per.week"]

train_data["capital"] = train_data["capital.gain"] - train_data["capital.loss"]

train_data = train_data.drop(columns=["capital.gain", "capital.loss"])

sparse_num_features = ["capital"]



from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler

scaler_ns = StandardScaler()

train_data[dense_num_features] = scaler_ns.fit_transform(train_data[dense_num_features])

scaler_s = RobustScaler()

train_data[sparse_num_features] = scaler_s.fit_transform(train_data[sparse_num_features])

train_data.head()
X_train = train_data.drop(columns=["income", "native.country_US", "native.country_others"])

X_train.head()
Y_train = train_data["income"]

Y_train.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

scores_mean = []

best_nb = 26

acc = 0

for nb in range(15, 37, 1):

    knn_model = KNeighborsClassifier(n_neighbors=nb, n_jobs=2, algorithm='brute')

    score = np.mean(cross_val_score(knn_model, X_train, Y_train, cv=10))

    scores_mean.append(score)

    if score > acc:

        acc = score

        best_nb = nb
plt.plot(range(15, 37, 1), scores_mean, marker=".")

plt.ylabel("mean score")

plt.xlabel("n_neighbors value")

plt.xticks(range(15, 37, 1))
print(best_nb, acc)
knn_model = KNeighborsClassifier(n_neighbors=best_nb, algorithm='brute')

knn_model.fit(X_train, Y_train)
test_data = pd.read_csv("../input/adult-pmr3508/test_data.csv", sep=r'\s*,\s*', engine='python', na_values="?")



test_data["workclass"] = test_data["workclass"].fillna('Private')

test_data["occupation"] = test_data["occupation"].fillna('Prof-specialty')

test_data["native.country"] = test_data["native.country"].fillna('United-States')

test_data.isna().any()
val = test_data["native.country"].value_counts().index[:1]

test_data["native.country"] = np.where(test_data["native.country"].isin(val), "US", "others")



test_data = test_data.drop(columns=["Id", "fnlwgt", "education"])

test_data = pd.get_dummies(test_data, columns=categorical_features, prefix=categorical_features)



test_data["capital"] = test_data["capital.gain"] - test_data["capital.loss"]

test_data = test_data.drop(columns=["capital.gain", "capital.loss"])



test_data[dense_num_features] = scaler_ns.fit_transform(test_data[dense_num_features])

test_data[sparse_num_features] = scaler_s.fit_transform(test_data[sparse_num_features])

test_data = test_data.drop(columns=["native.country_US", "native.country_others"])
X_test = test_data

X_test.head()
Y_test = knn_model.predict(X_test)

predictions = {'income': Y_test}

output = pd.DataFrame(predictions)
output.to_csv("submission.csv", index=True, index_label='Id')