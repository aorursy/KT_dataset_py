import pandas as pd
import numpy as np

import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

import seaborn as sns

# for progress bar
from tqdm import tqdm_notebook as tqdm
# from progress.bar import Bar
adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",
                        header=0,
                        index_col=0,
                        na_values='?',
                        names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
                                 "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                                "Hours per week", "Country", "Target"])

adult_test = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",
                        header=0,
                        index_col=0,
                        na_values='?',
                        names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
                                 "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                                "Hours per week", "Country"])

print(adult.shape)
print(adult_test.shape)
print("for training we have: ", adult.shape)
print("and for testing: ", adult_test.shape)

adult.head()
adult.describe()
adult.info()
# adult_test.head()
# adult.reset_index(drop=True, inplace = True)
# adult_test.reset_index(drop=True, inplace = True)
# pass
sns.set()

adult_analysis = adult.copy()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

adult_analysis["Target"] = le.fit_transform(adult_analysis["Target"])
mask = np.triu(np.ones_like(adult_analysis.corr(), dtype=np.bool))

plt.figure(figsize=(10,10))

sns.heatmap(adult_analysis.corr(), mask=mask, square = True, annot=True, vmin=-1, vmax=1, cmap='autumn')
plt.show()

print(sns.__version__)
sns.distplot(adult_analysis["Age"], bins=15)
sns.catplot(x="Target",  y = "Age", data=adult_analysis, kind="violin")
sns.catplot(x="Education-Num", y="Target", kind="bar", data=adult_analysis)
# sns.catplot(x="Hours per week", y="Target", kind="", data=adult_analysis)
sns.lineplot(x="Hours per week", y="Target", data=adult_analysis)
sns.catplot(x="Target", y="Capital Gain", data=adult_analysis, kind="boxen")
sns.catplot(x="Target", y="Capital Loss", data=adult_analysis, kind="boxen")
adult["Country"].value_counts()
sns.catplot(y="Country", x="Target", data=adult_analysis, kind="bar")
chart = sns.catplot(x="Marital Status", y="Target", data=adult_analysis, kind="bar")
chart.set_xticklabels(rotation=45)
chart = sns.catplot(x="Occupation", y="Target", data=adult_analysis, kind="bar")
chart.set_xticklabels(rotation=90)
chart = sns.catplot(x="Relationship", y="Target", data=adult_analysis, kind="bar")
chart.set_xticklabels(rotation=45)
chart = sns.catplot(x="Race", y="Target", data=adult_analysis, kind="bar")
chart.set_xticklabels(rotation=45)
chart = sns.catplot(x="Sex", y="Target", data=adult_analysis, kind="bar")
chart.set_xticklabels(rotation=0)
nadult = adult.dropna()
# nadult_test = adult.dropna()

print("We are dropping ", len(adult) - len(nadult), " out of ", len(adult), "rows")
# nadult.head()
nadult = adult.dropna()
print("We are dropping ", len(adult) - len(nadult), " out of ", len(adult), "rows from the training set")
# nadult.head()
test_numerical = adult_test.select_dtypes(exclude="object")
test_numerical.fillna(test_numerical.mean(), 
                        inplace=True)

test_categorical = adult_test.select_dtypes(include="object")
test_categorical.fillna(test_categorical.value_counts()[0],
                        inplace=True)

X_test = pd.concat((test_numerical, test_categorical), axis=1, ignore_index=True)
X_test.columns = test_numerical.columns.to_list() + test_categorical.columns.to_list()

adult_test_completed = X_test[adult_test.columns.to_list()]

print(adult_test_completed.shape)
# print(X_test.columns)
# print(adult_test.columns)
# X_test.head()
X_adult = nadult.drop(labels=["Target"], axis=1)
Y_train_adult = nadult["Target"]

X_test_adult = adult_test_completed.copy()

# print(X_adult.shape)
# print(X_test_adult.shape)
# print(Y_train_adult.shape)
X_categorical = X_adult.select_dtypes(include=['object'],)
X_categorical = pd.DataFrame(X_categorical.values, dtype="string", columns=X_categorical.columns)
X_test_categorical = X_test_adult.select_dtypes(include=['object'])
X_test_categorical = pd.DataFrame(X_test_categorical.values, dtype="string", columns=X_test_categorical.columns)

X_numerical = X_adult.select_dtypes(exclude=['object'])
X_test_numerical = X_test_adult.select_dtypes(exclude=['object'])

print(X_categorical.shape)
print(X_test_categorical.shape)
print(X_numerical.shape)
print(X_test_numerical.shape)
X_categorical.drop(labels=["Education", "Country"],
                    axis=1,
                    inplace=True)

X_test_categorical.drop(labels=["Education", "Country"],
                    axis=1,
                    inplace=True)
print(X_categorical.shape)
print(X_test_categorical.shape)
X_temp = X_categorical.append(X_test_categorical, ignore_index=True, verify_integrity=True)
# print(pd.api.types.is_string_dtype(X_test_categorical.values))

# for c in X_test_categorical.columns:
#     print(c, ": ", pd.api.types.is_string_dtype(X_test_categorical[c]))
from sklearn.preprocessing import OneHotEncoder

categorical_encoder = sklearn.preprocessing.OneHotEncoder()
categorical_encoder.fit(X_temp)

categories = [val for sublist in categorical_encoder.categories_ for val in sublist]

X_categorical_1hot = pd.DataFrame(categorical_encoder.transform(X_categorical).toarray(), columns=categories)
X_test_categorical_1hot = pd.DataFrame(categorical_encoder.transform(X_test_categorical).toarray(), columns=categories)

print(X_categorical_1hot.shape)
print(X_test_categorical_1hot.shape)
print(categories)
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X_numerical)

X_numerical_scaled = pd.DataFrame(scaler.transform(X_numerical), columns=X_numerical.columns)
X_test_numerical_scaled = pd.DataFrame(scaler.transform(X_test_numerical), columns=X_numerical.columns)

# X_test_numerical_scaled.head()
X_train_adult = pd.concat((X_categorical_1hot, X_numerical_scaled), axis=1, ignore_index=True)
X_test_adult = pd.concat((X_test_categorical_1hot, X_test_numerical_scaled), axis=1, ignore_index=True)

print(X_train_adult.shape, X_test_adult.shape)
# Ks_to_try = 2**np.arange(3,8)
Ks_to_try = np.arange(20, 45, 3)
folds = 10

scores_from_cross_val = np.zeros((len(Ks_to_try), folds))
# bar = Bar("Progress", max=len(Ks_to_try))
for i,k in enumerate(Ks_to_try):
    print(i, "out of ", len(Ks_to_try), " ...")
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn,
                            X_train_adult,
                            Y_train_adult,
                            cv = folds)
    scores_from_cross_val[i,:] = score
print(i+1, "out of ", len(Ks_to_try))

# bar.finish()
# scores = np.mean(scores_from_cross_val, axis=1)
scores = np.mean(scores_from_cross_val, axis=1)
plt.plot(Ks_to_try, scores)

for i in range(0,len(scores)):
    x,y = Ks_to_try[i], scores[i]
    plt.text(x, y, "{:.3f}".format(y))

plt.ylabel("accuracy")
plt.xlabel("k")
plt.show()
k_best = np.argmax(scores)

trained_knn = knn.fit(X_train_adult, Y_train_adult)
Y_test_predicted = trained_knn.predict(X_test_adult)
X_dropped_1 = nadult.drop(labels=["fnlwgt", "Capital Loss", "Hours per week", "Target"], axis=1, errors="ignore")
X_dropped_1_test =  adult_test_completed.copy().drop(labels=["fnlwgt", "Capital Loss", "Hours per week"], axis=1, errors="ignore")

# split categorical and numerical
X_categorical = X_dropped_1.select_dtypes(include=['object'],)
X_categorical = pd.DataFrame(X_categorical.values, dtype="string", columns=X_categorical.columns)
X_test_categorical = X_dropped_1_test.select_dtypes(include=['object'])
X_test_categorical = pd.DataFrame(X_test_categorical.values, dtype="string", columns=X_test_categorical.columns)

X_numerical = X_dropped_1.select_dtypes(exclude=['object'])
X_test_numerical = X_dropped_1_test.select_dtypes(exclude=['object'])

# prepare one-hot features
X_temp = X_categorical.append(X_test_categorical, ignore_index=True, verify_integrity=True)
categorical_encoder = sklearn.preprocessing.OneHotEncoder()
categorical_encoder.fit(X_temp)

categories = [val for sublist in categorical_encoder.categories_ for val in sublist]

X_categorical_1hot = pd.DataFrame(categorical_encoder.transform(X_categorical).toarray(), columns=categories)
X_test_categorical_1hot = pd.DataFrame(categorical_encoder.transform(X_test_categorical).toarray(), columns=categories)

# normalize numerical features
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X_numerical)

X_numerical_scaled = pd.DataFrame(scaler.transform(X_numerical), columns=X_numerical.columns)
X_test_numerical_scaled = pd.DataFrame(scaler.transform(X_test_numerical), columns=X_test_numerical.columns)

# rebuild features
X_dropped_1 = pd.concat((X_categorical_1hot, X_numerical_scaled), axis=1, ignore_index=True)
X_dropped_1_test = pd.concat((X_test_categorical_1hot, X_test_numerical_scaled), axis=1, ignore_index=True)
knn = KNeighborsClassifier(n_neighbors=k_best)
score = cross_val_score(knn,
                        X_dropped_1,
                        Y_train_adult,
                        cv = folds)
print("for this selection of features we have got a score of {:.3f}".format(np.mean(score)))
Y_test_predicted = trained_knn.predict(X_test_adult)
output = pd.DataFrame(Y_test_predicted, columns=["Income"])
# output.to_csv("output.csv", index_label="Id")