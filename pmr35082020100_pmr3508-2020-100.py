import matplotlib.pyplot as plt

import sklearn

import pandas as pd

import numpy as np



df_train = pd.read_csv(

    "/kaggle/input/adult-pmr3508/train_data.csv",

    index_col=['Id'],

    engine="python",

    na_values="?",

)



df_test = pd.read_csv(

    "/kaggle/input/adult-pmr3508/test_data.csv",

    index_col=['Id'],

    engine="python",

    na_values="?",

)
df_train.shape
df_train.head()
df_test.shape
df_test.head()
df_train.groupby("income").age.hist()

plt.legend(["<=50k", ">50k"])

plt.title("Histogram of 'age' by 'income'")

plt.xlabel("Age")

plt.ylabel("Number of occurrences")
plt.figure(figsize=(15, 7))

df_train["workclass"].value_counts().plot(kind = 'pie')
workclass_top_value = df_train["workclass"].describe().top

df_train["workclass"] = df_train["workclass"].fillna(workclass_top_value)



workclass_top_value = df_test["workclass"].describe().top

df_test["workclass"] = df_test["workclass"].fillna(workclass_top_value)
df_train.groupby("income").workclass.hist()

plt.legend(["<=50k", ">50k"])

plt.title("Histogram of 'workclass' by 'income'")

plt.xlabel("Workclass")

plt.ylabel("Number of occurrences")
df_train.groupby("income").fnlwgt.hist()

plt.legend(["<=50k", ">50k"])

plt.title("Histogram of 'fnlwgt' by 'income'")

plt.xlabel("Fnlwgt")

plt.ylabel("Number of occurrences")
df_train.groupby("income")["education.num"].hist()

plt.legend(["<=50k", ">50k"])

plt.title("Histogram of 'education num' by 'income'")

plt.xlabel("Education num")

plt.ylabel("Number of occurrences")
df_train.groupby("income")["marital.status"].hist()

plt.legend(["<=50k", ">50k"])

plt.title("Histogram of 'marital status' by 'income'")

plt.xlabel("Marital status")

plt.ylabel("Number of occurrences")
plt.figure(figsize=(15, 7))

df_train["occupation"].value_counts().plot(kind = 'pie')
df_train["occupation"] = df_train["occupation"].fillna("Unknown")



df_test["occupation"] = df_test["occupation"].fillna("Unknown")
occupation = df_train.groupby(['occupation', 'income']).size().unstack()

occupation['sum'] = df_train.groupby('occupation').size()

occupation = occupation.sort_values('sum', ascending = False)[['<=50K', '>50K']]

occupation.plot(kind = 'bar', stacked = True)
df_train.groupby("income").relationship.hist()

plt.legend(["<=50k", ">50k"])

plt.title("Histogram of 'relationship' by 'income'")

plt.xlabel("Relationship")

plt.ylabel("Number of occurrences")
df_train.groupby("income").race.hist()

plt.legend(["<=50k", ">50k"])

plt.title("Histogram of 'race' by 'income'")

plt.xlabel("Race")

plt.ylabel("Number of occurrences")
df_train.groupby("income").sex.hist()

plt.legend(["<=50k", ">50k"])

plt.title("Histogram of 'sex' by 'income'")

plt.xlabel("Sex")

plt.ylabel("Number of occurrences")
df_train.groupby("income")["capital.gain"].hist()

plt.legend(["<=50k", ">50k"])

plt.title("Histogram of 'capital gain' by 'income'")

plt.xlabel("Capital gain")

plt.ylabel("Number of occurrences")
df_train.groupby("income")["capital.loss"].hist()

plt.legend(["<=50k", ">50k"])

plt.title("Histogram of 'capital loss' by 'income'")

plt.xlabel("Capital loss")

plt.ylabel("Number of occurrences")
df_train.groupby("income")["hours.per.week"].hist()

plt.legend(["<=50k", ">50k"])

plt.title("Histogram of 'hours per week' by 'income'")

plt.xlabel("Hours per week")

plt.ylabel("Number of occurrences")
plt.figure(figsize=(15, 7))

df_train["native.country"].value_counts().plot(kind = 'pie')
nativecountry_top_value = df_train["native.country"].describe().top

df_train["native.country"] = df_train["native.country"].fillna(nativecountry_top_value)



nativecountry_top_value = df_test["native.country"].describe().top

df_test["native.country"] = df_test["native.country"].fillna(nativecountry_top_value)
df_train.groupby("income")["native.country"].hist()

plt.legend(["<=50k", ">50k"])

plt.title("Histogram of 'native country' by 'income'")

plt.xlabel("Native country")

plt.ylabel("Number of occurrences")
df_train = df_train.drop(['fnlwgt', 'native.country', 'education'], axis=1)

df_test = df_test.drop(['fnlwgt', 'native.country', 'education'], axis=1)
from sklearn.preprocessing import LabelEncoder



df_train = df_train.apply(LabelEncoder().fit_transform)

df_test = df_test.apply(LabelEncoder().fit_transform)
from sklearn.preprocessing import Normalizer



cols_to_norm = ["age", "education.num", "capital.gain", "capital.loss", "hours.per.week"]

df_train[cols_to_norm] = df_train[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))



df_test[cols_to_norm] = df_test[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



Y_train = df_train.pop("income")

X_train = df_train

X_test = df_test



def search_best_K(X_train, Y_train, k_min, k_max, folds):



    knn = KNeighborsClassifier()

    param_grid = {'n_neighbors': np.arange(k_min, k_max)}

    knn_gscv = GridSearchCV(knn, param_grid, cv=folds)

    knn_gscv.fit(X_train, Y_train)

    print("Best K: ", knn_gscv.best_params_["n_neighbors"])

    print("Best score: ", knn_gscv.best_score_)

    best_K = knn_gscv.best_params_["n_neighbors"]



    return best_K



K = search_best_K(X_train, Y_train, 10, 41, 10) 

#  Testaremos valores de K entre 10 a 40 com validação cruzada entre 10 folds
maxKnn = KNeighborsClassifier(n_neighbors=K)

maxKnn.fit(X_train, Y_train)



numPrediction = maxKnn.predict(X_test)
backSubstitution = {0: '<=50K', 1: '>50K'}

prediction = np.array([backSubstitution[i] for i in numPrediction], dtype=object)

print(prediction)
submission = pd.DataFrame()

submission[0] = df_test.index

submission[1] = prediction

submission.columns = ['Id', 'income']



submission.to_csv('submission.csv', index=False)