import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
dataset=pd.read_csv('../input/train.csv')

testset=pd.read_csv('../input/test.csv')

dataset.dtypes
dataset.head(20)
Surived_Ages = dataset.Age[dataset.Survived == 1]

plt.hist(Surived_Ages, bins = 20)

plt.title("Survived by Age")

plt.show()



Surived_Ages = dataset.Age[dataset.Survived == 0]

plt.hist(Surived_Ages, bins = 20)

plt.title("Non -Survived by Age")

plt.show()
Survived_m = dataset.Survived[dataset.Sex == "male"].value_counts()

Survived_f = dataset.Survived[dataset.Sex == "female"].value_counts()

Survived_P1 = dataset.Survived[dataset.Pclass == 1].value_counts()

Survived_P2 = dataset.Survived[dataset.Pclass == 2].value_counts()

Survived_P3 = dataset.Survived[dataset.Pclass == 3].value_counts()



Survived_Ebk_S = dataset.Survived[dataset.Embarked == "S"].value_counts()

Survived_Ebk_C = dataset.Survived[dataset.Embarked == "C"].value_counts()

Survived_Ebk_Q = dataset.Survived[dataset.Embarked == "Q"].value_counts()



gender_df = pd.DataFrame({"S_Male": Survived_m, "S_Female": Survived_f})

pclass_df = pd.DataFrame({"S_P1": Survived_P1, "S_P2": Survived_P2, "S_P3": Survived_P3})

embark_df = pd.DataFrame({"Emk_S": Survived_Ebk_S, "Emk_C": Survived_Ebk_C, "Emk_Q": Survived_Ebk_Q })





gender_df.plot(kind="bar", stacked = True)

plt.title("Survived by Gender")

plt.xlabel("0:Non-Survived      1: Survived")

plt.ylabel("Count")



pclass_df.plot(kind="bar", stacked = True)

plt.title("Survived by PClass")

plt.xlabel("0:Non-Survived      1: Survived")

plt.ylabel("Count")



embark_df.plot(kind="bar", stacked = True)

plt.title("Survived by Embark location")

plt.xlabel("0:Non-Survived      1: Survived")

plt.ylabel("Count")

plt.show()
select_columns = ["Age", "Sex", "Pclass", "Embarked", "Fare"]

target_column = ["Survived"]



data = dataset[select_columns]

label = dataset[target_column]

testdata = testset[select_columns]



data.head()
def cleanNA(data):

    data_copy = data.copy(deep = True)

    data_copy["Age"] = data_copy["Age"].fillna(data_copy.Age.median())    

    data_copy["Sex"] = data_copy["Sex"].fillna("female")

    data_copy["Pclass"] = data_copy["Pclass"].fillna(data_copy["Pclass"].median())

    data_copy["Embarked"] = data_copy["Embarked"].fillna("S")

    data_copy.loc[:,'Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())

    return data_copy

    

def transfer_data(data):



    data_copy = data.copy(deep = True)

    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0

    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1

    

    data_copy.loc[data_copy['Embarked'] == "Q", "Embarked"] = 0

    data_copy.loc[data_copy['Embarked'] == "S", "Embarked"] = 1

    data_copy.loc[data_copy['Embarked'] == "C", "Embarked"] = 2



    return data_copy

    

data_no_nan = cleanNA(data)

data_no_str = transfer_data(data_no_nan)



test_no_nan = cleanNA(testdata)

test_no_str = transfer_data(test_no_nan)



x_data = data_no_str.values

y_data = label.values

x_test = test_no_str.values
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
X_train, X_valid, Y_train, Y_valid = train_test_split(x_data, y_data, test_size = 0.2, random_state = 0)



krange = range(1, 50)

scores = []

best_k, best_score = 1, 0

for k in krange:

    knn_model = KNeighborsClassifier(n_neighbors = k)

    knn_model.fit(X_train, Y_train.ravel())

    predictions = knn_model.predict(X_valid)

    score = accuracy_score(predictions, Y_valid)

    scores.append(score)

    if score > best_score:

        best_k, best_score = k, score

    print("K=" + str(k) + ", accuracy=" + str(score), "(So far Best_K=" + str(best_k) + " accuracy=",best_score,")")
plt.plot(krange, scores)

knn_model = KNeighborsClassifier(n_neighbors = best_k)

knn_model.fit(x_data, y_data.ravel())
# 预测

result = knn_model.predict(x_test)

print(result)
out_df = pd.DataFrame({"PassengerId":testset["PassengerId"], "Survived": result})



out_df.to_csv("submission.csv", header=True, index=False)

out_df.head(20)