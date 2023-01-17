import pandas as pd
adultTrain = pd.read_csv("adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")



adultTest = pd.read_csv("adult-pmr3508/test_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
print(adultTrain.shape)

print(adultTest.shape)
adultTrain.columns = ["Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"]
adultTrain
adultTest.columns = ["Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"]
adultTest
nadultTrain = adultTrain.dropna()

nadultTest = adultTest.dropna()
print("Train - Number of rows before removing NAs : " + str(adultTrain.shape[0]))

print("      - Number of rows after  removing NAs : " + str(nadultTrain.shape[0]))

print()

print("Test  - Number of rows before removing NAs : " + str(adultTest.shape[0]))

print("      - Number of rows after  removing NAs : " + str(nadultTest.shape[0]))
XadultTrain = nadultTrain[["Age", "Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

YadultTrain = nadultTrain.Target



XadultTest = nadultTest[["Age", "Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, XadultTrain, YadultTrain, cv=10)
scores
knn.fit(XadultTrain,YadultTrain)
YadultTest = knn.predict(XadultTest)

YadultTest
iDs = nadultTest["Id"]

results = pd.DataFrame(iDs)
results["Target"] = YadultTest
results
results.to_csv("resultsAdults.csv", index =False)
crTrain = pd.read_csv("costa-rican-household-poverty-prediction/train.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")



crTest = pd.read_csv("costa-rican-household-poverty-prediction/test.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
print(crTrain.shape)

print(crTest.shape)
ncrTrain = crTrain.dropna()

ncrTest = crTest.dropna()

ncrTrain
XcrTrain = ncrTrain[["v2a1", "v18q1", 'rooms', 'SQBescolari', 'SQBedjefe', 'SQBovercrowding']]

YcrTrain = ncrTrain.Target



XcrTest = ncrTest[["v2a1", "v18q1", 'rooms', 'SQBescolari', 'SQBedjefe', 'SQBovercrowding']]
crKnn = KNeighborsClassifier(n_neighbors=3)
crScores = cross_val_score(crKnn, XcrTrain, YcrTrain, cv=5)
crScores
crKnn.fit(XcrTrain,YcrTrain)
YcrTest = crKnn.predict(XcrTest)

YcrTest
crIDs = ncrTest["Id"]
crResults = pd.DataFrame(crIDs)
crResults["Target"] = YcrTest

crResults
crResults.to_csv("resultsCostaRica.csv", index =False)