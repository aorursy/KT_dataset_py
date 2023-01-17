import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt
def load_csv(path, limit = 0, labelName=0):

    csvData = pd.read_csv(path)

    #print(csvData.head())

    if limit != 0 and labelName != 0:

        data = csvData.loc[0:limit-1, csvData.columns != labelName]

        label = csvData.loc[0:limit-1, labelName]

        return data, label

    else:

        return csvData



csvData = load_csv("../input/train.csv")



csvData.head()

csvData.shape
maleRate = csvData.Survived[csvData.Sex == "male"].value_counts()

femaleRate = csvData.Survived[csvData.Sex == "female"].value_counts()



SurvivedBySex = pd.DataFrame({"male":maleRate, "femaleRate":femaleRate})

SurvivedBySex.plot(kind="bar", stacked="true")

plt.title("Survived by Sex")

plt.xlabel("Survived")

plt.ylabel("Numbers")

plt.show()
age = csvData["Age"]

age.hist(label="Total Number By Age")



plt.title("Age Distribution")

plt.xlabel("Age")

plt.ylabel("Number")

plt.show()
age = csvData["Age"]

age.hist()



survivedByAge = csvData[csvData.Survived== 1]["Age"]

survivedByAge.hist()



#rate = pd.DataFrame({"Total":age, "Survived":survivedByAge})

#rate.plot(kind="bar", stacked=True)



plt.title("Survived By Age")

plt.xlabel("Age")

plt.ylabel("Number")

plt.show()
fare = csvData["Fare"]

fare.hist()





survived = csvData[csvData.Survived==1]["Fare"]

survived.hist()



plt.title("Survived By Fare")

plt.xlabel("Fare")

plt.ylabel("Number")

plt.show()
pclass = csvData["Pclass"].hist()



survived = csvData[csvData.Survived==1]["Pclass"].hist()



plt.title("Survived By Pclass")

plt.xlabel("Class")

plt.ylabel("Number")

plt.show()
c = csvData.Survived[csvData["Embarked"] == "C"].value_counts()

q = csvData.Survived[csvData["Embarked"] == "Q"].value_counts()

s = csvData.Survived[csvData["Embarked"] == 'S'].value_counts()



print(c)

print(q)

print(s)





stack = pd.DataFrame({"C":c, "Q":q, "S":s})

stack.plot(kind="bar", stacked = True)

plt.title("Survived By Embarked")

plt.xlabel("Survived")

plt.ylabel("Number")

plt.show()
label = csvData.loc[:,"Survived"]



def selectCols(csvData):

    return csvData.loc[:,["Pclass", "Sex", "Age", "Fare", "Embarked"]]



data = selectCols(csvData)

data.head()
#print(data.isnull().any(axis=1))

#print(data.values[5])



def handelNonNumber(data):

    data = data.copy(deep = True)

    data.loc[data['Sex'] == 'female', 'Sex'] = 0

    data.loc[data['Sex'] == 'male', 'Sex'] = 1



    data.loc[data['Embarked'] == 'S', 'Embarked'] = 0

    data.loc[data['Embarked'] == 'Q', 'Embarked'] = 1

    data.loc[data['Embarked'] == 'C', 'Embarked'] = 2

    

    return data



data = handelNonNumber(data)

data.head()

def fillNaN(data):

    for col in data.columns:

        data[col] = data[col].fillna( int(data[col].median()) )



fillNaN(data)



print(data.isnull().values.any())

data.head(10)
from sklearn.model_selection import train_test_split



trainSet, validSet, trainLebel, validLabel = train_test_split(data, label, test_size = .2)



print(trainSet.shape, validSet.shape, trainLebel.shape, validLabel.shape)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



k_range = range(1, 51)

scores = []

maxScore = 0

maxK = 0

for k in k_range:

        knn = KNeighborsClassifier(n_neighbors = k)

        knn.fit(trainSet, trainLebel)

        result = knn.predict(validSet)

        score = accuracy_score(validLabel, result)

        scores.append(score)

        

        if(score > maxScore):

            maxScore = score

            maxK = k

        #print(k, score)

        

plt.plot(k_range, scores)

plt.xlabel("K")

plt.ylabel("accuracy")

plt.show()



print(maxK)
testData = load_csv("../input/test.csv")



testSet = selectCols(testData)

testSet = handelNonNumber(testSet)

fillNaN(testSet)



#testData.head()

knn = KNeighborsClassifier(n_neighbors = maxK)

knn.fit(data, label)

prediction = knn.predict(testSet)





print(prediction)
result = pd.DataFrame({"Passenger ID:": testData["PassengerId"], "Survived": prediction})

result.to_csv("submission.csv")