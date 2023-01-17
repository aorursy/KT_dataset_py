import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB  # Gaussian naive Bayes classifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

import matplotlib.pyplot as plt

import seaborn as sns

flatui = ["#9b59b6", "#3498db",  "#e74c3c", "#34495e", "#2ecc71"]

# sns.palplot(sns.color_palette("GnBu_d"))

sns.set_palette(flatui)

import numpy as np

np.random.seed(455)



iris = pd.read_csv("../input/Iris.csv",index_col="Id")
iris.head()
iris.describe()
iris.isnull().sum()
# Transform the symbolic species names into numbers suitable for the Bayes classifier

le = LabelEncoder()

le.fit(iris['Species'])

iris['Species'] = le.transform(iris['Species'])



# Split the dataset into 2/3 training data and 1/3 test data

trainSet, testSet = train_test_split(iris, test_size = 0.20)

print(trainSet.shape)

print(testSet.shape)

trainSet.head()
print(le.classes_)

le.inverse_transform([0,1,2])
# Format the data and expected values for SKLearn

trainData = pd.DataFrame.as_matrix(trainSet[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])

trainTarget = pd.DataFrame.as_matrix(trainSet[['Species']]).ravel()

testData = pd.DataFrame.as_matrix(testSet[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])

testTarget = pd.DataFrame.as_matrix(testSet[['Species']]).ravel()



classifiers = [GaussianNB(),MultinomialNB(),ComplementNB()]



mse = []

f1score = []

accuracy = []

for classifier in classifiers:    

    classifier.fit(trainData, trainTarget)

    predictedValues =  classifier.predict(testData)

    mse.append(mean_squared_error(testTarget,predictedValues))

    accuracy.append(accuracy_score(testTarget,predictedValues)) 

    f1score.append(f1_score(testTarget,predictedValues, average="macro"))
for i,classifier in enumerate(classifiers):

    print("*"*100)

    print(classifier)

    print("MSE "+ str(mse[i]))

    print("Accuracy "+str(accuracy[i]))

    print("F1 Score "+ str(f1score[i]))

cls = ["Gaussian","Multinomail","Complement"]

plt.figure(figsize=(20,5))

sns.barplot(mse,cls)
plt.figure(figsize=(20,5))

sns.barplot(accuracy,cls)
plt.figure(figsize=(20,5))

sns.barplot(f1score,cls)