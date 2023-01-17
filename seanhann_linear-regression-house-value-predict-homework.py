import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
csvData = pd.read_csv("../input/kc_house_data.csv")

csvData.head()
csvData.dtypes
data = csvData.loc[:,["bedrooms","bathrooms","sqft_living","floors","yr_built"]]

label = csvData["price"]

print(data.shape)
from sklearn.model_selection import train_test_split

trainSet, validateSet, trainLabel, validateLabel = train_test_split(data, label, test_size=1/3, random_state = 0)

print(trainSet.shape, validateSet.shape, trainLabel.shape, validateLabel.shape)

plt.scatter(data["sqft_living"], label)

plt.xlabel("sqft")

plt.ylabel("price")

plt.show()
data["sqft_living"].hist()

plt.show()
from sklearn.linear_model import LinearRegression

classifier = LinearRegression()

classifier.fit(trainSet, trainLabel)
#print(list(zip(trainSet.columns, np.transpose(classifier.coef_))))



pd.DataFrame(list(zip(trainSet.columns, np.transpose(classifier.coef_))))
classifier.intercept_
#一个房子，3个卧室，2个卫生间，2500sqft，2层楼，预测房价

floor1 = classifier.predict([[1,1,1200,1,1987]])

floor2 = classifier.predict([[1,1,1200,2,1987]])

print(floor1 - floor2)
predictions = classifier.predict(trainSet)



print( ((trainLabel - predictions)* (trainLabel - predictions)).sum()/len(trainSet) )

print( ((trainLabel - predictions)**2).sum()/len(trainSet) )



pd.DataFrame({"Predictions":predictions, "Real":trainLabel.values}).head()
(abs(predictions-trainLabel)/trainLabel).sum() / len(trainLabel)
predtest = classifier.predict(validateSet)

((predtest-validateLabel)**2).sum() / len(validateSet)
(abs(predtest-validateLabel)/validateLabel).sum() / len(validateLabel)