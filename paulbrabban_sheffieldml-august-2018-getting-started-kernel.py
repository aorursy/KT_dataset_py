import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
train[["OverallQual", "SalePrice"]].corr()
import seaborn as sns
result = sns.regplot(train["OverallQual"], train["SalePrice"])
from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(train["OverallQual"].values.reshape(-1, 1), train["SalePrice"].values)
model.predict([[1], [5], [10]]) # argument same as [1, 5, 10].reshape(-1, 1)
test = pd.read_csv("../input/test.csv")

predicted = model.predict(test["OverallQual"].values.reshape(-1, 1))

my_submission = pd.DataFrame({'Id': test["Id"], 'SalePrice': predicted})

my_submission.to_csv('my_submission.csv', index=False)