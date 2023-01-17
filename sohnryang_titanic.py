import pandas as pd
data = pd.read_csv("../input/titanic/train.csv")
data
data.info()
%matplotlib inline
import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(20, 15))
plt.show()
attrs = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
data[attrs]
from pandas.plotting import scatter_matrix
scatter_matrix(data[attrs], figsize=(12, 8))