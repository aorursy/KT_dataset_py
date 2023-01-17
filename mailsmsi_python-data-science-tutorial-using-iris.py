



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#from sklearn import model_selection

#from sklearn.metrics import accuracy_score

#from sklearn.linear_model import LogisticRegression

#from sklearn.linear_model import LinearRegression

#from sklearn.ensemble import RandomForestClassifier

#from sklearn.neighbors import KNeighborsClassifier

#from sklearn.svm import SVC


df = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")

                




df.head(5)





df.info()


df.shape





df.describe()



df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)



plt.show()


df.hist()

plt.show()


from pandas.plotting import scatter_matrix

scatter_matrix(df)

plt.show()
df.isnull().sum()
df.isnull()
df.corr(method='pearson') 