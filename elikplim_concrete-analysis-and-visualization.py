import numpy

import pandas

from sklearn.feature_selection import RFE

from sklearn.ensemble import ExtraTreesRegressor





import matplotlib.pyplot as plt

from pandas.tools.plotting import scatter_matrix





# fix random seed for reproducibility

seed = 7

numpy.random.seed(seed)
# load dataset

dataframe = pandas.read_csv(r"../input/concrete_data.csv")
dataframe.head()




print("Statistical Description:") 

dataframe.describe()
print("Shape:", dataframe.shape)
print("Data Types:", dataframe.dtypes)
print("Correlation:") 

dataframe.corr(method='pearson')
dataset = dataframe.values





X = dataset[:,0:8]

Y = dataset[:,8] 
#Feature Selection

model = ExtraTreesRegressor()

rfe = RFE(model, 3)

fit = rfe.fit(X, Y)



print("Number of Features: ", fit.n_features_)

print("Selected Features: ", fit.support_)

print("Feature Ranking: ", fit.ranking_) 
plt.hist((dataframe.concrete_compressive_strength))
dataframe.hist()
dataframe.plot(kind='density', subplots=True, layout=(3,4), sharex=False, sharey=False)
dataframe.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)

scatter_matrix(dataframe)