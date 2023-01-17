import pandas as pd





file_path = '../input/2012-18_teamBoxScore.csv'

X2012_18_teamBoxScore = pd.read_csv(file_path) 

X2012_18_teamBoxScore.describe()

y = X2012_18_teamBoxScore.teamPTS
nba_features = ['teamSTL', 'teamAST', 'teamTO']

x = X2012_18_teamBoxScore[nba_features]

x_2 = X2012_18_teamBoxScore["teamSTL"] + X2012_18_teamBoxScore["teamAST"] - X2012_18_teamBoxScore["teamTO"]
x.describe()
import statsmodels.api as sm

import numpy as np

from sklearn import linear_model





model = sm.OLS(y, x).fit()

predictions = model.predict(x)
lm = linear_model.LinearRegression()

model_2 = lm.fit(x,y)

predictions_2 = model_2.predict(x)
print("The predictions are")

print(predictions_2)

print("The predictions are")

print(predictions)

print(model.summary())
import matplotlib.pyplot as plt 



plt.scatter(x_2,y)
import matplotlib.pyplot as plt 



plt.scatter(x_2,y)

plt.plot(x_2, predictions,color='green')

plt.plot(x_2,predictions_2, color='red')



plt.show() 

nba_features = ['teamSTL', 'teamAST', 'teamTO', 'teamFGA', 'teamFGM']

z = X2012_18_teamBoxScore[nba_features]

z_2 = X2012_18_teamBoxScore["teamSTL"] + X2012_18_teamBoxScore["teamAST"] - X2012_18_teamBoxScore["teamTO"] + X2012_18_teamBoxScore["teamFGA"] - X2012_18_teamBoxScore["teamFGM"]





model_3 = sm.OLS(y, z).fit()

predictions_3 = model_3.predict(z)



print(model_3.summary())



plt.scatter(z_2,y)

plt.plot(z_2,predictions_3, color='yellow')



plt.show() 
