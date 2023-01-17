import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import stats



%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))
#To Display all the columns

pd.set_option('display.max_columns',None)
#Loading the Dataset

summary = pd.read_csv("../input/weatherww2/Summary of Weather.csv")
summary.head()
#Identifying the Null values 

summary.isnull().sum()
summary.columns
#Visualizing the Linear Regregression Line

sns.lmplot(x='MinTemp',y='MaxTemp',data=summary)

plt.show()
sns.boxplot(summary['MaxTemp'])

plt.show()
sns.boxplot(summary['MinTemp'])

plt.show()
plt.figure(figsize=(20,20))

sns.heatmap(summary.corr(), annot=True)
summary['MinTemp'].corr(summary['MaxTemp'])
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(summary[['MinTemp']],summary[['MaxTemp']],

                                             test_size = 0.30,

                                            random_state = 123)
xtrain.size
xtest.size
lr = LinearRegression()
lr.fit(xtrain, ytrain)
ypred = lr.predict(xtest)
ypred
plt.scatter(xtrain,ytrain, color='red')

plt.plot(xtrain,lr.predict(xtrain))

plt.xlabel('Minimum Temperature')

plt.ylabel('Maximum Temperature')

plt.title('Minimum Vs Maximum Temperature')

plt.show()
plt.scatter(xtest,ytest, color='red')

plt.plot(xtest,lr.predict(xtest))

plt.xlabel('Minimum Temperature')

plt.ylabel('Maximum Temperature')

plt.title('Minimum Vs Maximum Temperature')

plt.show()
from sklearn.metrics import r2_score, mean_squared_error
r2_score(ytest,ypred)
np.sqrt(mean_squared_error(ytest,ypred)) 