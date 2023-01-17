#import the libraries
import numpy as np 
import pandas as pd 

#Python library for Vidualization
import seaborn as sns  
import matplotlib.pyplot as plt

#For modeling
from sklearn import linear_model
from regressors import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
#Import the dataset
dataset = pd.read_csv('../input/Mall_Customers.csv')
dataset.head(10) #Printing first 10 rows of the dataset
#Rename the data
dataset.columns=['CustomerID','Gender','Age','Annual_Income','Spending_Score']
dataset.dtypes
mod = smf.ols(formula='Spending_Score ~ Annual_Income + Age +C(Gender)', data=dataset)
res = mod.fit()
print(res.summary())
mod1 = smf.ols(formula='Spending_Score ~ Age' , data=dataset)
res1 = mod1.fit()
print(res1.summary())
sns.lmplot(x="Age", y="Spending_Score", data=dataset)
