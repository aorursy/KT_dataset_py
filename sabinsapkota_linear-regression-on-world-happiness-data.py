#loads numpy and pandas for data manipulation
import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

df=pd.read_csv('../input/2015.csv')
df.head()

df.info()
df.describe()

#Renames all the attributes/columns
df.columns = ["Country","Region","Rank","Score","StandardError","GDP","Family",
                "Health","Freedom","Trust","Generosity","Dystopia"]
df.head()

y=df.Score #response
x=df.GDP #predictor
x=sm.add_constant(x) #Adds a constant term
x.head()

est=sm.OLS(y,x)
est=est.fit()
est.summary()

est.params

%pylab inline

#picks 100 points equally from min to max
x_prime=np.linspace(x.GDP.min(), x.GDP.max(),100)[:, np.newaxis]
x_prime=sm.add_constant(x_prime) #add constant

#calculates the predicted values
y_hat=est.predict(x_prime)

plt.scatter(x.GDP,y,alpha=0.3) #plots the raw data
plt.xlabel("Gross Domestic Product Per Capita")
plt.ylabel("Happiness Score")
plt.plot(x_prime[:,1],y_hat,'r',alpha=0.9) #Adds the regression colored in red

#formula: response ~ predictors
est=smf.ols(formula='Score~GDP',data=df).fit()
est.summary()
