import numpy as np

import pandas as pd

import matplotlib.pyplot as mp

import statsmodels.api as sm

import seaborn

seaborn.set()
raw_data=pd.read_csv('../input/Fifa19.csv',nrows=200)

raw_data#Contains all data
y=raw_data['Wage']

x1=raw_data['Overall']

mp.scatter(y,x1)

mp.show()
#Lets apply linear regression to see how apt the result can be

x=sm.add_constant(x1)

result=sm.OLS(y,x).fit()

result.summary()
new_data=pd.DataFrame({'const':1, 'Overall':[90,94]})

new_data
prediction=result.predict(new_data)

prediction
#Clearly, this is not working