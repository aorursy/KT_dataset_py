import pandas as pd
import numpy as np
from sklearn import linear_model
# initialize list of lists 
data = [[2600, 550000], [3000, 565000], [3200, 610000],[3200, 550000],[3600, 680000],[4000, 550000],[4400, 725000]] 
  
# Create the pandas DataFrame 
df = pd.DataFrame(data, columns = ['area', 'price']) 
  
# print dataframe. 
df
#df = pd.read_csv("homeprices.csv")
df.head()
model = linear_model.LinearRegression()
model.fit(df[['area']],df.price)
model.coef_
model.intercept_
model.predict([[5000]])
import pickle
with open('model_pickle','wb') as file:
    pickle.dump(model,file)
with open('model_pickle','rb') as file:
    mp = pickle.load(file)
mp.coef_
mp.intercept_
mp.predict([[5000]])
#!pip install joblib
!pip install sklearn.externals
import sklearn.external.joblib as extjoblib
import joblib

from sklearn.externals import joblib
joblib.dump(model, 'model_joblib')
mj = joblib.load('model_joblib')
mj.coef_
mj.intercept_
mj.predict([[5000]])
