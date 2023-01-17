##Is there a relationship between the daily minimum and maximum temperature? 
##Can we predict the maximum temperature given the minimum temperature?
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('../input/Summary of Weather.csv')
print(df.head())
lr = LinearRegression()
x=df.MinTemp.values
x = x.reshape(-1,1)
y = df.MaxTemp.values.reshape(-1,1)
lr.fit(x,y)
X = np.array([10,20,30,40,50]).reshape(-1,1)
for i in X:
    print("Min:",i,"Predcted max:",lr.predict([i]));

y = lr.predict(X)
plt.scatter(X,y,color="red")
plt.show()