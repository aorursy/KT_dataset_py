import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import os
print(os.listdir("../input"))
data = pd.read_csv("../input/column_2C_weka.csv")
data.head()
data.info()
data.describe()
data.corr()
f,axis = plt.subplots(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True, linewidths=.4, fmt= '.2f', ax = axis)
plt.show()
linear_reg = LinearRegression()

reshaped_x = data.pelvic_incidence.values.reshape(-1,1)
reshaped_y = data.sacral_slope.values.reshape(-1,1)

linear_reg.fit(reshaped_x,reshaped_y)

y_predicted = linear_reg.predict(reshaped_x)
plt.figure(figsize = (12,10))
plt.scatter(reshaped_x,reshaped_y,color='blue')
plt.plot(reshaped_x,y_predicted,color='red')
plt.show()
print(r2_score(y_predicted, reshaped_y))
data