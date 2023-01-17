import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('../input/real-estate-dataset/data.csv')

df
CRIM = df.iloc[:, 0]

INDUS= df.iloc[:, 1]

plt.scatter(CRIM, INDUS)

plt.show()
m = 0

c = 0

L = 0.0001  

epochs = 1000  

n = float(len(CRIM)) 
for i in range(epochs): 

    Y_pred = m*CRIM + c  

    D_m = (-2/n) * sum(CRIM * (INDUS - Y_pred))  

    D_c = (-2/n) * sum(INDUS - Y_pred) 

    m = m - L * D_m  

    c = c - L * D_c  

    

print (m, c)
print (m, c)
Y_pred = m*CRIM + c



plt.scatter(CRIM,INDUS)

plt.plot([min(CRIM), max(CRIM)], [min(Y_pred), max(Y_pred)], color='red') 

plt.show()