# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set()
data = {'Size':[6,8,12,14,18],'Price':[350,775,1150,1395,1675]}
df = pd.DataFrame(data)
df
sns.lmplot(data=df, x='Size',y='Price')
df['Size_Price'] = df.Size*df.Price
df['Size_Squared'] = df.Size ** 2
Size_bar = np.sum(df.Size) / len(df.Size)
Price_bar = np.sum(df.Price) / len(df.Price)
Size_Price_bar = np.sum(df.Size_Price) / len(df.Size_Price)
Size_Squared_bar = np.sum(df.Size_Squared) / len(df.Size_Squared)
print(Size_bar, Price_bar,Size_Price_bar, Size_Squared_bar)
m = ((Size_bar * Price_bar) - (Size_Price_bar)) / ((Size_bar ** 2 ) - (Size_Squared_bar))
m
c = Price_bar - m * Size_bar
c
def y(m,x,c):
    return m * x + c
print(y(m, 17, c))
print(y(m,16,c))
R_value = []
for i in range(df.shape[0]):
    R_value.append(y(m,df.Size[i], c))
print(R_value)
df['R_value'] = R_value
def R_Squared_Value(y, y_bar, y_cap):
    upper = []
    lower = []
    for i in range(y_cap.shape[0]):
        upper.append((y_cap[i] - y_bar) ** 2)
        lower.append((y[i] - y_bar) ** 2)
    return np.sum(upper) / np.sum(lower), upper, lower
R_square = R_Squared_Value(df.Price, Price_bar, df.R_value)
print(r"R^2 Value is {0:.2f}% and the Upper value is {1} and Lower Value is {2}".format(R_square[0], R_square[1],R_square[2]))
#print(R_Squared_Value(df.Price, Price_bar, df.R_value)[0])
