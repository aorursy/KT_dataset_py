# Loading the packages

import numpy as np

import pandas as pd 
df_train = pd.read_csv("../input/train.csv")

df_train.head()
target = df_train["target"]

df_train.pop("target");

df_train.pop("id");
df_train.shape
target.value_counts()
print((df_train.dtypes).head())
print((df_train.dtypes).tail())
types_columns = df_train.dtypes

counter = 0 



for x in types_columns:

    if x == "float64":

        counter += 1 

print(counter) # if counter is 300 then all the variables are float
for x in df_train.columns:

    print("Minimum {}, Maximum {}, Mean {}, Standard Deviation {}".format(df_train[x].min(), df_train[x].max(), df_train[x].mean(), df_train[x].std() ))
import matplotlib.pyplot as plt 

plt.hist(df_train["0"])

plt.show()
import matplotlib.pyplot as plt 

plt.hist(df_train["1"])

plt.show()
import matplotlib.pyplot as plt 

plt.hist(df_train["2"])

plt.show()
import matplotlib.pyplot as plt 

plt.hist(df_train["3"])

plt.show()
import matplotlib.pyplot as plt 

plt.hist(df_train["4"])

plt.show()
import matplotlib.pyplot as plt 

plt.hist(df_train["5"])

plt.show()