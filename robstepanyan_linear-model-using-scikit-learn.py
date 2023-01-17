import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import sklearn.linear_model



import os

print(os.listdir("../input"))
gdp = pd.read_excel("../input/gdp-per-capita-19902017/GDP(PPP) Per capita 1990-2017.xlsx")

gdp.head()
lss = pd.read_excel("../input/life-satisfaction-score-20152017/Life Satisfaction (2015-2017).xlsx")

lss.columns = lss.iloc[0]

lss = lss.iloc[1:].reset_index(drop=True)

lss.head()
# Filtering  Data from 2015-2017

gdp = gdp[["Country Name", 2015, 2016, 2017]]

# Getting average value

gdp["2015-2017"] = gdp[[2015, 2016, 2017]].mean(axis=1)

# Deleting rows: 2015, 2016, 2017

gdp = gdp[["Country Name", "2015-2017"]]



gdp.head()
gdp_for_lss = []

for country in lss["Country"]:

    gdp_for_lss.append(list(gdp[gdp["Country Name"] == country]["2015-2017"]))



# Empty values -> None (to avoid future problems in DataFrames)

for i in range(len(gdp_for_lss)):

    if gdp_for_lss[i] == []:

        gdp_for_lss[i] = None

    else:

        gdp_for_lss[i] = round(gdp_for_lss[i][0], 2)



# First 5 values

gdp_for_lss[:5]
gdp_lss = pd.DataFrame({"Country": lss["Country"], 

                        "LSS": lss["Life Satisfaction AVG Score"],

                       "GDP(PPP)": gdp_for_lss})



# Deleting rows containing empty values in GDP(PPP) column

gdp_lss = gdp_lss[np.isfinite(gdp_lss["GDP(PPP)"])].reset_index(drop=True)



# Saving this dataframe to share with other kaggle users

gdp_lss.to_csv("gdp_lss.csv")



# First 5 values of our dataframe

gdp_lss.head()
plt.figure(dpi = 200)

plt.scatter(x = gdp_lss["GDP(PPP)"], y = gdp_lss["LSS"])

plt.xlabel("GDP(PPP) per capita")

plt.ylabel("Life Satisfaction Score")
# France's index in our dataframe (df) is 8, GDP is 42269.59, LSS is 6.5

gdp_lss = gdp_lss.drop(8).reset_index(drop=True)

# Printing first 10 values

gdp_lss[:10]
x = pd.DataFrame(gdp_lss["GDP(PPP)"])

y = pd.DataFrame(gdp_lss["LSS"])
model = sklearn.linear_model.LinearRegression()
model.fit(x, y)
x_new = [[42269.59]] # France's GDP Per capita 

print("Predicted LSS: ", model.predict(x_new))