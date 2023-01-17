## Data analysis of "Gender Equality Index" on countries of European Union (Data from 2017)

## Study done by Özgür Can Arıcan
## Accessed to the data set from "EU Open Data Portal"
## link to the data set: http://data.europa.eu/88u/dataset/gender-equality-index
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#imported required libraries
#some rows shows the sources of the data, skipped them while importing into notebook.

df = pd.read_excel("../input/gender-equality-index-2017-eu/gender-equality-index-2017.xls", skiprows = [0, 31, 32, 33, 34, 35, 36])
df.head()
df.tail()
df.describe()
df.info()

#There is no any NA value in dataframe. So, no need to use dropna, fillna etc. for cleaning.
#Find the highest GEI and the country that have the highest GEI.

df.loc[df["Gender Equality Index"].idxmax][["Gender Equality Index", "Country"]]
#Find the lowest GEI and the country that have the lowest GEI.

df.loc[df["Gender Equality Index"].idxmin][["Gender Equality Index", "Country"]]
#Demonstrate the top 5 countries that have the highest GEI in bar plot

highestFive = df.set_index("Country")["Gender Equality Index"].sort_values(ascending = False).head(5)
highestFive.plot.bar()
#Demonstrate the top 5 countries that have the lowest GEI in bar plot

lowestFive = df.set_index("Country")["Gender Equality Index"].sort_values(ascending = True).head(5)
lowestFive.plot.bar()
#Demonstrate GEIs of all countries from highest to lowest in horizantal bar plot

allCountries = df.set_index("Country")["Gender Equality Index"].sort_values(ascending = True)
allCountries.plot.barh(figsize = (7, 7))
#Demonstrate:
#1) Full-time equivalent employment rate of women
#2) Mean monthly earnings of women
#3) Graduates of tertiary educations (women)
#4) Share of members of parliament (women)
#in four different bar plots in one figure.
fig, axs = plt.subplots(2, 2, figsize = (20, 7))

axs[0, 0].bar(df["Country"], df["Full-time equivalent employment rate (%) W"], color = "blue")
axs[0, 1].bar(df["Country"], df["Mean monthly earnings (PPS) W"], color = "red")
axs[1, 0].bar(df["Country"], df["Graduates of tertiary education (%) W"], color = "yellow")
axs[1, 1].bar(df["Country"], df["Share of members of parliament (%) W"], color = "green")
#Search for the correlation between GEI and full-time equivalent employment rate, mean monthly earnings,
#graduates of tertiary educations, share of members of parliament.
#Find which feature is most correlated with GEI.
df.plot.scatter(x = "Full-time equivalent employment rate (%) W", y = "Gender Equality Index", c = "blue")
np.corrcoef(df["Full-time equivalent employment rate (%) W"], df["Gender Equality Index"])
df.plot.scatter(x = "Mean monthly earnings (PPS) W", y = "Gender Equality Index", c = "red")
np.corrcoef(df["Mean monthly earnings (PPS) W"], df["Gender Equality Index"])
df.plot.scatter(x = "Graduates of tertiary education (%) W", y = "Gender Equality Index", c = "yellow")
np.corrcoef(df["Graduates of tertiary education (%) W"], df["Gender Equality Index"])
df.plot.scatter(x = "Share of members of parliament (%) W", y = "Gender Equality Index", c = "green")
np.corrcoef(df["Share of members of parliament (%) W"], df["Gender Equality Index"])
#At the end, the correlation coefficients of;
#1) employment rate -----------> 0.143
#2) monthly earnings ----------> 0.788
#3) education -----------------> 0.510
#4) be a member of parliament--> 0.795
#Altought, being a member of the parliament and the mothly earnings shows a strong correlation with GEI,
#the high education level shows a weaker correlation and employment rate shows "nearly" no correlation.
