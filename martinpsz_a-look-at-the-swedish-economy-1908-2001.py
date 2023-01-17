import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
dat = pd.read_csv("../input/Interestrate and inflation Sweden 1908-2001.csv")
dat.head()
#Rename the interest rate variable:

dat.rename(columns=lambda x: x.replace('Central bank interest rate diskonto average', 

                                       'Interest rate'), inplace=True)
#Replaces spaces with _ in the column names:

dat.columns = dat.columns.str.replace(" ", "_")
#we are missing values for each of the 4 variables:

dat.info()
#It appears that data includes from metadata and information that is 

#unimportant to this analysis. I will drop these rows in the next cell.

dat.loc[94: , :].head()
#dropping the above rows from the dataset:

dat.drop(dat.loc[94: , :].index, inplace=True)
#We now that a complete data set with our values of interest from 1908-2001.

dat.info()
#I notice that Interest_rate and price level are stored as objects rather than

#a float and int, respectively. Also, the interest rate values run to 4 

#decimal places, I will round that down to 2. So I make these alterations here:



dat.Interest_rate = round(dat.Interest_rate.astype("float64"), 2)

dat.Price_level = dat.Price_level.str.replace(",", "")

dat.Price_level = dat.Price_level.astype("float64")

dat.Period = dat.Period.astype("int64")
dat.head()
fig, ax = plt.subplots(3, figsize=(6,10))

fig.subplots_adjust(hspace=0.25)

ax[0].plot(dat.Period, dat.Interest_rate)

ax[0].set_title("Swedish Interest rates (1908-2001)")

ax[0].set_ylabel("Interest rate")

ax[0].set_xlim([1905, 2005])

ax[0].set_axis_bgcolor("white")

ax[1].plot(dat.Period, dat.Inflation)

ax[1].set_title("Swedish Inflation rates (1908-2001)")

ax[1].set_ylabel("Inflation rate")

ax[1].set_xlim([1905, 2005])

ax[1].set_axis_bgcolor("white")

ax[2].plot(dat.Period, dat.Price_level)

ax[2].set_title("Swedish Price levels (1908-2001)")

ax[2].set_ylabel("Price level")

ax[2].set_xlim([1905, 2005])

ax[2].set_axis_bgcolor("white");