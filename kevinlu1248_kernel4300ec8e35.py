import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
quantity_kg = pd.read_csv("../input/covid19-healthy-diet-dataset/Food_Supply_Quantity_kg_Data.csv")



# Lets see what our data looks like

quantity_kg
# Lets add a few more columns

quantity_kg["Mortality"] = quantity_kg["Deaths"] / quantity_kg["Confirmed"] * 100

quantity_kg["Recovery Rate"] = quantity_kg["Recovered"] / quantity_kg["Confirmed"] * 100 # change recovery to percentage

quantity_kg.dropna(inplace=True)

quantity_kg
quantity_kg.columns
import matplotlib.pyplot as plt

from numpy import cov

for index in quantity_kg.columns[1:25]:

    plt.plot(quantity_kg[index], quantity_kg["Mortality"], 'ro', label="Mortality Rate")

    plt.plot(quantity_kg[index], quantity_kg["Recovery Rate"], 'go', label="Recovery Rate")

    plt.xlabel("{} consumption (%)".format(index))

    plt.ylabel('Rates')

    plt.show()
for index in quantity_kg.columns[1:25]:

    plt.plot(quantity_kg[index], quantity_kg["Mortality"], 'ro', label="Mortality Rate")

    m, b = np.polyfit(quantity_kg[index], quantity_kg["Mortality"], 1)

    plt.plot(quantity_kg[index], m * quantity_kg[index]  + b)

    plt.xlabel("{} consumption (%)".format(index))

    plt.ylabel('Rates')

    plt.show()
kcal = pd.read_csv("../input/covid19-healthy-diet-dataset/Food_Supply_kcal_Data.csv")

kcal["Mortality"] = kcal["Deaths"] / kcal["Confirmed"] * 100

kcal["Recovery Rate"] = kcal["Recovered"] / kcal["Confirmed"] * 100 # change recovery to percentage

for index in kcal.columns[1:25]:

    plt.plot(kcal[index], kcal["Mortality"], 'ro', label="Mortality Rate")

    plt.plot(kcal[index], kcal["Recovery Rate"], 'go', label="Recovery Rate")

    plt.xlabel("{} consumption (kcal)".format(index))

    plt.ylabel('Rates')

    plt.show()
for index in kcal.columns[1:25]:

    plt.plot(kcal[index], kcal["Mortality"], 'ro', label="Mortality Rate")

    plt.xlabel("{} consumption (kcal)".format(index))

    plt.ylabel('Rates')

    plt.show()
corr = quantity_kg.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(quantity_kg.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(quantity_kg.columns)

ax.set_yticklabels(quantity_kg.columns)

plt.show()
correlation_rates = pd.Series(data=np.nan, index=quantity_kg.columns[1:24])

for index in quantity_kg.columns[1:24]:

    # could have also used list comprehension here but its a bit ugly

    m, b = np.polyfit(quantity_kg[index], quantity_kg["Mortality"], 1)

    correlation_rates[index] = m

correlation_rates = correlation_rates.sort_values(ascending=False)

correlation_rates.head(10)
correlation_rates.tail(10)