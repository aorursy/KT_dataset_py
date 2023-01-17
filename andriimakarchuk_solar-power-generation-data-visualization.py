import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
dataGen = pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv").dropna()

dataGen.head()
dataGen = dataGen[["DATE_TIME", "DC_POWER", "DAILY_YIELD"]].groupby("DATE_TIME").mean()

dataGen.head()
dataGen.corr()
dataSensor = pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv").dropna()

dataSensor.head()
dataSensor = dataSensor[["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]

dataSensor.head()
dataSensor.corr()
data = pd.DataFrame(index=dataGen.index, columns=["DATE_TIME", "DC_POWER", "DAILY_YIELD", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"])

data.head()
for i in range(len(dataGen)):

    data["DATE_TIME"][i]           = i

    data["DC_POWER"][i]            = dataGen["DC_POWER"][i]

    data["DAILY_YIELD"][i]         = dataGen["DAILY_YIELD"][i]

    data["AMBIENT_TEMPERATURE"][i] = dataSensor["AMBIENT_TEMPERATURE"][i]

    data["MODULE_TEMPERATURE"][i]  = dataSensor["MODULE_TEMPERATURE"][i]

    data["IRRADIATION"][i]         = dataSensor["IRRADIATION"][i]

data.head()
plt.plot(data["DATE_TIME"], data["DC_POWER"])

plt.xlabel("Registration number")

plt.ylabel("DC power")

plt.show()
plt.plot(data["DATE_TIME"], data["DAILY_YIELD"])

plt.xlabel("Registration number")

plt.ylabel("Daily yield")

plt.show()
plt.plot(data["DATE_TIME"], data["AMBIENT_TEMPERATURE"])

plt.xlabel("Registration number")

plt.ylabel("Ambient temperature")

plt.show()
plt.plot(data["DATE_TIME"], data["MODULE_TEMPERATURE"])

plt.xlabel("Registration number")

plt.ylabel("Module temperature")

plt.show()
plt.plot(data["DATE_TIME"], data["IRRADIATION"])

plt.xlabel("Registration number")

plt.ylabel("Irradiation")

plt.show()
dataGen = pd.read_csv("../input/solar-power-generation-data/Plant_2_Generation_Data.csv").dropna()

dataGen.head()
dataGen = dataGen[["DATE_TIME", "DC_POWER", "DAILY_YIELD"]].groupby("DATE_TIME").mean()

dataGen.head()
dataGen.corr()
dataSensor = pd.read_csv("../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv").dropna()

dataSensor.head()
dataSensor = dataSensor[["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]

dataSensor.head()
dataSensor.corr()
data = pd.DataFrame(index=dataGen.index, columns=["DATE_TIME", "DC_POWER", "DAILY_YIELD", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"])

data.head()
for i in range(len(dataGen)):

    data["DATE_TIME"][i]           = i

    data["DC_POWER"][i]            = dataGen["DC_POWER"][i]

    data["DAILY_YIELD"][i]         = dataGen["DAILY_YIELD"][i]

    data["AMBIENT_TEMPERATURE"][i] = dataSensor["AMBIENT_TEMPERATURE"][i]

    data["MODULE_TEMPERATURE"][i]  = dataSensor["MODULE_TEMPERATURE"][i]

    data["IRRADIATION"][i]         = dataSensor["IRRADIATION"][i]

data.head()
plt.plot(data["DATE_TIME"], data["DC_POWER"])

plt.xlabel("Registration number")

plt.ylabel("DC power")

plt.show()
plt.plot(data["DATE_TIME"], data["DAILY_YIELD"])

plt.xlabel("Registration number")

plt.ylabel("Daily yield")

plt.show()
plt.plot(data["DATE_TIME"], data["AMBIENT_TEMPERATURE"])

plt.xlabel("Registration number")

plt.ylabel("Ambient temperature")

plt.show()
plt.plot(data["DATE_TIME"], data["MODULE_TEMPERATURE"])

plt.xlabel("Registration number")

plt.ylabel("Module temperature")

plt.show()
plt.plot(data["DATE_TIME"], data["IRRADIATION"])

plt.xlabel("Registration number")

plt.ylabel("Irradiation")

plt.show()