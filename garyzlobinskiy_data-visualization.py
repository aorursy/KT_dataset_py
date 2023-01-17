import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
filepath = "../input/data-for-datavis/fifa.csv"
data = pd.read_csv(filepath,index_col="Date",parse_dates=True)
data.head()
plt.figure(figsize=(16,6))
sns.lineplot(data=data["BRA"])
spotifyFilepath = "../input/data-for-datavis/spotify.csv"
spotifyData = pd.read_csv(spotifyFilepath, index_col="Date",parse_dates=True)
spotifyData.head()
plt.figure(figsize=(14,6))
plt.title("Daily streams of 5 songs")
plt.xlabel("Date")
plt.ylabel("Streams in 10's of millions")
sns.lineplot(data=spotifyData['Despacito'], label="Despacito")
spotifyData.head()
spotifyData.tail()
list(spotifyData.columns)
spotifyData["Despacito"].head()
flightDelaysFilepath = "../input/data-for-datavis/flight_delays.csv"
flightDelaysData = pd.read_csv(flightDelaysFilepath, index_col="Month")
plt.figure(figsize=(10,6))
plt.title("Fifa Heatmap")
# sns.barplot(x=flightDelaysData.index,y=flightDelaysData['NK'])
sns.heatmap(data=data, annot=False)
flightDelaysData.head()
#flightDelaysData.NK.hist(bins=20)
insuranceFilepath = "../input/data-for-datavis/insurance.csv"
insuranceData = pd.read_csv(insuranceFilepath)
insuranceData.head()
sns.scatterplot(x=insuranceData['bmi'], y=insuranceData['charges'])
sns.regplot(x=insuranceData['bmi'], y=insuranceData['charges'])
sns.scatterplot(x=insuranceData['bmi'], y=insuranceData['charges'], hue=insuranceData['smoker'], size=insuranceData['age'], style=insuranceData['children'])
sns.scatterplot(x=data.index, y=data['BRA'])
sns.lmplot(x="bmi", y="charges", hue="children", data=insuranceData)
sns.swarmplot(x=insuranceData['smoker'], y=insuranceData["charges"])
irisFilepath = "../input/data-for-datavis/iris.csv"
irisData = pd.read_csv(irisFilepath, index_col="Id")
irisData.head()
sns.distplot(a=irisData["Petal Length (cm)"], kde=False)
# kde = Kernel density estimate