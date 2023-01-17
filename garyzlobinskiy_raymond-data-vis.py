import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
filepath = "../input/data-for-datavis/fifa.csv"
data1 = pd.read_csv(filepath, index_col="Date", parse_dates=True)
data1.head()
plt.figure(figsize=(16,6))
sns.lineplot(data=data1)
spotifyFilepath = "../input/data-for-datavis/spotify.csv"
spotifyData = pd.read_csv(spotifyFilepath, index_col='Date', parse_dates=True)
spotifyData.head(50)
plt.figure(figsize=(14,6))
plt.title("Daily Streams of 5 Songs Over the Span of 11 Years")
plt.xlabel("Date")
plt.ylabel("Steams")
sns.lineplot(data=spotifyData['Shape of You'], label="Shape of You")
sns.lineplot(data=spotifyData['Despacito'], label="Despacito")
list(spotifyData.columns)
spotifyData['Shape of You'].head()
fdData = pd.read_csv("../input/data-for-datavis/flight_delays.csv", index_col='Month')
fdData.head(50)
# 3,2,5,7,7,3,2,5,7,7,8,13

# 2,2,3,3,5,5,7,7,7,7,8,13

# 2+2+3+3+5+5+7+7+7+7+8+13=69

# 69/12 = 5.75

# Average (Mean) - 5.75
# Median - 6
# Mode - 7
# Range - 11
fdData.index
fdData.tail(10)
plt.figure(figsize=(10,6))
plt.title("Average Delay for Spirit Flights by Month")
sns.barplot(x=fdData.index,y=fdData['NK'])
plt.figure(figsize=(14,7))
plt.title("Average Delay time for all Airlines by Month")
sns.heatmap(data=fdData, annot=True)
plt.xlabel("Airline Name")
iData = pd.read_csv("../input/data-for-datavis/insurance.csv")
iData.head(50)
plt.figure(figsize=(10,10))
sns.scatterplot(x=iData['bmi'], y=iData['charges'])
plt.figure(figsize=(10,10))
sns.regplot(x=iData['bmi'], y=iData['charges'])
plt.figure(figsize=(10,10))
sns.scatterplot(x=iData['bmi'], y=iData['charges'], hue=iData['smoker'])
sns.lmplot(x='bmi', y='charges', hue='smoker', data=iData)
sns.swarmplot(x=iData['smoker'], y=iData['charges'])
irisData = pd.read_csv("../input/data-for-datavis/iris.csv", index_col="Id")
irisData.head(50)
sns.distplot(a=irisData['Petal Length (cm)'], kde=False)
sns.distplot(a=irisData['Petal Length (cm)'], kde=True)
sns.kdeplot(data=irisData['Petal Length (cm)'], shade=True)
sns.jointplot(x=irisData['Petal Length (cm)'], y=irisData['Sepal Width (cm)'], kind='kde')
irisSetosa = pd.read_csv("../input/data-for-datavis/iris_setosa.csv", index_col="Id")
irisVersi = pd.read_csv("../input/data-for-datavis/iris_versicolor.csv", index_col="Id")
irisVirginica = pd.read_csv("../input/data-for-datavis/iris_virginica.csv", index_col="Id")
sns.kdeplot(data=irisSetosa['Petal Length (cm)'], label="Setosa", shade=True)
sns.kdeplot(data=irisVersi['Petal Length (cm)'], label="Versicolor", shade=True)
sns.kdeplot(data=irisVirginica['Petal Length (cm)'], label="Virginica", shade=True)
sns.set_style("ticks")
plt.legend()