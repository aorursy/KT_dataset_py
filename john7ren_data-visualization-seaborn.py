import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
fifa_filepath = "../input/fifa.csv"

spotify_filepath = "../input/spotify.csv"

flight_filepath = "../input/flight_delays.csv"

insurance_filepath = "../input/insurance.csv"

iris_filepath = "../input/iris.csv"



fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)

spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)

flight_data = pd.read_csv(flight_filepath, index_col="Month") # Since the labels don't correspond to dates, we remove 'parse_dates'

insurance_data = pd.read_csv(insurance_filepath)

iris_data = pd.read_csv(iris_filepath, index_col="Id")
list(spotify_data.columns)
# Plot of entire dataset.

plt.figure(figsize=(14,6))

plt.title("Daily Global Streams of Popular Songs in 2017-2018")



sns.lineplot(data=spotify_data)
# Plot comparing two columns in the dataset.

plt.figure(figsize=(14,6))

plt.title("Daily Global Streams of Popular Songs in 2017-2018")



# Line chart showing daily global streams of 'Shape of You'

sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")



# Line chart showing daily global streams of 'Despacito'

sns.lineplot(data=spotify_data['Despacito'], label="Despacito")



plt.xlabel("Date")

plt.ylabel("Views (in millions)")
# Bar chart showing the average arrival delay for Spirit Airlines (airline code: NK).

plt.figure(figsize=(10,6))

plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=flight_data.index, y=flight_data['NK'])



plt.ylabel("Arrival delay (in minutes)")
# Heatmap

plt.figure(figsize=(14,7))

plt.title("Average Arrival Delay for Each Airline, by Month")



# Heatmap showing average arrival delay for each airline by month

sns.heatmap(data=flight_data, annot=True) # 'annot' values/numbers for each cell appears on the chart



plt.xlabel("Airline")
# Scatter plot (customers with higher BMI typically also tend to pay more in insurance costs)

sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
# Color-coded scatter plot (to display the relationships between not two, but... three variables!)

sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
# categorical scatter plot

sns.swarmplot(x=insurance_data['smoker'],

              y=insurance_data['charges'])
iris_set_filepath = "../input/iris_setosa.csv"

iris_ver_filepath = "../input/iris_versicolor.csv"

iris_vir_filepath = "../input/iris_virginica.csv"



iris_set_data = pd.read_csv(iris_set_filepath, index_col="Id")

iris_ver_data = pd.read_csv(iris_ver_filepath, index_col="Id")

iris_vir_data = pd.read_csv(iris_vir_filepath, index_col="Id")
# Histograms for each species

sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris-setosa", kde=False)

sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", kde=False)

sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", kde=False)



plt.title("Histogram of Petal Lengths, by Species")



# Force legend to appear

plt.legend()
# KDE plots for each species

sns.kdeplot(data=iris_set_data['Petal Length (cm)'], label="Iris-setosa", shade=True)

sns.kdeplot(data=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", shade=True)

sns.kdeplot(data=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", shade=True)



plt.title("Distribution of Petal Lengths, by Species")