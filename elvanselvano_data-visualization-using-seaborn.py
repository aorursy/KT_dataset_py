import pandas as pd # pandas <- matplotlib and numpy
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns # seaborn <- matplotlib
print("Setup complete")
# Path of the file to read
fifa_filepath = "../input/data-for-datavis/fifa.csv"

# Read the file into a variable fifa_data
fifa_data = pd.read_csv(fifa_filepath, index_col = "Date", parse_dates = True)
# Print the first 5 rows of the data
fifa_data.head()
# Set the width and height of the figure
plt.figure(figsize = (16, 6))

# Line chart showing how FIFA rankings evolved over time
sns.lineplot(data = fifa_data)
# Path of the file to read
spotify_filepath = "../input/data-for-datavis/spotify.csv"

# Read the file into a variable spotify_data
spotify_data = pd.read_csv(spotify_filepath, index_col = "Date", parse_dates = True)
# Print the first 5 rows of the data
spotify_data.head()
# Print the last five rows of the data
spotify_data.tail()
# Line chart showing daily global streams of each song
sns.lineplot(data = spotify_data)
# Change the style of the figure to the "dark" theme
sns.set_style("whitegrid")

# Set the width and height of the figure
plt.figure(figsize = (16, 6))

# Add title
plt.title("Daily Global Streams of Popular Songs in 2017 - 2018")

# Line chart showing daily global streams of each song
sns.lineplot(data = spotify_data)
list(spotify_data.columns)
# Set the width and height of the figure
plt.figure(figsize = (14, 6))

# Add title
plt.title("Daily Global Streams of Popular Songs in 2017 - 2018")

# Line chart showing daily global streams of "Shape of You"
sns.lineplot(data = spotify_data["Shape of You"], label = "Shape of You")

# Line chart showing daily global streams of "Despacito"
sns.lineplot(data = spotify_data["Despacito"], label = "Despacito")

# Add label for horizontal axis
plt.xlabel("Date")


# Path of the file to read
museum_filepath = "../input/data-for-datavis/museum_visitors.csv"

# Fill in the line below to read the file into a variable museum_data
museum_data = pd.read_csv(museum_filepath, index_col = "Date", parse_dates = True)
# Print the last give rows of the data
print(museum_data.tail())
# Line chart showing the number of visitors to each museum over time
sns.lineplot(data = museum_data)
# Set the width and height of the figure
plt.figure(figsize = (12, 6))

# Line plot showing the number of visitors to Avila Adobe over time
sns.lineplot(data = museum_data["Avila Adobe"])
# Path of the file to read
flight_filepath = "../input/data-for-datavis/flight_delays.csv"

# Read the file into a variable flight_data
flight_data = pd.read_csv(flight_filepath, index_col = "Month")
flight_data
# Set the width and height of the figure
plt.figure(figsize = (10, 6))

# Add title
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")

# Bar chart showing average delay for Spirit Airlines Flight by month
sns.barplot(x = flight_data.index, y = flight_data["NK"])

# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")
# Set the width and height of the figure
plt.figure(figsize = (14, 7))

# Add title
plt.title("Average Arrival Delay for Each Airline, by Month")

# Bar chart showing average delay for Spirit Airlines Flight by month
sns.heatmap(data = flight_data, annot = True)

# Add label for horizontal axis
plt.xlabel("Airline")
# Path of the file to read
ign_filepath = "../input/data-for-datavis/ign_scores.csv"

# Fill in the line below to read the file into a variable ign_data
ign_data = pd.read_csv(ign_filepath, index_col = "Platform")
# Print the data
ign_data
ign_data.describe()
# Set the width and height of the figure
plt.figure(figsize = (8, 6))

# Add title
plt.title("Average Score for Racing Games, by Platform")

# Bar chart showing average delay for Spirit Airlines Flight by month
sns.barplot(x = ign_data["Racing"], y = ign_data.index)

# Add label for horizontal axis
plt.xlabel("Rating")
# Set the width and height of the figure
plt.figure(figsize = (10, 10))

# Add title
plt.title("Average Game Score, by Platform and Genre")

# Bar chart showing average delay for Spirit Airlines Flight by month
sns.heatmap(data = ign_data, annot = True)

# Add label for horizontal axis
plt.xlabel("Genre")
# Path of the file to read
insurance_filepath = "../input/data-for-datavis/insurance.csv"

# Read the file into a variable insurance_data
insurance_data = pd.read_csv(insurance_filepath)
# Read the first 5 rows
insurance_data.head()
# Generate scatter plot based on bmi and charges
sns.scatterplot(x = insurance_data["bmi"], y = insurance_data["charges"])
# Generate regression plot
sns.regplot(x = insurance_data["bmi"], y = insurance_data["charges"])
# Generate scatter plot with hue
sns.scatterplot(x = insurance_data["bmi"], y = insurance_data["charges"], hue = insurance_data["smoker"])
# Generate regression plot with hue
sns.lmplot(x = "bmi", y = "charges", hue = "smoker", data = insurance_data)
# Generate swarm plot
sns.swarmplot(x = insurance_data["smoker"], y = insurance_data["charges"])
# Path of the file to read
iris_filepath = "../input/data-for-datavis/iris.csv"

# Read the file into a variable iris_data
iris_data = pd.read_csv(iris_filepath, index_col = "Id")

# Print the first 5 rows of the data
iris_data.head()
# Generate histogram
sns.distplot(a = iris_data["Petal Length (cm)"], kde = False)
# KDE Plot
sns.kdeplot(data = iris_data["Petal Length (cm)"], shade = True)
# 2D KDE Plot
sns.jointplot(x = iris_data["Petal Length (cm)"], y = iris_data["Sepal Width (cm)"], kind = "kde")
# Paths of the files to read
iris_set_filepath = "../input/data-for-datavis/iris_setosa.csv"
iris_ver_filepath = "../input/data-for-datavis/iris_versicolor.csv"
iris_vir_filepath = "../input/data-for-datavis/iris_virginica.csv"

# Read the files into variables
iris_set_data = pd.read_csv(iris_set_filepath, index_col = "Id")
iris_ver_data = pd.read_csv(iris_ver_filepath, index_col = "Id")
iris_vir_data = pd.read_csv(iris_vir_filepath, index_col = "Id")

# Print the first 5 rows of the Iris versicolor data
iris_ver_data.head()
# Histograms for each species
sns.distplot(a = iris_set_data["Petal Length (cm)"], label = "Iris-setosa", kde = False)
sns.distplot(a = iris_ver_data["Petal Length (cm)"], label = "Iris-versicolor", kde = False)
sns.distplot(a = iris_vir_data["Petal Length (cm)"], label = "Iris-virginica", kde = False)

# Add title
plt.title("Histogram of Petal Lengths, by Species")

# Force legend to appear
plt.legend()
# KDE Plots for each species
sns.kdeplot(data = iris_set_data["Petal Length (cm)"], label = "Iris-setosa", shade = True)
sns.kdeplot(data = iris_ver_data["Petal Length (cm)"], label = "Iris-versicolor", shade = True)
sns.kdeplot(data = iris_vir_data["Petal Length (cm)"], label = "Iris-virginica", shade = True)

# Add title
plt.title("Distribution of Petal Lengths, by Species")