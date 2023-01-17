import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("setup complete")
# Path of the file to read

fifa_filepath = "../input/data-for-datavis/fifa.csv"



# Read the file into a variable fifa_data

fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)

fifa_data.head()
# Set the width and height of the figure

plt.figure(figsize=(16,6))



# Line chart showing how FIFA rankings evolved over time

sns.lineplot(data=fifa_data)
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
spotify_filepath = "../input/data-for-datavis/spotify.csv"



# Read the file into a variable spotify_data

spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)
sns.lineplot(data=spotify_data)
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
museum_filepath = "../input/data-for-datavis/museum_visitors.csv"

# Fill in the line below to read the file into a variable museum_data

museum_data = pd.read_csv(museum_filepath,index_col="Date",parse_dates = True)



# Line chart showing the number of visitors to each museum over time

sns.lineplot(data=museum_data)
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Path of the file to read

flight_filepath = "../input/data-for-datavis/flight_delays.csv"



# Read the file into a variable flight_data

flight_data = pd.read_csv(flight_filepath, index_col="Month")
# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=flight_data.index, y=flight_data['NK'])



# Add label for vertical axis

plt.ylabel("Arrival delay (in minutes)")
# Set the width and height of the figure

plt.figure(figsize=(14,7))



# Add title

plt.title("Average Arrival Delay for Each Airline, by Month")



# Heatmap showing average arrival delay for each airline by month

sns.heatmap(data=flight_data, annot=True)



# Add label for horizontal axis

plt.xlabel("Airline")
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Path of the file to read

ign_filepath = "../input/data-for-datavis/ign_scores.csv"



# Fill in the line below to read the file into a variable ign_data

ign_data = pd.read_csv(ign_filepath,index_col = "Platform")



# Bar chart showing average score for racing games by platform

plt.figure(figsize=(10,6))

plt.title("Average score for racing games by platform")

sns.barplot(x=ign_data['Racing'],y=ign_data.index)

# Heatmap showing average game score by platform and genre

plt.figure(figsize=(18,7))

sns.heatmap(ign_data, annot=True)
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Path of the file to read

candy_filepath = "../input/data-for-datavis/candy.csv"



# Fill in the line below to read the file into a variable candy_data

candy_data = pd.read_csv(candy_filepath, index_col="id")

# Scatter plot showing the relationship between 'sugarpercent' and 'winpercent'

plt.figure(figsize=(14,7))

sns.scatterplot(x=candy_data['sugarpercent'],y = candy_data['winpercent'])
sns.regplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])

# Scatter plot showing the relationship between 'pricepercent', 'winpercent', and 'chocolate'

sns.scatterplot(x=candy_data['pricepercent'],y=candy_data['winpercent'],hue=candy_data['chocolate']) # Your code here



sns.lmplot(x="pricepercent", y="winpercent", hue="chocolate", data=candy_data) # Your code here

sns.swarmplot(x=candy_data['chocolate'], y=candy_data['winpercent'])
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Path of the file to read

insurance_filepath = "../input/data-for-datavis/insurance.csv"



# Read the file into a variable insurance_data

insurance_data = pd.read_csv(insurance_filepath)
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)
sns.swarmplot(x=insurance_data['smoker'],

              y=insurance_data['charges'])
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Path of the file to read

iris_filepath = "../input/data-for-datavis/iris.csv"



# Read the file into a variable iris_data

iris_data = pd.read_csv(iris_filepath, index_col="Id")
sns.distplot(a=iris_data['Petal Length (cm)'], kde=False)
# KDE plot 

sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)
sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")
# Paths of the files to read

iris_set_filepath = "../input/data-for-datavis/iris_setosa.csv"

iris_ver_filepath = "../input/data-for-datavis/iris_versicolor.csv"

iris_vir_filepath = "../input/data-for-datavis/iris_virginica.csv"



# Read the files into variables 

iris_set_data = pd.read_csv(iris_set_filepath, index_col="Id")

iris_ver_data = pd.read_csv(iris_ver_filepath, index_col="Id")

iris_vir_data = pd.read_csv(iris_vir_filepath, index_col="Id")
# Histograms for each species

sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris-setosa", kde=False)

sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", kde=False)

sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", kde=False)



# Add title

plt.title("Histogram of Petal Lengths, by Species")



# Force legend to appear

plt.legend()
# KDE plots for each species

sns.kdeplot(data=iris_set_data['Petal Length (cm)'], label="Iris-setosa", shade=True)

sns.kdeplot(data=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", shade=True)

sns.kdeplot(data=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", shade=True)



# Add title

plt.title("Distribution of Petal Lengths, by Species")
sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Paths of the files to read

cancer_b_filepath = "../input/data-for-datavis/cancer_b.csv"

cancer_m_filepath = "../input/data-for-datavis/cancer_m.csv"



# Fill in the line below to read the (benign) file into a variable cancer_b_data

cancer_b_data = pd.read_csv(cancer_b_filepath, index_col = "Id")

cancer_m_data = pd.read_csv(cancer_m_filepath, index_col = "Id")
"""# Histograms for benign and maligant tumors

sns.distplot(a=cancer_b_data['Area (mean)'],kde=False)

sns.distplot(a= cancer_m_data['Area (mean)'],kde=False)"""



"""sns.kdeplot(data=cancer_b_data['Radius (worst)'], shade=True, label="Benign")

sns.kdeplot(data=cancer_m_data['Radius (worst)'], shade=True, label="Malignant")"""
"""import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("set up complete")"""
"""fifa_filepath = "../input/data-for-datavis/spotify.csv"

fifa_data = pd.read_csv(fifa_filepath, index_col = "Date" , parse_dates = True)"""
fifa_data.head()

plt.figure(figsize=(16,6))

sns.lineplot(data = fifa_data)
print("heallo world")