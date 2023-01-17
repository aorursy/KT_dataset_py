import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# A dataset from the Los Angeles Data Portal that tracks monthly visitors between Jan 2014 and Nov 2018 to four different museums in the City of Los Angeles.

museum_data = pd.read_csv("../input/data-for-datavis/museum_visitors.csv", index_col="Date", parse_dates=True)

museum_data.head()
plt.figure(figsize=(8,6))



sns.lineplot(data=museum_data) 



plt.title("Monthly number of visitors to each museum in 2014-2018")

plt.xlabel("Date")

plt.ylabel("Number of visitors")
plt.figure(figsize=(8,6))



sns.lineplot(data=museum_data['Avila Adobe'], label='Avila Adobe') 



plt.title("Monthly number of visitors to Avila Adobe museum in 2014-2018")

plt.xlabel("Date")

plt.ylabel("Number of visitors")
# A dataset from the US Department of Transportation that tracks flight delays in 2015.

# Each entry shows the average arrival delay (in minutes) for a different airline and month. Negative entries denote flights that (on average) tended to arrive early.

flight_data = pd.read_csv("../input/data-for-datavis/flight_delays.csv", index_col="Month")

flight_data.head()
plt.figure(figsize=(8,6))



sns.barplot(x=flight_data.index, y=flight_data['NK'])



plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month, in 2015")

plt.xlabel("Month")

plt.ylabel("Arrival delay (in minutes)")
plt.figure(figsize=(8,6))



sns.heatmap(data=flight_data, annot=True) # annot: to annotate values



plt.title("Average Arrival Delay for Each Airline, by Month, in 2015")

plt.xlabel("Airline")

plt.ylabel("Month")
# A (synthetic) dataset of insurance charges with relevant customer information.

insurance_data = pd.read_csv("../input/data-for-datavis/insurance.csv")

insurance_data.head()
plt.figure(figsize=(8,6))



sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])



plt.title("BMI impact on insurances charges")

plt.xlabel("Body Mass Index")

plt.ylabel("Charges")
plt.figure(figsize=(8,6))



sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])



plt.title("BMI and smoking impact on insurances charges")

plt.xlabel("Body Mass Index")

plt.ylabel("Charges")
plt.figure(figsize=(8,6))



sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])



plt.title("BodyMI impact on insurances charges")

plt.xlabel("Body Mass Index")

plt.ylabel("Charges")
plt.figure(figsize=(8,6))



sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)



plt.title("BMI and smoking impact on insurances charges")

plt.xlabel("Body Mass Index")

plt.ylabel("Charges")
plt.figure(figsize=(8,6))



sns.swarmplot(x=insurance_data['smoker'], y=insurance_data['charges'])



plt.title("Smoking impact on insurances charges")

plt.xlabel("Smoker")

plt.ylabel("Charges")
# A dataset of 150 different flowers, or 50 each from three different species of iris (Iris setosa, Iris versicolor, and Iris virginica).

# Each row in the dataset corresponds to a different flower. There are four measurements: the sepal length and width, along with the petal length and width.

iris_data = pd.read_csv("../input/data-for-datavis/iris.csv", index_col="Id")

iris_data.head()
plt.figure(figsize=(8,6))



sns.distplot(a=iris_data['Petal Length (cm)'], kde=False) #  kde: kernel density estimate



plt.title("Histogram of Petal Length")

plt.xlabel("Petal Length (cm)")
# Three datasets, each of 50 different flowers, associated with different species of iris (Iris setosa, Iris versicolor, and Iris virginica).

# Each row in the dataset corresponds to a different flower. There are four measurements: the sepal length and width, along with the petal length and width.

iris_set_data = pd.read_csv("../input/data-for-datavis/iris_setosa.csv", index_col="Id")

iris_ver_data = pd.read_csv("../input/data-for-datavis/iris_versicolor.csv", index_col="Id")

iris_vir_data = pd.read_csv("../input/data-for-datavis/iris_virginica.csv", index_col="Id")

iris_ver_data.head()
plt.figure(figsize=(8,6))



sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris-setosa", kde=False)

sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", kde=False)

sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", kde=False)



plt.title("Histogram of Petal Lengths, by Species")

plt.xlabel("Petal Length (cm)")

plt.legend()
plt.figure(figsize=(8,6))



sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True) # shade: to color the area below the curve



plt.title("KDE plot of Petal Length")

plt.xlabel("Petal Length (cm)")
plt.figure(figsize=(8,6))



sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")