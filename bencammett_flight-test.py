import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
flight_filepath = "../input/flight_training_data.csv"



flight_data = pd.read_csv(flight_filepath, index_col="Month")



flight_data.head()
plt.figure(figsize=(10,6))



plt.title("Average Arrival Delay for Spirit Airlines Flights, By Month")



sns.barplot(x=flight_data.index, y=flight_data['NK'])



plt.ylabel("Arrival Delay (in minutes)")
plt.figure(figsize=(14,7))



plt.title("Average Arrival Delay for Each Airline, By Month")



sns.heatmap(data=flight_data, annot=True)



plt.xlabel("Airline")
print(list(flight_data.columns))

plt.figure(figsize=(10,6))

for flight in flight_data:

    sns.lineplot(data=flight_data[flight], label="{}".format(flight))

plt.title("Average Arrival Delay for Each Airline, By Month")

plt.ylabel("Arrival Delay (in minutes)")

plt.xlabel("Month")