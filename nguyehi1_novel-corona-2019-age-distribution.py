import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print('Setup Complete')
my_filepath = "../input/novel-corona-virus-2019-dataset/covid_19_data.csv"



my_data = pd.read_csv(my_filepath)

print(my_data.shape)



# my_data.head()

my_data.tail()
my_data_Jul6 = my_data[my_data['ObservationDate']=='07/06/2020']

print(my_data_Jul6.shape)

my_data_Jul6.head()
# Set the width and height of the figure

plt.figure(figsize=(20,6))



# Add title 

plt.title("Number of COVID-19 confirmed cases and deaths as of July 6")



# Distribution

# sns.distplot(a=my_data_Jul6['Confirmed'], label='Confirmed cases', kde=False)

# sns.distplot(a=my_data_Jul6['Deaths'], label='Deaths', kde=False)



# plot

sns.kdeplot(data=my_data_Jul6['Deaths'], shade=True)

sns.kdeplot(data=my_data_Jul6['Confirmed'], shade=True)



# Force legend to appear

# plt.legend()

my_data_Jul6_US = my_data_Jul6[my_data_Jul6['Country/Region'] == 'US']

print(my_data_Jul6_US.shape)

my_data_Jul6_US
# Bar chart

plt.figure(figsize=(6,10))

sns.barplot(x='Deaths', y='Province/State', data=my_data_Jul6_US)



# Add title

plt.title("Number of COVID-19 deaths as of July 6 in the US")



# Add label for  axis

# plt.ylabel("Number of deaths")

# plt.xlabel('Date')