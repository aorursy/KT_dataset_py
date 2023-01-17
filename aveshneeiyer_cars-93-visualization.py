# DS L2T12

# 7 Visualisations



# Import packages

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import matplotlib as mpl

%matplotlib inline

plt.style.use('classic')



import seaborn as sns

plt.style.use('seaborn-whitegrid')



# Read the data file into a df

car_data = pd.read_csv('../input/cars93/Cars93.csv')



# Read the data file into a df

#car_data = pd.read_csv('Cars93.csv', delimiter = ',')
# Print columns and first 5 rows of df

print(car_data.columns)

car_data.head()
# Drop the Unnamed column

car_data = car_data.drop('Unnamed: 0', axis=1)
car_data.info()  # some missing data in Rear.seat.room and luggage room
# Group by Type of vehicle, Calculate average

group_by_Type = car_data.groupby(by=['Type'])



car_data_avg = round(group_by_Type.mean(),0)

car_data_avg



plt.imshow(car_data_avg)
# Comparing No of passenger to Vehicle type and Min price

# Correlation dataframe using a Heat Map

corr_df = pd.DataFrame({'Min price': car_data_avg['Min.Price'],

                                      'Passengers': car_data_avg['Passengers']})

corr_df



# Plt

ax = sns.heatmap(car_data_avg)
# Heatmap showing correlation between features

plt.figure(figsize=(15,5))

c= car_data.corr()

sns.heatmap(c,cmap='BrBG',annot=True)
# Visualise Price histogram in seaborn

sns.distplot(car_data['Price'])
# Visualise Length histogram in seaborn 

sns.distplot(car_data['Length'])
# Visualise Weight histogram in seaborn 

sns.distplot(car_data['Weight'])
# Plot Manufacturer 

year = car_data['Manufacturer'][:10]

box = {

    'min': car_data['Min.Price'][:10],

    'max': car_data['Max.Price'][:10],

    'avg': car_data['Price'][:10],

}



fig, ax = plt.subplots()

ax.stackplot(year, box.values(),

             labels=box.keys())

ax.legend(loc='upper right')

ax.set_title('Price')

ax.set_xlabel('Manufacturer')

ax.set_ylabel('Price')



plt.show()
# Box plot

ax = sns.boxplot(x=car_data['Weight'])
# Histogram for Price

plt.hist(car_data['Price'], 20, density =1,facecolor="aqua", alpha=0.7)

plt.show()
# Line graph plotting passenger capacity to make

make = car_data ['Make'].tolist()

num_pass  = car_data ['Passengers'].tolist()

plt.plot(num_pass, make, label = 'Make to passengers')

plt.xlabel('Make of Vehicle')

plt.ylabel('Number of Passengers')

plt.xticks(num_pass)

plt.title('Make to passenger capacity')

plt.yticks([10, 20, 30, 40, 50, 60])

plt.show()
fig, ax = plt.subplots(figsize=(5,5))

ax.scatter(car_data['Wheelbase'], car_data['Width'])

plt.title('Scatter plot between Wheelbase and Width')

ax.set_xlabel('Wheelbase')

ax.set_ylabel('Width')

plt.show()
# Histogram

# iloc - takes all the rows and the zeroth column

car_data_count_series = car_data_avg.iloc[5:5,1]



# relevant features to plot  

features_of_interest = pd.DataFrame({'EngineSize': car_data_avg['EngineSize'],

                                     'Passengers': car_data_avg['Passengers']}                                   )



# plot a few of the manufactures for visibility

features_of_interest = features_of_interest.iloc[:10,]

features_of_interest.plot(kind='hist')