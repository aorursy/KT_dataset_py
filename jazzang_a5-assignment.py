# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# original data 



dataset = pd.read_csv("../input/opioid-overdose-deaths/Multiple Cause of Death 1999-2014 v1.1.csv")

df = pd.DataFrame(dataset)

df
# cleaned up and renamed one of the columns to make it cleaner and readable

updated_df = df.rename(columns={"Prescriptions Dispensed by US Retailers in that year (millions)": "Prescriptions (mils)"})



# drop the columns on confidence intervals 

df = updated_df.drop(columns=['Crude Rate Lower 95% Confidence Interval','Crude Rate Upper 95% Confidence Interval'])

df
# delete all rows in Deaths and Crude Rate if it contains 'Suppressed' and 'Unreliable'

drop_rows = df[ (df['Deaths'] == 'Suppressed') | (df['Deaths'] == 'Unreliable') | (df['Crude Rate'] == 'Suppressed') | (df['Crude Rate'] == 'Unreliable') ].index

df.drop(drop_rows , inplace=True)

df
plt.style.use('seaborn-dark')



# plotting a bar chart that compares opioid deaths by segmenting states over the 15 years period

df[['Deaths', 'Population','Crude Rate','Prescriptions (mils)']] = df[['Deaths', 'Population','Crude Rate','Prescriptions (mils)']].apply(pd.to_numeric)

avg_deaths = df.groupby('State')['Deaths','Prescriptions (mils)'].mean().sort_values(by='Deaths', ascending=False)

avg_deaths.head(10) # top 10 states that has the highest average deaths due to opioid
# plotting the average deaths with states in a horizontal bar chart

ax = avg_deaths.plot(kind = 'barh', figsize=(20,20))

plt.title('Annual Average Opioid-related Deaths Across States',fontsize = 25, fontweight='bold')

plt.ylabel('States',fontsize = 18, fontweight='bold')

plt.xlabel('Average Number of Deaths due to Opioid', fontsize = 18, fontweight='bold')



# display the value of each state's death numbers

# reference: https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh

for i, v in enumerate(avg_deaths['Deaths'].round()):

    ax.text(v + 3, i - .35, str(v), color='black', fontweight='regular')

plt.show()
# store the dataframe of grouped state vs average crude rates in avg_crude df

avg_crude = df.groupby('State')['Crude Rate'].mean().sort_values(ascending=False)

crude_bar = avg_crude.plot(kind = 'bar', figsize=(20,5), legend = True)



# plot the bar chart to show the average crude rate across states  

plt.title('Average no. of new cases/deaths per 100,000 people in each state in a year',fontsize = 25, fontweight='bold')

plt.ylabel('No. of deaths/100,000 people',fontsize = 18, fontweight='bold')

plt.xlabel('States', fontsize = 18, fontweight='bold')

plt.xticks(rotation=75)

plt.show()
# store the dataframe of grouped years vs prescriptions in millions & deaths across states over the years 

avg_prescrip = df.groupby('Year')['Deaths','Prescriptions (mils)'].sum().sort_values(by = 'Year', ascending=False)

prescrip_effect = avg_prescrip.plot(kind = 'line', figsize=(20,10), legend = True, linewidth=10)



# plot the line graph  

plt.title('No. of Opiate-related Deaths & Prescription (Millions) over 15 years',fontsize = 25, fontweight='bold')

plt.ylabel('No. of opiate deaths & opiate prescription',fontsize = 18, fontweight='bold')

plt.xlabel('Year 1994 to 2014', fontsize = 18, fontweight='bold')

plt.xticks(rotation=75)

plt.show()
# to calculate the Opioid Prescribing Rate per 100, 

# i would need to take the Prescription (in mils) divide by the the population and then divide it by 100



# create a new column that calculates OPR per person

df['OPR'] = df['Prescriptions (mils)'] * 1000000 / df['Population']

df['OPR'] = df['OPR'].round(decimals=2)

df.sort_values(by = 'OPR', ascending=False)
# create opr ranking system

def opr_rates(opr):

    if opr > 300:

        return "Too High Prescription"

    elif opr < 300 and opr >= 100:

        return "High Prescription"

    elif opr < 100 and opr >= 50:

        return "Moderate Prescription"

    elif opr < 50:

        return "Low Prescription"

    

# store the dataframe of grouped states vs OPR across states over the years 

avg_opr = df.groupby('State')['OPR','Deaths'].mean().sort_values(by = 'State', ascending=False)

avg_opr['OPR'].apply(opr_rates)
# if we visualize the average OPR across different states over the 15 years in a piechart

plt.title('Nation OPR Ratings',fontsize = 18, fontweight='bold')

opr_pie = avg_opr['OPR'].apply(opr_rates).value_counts()

opr_pie.plot(kind='pie',figsize=(10,8))
# to have a more in-depth comparison across the two variables, OPR & opiate-related deaths, we plot a hbar across all the states



opr_effect = avg_opr.plot(kind = 'barh', figsize=(20,10), legend = True, linewidth=20)



# plot the bar chart

plt.title('OPR vs Opiate-Death Levels Across States',fontsize = 18, fontweight='bold')

plt.ylabel('States',fontsize = 18, fontweight='bold')

plt.xlabel('Average Opiate Prescription Per Person & Opiate-related Death Numbers ',fontsize = 18, fontweight='bold')

plt.xticks(rotation=75)

plt.show()
# reshaping the table to be a pandas pivot table 

heatmap_data = pd.pivot_table(df, values='OPR', index=['State'], columns=['Year'])

heatmap_data
import seaborn as sb

# plotting a heatmap based on the pivot table 

fig, ax = plt.subplots(figsize=(20,15))

sb.heatmap(heatmap_data, cmap="BuGn",linewidths=.5, ax=ax)



# putting titles and labels 

plt.title('OPR Values Across States and Years',fontsize = 18, fontweight='bold')

plt.ylabel('States',fontsize = 18, fontweight='bold')

plt.xlabel('Years ',fontsize = 18, fontweight='bold')



plt.show()
# pip install opencage # also added in the console
# # pip installed geopy and gmplot via the console already

# import gmplot

# # For improved table display in the notebook

# from IPython.display import display



# from kaggle_secrets import UserSecretsClient

# user_secrets = UserSecretsClient()

# secret_value_1 = user_secrets.get_secret("opencage") # make sure this matches the Label of your key

# key1 = secret_value_1



# from opencage.geocoder import OpenCageGeocode

# geocoder = OpenCageGeocode(key1)



# for i in df['State']:

#     query = i  

#     results = geocoder.geocode(query)

#     lat = str(results[0]['geometry']['lat'])

#     lng = str(results[0]['geometry']['lng'])

    





# gmap = gmplot.GoogleMapPlotter(34.0522, -118.2437, 10)

# gmap.draw("my_heatmap.html")