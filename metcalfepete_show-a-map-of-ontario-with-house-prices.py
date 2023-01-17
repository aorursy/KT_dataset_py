import pandas as pd

import matplotlib.pyplot as plt

import warnings



warnings.filterwarnings('ignore')



df = pd.read_csv('../input/properties.csv', index_col=0)



# Only consider houses worth more than $50K

df = df[df['Price ($)'] > 49000]





# Parse the address and create a city column

df['City'] = df['Address'].str.replace(', ON','')

df['City'] = df['City'].str.split(' ').str.get(-1)



# Remove areas that only have a few houses

df1 = df.groupby(['City']).filter(lambda x: len(x) > 10)



# Get the Average Price for a city

AvePrice = df1.groupby(['City']).mean()



print(AvePrice.head(10))
import folium



map_hooray = folium.Map(location= [45.65, -83],

                    zoom_start = 7) # Uses lat then lon. The bigger the zoom number, the closer in you get



# Get the highest average house price

maxave = int(AvePrice['Price ($)'].max())

print("Highest City House Price is: ", maxave)



# Create a color map to match house prices. White - low price, Black - high price

colormap = ['white','lightgray','pink','lightred','orange','darkred','red','purple','darkpurple','black']



# Add marker info 

for index, row in AvePrice.iterrows(): 

    # Set icon color based on price

    theCol = colormap[ int((len(colormap) - 1 ) *  float( row['Price ($)']) / maxave) ]

    # Create a marker text with City name and average price

    markerText =  str(index) + ' ${:,.0f}'.format(row['Price ($)'])

    

    folium.Marker([row['lat'],row['lng']], popup = markerText, 

                  icon=folium.Icon(color= theCol)).add_to(map_hooray)



map_hooray