# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



# ESTABLISHMENT CODES, ESTABLISHMENT TYPES

# AND LICENSE OR FEE ARTICLES

# EST. CODES & TYPES ARTICLES EST. CODES & TYPES ARTICLES

# A(3)–Store 28-A M – Salvage Dealer 17-B

# B – Bakery 20-C N – Wholesale Produce Packer

# C – Food Manufacturer 20-C O – Produce Grower/Packer/Broker, Storage

# D – Food Warehouse 28-D P – C.A. Room

# E – Beverage Plant 20-C Q – Feed Mill/Medicated 8

# F – Feed Mill/Non-Medicated 8 R – Pet Food Manufacturer 8

# G - Processing Plant 20 S – Feed Warehouse and/or Distributor 8

# H - Wholesale Manufacturer 20-C T – Disposal Plant 5-C

# I - Refrigerated Warehouse 19 U - Disposal Plant/Transportation Service 5-C

# J – Multiple Operations V – Slaughterhouse 5-A

# K - Vehicle W – Farm Winery-Exempt 20-C, for OCR Use

# L - Produce Refrigerated Warehouse 19 Z - Farm Product Use Only

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import folium

import matplotlib.pyplot as plt
ny_retail = pd.read_csv('../input/retail-food-stores.csv')
ny_retail.head()
ny_retail.info()
ny_retail.columns
#Top 20 stores(Entity Name) by establishment

ny_retail.groupby('Establishment Type')['Entity Name'].value_counts()[:20]
#Top 10 Stores in NY

ny_retail['Entity Name'].value_counts()[:10]
plt.figure(figsize=(20,10),dpi = 100)

labels = ny_retail['Entity Name'].value_counts()[:10].index

sizes = ny_retail['Entity Name'].value_counts()[:10].values

#colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

# fontsize=12

#explode=(0, 0, 0, 0, 0.15)

plt.pie(sizes,labels=labels,autopct='%1.1f%%',startangle=90)

plt.legend(labels,loc="best")

# View the plot drop above

plt.axis('equal', fontsize=12)

# View the plot

plt.tight_layout()

plt.show()
#All stores in NY by County

ny_retail.groupby('County')['Entity Name'].value_counts()
ny_retail[ny_retail['Entity Name'].str.contains('BJS WHOLESALE CLUB INC')]['County'].value_counts()
ny_retail[ny_retail['Entity Name'].str.contains('COSTCO')]['County'].value_counts()
#All stores in NY by EstablishmentType. EstablishmentType code reveals what type of store it it ex: id D is present,it is a Wholesale

ny_retail.groupby('Establishment Type')['Entity Name'].value_counts()
#JABC

#Subset of Albany county data from complete dataset 



Albany = ny_retail[ny_retail['County'] == 'Albany']

Albany.head(2)
Albany.info()
location = ny_retail[ny_retail['Zip Code'] == 12047]
location.info()
#Top 10 Establishment type in Albany County

Albany['Establishment Type'].value_counts()[:10]



plt.figure(figsize=(20,10))

labels = Albany['Establishment Type'].value_counts()[:5].index

sizes = Albany['Establishment Type'].value_counts()[:5].values

plt.pie(sizes,labels=labels,autopct='%1.1f%%',shadow=True, startangle=140)

plt.legend(labels,loc="best")

plt.axis('equal', fontsize=12)

plt.tight_layout()

plt.show()
#Top 10 Establishment Type in NY. JAC :

# J – Multiple Operations V – Slaughterhouse 5-A

# A(3)–Store 28-A M – Salvage Dealer 17-B

# C – Food Manufacturer 20-C O – Produce Grower/Packer/Broker, Storage



ny_retail['Establishment Type'].value_counts()[:10]
plt.figure(figsize=(20,10),dpi=100)

labels = ny_retail['Establishment Type'].value_counts()[:5].index

sizes = ny_retail['Establishment Type'].value_counts()[:5].values

#colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

# fontsize=12

plt.pie(sizes,labels=labels,autopct='%1.1f%%',shadow=False, startangle=140)

plt.legend(labels,loc="best")

plt.axis('equal', fontsize=12)

plt.tight_layout()

plt.show()
#Find the all CVS and Walgreens/County

#Find all BJs and Costco /County

#Find all Top Esatblishmnet type and ZIpcode and plot
# No of Costco Wholesale stores in NY

store_name = ['COSTCO']

def find_store(entity):

    if ('COSTCO') in entity.split():

        return True

    else:

        return False

    

print(sum((ny_retail['Entity Name'].apply(lambda x:find_store(x)))))

# No of BJS Wholesale stores in NY

bjs_no = ny_retail[ny_retail['Entity Name'].str.contains('BJS WHOLESALE CLUB INC')]['County'].nunique()

print(bjs_no)
costco = ny_retail[ny_retail['Entity Name'].str.contains('COSTCO')]['County']

print(costco.head(1))

c = costco.size
def process_location(loc):

    loc_list=[]

    codes=loc

    location=eval(codes)

    try:

        loc_list.append(location['longitude'] + "," + location['latitude']+","+location['human_address'])

#   loc_list.append((location['longitude'])+ ","+(location['latitude']))

    except:

        # append a missing value to lat

        loc_list.append(np.NaN)



    return loc_list
costco= [process_location(ny_retail['Location'][i]) for i in range(0,10)]

print(costco)

df=pd.DataFrame(costco,columns = ['geo'])

print(df)

# Create two lists for the loop results to be placed( Thanks to Chris Ablon)

lon = []

lat = []

address = []

# For each row in a varible,

for row in df['geo']:

    # Try to,

    try:

        # Split the row by comma and append

        # everything before the comma to lat

        lat.append(row.split(',')[1])

        # Split the row by comma and append

        # everything after the comma to lon

        lon.append(row.split(',')[0])

        address.append(row.split(',')[2])

    # But if you get an error

    except:

        # append a missing value to lat

        lat.append(np.NaN)

        # append a missing value to lon

        lon.append(np.NaN)

        # append a missing value to address

        address.append(np.NaN)



# Create two new columns from lat and lon



df['longitude'] = lon

df['latitude'] = lat

df['address'] = address

print(df)
df.dropna(how='all', inplace=True)

df
locations = df[['latitude','longitude']].astype(float)

locations

locationlist = locations.values.tolist()

len(locationlist)


locations
locationlist
addresslist=df[['address']]
result = pd.concat([locations,addresslist],axis=1)

result
#Marking top 10 Costo locations on Map

mark=folium.Map(location = [42.73063,-73.703443],zoom_start = 12)

for i in range(0,len(result)):

    folium.Marker([result.iloc[i]['latitude'], result.iloc[i]['longitude']], popup=result.iloc[i]['address']).add_to(mark)

mark    

    

    