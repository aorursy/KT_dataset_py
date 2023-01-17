# A common and complex challenge in the home buying/lending industry is the problem of accurately predicting property values 

# and home sales prices. Home prices/values depend on many factors: (1) property characteristics such as size of home, year 

# of construction, number of rooms, property condition, etc.; (2) spatial neighborhood characteristics such as neighboring 

# home sales and other property listings, quality of schools, etc. ; (3) temporal characteristics such as local property 

# appreciation or depreciation, seasonality, etc.. Solving this spatio-temporal statistical problem allows home buyers, 

# sellers, and lenders better evaluate the decisions they are making when purchasing, selling, or lending money for a home. 



# The data provided here includes home sales over the past 13 years for the states of MA, PA & RI. The goal is to construct 

# a model using data for home sales prior to October 1, 2018, and then use that model to predict sales prices for the most 

# recent three months of home sales (sale dates of 10/1/2018 to 12/31/2018).  The attributes provided include property 

# characteristics, sale dates, and geographic location of each property sold. This should provide a solid baseline of 

# predictors that can be used to understand how home sales vary spatially, temporally, and as a function of property 

# characteristics. Analysts should also feel free to add in additional explanatory data that might be useful in 

# predicting home sales (e.g., local economic factors, household average incomes, neighborhood demographic 

# information, etc.) in order to create the most accurate prediction. The minimum acceptable accuracy for a 

# commercially viable valuation model is 50% of home sales are within 10% of the sale value. Prediction accuracy for 

# this competition will be determined by the entry with the highest proportion of predicted home sales that are 

# within 10% of the actual home sale value.  
import pandas as pd

import pickle

import numpy as np

import datetime as dt

import matplotlib.pyplot as plt



# You may not have this dependency downloaded! It cannot be installed traditionally with the pip command.

# You may have to create a python virtual environment to get this dependency. However, it is a visualization

# tool for the demo and you do not need it for your own project. Our suggestions is to view the demoNotebook.html

# file in browser to see the output of the demo.

import mpl_toolkits

 

from mpl_toolkits.basemap import Basemap

from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection

from matplotlib.colors import Normalize    

%matplotlib inline



#import os

#print(os.listdir("../input/citizens/Citizens"))

#import datetime

#Read in the data

test = open('../input/citizens/Citizens/datathon_propattributes.obj', 'rb')

test = pickle.load(test)



 







predict = test[test['transaction_dt'] > '2018-10-01']

test[test['transaction_dt'] == '2018-10-01']
pd.set_option('display.max_rows', 200)

#test.dtypes

test['roof_cover'].value_counts(dropna=False)
pd.set_option("display.colheader_justify", 'left')

test2=pd.read_excel('../input/citizens/Citizens/FileLayout.xlsx')

pd.set_option('display.max_rows', 72)

pd.set_option('max_colwidth', 500)

test2 
#What are the basic statistics on the dataset?

(test[["building_square_feet" ,"year_built" 

   ,"total_living_square_feet","bedrooms"                     

   ,"total_rooms", "total_ground_floor_square_feet"

   ,"total_basement_square_feet" ,"total_garage_parking_square_feet"

   ,"effective_year_built"

   ,"assessed_total_value"        ,"assessed_land_value" 

   ,"assessed_improvement_value"  ,"tax_year" 

   ]]).describe()

#What is the sale price distribution?

#no_outliers = test[np.abs(test.sale_amt-test.sale_amt.mean())<=(3*test.sale_amt.std())]

#testdate=test[(test['transaction_dt'] >= '01/01/18') & (test['transaction_dt'] < '01/01/19')]

no_outliers = test[( test.sale_amt  <=1000000 ) & ( test.sale_amt  > 0 ) & (test.dwelling_type == 'Single Family Residential') &

(test['transaction_dt'] >= '01/01/18') & (test['transaction_dt'] < '01/01/19')  ]



fig = plt.figure(figsize=(20, 10))

ax = plt.subplot(111)

ax.spines["top"].set_visible(False) 

ax.spines["right"].set_visible(False)

ax.spines['left'].set_color('#1a1a1a')

ax.spines['bottom'].set_color('#1a1a1a')



ax.get_xaxis().tick_bottom()  

ax.get_yaxis().tick_left()



plt.xticks(fontsize=14, color='#1a1a1a')  

plt.yticks(fontsize=14, color='#1a1a1a')



plt.title('MA, PA, RI  property price distribution 2018 (outliers removed)', fontsize=30)

plt.xlabel('Price ($)', fontsize=20)

plt.ylabel('Frequency', fontsize=20)



plt.hist(no_outliers.sale_amt, bins=100);

#Plot the sales on a zipcode based heatmap



from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection

from matplotlib.colors import Normalize



import matplotlib.cm





fig, ax = plt.subplots(figsize=(10,20))

m = Basemap(resolution='f', # c, l, i, h, f or None

            projection='merc', 

            lat_0=54.5, lon_0=-4.36,

            llcrnrlon=-80.94, llcrnrlat= 39.5, urcrnrlon=-69.16, urcrnrlat=43.5)

           

m.drawmapboundary(fill_color='#46bcec')

m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')

m.drawcoastlines()

 

#The shapefile is not included (it is somewhat large) but can be obtained from the following location if you wish:

#https://www.census.gov/geo/maps-data/data/cbf/cbf_zcta.html

#m.readshapefile('C:\Users\J056586\cb_2017_us_zcta510_500k', 'areas')

m.readshapefile('/cbrdm_mod_anlyst/j056586-grashorn/cb_2017_us_zcta510_500k/cb_2017_us_zcta510_500k', 'areas')

#m.readshapefile('cb_2017_us_zcta510_500k/cb_2017_us_zcta510_500k', 'areas')



df_poly = pd.DataFrame({

        'shapes': [Polygon(np.array(shape), True) for shape in m.areas],

        'area': [area['ZCTA5CE10'] for area in m.areas_info]

    })

df_poly = df_poly.merge(new_areas, on='area', how='left')



cmap = plt.get_cmap('Oranges')   

pc = PatchCollection(df_poly.shapes, zorder=2)

norm = Normalize()



pc.set_facecolor(cmap(norm(df_poly['count'].fillna(0).values)))

ax.add_collection(pc)



mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)



mapper.set_array(df_poly['count'])

plt.colorbar(mapper, shrink=0.4)



m


