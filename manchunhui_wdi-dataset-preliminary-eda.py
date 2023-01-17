import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Path of the file to read

filepath = "/kaggle/input/world-development-indicators/WDI_csv/WDISeries.csv"



# Read the file into df

df1 = pd.read_csv(filepath)

df1.head()
df1.describe()
#Path of the file to read

filepath = "/kaggle/input/world-development-indicators/WDI_csv/WDICountry.csv"



# Read the file into df

df2 = pd.read_csv(filepath)

df2.head()
df2.describe()
X = ['No. Of Unique Country Codes','No. Of Country Codes']

Y = [len(df2['Country Code'].unique()), len(df2['Country Code'])]

x_pos = [i for i, _ in enumerate(X)]

ax1=plt.subplot(111)

ax1.bar(x_pos, Y, color='green')

plt.ylabel("Count of Country Codes")

plt.xticks(x_pos, X)

plt.show()
#Path of the file to read

filepath = "/kaggle/input/world-development-indicators/WDIData_T.csv"



# Read the file into df

df3 = pd.read_csv(filepath)

df3.head()
nRow, nCol = df3.shape

print(f'There are {nRow} rows and {nCol} columns')
df3.describe()
countries = df3['CountryName'].unique().tolist()

countryCodes = df3['CountryCode'].unique().tolist()

indicators = df3['IndicatorName'].unique().tolist()

years = df3['Year'].unique().tolist()

print(f'There is data on {len(countries)} unique countries with {len(indicators)} different types of development indicators')
print(f'And the data start from {min(years)} to {max(years)}')
[indicator for indicator in indicators if 'emissions' in indicator]
# select CO2 emissions for ten countries

hist_indicator = 'CO2 emissions \(metric'

hist_country = 'USA|GBR|FRA|CHN|JPN|DEU|IND|ITA|BRA|CAN'

mask1 = df3['IndicatorName'].str.contains(hist_indicator) 

mask2 = df3['CountryCode'].str.contains(hist_country)



# stage is a filtered dataset of CO2 emissions over time for countries with country codes USA|GBR|FRA|CHN|JPN|DEU|IND|ITA|BRA|CAN.

stage = df3[mask1 & mask2]
stage.head()
#These are the "Tableau 20" colors as RGB.    

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    

             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    

             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    

             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    

             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    

  

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    

for i in range(len(tableau20)):    

    r, g, b = tableau20[i]    

    tableau20[i] = (r / 255., g / 255., b / 255.) 
countrylist = hist_country.split('|')

plt_vals=[]



plt.figure(figsize=(14,8))

ax = plt.subplot(111)    

ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False)

ax.get_xaxis().tick_bottom()    

ax.get_yaxis().tick_left() 

for y in range(5, 30, 5):    

    plt.plot(range(1959, 2017), [y] * len(range(1959, 2017)), "--", lw=1.0, color="black", alpha=0.3)



j = 0

for i in countrylist:

    plt_vals = stage.loc[stage['CountryCode'] == i]

    plt.plot(plt_vals['Year'].values, plt_vals['Value'].values, linewidth=1.5, color=tableau20[j])

    y_pos = plt_vals['Value'].values[-1] -0.3

    if plt_vals['CountryCode'].iloc[0] == 'USA':

        y_pos +=0.3

    elif plt_vals['CountryCode'].iloc[0] == 'DEU':

        y_pos -=0.3

    elif plt_vals['CountryCode'].iloc[0] == 'JPN':

        y_pos +=0.3

    elif plt_vals['CountryCode'].iloc[0] == 'GBR':

        y_pos +=0.2

    elif plt_vals['CountryCode'].iloc[0] == 'BRA':

        y_pos +=0.3

    plt.text(2016.5, y_pos, plt_vals['CountryName'].iloc[0], fontsize=12, color=tableau20[j])

    j+=1

    

plt.title('CO2 Emissions (metric tons per capita)')

plt.axis([1959, 2020,0,25])

plt.show()