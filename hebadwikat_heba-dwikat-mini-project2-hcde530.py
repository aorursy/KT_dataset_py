# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Loading libraries 

 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

#% matplotlib inline

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)

sns.set(font_scale=1)



#for regression Model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression #to perform the linear regression

from sklearn.model_selection import train_test_split

from sklearn.metrics import explained_variance_score
#logisitc Regression

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score
import pandas as pd

rent = pd.read_csv("../input/zillow-rent-index-zri-us-20102020/Neighborhood_Zri_AllHomesPlusMultifamily (1).csv") #this will include all the Zillow rent index state rents by neighbourhood

print ('There are {0} rows and {1} attributes.'.format(rent.shape[0], rent.shape[1]))

rent.head()
rent.info()

rent.describe()
#Filtering out the data to only take in seattle regions

seattle = rent[rent.City=='Seattle']



#average the price for each year so we do not have too many months to work with

#we used a regular expression to match multiple columns that correspond to the same year

seattle['c2010'] = seattle.filter(regex = '2010').mean(axis=1)

seattle['c2011'] = seattle.filter(regex = '2011').mean(axis=1)

seattle['c2012'] = seattle.filter(regex = '2012').mean(axis=1)

seattle['c2013'] = seattle.filter(regex = '2013').mean(axis=1)

seattle['c2014'] = seattle.filter(regex = '2014').mean(axis=1)

seattle['c2015'] = seattle.filter(regex = '2015').mean(axis=1)

seattle['c2016'] = seattle.filter(regex = '2016').mean(axis=1)

seattle['c2017'] = seattle.filter(regex = '2017').mean(axis=1)

seattle['c2018'] = seattle.filter(regex = '2018').mean(axis=1)

seattle['c2019'] = seattle.filter(regex = '2019').mean(axis=1)

seattle['c2020'] = seattle.filter(regex = '2020').mean(axis=1)

#remove unwanted columns

regressionData = seattle.filter(['RegionName', 'c2010', 'c2011', 'c2012',

                                 'c2013', 'c2014', 'c2015', 'c2016', 'c2017', 'c2018', 'c2019', 'c2020'])

#rename year columns to use in regression

regressionData = regressionData.rename(columns = {

    'c2010': '2010',

    'c2011': '2011',

    'c2012': '2012',

    'c2013': '2013',

    'c2014': '2014',

    'c2015': '2015',

    'c2016': '2016',

    'c2017': '2017',

    'c2018': '2018',

    'c2019': '2019',

    'c2020': '2020'

    

})



#setting the index for plotting

plotData = regressionData.set_index('RegionName')
plotData.head(15).T.plot(figsize=(20, 10))

plt.xlabel("Year")

plt.ylabel("Price")

plt.title("Rent Prices per neighbourhood")
#rdt = regressionDataTransposed

#transposed data so we can manipulate it easier

rdt = plotData.T

rdt.head()



#applied a polynomial for to the data, with year as the x and price as the y

#we used logarithmic scale

#trying to use logisitical regression in a different way (maybe not the best way)

regressionModel = np.polyfit(np.log(rdt.index.astype(str).astype(int)), rdt['Capitol Hill'], 1)
#plot actual pricing for comparison #Lets use Capitol Hill as an example

plotData.T['Capitol Hill'].plot()

plt.xlabel("Year")

plt.ylabel("Price")

plt.title("Current Rent Prices in Capitol Hill")
#apply the model to the data to find prediction



# logarithmic function

def func(x, p1,p2):

  return p1*np.log(x)+p2



#setting the prediction data

predictionData = pd.DataFrame({'x' : [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018

                             , 2019, 2020, 2021, 2022, 2023, 2024, 2025]})



#doing the prediction

predictionData['y'] = func(predictionData['x'], regressionModel[0], regressionModel[1])



#plotting

predictionData.set_index('x').plot(xlim=(2010,2025), ylim=(1250, 3000),color='red')

plt.xlabel("Year")

plt.ylabel("Price")

plt.title("Predicted Rent Prices for Capitol Hill")
#regressionData.head()

#I would like to figure out the top highest rent neighbourhood in Seattle in 2020

topfive = regressionData.sort_values(by=['2020'], axis=0, ascending=False)
topfive = topfive.head(5)

topfive
#Transposing the top five areas

regressionMod= topfive.set_index("RegionName").T

regressionMod
X = regressionMod.index.astype(str).astype(int).values.reshape(-1, 1)  # values converts it into a numpy array #first column of the data frame

Y = regressionMod.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column #second column of the data frame



linear_regressor = LinearRegression()  # create object for the class

linear_regressor.fit(X, Y)  # perform linear regression

Y_pred = linear_regressor.predict(X)  # make predictions

Y_pred
#prediciting the model for Madrona

plt.scatter(X,Y)

#plt.plot(array,y_head,color='red')

plt.xlabel("Madrona")

plt.ylabel("Price")

plt.plot(X, Y_pred,color='red')

plt.show()
seattle["RegionName"].unique() #All the neighbourhoods from the first dataset
seattle["RegionName"].count()
crimeRate = pd.read_csv("/kaggle/input/zillow-rent-index-zri-us-20102020/Crime_Data.csv")#using the second dataset 

crimeRate = crimeRate.drop(['Report Number', 'Precinct',  'Sector', 'Beat','Occurred Date','Report Number'], axis=1)#dropping any unneeded columns

crimeRate = crimeRate.dropna(how='any')



#I would like to get rid of any unknown values in the data

crimeRate = crimeRate.loc[crimeRate['Neighborhood'] != 'UNKNOWN']

crimeRate.head()
crimeRate['Year'] = pd.DatetimeIndex(crimeRate['Reported Date']).year #reducing the date to the year

crimeRate.head()

#greater than the start date and smaller than the end date

#the latest year between the two datasets is 2019,we will be using that

crimeRate = crimeRate.loc[crimeRate['Year'] == 2019]

crimeRate
crimeRate["Neighborhood"].unique()

#finding the Neighborhoods in the crime dataset.

processedCrimeRates = crimeRate

processedCrimeRates['RegionName'] = processedCrimeRates['Neighborhood']





processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'CENTRAL AREA/SQUIRE PARK', 'RegionName'] = 'Minor'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'QUEEN ANNE', 'RegionName'] = 'East Queen Anne'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'NORTHGATE', 'RegionName'] = 'Haller Lake'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'LAKEWOOD/SEWARD PARK', 'RegionName'] = 'Seward Park'

#>_<

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'ALKI', 'RegionName'] = 'Alki'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'HIGH POINT', 'RegionName'] = 'Fairmount Park'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'ROXHILL/WESTWOOD/ARBOR HEIGHTS', 'RegionName'] = 'Roxhill'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'BRIGHTON/DUNLAP', 'RegionName'] = 'Brighton'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'FIRST HILL', 'RegionName'] = 'First Hill'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'SLU/CASCADE', 'RegionName'] = 'South Lake Union'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'ALASKA JUNCTION', 'RegionName'] = 'Fairmount Park'

#>_<

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'HIGHLAND PARK', 'RegionName'] = 'Highland Park'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'SANDPOINT', 'RegionName'] = 'View Ridge'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'NORTH ADMIRAL', 'RegionName'] = 'Admiral'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'SOUTH PARK', 'RegionName'] = 'South Park'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'CAPITOL HILL', 'RegionName'] = 'Capitol Hill'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'LAKECITY', 'RegionName'] = 'Victory Heights'

#Break,I hope there is a better way to do this,this turned out to be really lengthy

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'CLAREMONT/RAINIER VISTA', 'RegionName'] = 'Rainier View'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'ROOSEVELT/RAVENNA', 'RegionName'] = 'Roosevelt'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'HILLMAN CITY', 'RegionName'] = 'Hillman City'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'SODO', 'RegionName'] = 'Downtown'

#>_<

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'SOUTH BEACON HILL', 'RegionName'] = 'South Beacon Hill'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'WALLINGFORD', 'RegionName'] = 'Wallingford'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'FAUNTLEROY SW', 'RegionName'] = 'Fauntleroy'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'NORTH BEACON HILL', 'RegionName'] = 'North Beacon Hill'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'RAINIER VIEW', 'RegionName'] = 'Rainier View'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'MORGAN', 'RegionName'] = 'Seward Park'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'MAGNOLIA', 'RegionName'] = 'Magnolia'

#>_<

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'SOUTH DELRIDGE', 'RegionName'] = 'South Delridge'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'NORTH DELRIDGE', 'RegionName'] = 'North Delridge'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'GREENWOOD', 'RegionName'] = 'Greenwood'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'RAINIER BEACH', 'RegionName'] = 'Rainier Beach'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'MID BEACON HILL', 'RegionName'] = 'South Beacon Hill'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'MOUNT BAKER', 'RegionName'] = 'Mt. Baker'

#>_<

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'PHINNEY RIDGE', 'RegionName'] = 'Phinney Ridge'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'CHINATOWN/INTERNATIONAL DISTRICT', 'RegionName'] = 'Downtown'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'BELLTOWN', 'RegionName'] = 'Belltown'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'UNIVERSITY', 'RegionName'] = 'University District'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'PIONEER SQUARE', 'RegionName'] = 'Downtown'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'GENESEE', 'RegionName'] = 'Genesee'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'MILLER PARK', 'RegionName'] = 'Capitol Hill'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'BALLARD NORTH', 'RegionName'] = 'Sunset Hill'

#>_<

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'DOWNTOWN COMMERCIAL', 'RegionName'] = 'Downtown'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'BALLARD SOUTH', 'RegionName'] = 'Adams'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'MADRONA/LESCHI', 'RegionName'] = 'Madrona'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'JUDKINS PARK/NORTH BEACON HILL', 'RegionName'] = 'Judkins Park'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'FREMONT', 'RegionName'] = 'Fremont'

#>_<

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'COLUMBIA CITY', 'RegionName'] = 'Columbia City'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'MONTLAKE/PORTAGE BAY', 'RegionName'] = 'University District'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'NEW HOLLY', 'RegionName'] = 'South Beacon Hill'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'MADISON PARK', 'RegionName'] = 'Madison Park'

#omg THIS IS REALLY labor work

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'GEORGETOWN', 'RegionName'] = 'North Beacon Hill'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'EASTLAKE - WEST', 'RegionName'] = 'Eastlake'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'COMMERCIAL HARBOR ISLAND', 'RegionName'] = 'Admiral'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'COMMERCIAL DUWAMISH', 'RegionName'] = 'South Delridge'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'PIGEON POINT', 'RegionName'] = 'North Delridge'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'EASTLAKE - EAST', 'RegionName'] = 'Eastlake'

processedCrimeRates.loc[processedCrimeRates['RegionName'] == 'BITTERLAKE', 'RegionName'] = 'Bitter Lake'



processedCrimeRates

crimeTotals = processedCrimeRates.groupby(['RegionName']).count()[['Occurred Time']].sort_values(by=['Occurred Time'],ascending=False) #Grouping by in order to count crime in 2019

crimeTotals.rename(columns={"Occurred Time" : "Crimes"}, inplace=True)
rentData = regressionData[['RegionName', '2019']].set_index('RegionName') #RegionName will be our index to merge both datasets

rentData.rename(columns={"2019" : "Rent"}, inplace=True)

crimeRegression = rentData.join(crimeTotals).dropna() #joininf Crime and Rent from the two dataset



crimeRegression.sort_values(by=['Crimes'],ascending=False).head()

#Downtown is an outlier, lets remove it
crimeRegression = crimeRegression.drop('Downtown') #Im sorry Outlier ...
crimeRegression.corr()
#performing the regression model,using linear regression

XX = crimeRegression.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array #first column of the data frame

YY = crimeRegression.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column #second column of the data frame



rent_linear_regressor = LinearRegression()  # create object for the class

rent_linear_regressor.fit(XX, YY)  # perform linear regression



YY_pred = rent_linear_regressor.predict(XX)
predictedCrime = rent_linear_regressor.predict([[1500]]) #Lets predict using a certain rent price = 1500



predictedCrime
#Plotting the model

plt.scatter(XX,YY)

plt.xlabel("Price")

plt.ylabel("Crime")

plt.plot(XX, YY_pred,color='red')

plt.show()
from sklearn.cluster import KMeans

clusterer = KMeans(n_clusters=4)

clusterer.fit(crimeRegression) # index = regionName, price, lat, long, maybe crime count
plt.scatter(crimeRegression['Rent'], crimeRegression['Crimes'], c=clusterer.labels_, s=25, cmap="tab10")

plt.xlabel("Price")

plt.ylabel("Crime")