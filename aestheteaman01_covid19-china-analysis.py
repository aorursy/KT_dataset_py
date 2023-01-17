import pandas as pd  

from geopy.geocoders import Nominatim

import matplotlib.pyplot as plt
data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data.tail()
#Dropping of the last updated column

data.drop('Last Update',axis=1, inplace=True)



#Reseting the index

data.set_index('SNo',inplace = True)



#Viewing the dataset post ETL

data.head()
#Selecting only those data rows where country = China



China_Data = data['Country/Region'] == 'Mainland China'   #Selecting the country China

covid_china_data = data[China_Data]                       #Filtering the dataset

covid_china_data.head()
china_data = pd.read_csv('../input/china-covid19-data/China_edited_data.csv')

china_data.head()
#Viewing the generated grpahs for the reported cities in China



%pylab inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



plt.figure(figsize=(15,5))

img=mpimg.imread('../input/china-covid19-data/Anhui.png')

imgplot = plt.imshow(img)

plt.show()



plt.figure(figsize=(15,5))

img=mpimg.imread('../input/china-covid19-data/Beijing.png')

imgplot = plt.imshow(img)

plt.show()



for i in range(1,17):

    plt.figure(figsize=(15,5))

    img=mpimg.imread('../input/china-covid19-data/Screenshot ({}).png'.format(303+i))

    imgplot = plt.imshow(img)

    plt.show()
#Plotting the Graph for Hubei, China (Largest Number of reported cases)



plt.figure(figsize=(20,10))

img=mpimg.imread('../input/china-covid19-data/Screenshot (313).png')

imgplot = plt.imshow(img)

plt.show()