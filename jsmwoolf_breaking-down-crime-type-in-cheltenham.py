# This portion of code is boilerplate from the Cheltenham Crime Data Notebook

# This is a simple dataset.  

# When reading in the data, the only area that may requires 

# special attention is the date format.  You may want to use %m/%d/%Y %I:%M:00 %p format.

import pandas as pd

import numpy as np

import datetime





import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)





dateparse = lambda x: datetime.datetime.strptime(x,'%m/%d/%Y %I:%M:00 %p')



# Read data 

d=pd.read_csv("../input/crime.csv",parse_dates=['incident_datetime'],date_parser=dateparse)



# Display data that we retrieve from the CSV file

d.head()
# Count the number of crimes per category

crimeGraph = pd.value_counts(d['parent_incident_type'])

# Display the most common and least common 

numCrimeCat = len(crimeGraph)

print("The most common crime is {} with {:2f}% occurrances".format(

    crimeGraph.keys()[0],

    100*(crimeGraph[0] / sum(crimeGraph))

))

print("The least common crime is {} with {:2f}% occurrances".format(

    crimeGraph.keys()[numCrimeCat-1],

    100*(crimeGraph[numCrimeCat-1] / sum(crimeGraph))

))

# Plot the crimes in a bar graph

crimePlot = crimeGraph.plot(kind = "bar")

plt.title("Cheltenham Crime")

plt.xlabel("Type")

plt.ylabel("Occurrance")
cityGraph = pd.value_counts(d['city'])

cityPlot = cityGraph.plot(kind = "bar")

plt.title("City Crime")

plt.xlabel("City")

plt.ylabel("Occurrance")
theCities = d[['city','parent_incident_type']]

for city in d['city'].value_counts().keys()[:10]:

    myCity = theCities[theCities['city'] == city]['parent_incident_type'].value_counts()

    myCity.plot(kind = 'bar')

    plt.title("Crimes in " + city)

    plt.xlabel("Type")

    plt.ylabel("Occurrance")

    plt.show()
for crime in d['parent_incident_type'].value_counts().keys():

    myCrime = theCities[theCities['parent_incident_type'] == crime]['city'].value_counts()

    print("{} is most common in {}".format(crime,myCrime.keys()[0]))