import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



crimeData = pd.read_csv("../input/Crimes_2012-2016.csv")

print("Total number of crimes in the dataset: {}".format(len(crimeData)))

crimeData.head()
crimeByType = crimeData['CrmCd.Desc'].value_counts()

crimeByType
crimeData['year'] = pd.Series(crimeData['Date.Rptd'].str[-4:],index=crimeData.index)

crimeByYear = crimeData['year'].value_counts(sort=False).sort_index()

crimeByYear.plot(kind = 'line')
for year in crimeByYear.keys():

    crimeYear = crimeData[crimeData['year'] == year]['CrmCd.Desc'].value_counts()[:10]

    crimeYear = crimeYear.plot(kind = 'bar',title = "Crimes in " + year)

    plt.show()
crimeData['AREA.NAME'].value_counts()
crimeByArea = crimeData['AREA.NAME'].value_counts().sort_index()

crimeCommonType = {} # This dictionary is for later

for area in crimeByArea.keys():

    crimeArea = crimeData[crimeData['AREA.NAME'] == area]['CrmCd.Desc'].value_counts()[:10]

    for crType in crimeArea.keys():

        if not crType in crimeCommonType:

            crimeCommonType[crType] = [area]

        else:

            crimeCommonType[crType].append(area)

    crimeArea = crimeArea.plot(kind = 'bar',title = "Crimes in " + area)

    plt.show()
data = np.array([[False for i in list(crimeCommonType.keys())] for j in list(crimeByArea.keys())])

crimeOccur = pd.DataFrame(data, index= crimeByArea.keys(), columns=  crimeCommonType.keys())

crimeOccur.shape

for crimes in crimeCommonType.keys():

    for cities in crimeCommonType[crimes]:

        crimeOccur[crimes][cities] =  True

crimeOccur
crimeData["Date.Rptd"] = pd.to_datetime(crimeData["Date.Rptd"],infer_datetime_format=True)

crimeData["DATE.OCC"] = pd.to_datetime(crimeData["DATE.OCC"],infer_datetime_format=True)

crimeReportDelay = crimeData[crimeData["Date.Rptd"] != crimeData["DATE.OCC"]]

print(len(crimeReportDelay) / len(crimeData))

crimeReportDelay.head(10)
delays = abs(crimeReportDelay["Date.Rptd"] - crimeReportDelay["DATE.OCC"])
delays.describe()
crOcc = crimeData['DATE.OCC']

crOcc.value_counts().sort_index().plot(figsize=(10,8))

plt.title('Crimes Occurred')

plt.xlabel('Time')

plt.ylabel('Number of Crimes')

plt.show()

crRptd = crimeData['Date.Rptd']

crRptd.value_counts().sort_index().plot(color='r',figsize=(10,8))

plt.title('Crimes Reported')

plt.xlabel('Time')

plt.ylabel('Number of Crimes')

plt.show()
crRptd.value_counts().tail(1)
crOcc.value_counts().tail(1)
crimeData['Status.Desc'].value_counts().plot(kind = 'pie',autopct='%.2f',figsize=(6,6))
for status in crimeData['Status.Desc'].value_counts().keys():

    temp = crimeData[crimeData['Status.Desc'] == status]['CrmCd.Desc'].value_counts()

    print("Most common crime with {} is {}".format(status,temp.keys()[0]))