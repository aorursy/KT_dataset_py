#Comment this cell and restart the session after running it
"""!pip install --upgrade plotly
!pip install --upgrade geopandas
!pip install --upgrade pyshp
!pip install --upgrade shapely"""
#Import Essential Libraries
!pip3 install geopandas==0.3.0
!pip3 install pyshp==1.2.10
!pip3 install shapely==1.6.3
!pip install plotly-geo

import numpy as np
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
#Cleaning routine for files from USDA Economic Research Service
def cleanGovData(file, index):
    data = pd.read_csv(file)
    data.columns = data.iloc[index]
    data = data.drop([i for i in range(index + 1)], axis = 0)
    data.index -= index + 1
    data = data.rename(columns = {"FIPStxt" : "FIPS"})
    data["FIPS"] = list(map(int, data["FIPS"]))
    return data


#Creates a chloropleth map on a USA counties map template from any list
def plotMap(inputList):
    endPoints = list(np.percentile(inputList, [5 + (5 * i) for i in range(19)])) #5% intervals for range of numbers and colors
    i = 0
    while i < len(endPoints):
        if endPoints.count(endPoints[i]) > 1 or endPoints[i] == min(inputList) or endPoints[i] == max(inputList):
            del endPoints[i] #removes any repeating intervals or minimum/maximum for well balanced intervals
        else:
            i += 1
            
    colorScale, numColors = [], len(endPoints) + 1
    for i in range(numColors): #Creates as many colors between light blue and dark blue as needed (based on intervals)
        colorScale.append("rgb(" + str(247 - ((239 / (numColors - 1)) * i)) + ", " + str(251 - ((203 / (numColors - 1)) * i)) + ", " + str(255 - ((148 / (numColors - 1)) * i)) + ")")
        
    fig = ff.create_choropleth(fips = fipsList, values = inputList, colorscale = colorScale, binning_endpoints = endPoints)
    fig.layout.template = None 
    fig.show(rendering = "kaggle") #creates chloropeth map using the list, color scale, intervals, and fips values
    

    
#Generates a linear regression line for any two lists
def regLine(xList, yList):
    coeff = np.polyfit(xList, yList, 1)
    return [((coeff[0] * xList[i]) + coeff[1]) for i in range(len(xList))]


#Returns the correlation coefficient for any two lists
def coeff(xList, yList):
    return np.corrcoef(xList, yList)[0][1]


#Removes Manhattan and counties with negligible cases/deaths
def reduce(inputList):
    outputList = []
    for i in range(len(confirmedList)):
        if confirmedList[i] > 10 and deathList[i] > 5 and i != 1750:
            outputList.append(inputList[i])
    return outputList


#Min-Max normalizes/scales any list
def normalize(inputList):
    return ((np.array(inputList) - min(inputList)) / (max(inputList) - min(inputList)))
#Retrieving and cleaning general case info from Johns Hopkins data
covidData = pd.read_csv("/kaggle/input/06-07-2020.csv")
covidData = covidData.rename(columns = {"Admin2" : "County"})
covidData["FIPS"].fillna(999999999999, inplace = True)
covidData["FIPS"] = list(map(int, covidData["FIPS"]))

covidData.head()
#Retrieves and cleans files from USDA data
unemployData = cleanGovData("/kaggle/input/Unemployment.csv", 6)
povertyData = cleanGovData("/kaggle/input/Poverty.csv", 3)
populationData = cleanGovData("/kaggle/input/Population.csv", 1)
educationData = cleanGovData("/kaggle/input/Education.csv", 3)

unemployData.head()
#Retrieves racial/demographic data (already been cleaned from original census data)
racialData = pd.read_csv("/kaggle/input/racialData.csv")

racialData.head()
#List of all 2959 FIPS values to be used (already found using set intersection)
fipsList = list(racialData["FIPS"])

#All lists to be created from data sources
countyList, stateList, confirmedList, deathList, mortalityList = [], [], [], [], []
whiteAlone, blackAlone, nativeAlone, asianAlone, hispanic = [], [], [], [], []
medianList, unemployList, povallList, popList = [], [], [], []
lessHigh, onlyHigh, someCollege, bachelorAbove = [], [], [], []
confirmPopList, deathPopList = [], []
#Loops through all FIPS values
for fips in fipsList:
    
    #Finds rows from each data frame with current FIPS value
    covidRow = covidData.loc[covidData["FIPS"] == fips]
    unemployRow = unemployData.loc[unemployData["FIPS"] == fips]
    povertyRow = povertyData.loc[povertyData["FIPS"] == fips]
    populationRow = populationData.loc[populationData["FIPS"] == fips]
    educationRow = educationData.loc[educationData["FIPS"] == fips]
    racialRow = racialData.loc[racialData["FIPS"] == fips]

    #Case info from Johns Hopkins data
    countyList.append(list(covidRow["County"])[0]) #list of county names
    stateList.append(list(covidRow["Province_State"])[0]) #list of corresponding state names
    if list(covidRow["Confirmed"])[0] == 0:
        mortalityList.append(0) #adds 0 to mortality list if there are zero cases
    else:
        mortalityList.append(list((covidRow["Deaths"] / covidRow["Confirmed"]) * 100)[0]) #deaths/confirmed = mortality
    confirmedList.append(int(list(covidRow["Confirmed"])[0])) #number of total confirmed cases as of 06/07/2020
    deathList.append(int(list(covidRow["Deaths"])[0])) #number of total deaths as of 06/07/2020


    #Confirmed Cases and Deaths per 100,000 people
    confirmPopList.append(list(covidRow["Confirmed"])[0] / float(list(populationRow["POP_ESTIMATE_2018"])[0].replace(",", "")) * 100000)
    deathPopList.append(list(covidRow["Deaths"])[0] / float(list(populationRow["POP_ESTIMATE_2018"])[0].replace(",", "")) * 100000)

    #Economic information - median household income and unemployment rate
    medianList.append(int(list(unemployRow["Median_Household_Income_2018"])[0].replace(",", "")))
    unemployList.append(float(list(unemployRow["Unemployment_rate_2018"])[0]))

    #Poverty information - percent of total population living under poverty line
    povallList.append(float(list(povertyRow["PCTPOVALL_2018"])[0]))
    
    #Total Population 2018
    popList.append(int(list(populationRow["POP_ESTIMATE_2018"])[0].replace(",", "")))
    
    #Educational information (percent of adults)
    lessHigh.append(float(list(educationRow["Percent of adults with less than a high school diploma, 2014-18"])[0]))
    onlyHigh.append(float(list(educationRow["Percent of adults with a high school diploma only, 2014-18"])[0]))
    someCollege.append(float(list(educationRow["Percent of adults completing some college or associate's degree, 2014-18"])[0]))
    bachelorAbove.append(float(list(educationRow["Percent of adults with a bachelor's degree or higher, 2014-18"])[0]))

    #Racial data (percent of total population)
    whiteAlone.append(float(racialRow["White Alone"]))
    blackAlone.append(float(racialRow["Black Alone"]))
    nativeAlone.append(float(racialRow["Native Alone"]))
    asianAlone.append(float(racialRow["Asian Alone"]))
    hispanic.append(float(racialRow["Hispanic"]))
#Creates a new data frame from all of the lists
cleanData = pd.DataFrame({
                          "State" : stateList, "FIPS" : fipsList, "County" : countyList,
                          "Population 2018" : popList, "Median Household Income 2018 ($)" : medianList,
                          "Unemployment Rate 2018 (%)" : unemployList, "Poverty 2018 (%)" : povallList,
                          "Confirmed Cases" : confirmedList, "Confirmed Deaths" : deathList,
                          "Confirmed Cases Per 100,000 people" : confirmPopList,
                          "Deaths Per 100,000 people" : deathPopList, "Mortality Rate (%)" : mortalityList,
                          "White Alone (%)" : whiteAlone, "Black Alone (%)" : blackAlone,
                          "Native American Alone (%)" : nativeAlone, "Asian Alone (%)" : asianAlone,
                          "Hispanic (%)" : hispanic, "Less than a High School Diploma (%)" : lessHigh,
                          "Only a High School Diploma (%)" : onlyHigh, "Some College/Associate's Degree (%)" : someCollege,
                          "Bachelor's Degree or Higher (%)" : bachelorAbove
                        })

cleanData.head(10)
#YOU CAN SCROLL TO SEE THE REST OF THE MAP!
#Plotting COVID-19 mortality rate by county
plotMap(mortalityList)
#Plotting confirmed cases per 100,000 people by county
plotMap(confirmPopList)
#Plotting the confirmed cases per 100,000 people against the mortality rate
plt.rcParams["figure.figsize"] = (12, 8)

plt.xlabel("Confirmed Cases per 100,000 people")
plt.ylabel("Mortality Rate (%)")
plt.scatter(confirmPopList, mortalityList, 1)
plt.plot(confirmPopList, regLine(confirmPopList, mortalityList))
plt.show()

print("Correlation coefficient: " + str(coeff(confirmPopList, mortalityList)))
#Plotting deaths per 100,000 people by county
plotMap(deathPopList)
#Plotting the poverty rate by county
plotMap(povallList)
#Plotting poverty rate against the deaths per 100,000 people
plt.rcParams["figure.figsize"] = (16, 8)

plt.subplot(1, 2, 1)
plt.xlabel("Poverty Rate (%)")
plt.ylabel("Deaths per 100,000 people")
plt.scatter(povallList, deathPopList)
plt.plot(povallList, regLine(povallList, deathPopList))

plt.subplot(1, 2, 2)
plt.xlabel("Poverty Rate (%)")
plt.ylabel("Deaths per 100,000 people")
tempPov, tempDeaths = povallList[0 : 1750] + povallList[1751:], deathPopList[0 : 1750] + deathPopList[1751:]
plt.scatter(tempPov, tempDeaths)
plt.plot(tempPov, regLine(tempPov, tempDeaths))

plt.show()

print("Correlation coefficient with Manhattan: " + str(coeff(povallList, deathPopList)))
print("Correlation coefficient without Manhattan: " + str(coeff(tempPov, tempDeaths)))
#Removing negligible counties
plt.rcParams["figure.figsize"] = (8, 8)

x, y = reduce(povallList), reduce(deathPopList)
plt.xlabel("Poverty Rate (%)")
plt.ylabel("Deaths per 100,000 people")
plt.scatter(x, y)
plt.plot(x, regLine(x, y))

print("Correlation coefficient without Manhattan and negligible counties: " + str(coeff(x, y)) + "\n")
print("Number of counties in reduced list: " + str(len(x)) + " (" + str(len(x) / 2959 * 100) + "% of all 2959 counties)")
print("% of total confirmed cases that reduced list is: " + str(sum(reduce(confirmedList)) / sum(confirmedList) * 100))
print("% of total population that reduced list is: " + str(sum(reduce(popList)) / sum(popList) * 100))
#Plotting the unemployment rate by county
plotMap(unemployList)
#Plotting unemployment rate against deaths per 100,000 people
plt.rcParams["figure.figsize"] = (8, 8)

x, y = reduce(unemployList), reduce(deathPopList)
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Deaths per 100,000 people")
plt.scatter(x, y)
plt.plot(x, regLine(x, y))
plt.show()

print("Correlation coefficient: " + str(coeff(x, y)))
#Plotting the median household income by county
plotMap(medianList)
#Plotting median household income against deaths per 100,000 people
plt.rcParams["figure.figsize"] = (8, 8)

x, y = reduce(medianList), reduce(deathPopList)
plt.xlabel("Median Household Income ($)")
plt.ylabel("Deaths per 100,000 people")
plt.scatter(x, y)
plt.plot(x, regLine(x, y))
plt.show()

print("Correlation coefficient: " + str(coeff(x, y)))
#Plotting white alone % by county
plotMap(whiteAlone)
#Plotting white alone % against deaths per 100,000 people
plt.rcParams["figure.figsize"] = (8, 8)

x, y = reduce(whiteAlone), reduce(deathPopList)
plt.xlabel("White Alone (%)")
plt.ylabel("Deaths per 100,000 people")
plt.scatter(x, y)
plt.plot(x, regLine(x, y))
plt.show()

print("Correlation coefficient: " + str(coeff(x, y)))
#Plotting black alone % by county
plotMap(blackAlone)
#Plotting black alone % against deaths per 100,000 people
plt.rcParams["figure.figsize"] = (8, 8)

x, y = reduce(blackAlone), reduce(deathPopList)
plt.xlabel("Black Alone (%)")
plt.ylabel("Deaths per 100,000 people")
plt.scatter(x, y)
plt.plot(x, regLine(x, y))
plt.show()

print("Correlation coefficient: " + str(coeff(x, y)))
#Plotting hispanic % by county
plotMap(hispanic)
#Plotting hispanic % against deaths per 100,000 people
plt.rcParams["figure.figsize"] = (8, 8)

x, y = reduce(hispanic), reduce(deathPopList)
plt.xlabel("Hispanic (%)")
plt.ylabel("Deaths per 100,000 people")
plt.scatter(x, y)
plt.plot(x, regLine(x, y))
plt.show()

print("Correlation coefficient: " + str(coeff(x, y)))
#Plotting asian alone % by county
plotMap(asianAlone)
#Plotting asian alone % against deaths per 100,000 people
plt.rcParams["figure.figsize"] = (8, 8)

x, y = reduce(asianAlone), reduce(deathPopList)
plt.xlabel("Asian Alone (%)")
plt.ylabel("Deaths per 100,000 people")
plt.scatter(x, y)
plt.plot(x, regLine(x, y))
plt.show()

print("Correlation coefficient: " + str(coeff(x, y)))
#Plotting Native American alone % by county
plotMap(nativeAlone)
#Plotting Native American alone % against deaths per 100,000 people
plt.rcParams["figure.figsize"] = (8, 8)

x, y = reduce(nativeAlone), reduce(deathPopList)
plt.xlabel("Native American Alone (%)")
plt.ylabel("Deaths per 100,000 people")
plt.scatter(x, y)
plt.plot(x, regLine(x, y))
plt.show()

print("Correlation coefficient: " + str(coeff(x, y)))
#Plotting less than a high school diploma % by county
plotMap(lessHigh)
#Plotting less than a high school diploma % against deaths per 100,000 people
plt.rcParams["figure.figsize"] = (8, 8)

x, y = reduce(lessHigh), reduce(deathPopList)
plt.xlabel("Adults with less than a high school diploma (%)")
plt.ylabel("Deaths per 100,000 people")
plt.scatter(x, y)
plt.plot(x, regLine(x, y))
plt.show()

print("Correlation coefficient: " + str(coeff(x, y)))
#Plotting only a high school diploma % by county
plotMap(onlyHigh)
#Plotting only a high school diploma % against deaths per 100,000 people
plt.rcParams["figure.figsize"] = (8, 8)

x, y = reduce(onlyHigh), reduce(deathPopList)
plt.xlabel("Adults with a high school diploma only (%)")
plt.ylabel("Deaths per 100,000 people")
plt.scatter(x, y)
plt.plot(x, regLine(x, y))
plt.show()

print("Correlation coefficient: " + str(coeff(x, y)))
#Plotting some college/associate's degree % by county
plotMap(someCollege)
#Plotting some college/associate's degree % against deaths per 100,000 people
plt.rcParams["figure.figsize"] = (8, 8)

x, y = reduce(someCollege), reduce(deathPopList)
plt.xlabel("Adults with some college/associate's degree (%)")
plt.ylabel("Deaths per 100,000 people")
plt.scatter(x, y)
plt.plot(x, regLine(x, y))
plt.show()

print("Correlation coefficient: " + str(coeff(x, y)))
#Plotting bachelor's degree or higher % by county
plotMap(bachelorAbove)
#Plotting bachelor's degree or higher % against deaths per 100,000 people
plt.rcParams["figure.figsize"] = (8, 8)

x, y = reduce(bachelorAbove), reduce(deathPopList)
plt.xlabel("Adults with a bachelor's degree or higher (%)")
plt.ylabel("Deaths per 100,000 people")
plt.scatter(x, y)
plt.plot(x, regLine(x, y))
plt.show()

print("Correlation coefficient: " + str(coeff(x, y)))
#Plotting relationships between economic factors
plt.rcParams["figure.figsize"] = (18, 8)

plt.subplot(1, 3, 1)
plt.xlabel("Median Household Income ($)")
plt.ylabel("Poverty Rate (%)")
plt.scatter(medianList, povallList)

plt.subplot(1, 3, 2)
plt.xlabel("Median Household Income ($)")
plt.ylabel("Unemployment Rate (%)")
plt.scatter(medianList, unemployList)

plt.subplot(1, 3, 3)
plt.xlabel("Poverty Rate (%)")
plt.ylabel("Unemployment Rate (%)")
plt.scatter(povallList, unemployList)

plt.show()
#Plotting relationships between demographic factors (only significant correlations plotted)
plt.rcParams["figure.figsize"] = (18, 8)

plt.subplot(1, 3, 1)
plt.xlabel("White Alone (%)")
plt.ylabel("Black Alone (%)")
plt.scatter(whiteAlone, blackAlone)

plt.subplot(1, 3, 2)
plt.xlabel("White Alone (%)")
plt.ylabel("Asian Alone (%)")
plt.scatter(whiteAlone, asianAlone)

plt.subplot(1, 3, 3)
plt.xlabel("White Alone (%)")
plt.ylabel("Native Alone (%)")
plt.scatter(whiteAlone, nativeAlone)

plt.show()
#Plotting relationships between educational factors (only significant correlations plotted)
plt.rcParams["figure.figsize"] = (18, 8)

plt.subplot(1, 3, 1)
plt.xlabel("Adults with less than a high school diploma (%)")
plt.ylabel("Adults with some college/associate's degree (%)")
plt.scatter(lessHigh, someCollege)

plt.subplot(1, 3, 2)
plt.xlabel("Adults with less than a high school diploma (%)")
plt.ylabel("Adults with bachelor's degrees or higher (%)")
plt.scatter(lessHigh, bachelorAbove)

plt.subplot(1, 3, 3)
plt.xlabel("Adults with a high school diploma only (%)")
plt.ylabel("Adults with bachelor's degrees or higher (%)")
plt.scatter(onlyHigh, bachelorAbove)

plt.show()
#Plotting other relationships (only significant ones plotted)

plt.rcParams["figure.figsize"] = (18, 28)

plt.subplot(4, 3, 1)
plt.xlabel("Poverty Rate (%)")
plt.ylabel("White Alone (%)")
plt.scatter(povallList, whiteAlone)

plt.subplot(4, 3, 2)
plt.xlabel("Poverty Rate (%)")
plt.ylabel("Black Alone (%)")
plt.scatter(povallList, blackAlone)

plt.subplot(4, 3, 3)
plt.xlabel("Poverty Rate (%)")
plt.ylabel("Adults with less than a high school diploma (%)")
plt.scatter(povallList, lessHigh)

plt.subplot(4, 3, 4)
plt.xlabel("Poverty Rate (%)")
plt.ylabel("Adults with bachelor's degrees or higher (%)")
plt.scatter(povallList, bachelorAbove)

plt.subplot(4, 3, 5)
plt.xlabel("Median Household Income ($)")
plt.ylabel("Asian Alone (%)")
plt.scatter(medianList, asianAlone)

plt.subplot(4, 3, 6)
plt.xlabel("Median Household Income ($)")
plt.ylabel("Adults with less than a high school diploma (%)")
plt.scatter(medianList, lessHigh)

plt.subplot(4, 3, 7)
plt.xlabel("Median Household Income ($)")
plt.ylabel("Adults with a high school diploma only (%)")
plt.scatter(medianList, onlyHigh)

plt.subplot(4, 3, 8)
plt.xlabel("Median Household Income ($)")
plt.ylabel("Adults with bachelor's degrees or higher (%)")
plt.scatter(medianList, bachelorAbove)

plt.subplot(4, 3, 9)
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Adults with less than a high school diploma (%)")
plt.scatter(unemployList, lessHigh)

plt.subplot(4, 3, 10)
plt.xlabel("Asian Alone (%)")
plt.ylabel("Adults with a high school diploma only (%)")
plt.scatter(asianAlone, onlyHigh)

plt.subplot(4, 3, 11)
plt.xlabel("Asian Alone (%)")
plt.ylabel("Adults with bachelor's degrees or higher (%)")
plt.scatter(asianAlone, bachelorAbove)

plt.subplot(4, 3, 12)
plt.xlabel("Hispanic (%)")
plt.ylabel("Adults with less than a high school diploma (%)")
plt.scatter(hispanic, lessHigh)

plt.show()
#Plotting combined index against deaths per 100,000 people
plt.rcParams["figure.figsize"] = (16, 8)

allLists = [povallList, unemployList, medianList, whiteAlone, blackAlone, hispanic, asianAlone, nativeAlone, lessHigh, onlyHigh, someCollege, bachelorAbove]
allCoeff = []


#Reduced lists against deaths per 100,000 people
y = reduce(deathPopList)
for i in range(len(allLists)):
    allCoeff.append(coeff(normalize(reduce(allLists[i])), y))

combinedIndex = np.array([0 for i in range(len(y))])
for i in range(len(allLists)):
    combinedIndex = combinedIndex + (allCoeff[i] * normalize(reduce(allLists[i])))

plt.subplot(1, 2, 1)
plt.xlabel("Combined Socio-Economic Index")
plt.ylabel("COVID-19 Deaths per 100,000 people")
plt.title("Correlation coefficient for reduced lists: " + str(coeff(combinedIndex, y)))
plt.scatter(combinedIndex, y)
plt.plot(combinedIndex, regLine(combinedIndex, y))


#All 2959 counties combined index against natural log of number of deaths per 100,000 people
y = ma.log(deathPopList)
allCoeff.clear()
for i in range(len(allLists)):
    allCoeff.append(coeff(normalize(allLists[i]), y))

combinedIndex = np.array([0 for i in range(len(y))])
for i in range(len(allLists)):
    combinedIndex = combinedIndex + (allCoeff[i] * normalize(allLists[i]))

plt.subplot(1, 2, 2)
plt.xlabel("Combined Socio-Economic Index")
plt.ylabel("Natural Log of COVID-19 Deaths per 100,000 people")
plt.title("Correlation coefficient for all lists: " + str(coeff(combinedIndex, y)))
plt.scatter(combinedIndex, y)
plt.plot(combinedIndex, regLine(combinedIndex, y))


plt.show()