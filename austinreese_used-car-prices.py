import pandas as pd

from IPython.core.display import display, HTML



data = pd.read_csv("../input/craigslistVehiclesFull.csv")

data.head()
def buildPercentileFrame(data, targetAttribute, groupOnAttribute):

    #Clean outlying data without affecting original dataframe (year category requires cleaning but not as much as other fields)

    tempData = data.copy()

    if groupOnAttribute != "year":

        tempData[groupOnAttribute] = tempData[groupOnAttribute][~((tempData[groupOnAttribute]-tempData[groupOnAttribute].mean()).abs() > 3*tempData[groupOnAttribute].std())]

    else:

        tempData[groupOnAttribute] = tempData[groupOnAttribute][~((tempData[groupOnAttribute]-tempData[groupOnAttribute].mean()).abs() > 10*tempData[groupOnAttribute].std())]

    if targetAttribute != "year":

        tempData[targetAttribute] = tempData[targetAttribute][~((tempData[targetAttribute]-tempData[targetAttribute].mean()).abs() > 3*tempData[targetAttribute].std())]

    else:

        tempData[targetAttribute] = tempData[targetAttribute][~((tempData[targetAttribute]-tempData[targetAttribute].mean()).abs() > 10*tempData[targetAttribute].std())]



    #Build a list of 11 percentiles ranging from 0 through 100 for the group on attribute

    xPercentiles = [tempData[groupOnAttribute].quantile((i+1)/10) for i in range(10)]

    xPercentiles.insert(0, 0)

    

    #percentileDict will later be used to store lists of means for all 10 group by attribute percentiles

    percentileDict = {"mean_between_percentiles": [f"{i}-{i+10}" for i in range(0, 100, 10)]}

    

    #Loop through all percentiles of group by attribute to find subsequent percentiles for target attribute

    for i in range(10):

        #Build a temporary frame of rows between two 'group by attribute' percentiles

        xPercentileFrame = tempData[tempData[groupOnAttribute].between(xPercentiles[i], xPercentiles[i + 1])]

        

        #Build a list of 11 percentiles ranging from 0 through 100 for the target attribute using the temporary group by frame just created

        yPercentiles = [xPercentileFrame[targetAttribute].quantile((j+1)/10) for j in range(10)]

        yPercentiles.insert(0, 0)

        

        #Gather means for target attribute at all 10 percentiles

        yMeans = []

        for j in range(10):

            yPercentileFrame = xPercentileFrame[xPercentileFrame[targetAttribute].between(yPercentiles[j], yPercentiles[j+1])]

            yMeans.append(int(yPercentileFrame[targetAttribute].mean()))

        

        #Finally, add the data to a dictionary

        if len(percentileDict) == 1:

            percentileDict[f"{groupOnAttribute} between {round(xPercentiles[i])} and {round(xPercentiles[i+1])}"] = yMeans

        else:

            percentileDict[f"{round(xPercentiles[i])}-{round(xPercentiles[i+1])}"] = yMeans

            

    del tempData

    

    #Return an html table of the data

    return HTML(pd.DataFrame(percentileDict).set_index("mean_between_percentiles").to_html())
display(buildPercentileFrame(data, "price", "odometer"))
fordData = data[data.manufacturer == 'ford']

display(buildPercentileFrame(fordData, "price", "odometer"))
midsizeFords = fordData[fordData["size"] == 'mid-size']

midsizeFordSedans = midsizeFords[midsizeFords.type == "sedan"]

execellentMidsizeFordSedans = midsizeFordSedans[midsizeFordSedans.condition == "excellent"]

display(buildPercentileFrame(execellentMidsizeFordSedans, "price", "odometer"))
newerMidsizeFordSedans = execellentMidsizeFordSedans[execellentMidsizeFordSedans.year >= 2015]

display(buildPercentileFrame(newerMidsizeFordSedans, "price", "odometer"))

newerMidsizeFordSedans.size