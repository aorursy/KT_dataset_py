import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
import time
from scipy.stats import norm
from scipy.integrate import odeint
%matplotlib inline

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

font = {'family' : 'DejaVu Sans', 'weight' : 'bold', 'size'   : 18}
plt.rc('font', **font)

data = pd.read_csv("../input/multipleChoiceResponses.csv", encoding='latin-1', dtype='object')
curr = pd.read_csv("../input/conversionRates.csv", encoding='latin-1', dtype='object')
currencies = curr.loc[:,['originCountry', 'exchangeRate']]
currencies.set_index('originCountry', inplace=True)
data.info()
def changeItemNamesInList(chosenList, newNames, oldNames, printToConsole = False):
    if len(chosenList) > 0:
        allEntered = False
        for x in range(len(chosenList)):
            for j in range(len(newNames)):
                if str(chosenList[x]) == str(oldNames[j]):
                    chosenList[x] = newNames[j]

                    allEntered = True
                    for k in range(len(newNames)):
                        if newNames[k] not in chosenList:
                            allEntered = False
                    if allEntered:
                        if printToConsole:
                            print("Item changed from:\n\t {0}\nto:\n\t{1}\n".format(oldNames[k],newNames[k]))
                        return chosenList
        if not allEntered:
            print("\nSome items were not changed:")
            for k in range(len(newNames)):
                if newNames[k] not in chosenList:
                    print("\t- {0}".format(newNames[k]))
            return chosenList
    else:
        print("List is empty.")
        return []
        
def drawNumbers(ax, rects, labels, char):
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 2, str(label)+ char, ha='center', va='bottom')

def returnLen( table, column, value):
    return len(table[table[column] == value].index)

def returnLenOfNullAndNaN( table, column ):
    return len(table[table[column].isnull()].index)

def gogo(x):
    printItems =False
    oldX = x
    if printItems:
        print("This is starting x: {0}".format(x))
    try:
        if ',' in x:
            while True:
                pos = x.find(',')
                diff = len(x) - pos - 1
                if printItems:
                    print("Pos and diff: {0} {1}".format(pos,diff))
                if diff > 2:
                    x = str(x[0:pos]) + str(x[pos+1:len(x)])
                    if printItems:
                        print("diff > 2 new x: {0}".format(x))
                else:
                    y = x[(pos+1):len(x)]
                    if printItems:
                        print("y: {0}".format(y))
                    if y.find(',') == -1:
                        x = x.replace(',','.')
                        if printItems:
                            print("New ',' on y not found, new x is: {0}".format(x))
                    else:
                        x = str(x[0:pos]) + str(x[pos+1:len(x)])
                        if printItems:
                            print("New ',' on y IS found, new x is: {0}".format(x))
                if x.find(',') == -1:
                    if printItems:
                        print("--------------------------------------------")
                    break
                if printItems:
                    print("-------------ONE MORE TIME -------------------------------")
        x = float(x)
        if printItems and x >= 1000000:
            print("this is old X: {0}".format(oldX))
            print("--------------------------------------------")
            print()
    except ValueError:
        x = np.nan
    finally:
        if np.isnan(x) or x < 35000:
            x = np.nan
        return x

def deleteNanAndCalculateSalary(df, currencies):
    df.dropna(subset=['CompensationAmount', 'CompensationCurrency'], inplace=True)
    df['CompensationAmount'] = df['CompensationAmount'].apply(lambda x: gogo(x))
    df.dropna(subset=['CompensationAmount'], inplace=True)
    for i in df.iterrows():
        row = list(i[1])
        if row[2] != 'SPL':
            df.at[i[0], 'CompensationAmount'] = round(float(df.at[i[0], 'CompensationAmount']) * float(currencies.loc[row[2]].values[0]),2)
            currencyU = currencies.loc[row[2]].values[0]
        else:
            df.at[i[0], 'CompensationAmount'] = round(float(df.at[i[0], 'CompensationAmount']) * 6.0,2)
            currencyU = 6.0
            
        if str(i[0]) == '804':
            print(i)
            print("This is the multiply: {0}".format(df.at[i[0], 'CompensationAmount']))
            print("This is currency: {0}".format(currencyU))
    return df

def stats(table, tableName, column, printInfo = False):
    selected = list(set(table[column].tolist()))
    if printInfo:
        print("*******************************************************************************")
        print("STATS FOR ---> TABLE `" + tableName + "`, COLUMN `" + column + "`")
        print("*******************************************************************************")
    
    totalTableLength = len(table.index)
    resultList = []
    
    for x in selected:
        if str(x) == 'nan':
            tempLength = returnLenOfNullAndNaN( table, column )
            string = "NAN_NULL"
        else:
            tempLength = returnLen( table, column, x)
            if x != "N/A, I did not receive any formal education":
                string = str(x).capitalize()
            else:
                string = x
            
        percent = round(tempLength*100.0/totalTableLength, 1)
        resultList.append((string,tempLength, percent))
    t = list(zip(*resultList))
    result_df = pd.DataFrame({'name': t[0], 'count': t[1], 'percentage': t[2]}, columns=['name', 'count', 'percentage'])
    result_df.set_index('name', drop=True, inplace=True)
    result_df.sort_values('percentage', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last')
    if printInfo:
        print(result_df)
        print()
    return result_df

def returnSeriesFromDataframeByIndex(df, listOfIndex, indexColumn, explodeColumn, explodeValue):
    resultDF = []
    for i in listOfIndex:
        if explodeValue == 'NAN_NULL':
            tempDF = df[(df[indexColumn] == i) & (df[explodeColumn].isnull())]
        else:
            tempDF = df[(df[indexColumn] == i) & (df[explodeColumn] == explodeValue)]
        resultDF.append(len(tempDF.index))
    return pd.Series(resultDF, name=explodeValue)

def returnDistributionDataframe(df, indexColumn, explodeColumn):
    allIndexes = list(set(df[indexColumn].tolist()))
    if np.nan in allIndexes:
        allIndexes.remove(np.nan)
    resultList = []
    dfTemp = stats(df, '', explodeColumn, False)
    dfExplodeColumnList = dfTemp.index
    df = df.dropna(subset=[indexColumn])
    
    resultList.append(pd.Series(allIndexes, name="name"))
    
    for i in dfExplodeColumnList:
        if i == 'NAN_NULL':
            temp = df[df[explodeColumn].isnull()]
            resultList.append(returnSeriesFromDataframeByIndex(temp, allIndexes, indexColumn, explodeColumn, i))
        else:
            temp = df[df[explodeColumn] == i]
            resultList.append(returnSeriesFromDataframeByIndex(temp, allIndexes, indexColumn, explodeColumn, i))
    
    df3 = pd.DataFrame(pd.Series(allIndexes, name="name"), columns=['name'])
    for i in resultList:
        df3.loc[:, i.name] = i
    df3.set_index('name', drop=True, inplace=True)
    return df3

def returnStatsForColumnCount(df, statsColumn, printInfo = False):
    totalSum = df[df.columns].sum().sum()
    columnSum = df[statsColumn].sum()
    percent = round(columnSum*100.0/totalSum,1)
    if printInfo:
        print("*******************************************************************************")
        print("STATS FOR COLUMN ---> `" + statsColumn + "`")
        print("*******************************************************************************\n")
        print("`{0}` sum: {1} or {2}%".format(statsColumn,df[statsColumn].sum(),percent))
        print("Total table sum: {0}\n".format(df[df.columns].sum().sum()))
        print("`{0}` found in:\n------------------------".format(statsColumn))
        for country in df.index:
            sumPerCountry = int(df.loc[country,statsColumn])
            percent = round(sumPerCountry*100.0/totalSum,2)
            if sumPerCountry > 0:
                print("- {0}: {1} or {2}%".format(country,sumPerCountry,percent))
        print()

def returnHistogramList(df, explodeColumn, valueColumn, histType = ''):
    explode =  list(set(df[explodeColumn].tolist()))
    resultList = []
    maxRows, maxIndex = 0, 0
    
    for i in np.arange(len(explode)):
        tempList = []
        
        if str(explode[i]) == 'nan' or str(explode[i]) == 'NAN_NULL':
            explode[i] = 'NAN_NULL'
            tempDF = list(df.loc[df[explodeColumn].isnull()].loc[:,valueColumn].values)
        else:
            tempDF = list(df.loc[df[explodeColumn] == str(explode[i])].loc[:,valueColumn].values)
            
        tempList = sorted(list(tempDF))
        
        if histType == 'age':
            for j in np.arange(len(tempList)):
                if int(tempList[j]) < 14 or int(tempList[j]) > 80:
                    tempList[j] = np.nan
        elif histType == 'salary':
            for j in np.arange(len(tempList)):
                if str(tempList[j]) != 'nan':
                    if int(tempList[j]) < 35000 or int(tempList[j]) > 1000000:
                        tempList[j] = np.nan
                
        resultList.append((str(explode[i]),tempList))
    resultDFList = []
    columns = []
    if len(explode) > 1:
        count=0
        sumi = 0
        for i in np.arange(len(resultList)):
            j,k = (*resultList[i],)
            sumi += len(k)
            if np.nan in k:
                k = [x for x in k if x is not np.nan]
            tempDF2 = pd.DataFrame({j:k}, columns=[j], index=np.arange(len(k)))
            resultDFList.append((j,tempDF2))
            if len(k) > maxRows:
                maxRows = len(k)
                maxIndex = i
        i,j = (*resultDFList[maxIndex],)
        finalDF = j
        finalDF.dropna(axis=0, inplace=True)
        columns.append(i)
        for i in np.arange(len(resultDFList)):
            if i != maxIndex:
                j,k = resultDFList[i]
                columns.append(j)
                finalDF = pd.concat((finalDF,k), ignore_index=True, axis=1)
        finalDF.columns = columns
        return finalDF

def legendTicksAndAxStuff(ax, legendEntries=[], legTitle="", xLabel="", yLabel="", fontSize=0, grid=False, xTicks=[], xMinorTicks=[], yTicks=[], yMinorTicks=[], xRot=0 , yRot=0, colors=[]):
    if len(colors) == 0:
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    # ticks 
    if len(xTicks) > 0:
        ax.set_xticks(xTicks, minor=False)
    if len(yTicks) > 0:
        ax.set_yticks(yTicks, minor=False)
    # Minor ticks
    if len(xMinorTicks) > 0 or len(yMinorTicks) > 0:
        ax.minorticks_on()
        if len(xMinorTicks) > 0:
            ax.set_xticks(xMinorTicks, minor=True)
        if len(yMinorTicks) > 0:
            ax.set_yticks(yMinorTicks, minor=True)
        if len(xMinorTicks) > 0 and len(yMinorTicks) == 0:
            ax.tick_params(axis='y',which='minor',left='off')
        elif len(xMinorTicks) == 0 and len(yMinorTicks) > 0:
            ax.tick_params(axis='x',which='minor',bottom='off')
    # labels
    if xLabel != "" or yLabel != "":
        label = ax.set(xlabel=xLabel, ylabel=yLabel)
    if grid:
        ax.grid()
        
    patches = []
    endSize = 14
    if len(legendEntries) > 0:
        for i in np.arange(len(legendEntries)):
            colorValue = i % 10
            if fontSize != 0:
                endSize = fontSize
            patches.append(mpatches.Patch(color=colors[colorValue], label=legendEntries[i]))
        ax.legend(handles=list(patches), title=legTitle, fontsize=endSize)
    else:
        ax.legend(title=legTitle, fontsize=endSize)
    
    if xRot != 0:
        for tick in ax.get_xticklabels():
            tick.set_rotation(xRot)
    if yRot != 0:
        for tick in ax.get_yticklabels():
            tick.set_rotation(yRot)

def drawMedianAndAverage(average, median):
    plt.axvline(x=average, color=colors[8], ls="--", label="Average", linewidth=2)
    plt.axvline(x=median, color=colors[9], ls="--", label="Median", linewidth=2)            

import timeit

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def calcMedianAverageForDFColumn(df, column):
    if column == -1:
        column = df.columns.values
    listValues = [float(x) for x in df[column].iloc[:].dropna().values]
    listValues = sorted(np.asarray(listValues))
    average = int(round(np.mean(listValues, axis=0),0))
    median = int(round(np.median(listValues, axis=0),0))
    count = len(listValues)
    return [average, median, count]

def checkIfNumber(potentialNumber):
    try:
        int(potentialNumber)
        return True
    except ValueError:
        return False

def returnNumberPartAndStringPartFromString(word):
    number = ''
    string = ''
    for i in np.arange(len(word)):
        try:
            value = str(int(word[i]))
            isNumber = True
        except ValueError:
            isNumber = False
        finally:
            if isNumber:
                number += value
            else:
                string += word[i]
    return (int(number), string)

def modifyDfToGiveLogAxisForSalaray(df):
    resultList = []
    
    df.drop('NAN_NULL', inplace=True)
    index = list(df.index.values)
    change = {"<1mb" : 0, "mb" : 1e6,  "gb" : 1e9, "tb": 1e12, "pb" : 1e15}
    
    for i in np.arange(len(index)):
        word = str(index[i])
        number = checkIfNumber(word[0])
        if word == "<1mb":
            index[i] = change[word]
            resultList.append(0)
        elif number:
            wordList = returnNumberPartAndStringPartFromString(word)
            num,char = wordList
            exponent = int(change[char])
            num = int(num) * exponent
            resultList.append(num)
    df.index = pd.Series(resultList, name="name")

def prepareForScatterPlot(df, xAxisColumn, yAxisColumn):
    tempDF = df.loc[:,[xAxisColumn, yAxisColumn]].dropna(subset=[xAxisColumn, yAxisColumn], axis=0)
    tempDF.loc[:,[xAxisColumn]] = tempDF.loc[:,[xAxisColumn]].apply(pd.to_numeric, errors='raise')
    tempDF.loc[:,[yAxisColumn]] = tempDF.loc[:,[yAxisColumn]].apply(pd.to_numeric, errors='raise')
    return tempDF

def returnFit(listToFit):
    filteredList = [x for x in listToFit if str(x) != 'nan']
    filteredList = sorted(filteredList)
    fitt = norm.pdf(filteredList, np.mean(filteredList), np.std(filteredList))
    return [filteredList, fitt]


### Plotting constants ###

# color palette we will be using
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

# string constants for changing columns/index names for plotting purposes
# GENDER
change_NB_GQ_GNC = [["N-B, GQ, or G-NC"], ["Non-binary, genderqueer, or gender non-conforming"]]
change_A_DIFF_ID = [["A diff. id."], ["A different identity"]]
# CURRENT EMPLOYMENT STATUS
change_IND_CONT_FREE_SE = [["Ind. cont., freelancer, or self-E"], ["Independent contractor, freelancer, or self-employed"]]
# EDUCATION IMPORTANCE
change_NA_I_DIDNT_REC_FE = [["N/A, I didn't receive any FE"], ["N/A, I did not receive any formal education"]]
allDataScientists = data.loc[data['CurrentJobTitleSelect'] == 'Data Scientist', :]
genderAllStats = stats(data.loc[:,['GenderSelect']], 'All', 'GenderSelect', True)
genderUsStats = stats(data.loc[data['Country'] == 'United States'], 'US_applicants', 'GenderSelect', False)
genderAllStats.index = changeItemNamesInList(list(genderAllStats.index.values), change_NB_GQ_GNC[0], change_NB_GQ_GNC[1])
genderUsStats.index = changeItemNamesInList(list(genderUsStats.index.values), change_NB_GQ_GNC[0], change_NB_GQ_GNC[1])

fig = plt.figure(figsize=(15, 8))

ax1 = plt.subplot2grid((1, 2), (0, 0))
genderAllStats['count'].plot(kind='bar', ax=ax1, legend=None, title='All applicants', ylim=(0,16000), color=colors[0])
legendTicksAndAxStuff(ax1, [], "", "", "Count", 18, True, [], [], np.arange(0,18000,2000), np.arange(0,16000,500), 0, 0, [])
labels = genderAllStats['percentage']
drawNumbers(ax1, ax1.patches, labels, "%")

ax2 = plt.subplot2grid((1, 2), (0, 1))
genderUsStats['count'].plot(kind='bar', ax=ax2, legend=None, title='US applicants', ylim=(0,4000), color=colors[1])
legendTicksAndAxStuff(ax2, [], "", "", "Count", 18, True, [], [], np.arange(0,4500,500), np.arange(0,4000,100), 0, 0, [])
labels = genderUsStats['percentage']
drawNumbers(ax2, ax2.patches, labels, "%")

text=plt.text(1, 1.3, "Gender distribution - All vs. US", horizontalalignment='center', fontsize=25, transform = ax1.transAxes)

plt.tight_layout()

genderAllDsStats = stats(allDataScientists.loc[:,['GenderSelect']], 'All Data scientists', 'GenderSelect', False)
genderUsDsStats = stats(allDataScientists.loc[allDataScientists['Country'] == 'United States'].loc[:,['GenderSelect']], 'US Data scientists', 'GenderSelect', False)

genderAllDsStats.index = changeItemNamesInList(list(genderAllDsStats.index.values), change_NB_GQ_GNC[0], change_NB_GQ_GNC[1])
genderUsDsStats.index = changeItemNamesInList(list(genderUsDsStats.index.values), change_NB_GQ_GNC[0], change_NB_GQ_GNC[1])

fig = plt.figure(figsize=(15, 8))

ax1 = plt.subplot2grid((1, 2), (0, 0))
genderAllDsStats['count'].plot(kind='bar', ax=ax1, legend=None, title='All applicants', ylim=(0,2500), color=colors[0])
legendTicksAndAxStuff(ax1, [], "", "", "Count", 18, True, [], [], np.arange(0,3000,500), np.arange(0,2500,100), 0, 0, [])
labels = genderAllStats['percentage']
drawNumbers(ax1, ax1.patches, labels, "%")

ax2 = plt.subplot2grid((1, 2), (0, 1))
genderUsDsStats['count'].plot(kind='bar', ax=ax2, legend=None, title='US applicants', ylim=(0,800), color=colors[1])
legendTicksAndAxStuff(ax2, [], "", "", "Count", 18, True, [], [], np.arange(0,900,100), np.arange(0,800,25), 0, 0, [])
labels = genderUsStats['percentage']
drawNumbers(ax2, ax2.patches, labels, "%")

text=plt.text(1, 1.3, "Gender distribution of Data scientists - All vs. US", horizontalalignment='center', fontsize=25, transform = ax1.transAxes)

plt.tight_layout()
col_names = ['GenderSelect', 'Country', 'CurrentJobTitleSelect', 'Age', 'TitleFit', 'EmploymentStatus', 'UniversityImportance', 'WorkDatasetSize', 'CompensationAmount']
countryAll = data.loc[:,['Country']]
resultCountry = stats(countryAll, "Country", "Country", False)

ax=resultCountry['count'].plot(kind='bar', figsize=(16,8), fontsize=16, ylim=(0,4500),
                        title="Applicant distribution per country", color=[colors])
label = ax.set(xlabel="", ylabel="Count")
ax.minorticks_on()
ax.tick_params(axis='x',which='minor',bottom='off')
plt.grid()
data.head(5)
countryGenderAll = data.loc[:,['Country', 'GenderSelect']]
countryGenAll = returnDistributionDataframe(countryGenderAll, 'Country', 'GenderSelect')
countryGenAll.columns = changeItemNamesInList(list(countryGenAll.columns), change_NB_GQ_GNC[0], change_NB_GQ_GNC[1])
returnStatsForColumnCount(countryGenAll, 'NAN_NULL', True)
countryGenAll.drop('NAN_NULL', axis=1, inplace=True)
countryGenAll.sort_values('Male', axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last')
countryGenAll.head(5)
fig = plt.figure(figsize=(12, 11))

half = int(len(countryGenAll.index)/2)
countryGenAll.sort_values("Male", axis=0, ascending=False, inplace=True, kind='quicksort')

ax1 = plt.subplot2grid((2, 1), (0,0))
countryGenAll.iloc[0:half].plot(kind="bar", ax=ax1, fontsize=15, ylim=(0,3500), title="Gender count per country")
legendTicksAndAxStuff(ax1, [], "", "", "Count", 12, False, [],  [], np.arange(0,4000, 500), np.arange(0,3500, 100), 0, 0, [])

ax2 = plt.subplot2grid((2, 1), (1,0))
countryGenAll.iloc[half:].plot(kind="bar", ax=ax2, fontsize=15, ylim=(0,3500), title="Gender count per country")
legendTicksAndAxStuff(ax2, [], "", "", "Count", 12, False, [],  [], np.arange(0,4000, 500), np.arange(0,3500, 100), 0, 0, [])

fig.tight_layout()
del(countryGenAll)
ageCountryGenderAll = data.loc[:,['Age', 'GenderSelect', 'Country', 'CurrentJobTitleSelect']]
print(len(ageCountryGenderAll.index))
ageCountryGenderAll.dropna(subset=['Age'], axis=0, inplace=True)
ageCountryGenderAll['Age'] = ageCountryGenderAll['Age'].apply(lambda x: int(x))
newLength = len(ageCountryGenderAll.index)
print(newLength)

nanSum = 0
for i in list(ageCountryGenderAll['GenderSelect']):
    if i is np.nan:
        nanSum += 1
percent = round(nanSum*100.0/newLength,2)
print("There are {0} NaN items, which is {1}% of new DF length (new DF length = {2})".format(nanSum, percent, newLength))
ageCountryGenderAll.head(5)
newDF = returnHistogramList(ageCountryGenderAll, 'GenderSelect', 'Age', 'age')
newDF.columns = changeItemNamesInList(list(newDF.columns), change_NB_GQ_GNC[0], change_NB_GQ_GNC[1], False)

fig = plt.figure(figsize=(15, 14))
bins = np.linspace(0, 100, 50)
print("{0} {1}".format(len(ageCountryGenderAll.index),len(newDF.index)))

ax1 = plt.subplot2grid((2, 4), (0,0), colspan=2)
ageCountryGenderAll.plot(kind='hist', bins=bins, ax=ax1, title="Unfiltered", alpha=0.9, xlim=(0,100))
legendTicksAndAxStuff(ax1, ['All gender'], "", "Age", "Count", 18, True, np.arange(0,110,10), np.arange(0,102,2), np.arange(0,2200,200), np.arange(0,2050,50), 0, 0)

ax2 = plt.subplot2grid((2, 4), (0,2), colspan=2)
newDF.plot(kind='hist', bins=bins, stacked=True, ax=ax2, title="Filtered", color=colors, xlim=(0,100))
legendTicksAndAxStuff(ax2, [], "Gender", "Age", "Count", 14,True, np.arange(0,110,10), np.arange(0,102,2), np.arange(0,2200,200), np.arange(0,2050,50), 0, 0)

ax3 = plt.subplot2grid((2, 4), (1,0), colspan=4)
ageCountryGenderAll.plot(kind='hist', bins=bins, ax=ax3, alpha=0.6, xlim=(0,100), color=colors[0])
newDF.plot(kind='hist', bins=bins, stacked=True, ax=ax3, alpha=0.6, xlim=(0,100), title="Unfiltered vs. Filtered", color=colors[1])
legendTicksAndAxStuff(ax3, ['Unfiltered', 'Filtered'], "", "Age", "Count", 18, True, np.arange(0,110,10), np.arange(0,102,2), np.arange(0,2200,200),np.arange(0,2050,50), 0, 0)

text=ax1.text(1, 1.2, "Age outlier filtering - unfiltered vs. filtered DataFrame", horizontalalignment='center', fontsize=30, transform = ax1.transAxes)

plt.tight_layout()
statsList = []
statsList.append(calcMedianAverageForDFColumn(pd.DataFrame(newDF.values.flatten().tolist(), columns=['All']), "All"))

for i in newDF.columns.values:
    statsList.append(calcMedianAverageForDFColumn(newDF, i))

newDF_MedAvgCount = pd.DataFrame(statsList, columns=['average' ,'median', 'count'], index=(['All'] + list(newDF.columns.values)))
newDF_MedAvgCount.sort_values(["count"], axis=0, ascending=False, inplace=True, kind='quicksort')

fig = plt.figure(figsize=(14, 10))

ax1 = plt.subplot2grid((3, 6), (0, 0), colspan=3)
newDF['Male'].plot(kind='hist', bins=bins, ax=ax1, title='Male age distribution', color=colors[0], xlim=(0,100), ylim=(0,1500))
drawMedianAndAverage(newDF_MedAvgCount.at['Male','average'], newDF_MedAvgCount.at['Male','median'])
legendTicksAndAxStuff(ax1, [], "", "Age", "Count", 18, True, np.arange(0,110,10), np.arange(0,100,2), np.arange(0,1800,300), np.arange(0,1500,100), 0, 0, [colors[0]])

ax2 = plt.subplot2grid((3, 6), (0, 3), colspan=3)
newDF['Female'].plot(kind='hist', bins=bins, ax=ax2, title='Female age distribution', color=colors[1], xlim=(0,100), ylim=(0,400))
drawMedianAndAverage(newDF_MedAvgCount.at['Female','average'], newDF_MedAvgCount.at['Female','median'])
legendTicksAndAxStuff(ax2, [], "", "Age", "Count", 18,True, np.arange(0,110,10), np.arange(0,100,2), np.arange(0,450,50), np.arange(0,400,10), 0, 0, [colors[1]])

ax3 = plt.subplot2grid((3, 6), (1, 0), colspan=2)
newDF['N-B, GQ, or G-NC'].plot(kind='hist', bins=bins, ax=ax3, title='N-B, GQ, or G-NC', color=colors[2], xlim=(0,100), ylim=(0,20))
drawMedianAndAverage(newDF_MedAvgCount.at['N-B, GQ, or G-NC','average'], newDF_MedAvgCount.at['N-B, GQ, or G-NC','median'])
legendTicksAndAxStuff(ax3, [], "", "Age", "Count", 12,True, np.arange(0,120,20), np.arange(0,100,5), np.arange(0,22,2), np.arange(0,20,1), 0, 0, [colors[2]])

ax4 = plt.subplot2grid((3, 6), (1, 2), colspan=2)
newDF['A different identity'].plot(kind='hist', bins=bins, ax=ax4, label='A diff. id.', title='A different identity', color=colors[3], xlim=(0,100), ylim=(0,20))
drawMedianAndAverage(newDF_MedAvgCount.at['A different identity','average'], newDF_MedAvgCount.at['A different identity','median'])
legendTicksAndAxStuff(ax4, [], "", "Age", "Count", 12,True, np.arange(0,120,20), np.arange(0,100,5), np.arange(0,22,2), np.arange(0,20,1), 0, 0, [colors[3]])

ax5 = plt.subplot2grid((3, 6), (1, 4), colspan=2)
newDF['NAN_NULL'].plot(kind='hist', bins=bins, ax=ax5, title='NAN_NULL', color=colors[4], xlim=(0,100), ylim=(0,20))
drawMedianAndAverage(newDF_MedAvgCount.at['NAN_NULL','average'], newDF_MedAvgCount.at['NAN_NULL','median'])
legendTicksAndAxStuff(ax5, [], "", "Age", "Count", 12,True, np.arange(0,120,20), np.arange(0,100,5), np.arange(0,22,2), np.arange(0,20,1), 0, 0, [colors[4]])

text=plt.text(1, 1.3, "Age distribution per gender - All applicants", horizontalalignment='center', fontsize=25, transform = ax1.transAxes)

plt.tight_layout()
plt.show()

newDF_MedAvgCount.head(6)
newDF = returnHistogramList(ageCountryGenderAll.loc[ageCountryGenderAll['Country'] == 'United States'].loc[:,['GenderSelect', 'Age']], 'GenderSelect', 'Age', 'age')
newDF.columns = changeItemNamesInList(list(newDF.columns), change_NB_GQ_GNC[0], change_NB_GQ_GNC[1], False)

statsList = []
statsList.append(calcMedianAverageForDFColumn(pd.DataFrame(newDF.values.flatten().tolist(), columns=['All']), "All"))

for i in newDF.columns.values:
    statsList.append(calcMedianAverageForDFColumn(newDF, i))

newDF_MedAvgCount = pd.DataFrame(statsList, columns=['average' ,'median', 'count'], index=(['All'] + list(newDF.columns.values)))
newDF_MedAvgCount.sort_values(["count"], axis=0, ascending=False, inplace=True, kind='quicksort')

fig = plt.figure(figsize=(14, 10))

ax1 = plt.subplot2grid((3, 6), (0, 0), colspan=3)
newDF['Male'].plot(kind='hist', bins=bins, ax=ax1, title='Male age distribution', color=colors[0], xlim=(0,100), ylim=(0,350))
drawMedianAndAverage(newDF_MedAvgCount.at['Male','average'], newDF_MedAvgCount.at['Male','median'])
legendTicksAndAxStuff(ax1, [], "", "Age", "Count", 18, True, np.arange(0,110,10), np.arange(0,100,2), np.arange(0,400,50), np.arange(0,350,10), 0, 0, [colors[0]])

ax2 = plt.subplot2grid((3, 6), (0, 3), colspan=3)
newDF['Female'].plot(kind='hist', bins=bins, ax=ax2, title='Female age distribution', color=colors[1], xlim=(0,100), ylim=(0,125))
drawMedianAndAverage(newDF_MedAvgCount.at['Female','average'], newDF_MedAvgCount.at['Female','median'])
legendTicksAndAxStuff(ax2, [], "", "Age", "Count", 18,True, np.arange(0,110,10), np.arange(0,100,2), np.arange(0,150,25), np.arange(0,125,5), 0, 0, [colors[1]])

ax3 = plt.subplot2grid((3, 6), (1, 0), colspan=2)
newDF['N-B, GQ, or G-NC'].plot(kind='hist', bins=bins, ax=ax3, title='N-B, GQ, or G-NC', color=colors[2], xlim=(0,100), ylim=(0,10))
drawMedianAndAverage(newDF_MedAvgCount.at['N-B, GQ, or G-NC','average'], newDF_MedAvgCount.at['N-B, GQ, or G-NC','median'])
legendTicksAndAxStuff(ax3, [], "", "Age", "Count", 12,True, np.arange(0,120,20), np.arange(0,100,5), np.arange(0,12,2), np.arange(0,10,1), 0, 0, [colors[2]])

ax4 = plt.subplot2grid((3, 6), (1, 2), colspan=2)
newDF['A different identity'].plot(kind='hist', bins=bins, ax=ax4, label='A diff. id.', title='A different identity', color=colors[3], xlim=(0,100), ylim=(0,10))
drawMedianAndAverage(newDF_MedAvgCount.at['A different identity','average'], newDF_MedAvgCount.at['A different identity','median'])
legendTicksAndAxStuff(ax4, [], "", "Age", "Count", 12,True, np.arange(0,120,20), np.arange(0,100,5), np.arange(0,12,2), np.arange(0,10,1), 0, 0, [colors[3]])

ax5 = plt.subplot2grid((3, 6), (1, 4), colspan=2)
newDF['NAN_NULL'].plot(kind='hist', bins=bins, ax=ax5, title='NAN_NULL', color=colors[4], xlim=(0,100), ylim=(0,10))
drawMedianAndAverage(newDF_MedAvgCount.at['NAN_NULL','average'], newDF_MedAvgCount.at['NAN_NULL','median'])
legendTicksAndAxStuff(ax5, [], "", "Age", "Count", 12,True, np.arange(0,120,20), np.arange(0,100,5), np.arange(0,12,2), np.arange(0,10,1), 0, 0, [colors[4]])

text=plt.text(1, 1.3, "Age distribution per gender - US applicants", horizontalalignment='center', fontsize=25, transform = ax1.transAxes)

plt.tight_layout()
plt.show()

newDF_MedAvgCount.head(6)
newDF = returnHistogramList(ageCountryGenderAll.loc[ageCountryGenderAll['CurrentJobTitleSelect'] == 'Data Scientist'].loc[:,['GenderSelect', 'Age']], 'GenderSelect', 'Age', 'age')
newDF.columns = changeItemNamesInList(list(newDF.columns), change_NB_GQ_GNC[0], change_NB_GQ_GNC[1], False)

statsList = []
statsList.append(calcMedianAverageForDFColumn(pd.DataFrame(newDF.values.flatten().tolist(), columns=['All']), "All"))

for i in newDF.columns.values:
    statsList.append(calcMedianAverageForDFColumn(newDF, i))

newDF_MedAvgCount = pd.DataFrame(statsList, columns=['average' ,'median', 'count'], index=(['All'] + list(newDF.columns.values)))
newDF_MedAvgCount.sort_values(["count"], axis=0, ascending=False, inplace=True, kind='quicksort')

fig = plt.figure(figsize=(14, 10))

ax1 = plt.subplot2grid((3, 6), (0, 0), colspan=3)
newDF['Male'].plot(kind='hist', bins=bins, ax=ax1, title='Male age distribution', color=colors[0], xlim=(0,100), ylim=(0,250))
drawMedianAndAverage(newDF_MedAvgCount.at['Male','average'], newDF_MedAvgCount.at['Male','median'])
legendTicksAndAxStuff(ax1, [], "", "Age", "Count", 18, True, np.arange(0,110,10), np.arange(0,100,2), np.arange(0,300,50), np.arange(0,250,10), 0, 0, [colors[0]])

ax2 = plt.subplot2grid((3, 6), (0, 3), colspan=3)
newDF['Female'].plot(kind='hist', bins=bins, ax=ax2, title='Female age distribution', color=colors[1], xlim=(0,100), ylim=(0,60))
drawMedianAndAverage(newDF_MedAvgCount.at['Female','average'], newDF_MedAvgCount.at['Female','median'])
legendTicksAndAxStuff(ax2, [], "", "Age", "Count", 18,True, np.arange(0,110,10), np.arange(0,100,2), np.arange(0,70,10), np.arange(0,60,2), 0, 0, [colors[1]])

ax3 = plt.subplot2grid((3, 6), (1, 0), colspan=2)
newDF['N-B, GQ, or G-NC'].plot(kind='hist', bins=bins, ax=ax3, title='N-B, GQ, or G-NC', color=colors[2], xlim=(0,100), ylim=(0,6))
drawMedianAndAverage(newDF_MedAvgCount.at['N-B, GQ, or G-NC','average'], newDF_MedAvgCount.at['N-B, GQ, or G-NC','median'])
legendTicksAndAxStuff(ax3, [], "", "Age", "Count", 12,True, np.arange(0,120,20), np.arange(0,100,5), np.arange(0,6,1), [], 0, 0, [colors[2]])

ax4 = plt.subplot2grid((3, 6), (1, 2), colspan=2)
newDF['A different identity'].plot(kind='hist', bins=bins, ax=ax4, label='A diff. id.', title='A different identity', color=colors[3], xlim=(0,100), ylim=(0,6))
drawMedianAndAverage(newDF_MedAvgCount.at['A different identity','average'], newDF_MedAvgCount.at['A different identity','median'])
legendTicksAndAxStuff(ax4, [], "", "Age", "Count", 12,True, np.arange(0,120,20), np.arange(0,100,5), np.arange(0,6,1), [], 0, 0, [colors[3]])

ax5 = plt.subplot2grid((3, 6), (1, 4), colspan=2)
newDF['NAN_NULL'].plot(kind='hist', bins=bins, ax=ax5, title='NAN_NULL', color=colors[4], xlim=(0,100), ylim=(0,6))
drawMedianAndAverage(newDF_MedAvgCount.at['NAN_NULL','average'], newDF_MedAvgCount.at['NAN_NULL','median'])
legendTicksAndAxStuff(ax5, [], "", "Age", "Count", 12,True, np.arange(0,120,20), np.arange(0,100,5), np.arange(0,6,1), [], 0, 0, [colors[4]])

text=plt.text(1, 1.3, "Age distribution per gender - All Data scientists", horizontalalignment='center', fontsize=25, transform = ax1.transAxes)

plt.tight_layout()
plt.show()

newDF_MedAvgCount.head(6)
newDF = returnHistogramList(ageCountryGenderAll.loc[ageCountryGenderAll['Country'] == 'United States'].loc[ageCountryGenderAll['CurrentJobTitleSelect'] == 'Data Scientist'].loc[:,['GenderSelect', 'Age']], 'GenderSelect', 'Age', 'age')
newDF.columns = changeItemNamesInList(list(newDF.columns), change_NB_GQ_GNC[0], change_NB_GQ_GNC[1], False)

statsList = []
statsList.append(calcMedianAverageForDFColumn(pd.DataFrame(newDF.values.flatten().tolist(), columns=['All']), "All"))

for i in newDF.columns.values:
    statsList.append(calcMedianAverageForDFColumn(newDF, i))

newDF_MedAvgCount = pd.DataFrame(statsList, columns=['average' ,'median', 'count'], index=(['All'] + list(newDF.columns.values)))
newDF_MedAvgCount.sort_values(["count"], axis=0, ascending=False, inplace=True, kind='quicksort')

fig = plt.figure(figsize=(14, 10))

ax1 = plt.subplot2grid((3, 6), (0, 0), colspan=3)
newDF['Male'].plot(kind='hist', bins=bins, ax=ax1, title='Male age distribution', color=colors[0], xlim=(0,100), ylim=(0,80))
drawMedianAndAverage(newDF_MedAvgCount.at['Male','average'], newDF_MedAvgCount.at['Male','median'])
legendTicksAndAxStuff(ax1, [], "", "Age", "Count", 18, True, np.arange(0,110,10), np.arange(0,100,2), np.arange(0,90,10), np.arange(0,80,2), 0, 0, [colors[0]])

ax2 = plt.subplot2grid((3, 6), (0, 3), colspan=3)
newDF['Female'].plot(kind='hist', bins=bins, ax=ax2, title='Female age distribution', color=colors[1], xlim=(0,100), ylim=(0,25))
drawMedianAndAverage(newDF_MedAvgCount.at['Female','average'], newDF_MedAvgCount.at['Female','median'])
legendTicksAndAxStuff(ax2, [], "", "Age", "Count", 18,True, np.arange(0,110,10), np.arange(0,100,2), np.arange(0,30,5), np.arange(0,25,1), 0, 0, [colors[1]])

ax3 = plt.subplot2grid((3, 6), (1, 0), colspan=2)
newDF['N-B, GQ, or G-NC'].plot(kind='hist', bins=bins, ax=ax3, title='N-B, GQ, or G-NC', color=colors[2], xlim=(0,100), ylim=(0,4))
drawMedianAndAverage(newDF_MedAvgCount.at['N-B, GQ, or G-NC','average'], newDF_MedAvgCount.at['N-B, GQ, or G-NC','median'])
legendTicksAndAxStuff(ax3, [], "", "Age", "Count", 12,True, np.arange(0,120,20), np.arange(0,100,5), np.arange(0,5,1), [], 0, 0, [colors[2]])

ax4 = plt.subplot2grid((3, 6), (1, 2), colspan=2)
newDF['A different identity'].plot(kind='hist', bins=bins, ax=ax4, label='A diff. id.', title='A different identity', color=colors[3], xlim=(0,100), ylim=(0,4))
drawMedianAndAverage(newDF_MedAvgCount.at['A different identity','average'], newDF_MedAvgCount.at['A different identity','median'])
legendTicksAndAxStuff(ax4, [], "", "Age", "Count", 12,True, np.arange(0,120,20), np.arange(0,100,5), np.arange(0,5,1), [], 0, 0, [colors[3]])

ax5 = plt.subplot2grid((3, 6), (1, 4), colspan=2)
newDF['NAN_NULL'].plot(kind='hist', bins=bins, ax=ax5, title='NAN_NULL', color=colors[4], xlim=(0,100), ylim=(0,4))
drawMedianAndAverage(newDF_MedAvgCount.at['NAN_NULL','average'], newDF_MedAvgCount.at['NAN_NULL','median'])
legendTicksAndAxStuff(ax5, [], "", "Age", "Count", 12,True, np.arange(0,120,20), np.arange(0,100,5), np.arange(0,5,1), [], 0, 0, [colors[4]])

text=plt.text(1, 1.3, "Age distribution per gender - US Data scientists", horizontalalignment='center', fontsize=25, transform = ax1.transAxes)

plt.tight_layout()
plt.show()

newDF_MedAvgCount.head(6)
del(newDF_MedAvgCount)
del(newDF)
result = stats(data.loc[:,['UniversityImportance']], "AllApplicants", "UniversityImportance")
result.index = changeItemNamesInList(list(result.index), change_NA_I_DIDNT_REC_FE[0], change_NA_I_DIDNT_REC_FE[1])

result2 = stats(data.loc[data['Country'] == 'United States'].loc[:,['UniversityImportance']], "USapplicants", "UniversityImportance")
result2.index = changeItemNamesInList(list(result2.index), change_NA_I_DIDNT_REC_FE[0], change_NA_I_DIDNT_REC_FE[1])


fig = plt.figure(figsize=(15, 24))

ax1 = plt.subplot2grid((2, 1), (0, 0))
ax1.pie(result.loc[:,['percentage']], autopct='%1.1f%%', explode = (0.05,0.05,0.05,0.05,0.05, 0.05,0.05), radius=2, shadow=True, startangle=10, labels=result.index.values)
ax1.axis('equal')

text=ax1.text(0.5, 1.05, "University importance - All applicants", horizontalalignment='center', fontsize=33, transform = ax1.transAxes)

ax2 = plt.subplot2grid((2, 1), (1, 0))
ax2.pie(result2.loc[:,['percentage']], autopct='%1.1f%%', explode = (0.05,0.05,0.05,0.05,0.05, 0.05,0.05), radius=2, shadow=True, startangle=10, labels=result2.index.values)
ax2.axis('equal')

text=ax2.text(0.5, 1.05, "University importance - US applicants", horizontalalignment='center', fontsize=33, transform = ax2.transAxes)
result = stats(data.loc[data['CurrentJobTitleSelect'] == 'Data Scientist'].loc[:,['UniversityImportance']], "AllDataScienceApplicants", "UniversityImportance")
result.index = changeItemNamesInList(list(result.index), change_NA_I_DIDNT_REC_FE[0], change_NA_I_DIDNT_REC_FE[1])

result2 = stats(data.loc[data['Country'] == 'United States'].loc[data['CurrentJobTitleSelect'] == 'Data Scientist'].loc[:,['UniversityImportance']], "USDataScienceApplicants", "UniversityImportance")
result2.index = changeItemNamesInList(list(result2.index), change_NA_I_DIDNT_REC_FE[0], change_NA_I_DIDNT_REC_FE[1])


fig = plt.figure(figsize=(15, 24))

ax1 = plt.subplot2grid((2, 1), (0, 0))
ax1.pie(result.loc[:,['percentage']], autopct='%1.1f%%', explode = (0.05,0.05,0.05,0.05,0.05, 0.05,0.05), radius=2, shadow=True, startangle=10, labels=result.index.values)
ax1.axis('equal')

text=ax1.text(0.5, 1.05, "University importance - All data scientists", horizontalalignment='center', fontsize=33, transform = ax1.transAxes)

ax2 = plt.subplot2grid((2, 1), (1, 0))
ax2.pie(result2.loc[:,['percentage']], autopct='%1.1f%%', explode = (0.05,0.05,0.05,0.05,0.05, 0.05,0.05), radius=2, shadow=True, startangle=10, labels=result2.index.values)
ax2.axis('equal')

text=ax2.text(0.5, 1.05, "University importance - US data scientists", horizontalalignment='center', fontsize=33, transform = ax2.transAxes)
result = stats(data.loc[:,['WorkDatasetSize']], "AllApplicants", "WorkDatasetSize")
result2 = stats(data.loc[data['Country'] == 'United States'].loc[:,['WorkDatasetSize']], "AllApplicants", "WorkDatasetSize")
result3 = stats(data.loc[data['CurrentJobTitleSelect'] == 'Data Scientist'].loc[:,['WorkDatasetSize']], "AllApplicants", "WorkDatasetSize")
result4 = stats(data.loc[data['Country'] == 'United States'].loc[data['CurrentJobTitleSelect'] == 'Data Scientist'].loc[:,['WorkDatasetSize']], "AllApplicants", "WorkDatasetSize")

fig = plt.figure(figsize=(16, 20))

ax1 = plt.subplot2grid((4, 1), (0, 0))
result['count'].plot(kind='bar', fontsize=22, ax=ax1, title="All applicants", ylim=(0, 12000), color=colors[0]);
drawNumbers(ax1,ax1.patches,result['percentage'],"%")
legendTicksAndAxStuff(ax1, ['All applicants'], "", "", "Count", 18, True, [], [], np.arange(0,15000,3000), np.arange(0,12000,1000), 0, 0, [colors[0]])

ax2 = plt.subplot2grid((4, 1), (1, 0))
result2['count'].plot(kind='bar', fontsize=22, ax=ax2, title="US applicants", ylim=(0, 3000), color=colors[1]);
drawNumbers(ax2,ax2.patches,result2['percentage'],"%")
legendTicksAndAxStuff(ax2, ['US applicants'], "", "", "Count", 18, True, [], [], np.arange(0,3600,600), np.arange(0,3000,200), 0, 0, [colors[1]])

ax3 = plt.subplot2grid((4, 1), (2, 0))
result3['count'].plot(kind='bar', fontsize=22, ax=ax3, title="All Data scientists", ylim=(0, 800), color=colors[2]);
drawNumbers(ax3,ax3.patches,result3['percentage'],"%")
legendTicksAndAxStuff(ax3, ['All Data scientists'], "", "", "Count", 18, True, [], [], np.arange(0,1000,200), np.arange(0,800,50), 0, 0, [colors[2]])

ax4 = plt.subplot2grid((4, 1), (3, 0))
result4['count'].plot(kind='bar', fontsize=22, ax=ax4, title="US Data scientists", ylim=(0, 200), color=colors[3]);
drawNumbers(ax4,ax4.patches,result4['percentage'],"%")
legendTicksAndAxStuff(ax4, ['US Data scientists'], "", "", "Count", 18, True, [], [], np.arange(0,250,50), np.arange(0,200,10), 0, 0, [colors[3]])

text=ax1.text(0.5, 1.25, "Work dataset size distribution", horizontalalignment='center', fontsize=33, transform = ax1.transAxes)

plt.tight_layout()
cols = ['CompensationAmount','GenderSelect', 'CompensationCurrency', 'Country', 'Age', 'EmploymentStatus', 'CurrentJobTitleSelect']
allApplicants = data.loc[:,cols]
allApplicants = deleteNanAndCalculateSalary(allApplicants.loc[:], currencies)
print(max(allApplicants.loc[:,'CompensationAmount'].values))
allApplStats = calcMedianAverageForDFColumn(allApplicants.loc[:], 'CompensationAmount')
indexCols = ['All applicants']
statsList = [allApplStats]
pd.DataFrame(statsList, columns=['average' ,'median', 'count'], index=indexCols)
for index, row in allApplicants.iterrows():
    if row[0] > 700000:
        print(row[0])
        allApplicants.drop(index, inplace=True)
usApplicants = allApplicants.loc[allApplicants['Country'] == 'United States']
allDataScienceApplicants = allApplicants.loc[allApplicants['CurrentJobTitleSelect'] == 'Data Scientist']
usDataScienceApplicants = usApplicants.loc[usApplicants['CurrentJobTitleSelect'] == 'Data Scientist']

newNames = change_A_DIFF_ID[0] + change_NB_GQ_GNC[0]
oldNames = change_A_DIFF_ID[1] + change_NB_GQ_GNC[1]

allAppl = returnHistogramList(allApplicants,'GenderSelect', 'CompensationAmount', 'salary')
allAppl.columns = changeItemNamesInList(list(allAppl.columns), newNames, oldNames, True)
allApplStats = calcMedianAverageForDFColumn(allApplicants.loc[:], 'CompensationAmount')

usAppl = returnHistogramList(usApplicants,'GenderSelect', 'CompensationAmount', 'salary')
usAppl.columns = changeItemNamesInList(list(usAppl.columns), newNames, oldNames, False)
usApplStats = calcMedianAverageForDFColumn(usApplicants.loc[:], 'CompensationAmount')

allDsAppl = returnHistogramList(allDataScienceApplicants,'GenderSelect', 'CompensationAmount', 'salary')
allDsAppl.columns = changeItemNamesInList(list(allDsAppl.columns), newNames, oldNames, False)
allDsApplStats = calcMedianAverageForDFColumn(allDataScienceApplicants.loc[:], 'CompensationAmount')

usDsAppl = returnHistogramList(usDataScienceApplicants,'GenderSelect', 'CompensationAmount', 'salary')
usDsAppl.columns = changeItemNamesInList(list(usDsAppl.columns), newNames, oldNames, False)
usDsApplStats = calcMedianAverageForDFColumn(usDataScienceApplicants.loc[:], 'CompensationAmount')

fig = plt.figure(figsize=(16, 25))

ax1 = plt.subplot2grid((5, 1), (0, 0))
allAppl.plot(kind='hist', bins=100, ax=ax1, stacked=True, title='All applicants', color=colors, xlim=(0,700000), ylim=(0,300))
drawMedianAndAverage(allApplStats[0], allApplStats[1])
legendTicksAndAxStuff(ax1, [], "", "Salary", "Count", 18,True, np.arange(0,750000,50000),np.arange(0,700000,10000), np.arange(0,400,100), np.arange(0,300,20), 290, 0, [])
fittAll = returnFit(allAppl.values.flatten().tolist())
    
ax2 = plt.subplot2grid((5, 1), (1, 0))
usAppl.plot(kind='hist', bins=100, ax=ax2, stacked=True, title='US applicants', color=colors, xlim=(0,700000), ylim=(0,300))
drawMedianAndAverage(usApplStats[0], usApplStats[1])
legendTicksAndAxStuff(ax2, [], "", "Salary", "Count", 18,True, np.arange(0,750000,50000),np.arange(0,700000,10000), np.arange(0,400,100), np.arange(0,300,20), 290, 0, [])
fittUS = returnFit(usAppl.values.flatten().tolist())

ax3 = plt.subplot2grid((5, 1), (2, 0))
allDsAppl.plot(kind='hist', bins=100, ax=ax3, stacked=True, title='All Data scientists', color=colors, xlim=(0,700000), ylim=(0,80))
drawMedianAndAverage(allDsApplStats[0], allDsApplStats[1])
legendTicksAndAxStuff(ax3, [], "", "Salary", "Count", 18,True, np.arange(0,750000,50000),np.arange(0,700000,10000), np.arange(0,100,20), np.arange(0,80,5), 290, 0, [])
fittAllDs = returnFit(allDsAppl.values.flatten().tolist())

ax4 = plt.subplot2grid((5, 1), (3, 0))
usDsAppl.plot(kind='hist', bins=100, ax=ax4, stacked=True, title='US Data scientists', color=colors, xlim=(0,700000), ylim=(0,80))
drawMedianAndAverage(usDsApplStats[0], usDsApplStats[1])
legendTicksAndAxStuff(ax4, [], "", "Salary", "Count", 18,True, np.arange(0,750000,50000),np.arange(0,700000,10000), np.arange(0,100,20), np.arange(0,80,5), 290, 0, [])
fittUsDs = returnFit(usDsAppl.values.flatten().tolist())

ax5 = plt.subplot2grid((5, 1), (4, 0))
plt.title("Probability density curves", loc='center')
plt.xlim( (0, 700000) )
plt.plot(fittAll[0], fittAll[1],'-o', alpha=0.4, color=colors[1], label="All applicants")
plt.plot(fittUS[0], fittUS[1],'-o', alpha=0.4, color=colors[0], label="US applicants")
plt.plot(fittAllDs[0], fittAllDs[1],'-o', alpha=0.8, color=colors[2], label="All Data scientists")
plt.plot(fittUsDs[0], fittUsDs[1],'-o', alpha=0.8, color=colors[3], label="US Data scientists")
legendTicksAndAxStuff(ax5, [], "", "Salary", "Probability", 18, True, np.arange(0,750000,50000), np.arange(0,700000,10000), [], [], 290, 0, [])

text=ax1.text(0.5, 1.25, "Salary distributions", horizontalalignment='center', fontsize=33, transform = ax1.transAxes)

plt.tight_layout()
statsList = [allApplStats, usApplStats, allDsApplStats, usDsApplStats]
indexCols = ['All applicants', 'US applicants', 'All Data scientists', 'US Data scientists']
pd.DataFrame(statsList, columns=['average' ,'median', 'count'], index=indexCols).head(4)
for index, row in allApplicants.iterrows():
    if row[0] > 600000:
        print(row[0])
scatterAll = prepareForScatterPlot(allApplicants, 'Age', 'CompensationAmount')
scatterUs = prepareForScatterPlot(usApplicants, 'Age', 'CompensationAmount')
scatterAllDs = prepareForScatterPlot(allDataScienceApplicants, 'Age', 'CompensationAmount')
scatterUsDs = prepareForScatterPlot(usDataScienceApplicants, 'Age', 'CompensationAmount')

fig = plt.figure(figsize=(16, 25))
xVals = np.linspace(15,75,50)

ax1 = plt.subplot2grid((5, 1), (0, 0))
scatterAll.plot.scatter(x='Age', y='CompensationAmount', color=colors[0], ax=ax1, title="All applicants", label='All applicants', xlim=(0,100), ylim=(0,700000))
k, l = np.polyfit(scatterUsDs.loc[:,'Age'].values, scatterUsDs.loc[:,'CompensationAmount'].values, 1)
plt.plot(xVals, k*xVals + l, '-', label="regression line", color=colors[6], linewidth=2)
legendTicksAndAxStuff(ax1, [], "", "Age", "Salary", 18,True, np.arange(0,105,5),np.arange(0,100,1), np.arange(0,800000,100000), np.arange(0,700000,10000), 0, 0, [])

ax2 = plt.subplot2grid((5, 1), (1, 0))
scatterUs.plot.scatter(x='Age', y='CompensationAmount', color=colors[1], ax=ax2, title="US applicants", label='US applicants', xlim=(0,100), ylim=(0,700000))
k, l = np.polyfit(scatterUsDs.loc[:,'Age'].values, scatterUsDs.loc[:,'CompensationAmount'].values, 1)
plt.plot(xVals, k*xVals + l, '-', label="regression line", color=colors[0], linestyle='dashed', marker='*', markerfacecolor=colors[0], markersize=12)
legendTicksAndAxStuff(ax2, [], "", "Age", "Salary", 18,True, np.arange(0,105,5),np.arange(0,100,1), np.arange(0,800000,100000), np.arange(0,700000,10000), 0, 0, [])

ax3 = plt.subplot2grid((5, 1), (2, 0))
scatterAllDs.plot.scatter(x='Age', y='CompensationAmount', color=colors[2], ax=ax3, title="All Data scientists", label='All Data scientists', xlim=(0,100), ylim=(0,700000))
k, l = np.polyfit(scatterUsDs.loc[:,'Age'].values, scatterUsDs.loc[:,'CompensationAmount'].values, 1)
plt.plot(xVals, k*xVals + l, '-', label="regression line", color=colors[5], linewidth=2)
legendTicksAndAxStuff(ax3, [], "", "Age", "Salary", 18,True, np.arange(0,105,5),np.arange(0,100,1), np.arange(0,800000,100000), np.arange(0,700000,10000), 0, 0, [])

ax4 = plt.subplot2grid((5, 1), (3, 0))
scatterUsDs.plot.scatter(x='Age', y='CompensationAmount', color='black', ax=ax4, title="US Data scientists", label='US Data scientists', xlim=(0,100), ylim=(0,700000))
k, l = np.polyfit(scatterUsDs.loc[:,'Age'].values, scatterUsDs.loc[:,'CompensationAmount'].values, 1)
plt.plot(xVals, k*xVals + l, '-', label="regression line", color=colors[3], linestyle='dashed', marker='*', markerfacecolor=colors[3], markersize=12)
legendTicksAndAxStuff(ax4, [], "", "Age", "Salary", 18,True, np.arange(0,105,5),np.arange(0,100,1), np.arange(0,800000,100000), np.arange(0,700000,10000), 0, 0, [])


ax5 = plt.subplot2grid((5, 1), (4, 0))
plt.title("All regression lines", loc='center')
k, l = np.polyfit(scatterAll.loc[:,'Age'].values, scatterAll.loc[:,'CompensationAmount'].values, 1)
plt.plot(xVals, k*xVals + l, '-', color=colors[6], label='All applicants', linewidth=2)

k, l = np.polyfit(scatterUs.loc[:,'Age'].values, scatterUs.loc[:,'CompensationAmount'].values, 1)
plt.plot(xVals, k*xVals + l, '-', color=colors[0], label='US applicants', linestyle='dashed', marker='*', markerfacecolor=colors[0], markersize=12)

k, l = np.polyfit(scatterAllDs.loc[:,'Age'].values, scatterAllDs.loc[:,'CompensationAmount'].values, 1)
plt.plot(xVals, k*xVals + l, '-', color=colors[5], label='All Data scientists', linewidth=2)

k, l = np.polyfit(scatterUsDs.loc[:,'Age'].values, scatterUsDs.loc[:,'CompensationAmount'].values, 1)
plt.plot(xVals, k*xVals + l, '-', color=colors[3], label='US Data scientists', linestyle='dashed', marker='*', markerfacecolor=colors[3], markersize=12)
legendTicksAndAxStuff(ax5, [], "Regression lines", "Age", "Salary", 18,True, np.arange(0,105,5),np.arange(0,100,1), np.arange(0,300000,50000), np.arange(0,250000,10000), 0, 0, [])

text=ax1.text(0.5, 1.25, "Salary vs. Age", horizontalalignment='center', fontsize=33, transform = ax1.transAxes)

plt.tight_layout()