import matplotlib as plt
import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot
bugs = pd.read_csv("../input/insect-light-trap/Thomsen_Jørgensen_et_al._JAE_All_data_1992-2009.csv", encoding = "latin1")
bugs.head(5)

bugs.info()
#Comparing count of the two orders found in this Data 
count = sns.countplot(x = 'order', data = bugs)
count.set_title("Order Count", size =20, y= 1.07)
viol = sns.violinplot(x='order', y = 'year', data = bugs)
viol.set_title("Order Count over Time", size =20, y = 1.07)
#Both orders seem to have similar cyclical patterns of coming to the light
#Creating new data set to analyze count of family
count = bugs['family'].value_counts().to_dict()
sBugs = sorted(count.items(), key=lambda x:x[1], reverse=True)
sBugs = sBugs[:20]
Bug = pd.DataFrame(sBugs, columns = ['Family', 'Count'])
pyplot.subplots(figsize=(10,15))
bar = sns.barplot(y = 'Family', x = 'Count', data = Bug)
bar.set(xlabel='Count')
bar.set_title("Family Count", size = 30, y= 1.01)
#Creating new data set to analyze count of Name
fig, ax = pyplot.subplots(figsize=(10,50))
count = bugs['name'].value_counts().to_dict()
sBugs = sorted(count.items(), key=lambda x:x[1], reverse=True)
sBugs = sBugs[:100]
sBug = pd.DataFrame(sBugs, columns = ['Name', 'Count'])
bar = sns.barplot(y = 'Name', x = 'Count', data = sBug)
bar.set_title("Name Count", size = 30, y= 1.005)
bar.set(xlabel='Count')
#Grouping df by year and family count
cgroups = bugs.groupby('year')['family'].value_counts()
cgroups = pd.DataFrame({'Year':cgroups.index, 'Count':cgroups.values})
#Using Bug dataframe to eliminate rows that are not significant
count = bugs['family'].value_counts().to_dict()
sBugs = sorted(count.items(), key=lambda x:x[1], reverse=True)
Bug = pd.DataFrame(sBugs, columns = ['Family', 'Count'])
listy = pd.DataFrame(cgroups['Year'].tolist())
cgroups['Year'] = listy[0]
cgroups['Family'] = listy[1]
cgroups = cgroups[['Year', 'Family', 'Count']]
Bug = Bug[Bug['Count'] >= 50]
#Only using insects that were significantly collected (min total count is 50)]
for i in cgroups['Family']:
    test = Bug[Bug['Family'] == i]
    if(bool(test.empty)):
        cgroups = cgroups[cgroups.Family != i]
#Calculating corrcoef and storing results into cFrame
cFrame = pd.DataFrame(index = range(0,44), columns = ['Family', 'Corr'])
cFrame['Family'] = cgroups['Family'].unique()
c = []
for i in range(len(cFrame)):
    fFrame = cgroups[cgroups['Family'] == cFrame['Family'][i]]
    c.append(np.corrcoef(fFrame['Year'], fFrame['Count'])[0,1])
cFrame['Corr'] = c
cFrame = cFrame.sort_values(by = 'Corr', ascending=False)
fig, ax = pyplot.subplots(figsize=(10,20))
bar = sns.barplot(y = 'Family', x = 'Corr', data = cFrame)
bar.set_title("Correlation of Count to Time in Study", size = 30, y= 1.005)
bar.set(xlabel='Correlation Coefficient')

#Families w/ Highest growth over study
cFrame.head(5)
#Families w/ Highest decline over study
cFrame.tail(5)
#Graphed top 5 familys that increased in count the most over the time period of the study
high = cgroups[(cgroups.Family == 'COCCINELLIDAE') |  (cgroups.Family =='NITIDULIDAE') \
              | (cgroups.Family == 'ARCTIIDAE') | (cgroups.Family == 'MOMPHIDAE') \
              | (cgroups.Family == 'SCARABAEIDAE')]
h = sns.pointplot(x='Year', y ='Count', data = high, hue = 'Family')
h.set(ylabel = 'Count')
h.set_title("Highest Growth over Study", size =20, y= 1.05)
#Graphed top 5 familys that decreased in count the most over the time period of the study
low = cgroups[(cgroups.Family == 'SCRAPTIIDAE') |  (cgroups.Family =='DYTISCIDAE') \
              | (cgroups.Family == 'DERMESTIDAE') | (cgroups.Family == 'MONOTOMIDAE') \
              | (cgroups.Family == 'NEPTICULIDAE')]
l = sns.pointplot(x='Year', y ='Count', data = low, hue = 'Family')
l.set(ylabel = 'Count')
l.set_title("Highest Decline over Study", size =20, y= 1.05)
"""Since the purpose of this collection was to find effects of climate change I compared the avg temp 
of Denmark, where the study took place, to the bug collection data set"""
clim = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv")
clim.info()
#Extracting only Denmark Data
den = clim[clim['Country'] == 'Denmark']
den['Year'] = pd.DatetimeIndex(den['dt']).year
den['dt'] = pd.to_datetime(den.dt)
#Setting date format to the same as the climate df
bugs['date1'] = pd.to_datetime(bugs.date1) 
bugs['date2'] = pd.to_datetime(bugs.date2)
#Cutting climate df to match same time period as bug df
den = den[den['Year'] >= 1992]
den = den[den['Year'] <= 2009]
yearly = bugs['year'].value_counts()
yearly = pd.DataFrame({'Year':yearly.index, 'Count':yearly.values})
yearly = yearly[['Year', 'Count']]
yearly = yearly.sort_values(by = 'Year')
#Grouping average temp by year
ygroups = den.groupby('Year')['AverageTemperature'].mean()
ygroups = pd.DataFrame({'Year':ygroups.index, 'AverageTemperature':ygroups.values})
ygroups = ygroups[['Year', 'AverageTemperature']]
ygroups = ygroups.sort_values(by = 'Year')
#merging the two data sets together
yearly = yearly.merge(ygroups, on = 'Year')
reg = sns.regplot(x = 'Year', y = 'Count', data = yearly)
reg.set_title('Count over Time', size =20, y =1.05)
reg = sns.regplot(x = 'Year', y = 'AverageTemperature', data = yearly, color = 'red')
reg.set_title('Average Temperature over Time', size =20, y =1.05)
reg = sns.regplot(x = 'AverageTemperature', y = 'Count', data = yearly, color = 'green')
reg.set_title("Average Temperature's Effect on Count", size =20, y =1.05)
yearly= yearly.astype(int)
color = plt.cm.summer
sns.plt.title("Correlation of Bug Data", size = 20, y = 1.1)
sns.heatmap(yearly.astype(float).corr(), linewidths = 0.3,vmax = 1.0, square = True, \
            cmap = color, linecolor = 'white', annot = True)
"""From this Heatmap and the regplot above it seems the total amount of bugs captured in the light trap
is completely irrelevant to the time and change in climate in Denmark"""
"""Though we determined Temp is irrelevant to total count lets dive into the change in what specific 
families are visiting the trap"""
"""Measuring greatest change in count by a family by using corrcoef between count and temp"""
cgroups =cgroups.reset_index(drop=True)
#Using the same cGroups and cFrame dataframes established in the 'Change in Count' section
"""Inserting average temp from denmark climate df into cgroups in order to calculate corrcoef
for temp vs count of families"""
in_temp = []
for i in range(len(cgroups)):
    tFrame = ygroups[ygroups['Year'] == cgroups['Year'][i]]
    in_temp.append(float(tFrame['AverageTemperature']))
cgroups['Temp'] = in_temp
#Calculating corrcoef and storing results into cFrame
c = []
for i in range(len(cFrame)):
    tFrame = cgroups[cgroups['Family'] == cFrame['Family'][i]]
    c.append(np.corrcoef(tFrame['Temp'], tFrame['Count'])[0,1])
cFrame['Corr'] = c
cFrame = cFrame.sort_values(by = 'Corr', ascending =False)
fig, ax = pyplot.subplots(figsize=(10,20))
bar = sns.barplot(y = 'Family', x = 'Corr', data = cFrame)
bar.set(xlabel='Correlation Coefficient')
bar.set_title("Temperature to Insect Family Population", size = 20, y= 1.005)
#The highest correlation of temp to a family count is only .61!!!!
cFrame.head(5)
fig, ax = pyplot.subplots(figsize=(10,10))
high = cgroups[(cgroups.Family == 'NOLIDAE') |  (cgroups.Family =='TENEBRIONIDAE') \
              | (cgroups.Family == 'ELACHISTIDAE') | (cgroups.Family == 'SCIRTIDAE') \
              | (cgroups.Family == 'OECOPHORIDAE')]
h = sns.pointplot(x='Year', y ='Count', data = high, hue ='Family')
h.set(ylabel = 'Count')
h.set_title("Highest Growth to Temp", size =20, y= 1.05)