import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir('../input'))
data = pd.read_csv("../input/TIMEGenderData.csv")
data.head()
data.describe()
type(data['Female %'][0])
data = data.drop (['Female %', 'Male %'], axis = 1)
femaleperc = []
femaleperc = data.Female/data.Total * 100

maleperc=[]
maleperc = data.Male/data.Total * 100
femaleperc = [int(x) for x in femaleperc]
maleperc = [int(x) for x in maleperc]
data = data.assign(FemalePerc = femaleperc)
data = data.assign(MalePerc = maleperc)
data.head()
for i,row in data.iterrows():
    sum = data.FemalePerc[i] + data.MalePerc[i]
    if sum > 100: #Check whether there is any sum value above 100
        diff = sum - 100 #Find out the difference
        data.MalePerc[i] = data.MalePerc[i] - diff
        #We modify the MalePerc values for adjusting the difference. 
    elif sum < 100: #Check whether there is any sum value less than 100
        diff = 100 - sum
        data.MalePerc[i] = data.MalePerc[i] + diff
for i,row in data.iterrows():
    sum = data.FemalePerc[i] + data.MalePerc[i]
    if sum != 100:
        print ("Error") #Print Error if anyone of the row's sum of percentages in not 100.
#Plotting the data using a Stacked Bar Graph
plt.figure(figsize=(25,15)) #Setting the figure size 
barWidth = 0.9 #Setting width of each bar
x_values = data.Year #For setting the x-axis values as the Years of the publications
plt.bar(x_values, data.FemalePerc, color='#b5ffb9', edgecolor='white', width=barWidth, label='Female')
plt.bar(x_values, data.MalePerc, bottom=data.FemalePerc, color='#f9bc86', edgecolor='white', width=barWidth, label='Male')
plt.xticks(x_values, rotation=90, fontsize=15)
plt.yticks(fontsize=18)
plt.legend(bbox_to_anchor=(1,1), loc=2, prop={'size':15})
#bbox_to_anchor makes legend visible outside the graph. The placement of the legend follows a different x and y-axes than the graph. For the axes which are followed by 
#legend, (0,0) is lower left point of the chart and (1,1) is the upper rightmost point of the chart. That is why the location is (1,1) such that the legend box
#is just at the upper rightmost part of the chart. loc=2 indicates upper right corner. And prop is the size of the legend.
plt.xlabel('Years', fontsize=20)
plt.ylabel('Percentage', fontsize=20, rotation=90)
plt.title('Analysis of male and female personalities on covers of TIME (1923-2013)', fontsize = 25)
plt.show()
import seaborn as sns
plt.figure(figsize=(22,13))
sns.set(color_codes = True)
sns.set_style("darkgrid")
ax = sns.regplot(x="Year", y="MalePerc", data = data, color='#FF7F50', label='Male' )
ax1 = sns.regplot(x="Year", y="FemalePerc", data = data, color='#008000', label='Female')
plt.xlabel('Year', fontsize = 15)
plt.ylabel('Percentage', fontsize = 15)
plt.title('Trend of male and female covers on Time (1923-2013)', fontsize = 20)
plt.legend(bbox_to_anchor=(1,1), loc=2, prop={'size':15})
plt.show()