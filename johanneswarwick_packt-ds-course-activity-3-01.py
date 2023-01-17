# DataFrame Manipulation library
import pandas as pd

# Declarative statistical visualization library
import altair as alt

# Packages to plot Stacked bars chart
import matplotlib.pyplot as plt
import numpy as np
file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter03/bank-full.csv'
bankData = pd.read_csv(file_url, sep=";")
bankData.head()
bankSub1 = bankData[bankData['y'] == 'yes'].groupby('job')['y'].agg(jobgrp='count').reset_index()
print(bankSub1)
# Visualising the relationship using altair
alt.Chart(bankSub1).mark_line().encode(x='job', y='jobgrp')
jobTot = bankData.groupby('job')['y'].agg(jobTot='count').reset_index()
print(jobTot)
# Getting all the details in one place
jobProp = bankData.groupby(['job','y'])['y'].agg(jobCat='count').reset_index()
print(jobProp)
# Merging both the data frames
jobComb = pd.merge(jobProp, jobTot,left_on = ['job'], right_on = ['job'])
jobComb['catProp'] = (jobComb.jobCat/jobComb.jobTot)*100
jobComb.head()
# Visualising the relationship using altair
alt.Chart(jobComb).mark_line().encode(x='job', y='catProp').facet(column='y')
# Visualising the relationship using stacked bars

# Get the length of x axis labels and arranging its indexes
xlabels = len(jobTot['job'])
print(xlabels)

ind = np.arange(xlabels)
print(ind)
# Get width of each bar
width = 0.55

# Get the numbers to the plot
jobYes = jobComb[jobComb['y'] == 'yes']
jobList = jobYes['job']
#print(jobList)
jobYes = jobYes['catProp']
#print(jobYes)

jobNo = jobComb[jobComb['y'] == 'no']
jobNo = jobNo['catProp']
#print(jobNo)

# Getting the plots
p1 = plt.bar(ind, jobYes, width)
p2 = plt.bar(ind, jobNo, width, bottom=jobYes)

# Getting the labels for the plots
plt.ylabel('Propensity for a Term Loan')
plt.title('Employment Status versus Propensity for Term Deposits')

# Defining the x label indexes and y label indexes
plt.xticks(ind, jobList)
plt.yticks(np.arange(0, 100, 5))

# Defining the legends
plt.legend((p1[0], p2[0]), ('Yes', 'No'))

# To rotate the axis labels
plt.xticks(rotation=90)

plt.show()
