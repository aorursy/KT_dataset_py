# Importing library files
import matplotlib.pyplot as plt
import numpy as np
# Create a simple list of categories
jobList = ['admin','scientist','doctor','management']

print(jobList)
# Getting two categories ( 'yes','No') for each of jobs
jobYes = [20,60,70,40]
jobNo = [80,40,30,60]
# Get the length of x axis labels and arranging its indexes
xlabels = len(jobList)
print(xlabels)

ind = np.arange(xlabels)
print(ind)
# Get width of each bar
width = 0.45

# Getting the plots
p1 = plt.bar(ind, jobYes, width)
p2 = plt.bar(ind, jobNo, width, bottom=jobYes)

# Getting the labels for the plots
plt.ylabel('Proportion of Jobs')
plt.title('Jobs')

# Defining the x label indexes and y label indexes
plt.xticks(ind, jobList)
plt.yticks(np.arange(0, 100, 5))

# Defining the legends
plt.legend((p1[0], p2[0]), ('Yes', 'No'))

# To rotate the axis labels
#plt.xticks(rotation=90)

plt.show()