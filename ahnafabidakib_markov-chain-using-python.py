#importing libraries we are going to use
import numpy as np
import random as rm


#defining states and their probabilities
#the statespace
states = ["Sleep","Report","CT"]
#possible sequence of events
transitionName = [["SS","SR","SC"],["RS","RR","RC"],["CS","CR","CC"]]
#probabilities matrix (transition matrix)
transitionMatrix = [[0.2,0.6,0.2],[0.1,0.6,0.3],[0.2,0.7,0.1]]
#showing error message
if sum(transitionMatrix[0])+sum(transitionMatrix[1])+sum(transitionMatrix[2]) != 3:
    print("Somewhere, something went wrong. Transition matrix, perhaps?")
else: print("Chill, everything is in order!! ;)")
    
def activity_forecast(days):
    # Choose the starting state
    activityToday = "Sleep"
    activityList = [activityToday]
    i = 0
    prob = 1
    while i != days:
        if activityToday == "Sleep":
            change = np.random.choice(transitionName[0],replace=True,p=transitionMatrix[0])
            if change == "SS":
                prob = prob * 0.2
                activityList.append("Sleep")
                pass
            elif change == "SR":
                prob = prob * 0.6
                activityToday = "Report"
                activityList.append("Report")
            else:
                prob = prob * 0.2
                activityToday = "CT"
                activityList.append("CT")
        elif activityToday == "Report":
            change = np.random.choice(transitionName[1],replace=True,p=transitionMatrix[1])
            if change == "RR":
                prob = prob * 0.6
                activityList.append("Report")
                pass
            elif change == "RS":
                prob = prob * 0.1
                activityToday = "Sleep"
                activityList.append("Sleep")
            else:
                prob = prob * 0.3
                activityToday = "CT"
                activityList.append("CT")
        elif activityToday == "CT":
            change = np.random.choice(transitionName[2],replace=True,p=transitionMatrix[2])
            if change == "CC":
                prob = prob * 0.1
                activityList.append("CT")
                pass
            elif change == "CS":
                prob = prob * 0.2
                activityToday = "Sleep"
                activityList.append("Sleep")
            else:
                prob = prob * 0.7
                activityToday = "Report"
                activityList.append("Report")
        i += 1    
    return activityList


# To save every activityList
list_activity = []
count = 0

# `Range` starts from the first count up until but excluding the last count
for iterations in range(1,10000):
        list_activity.append(activity_forecast(2))

# Check out all the `activityList` we collected    
#print(list_activity)

# Iterate through the list to get a count of all activities ending in state:'Report'
for smaller_list in list_activity:
    if(smaller_list[2] == "Report"):
        count += 1
# Calculate the probability of starting from state:'Sleep' and ending at state:'Report'
percentage = (count/10000) * 100
print("The probability of starting at state:'Sleep' and ending at state:'Report'= " + str(percentage) + "%")

# Calculate the probability of starting from state:'Sleep' and ending at state:'Run'
percentage = (count/10000) * 100
print("The probability of starting at state:'Sleep' and ending at state:'Report'= " + str(percentage) + "%")