# Problem 1
# Store blood group of 50 different patients (O+,A+,B+,AB+,O-,A-,B-,AB-). Graphically show how mnay patients have O- blood group.

from matplotlib import pyplot as plt

BLOODGROUPS=['O+','A+','B+','AB+','O-','A-','B-','AB-']

 
PATIENTS = [10,8,7,9,6,6,1,3]
  
tick_label = ['O+','A+','B+','AB+','O-','A-','B-','AB-'] 
  
plt.bar(BLOODGROUPS, PATIENTS, tick_label = tick_label, width = 0.8, color = ['green', 'green','green','green','red','green','green','green']) 
  
plt.xlabel('BLOODGROUP') 
plt.ylabel('PATIENTS')  
plt.title('Blood Group Graph')

# Problem 2
# For a certain student Store data of marks obtained by the student in English, Bengali, Hindi, Maths, History and Geography. 
#Graphically represent the data in a pie chart and bring out the slice having least marks. ( No subject should have same marks).

import matplotlib.pyplot as plt
slices = ['English','Bengali','Hindi','Maths','History','Geography']
marks = [90,50,91,99,88,87]
plt.pie(marks,labels=slices,startangle=90,shadow=True,explode=(0,0.1,0,0,0,0),autopct='%1.1f%%')
# Problem 3
# Store height of 50 students in inches. 
#Now while the data was being recorded manually there has been some typing mistake 
# and therefore height of 2 students have been recorded as 172 inch and 2 students have been recorded as 12 inch. 
#Graphically plot and show how you can seggregate correct data from abnormal data. 

import numpy as np
import random as rn
import matplotlib.pyplot as plt

height = np.array([])
    # creating sample data
for i in range(50):
    height = np.append(height , [rn.randint(62,75)])
height[21] = 172
height[13] = 172
height[10] = 2
height[43] = 2
    # sample data created
plt.boxplot(height)
plt.show()

