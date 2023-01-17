# Problem 1



from matplotlib import pyplot as plt



BLOODGROUPS=['O+','A+','B+','AB+','O-','A-','B-','AB-']



 

PATIENTS = [10,5,7,9,8,6,1,3]

  

tick_label = ['O+','A+','B+','AB+','O-','A-','B-','AB-'] 

  

plt.bar(BLOODGROUPS, PATIENTS, tick_label = tick_label, width = 0.8, color = ['green', 'green','green','green','red','green','green','green']) 

  

plt.xlabel('BLOODGROUP') 

plt.ylabel('PATIENTS')  

plt.title('Blood Group Graph')
# Problem 2





import matplotlib.pyplot as plt

slices = ['English','Bengali','Hindi','Maths','History','Geography']

marks = [88,83,94,97,82,98]

plt.pie(marks,labels=slices,startangle=90,shadow=True,explode=(0,0.1,0,0,0,0),autopct='%1.1f%%')
# Problem 3



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