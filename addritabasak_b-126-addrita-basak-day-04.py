#1. Store blood group of 50 different patients (O+,A+,B+,AB+,O-,A-,B-,AB-). 

#Graphically show how mnay patients have O- blood group.



import matplotlib.pyplot as plt

blood_grps=['O+','A+','A-','O+','B-','AB+','AB-','O+','0-','O-','O+','A+','B+','O+','A+','A-','O+','B-','AB+','AB-','O+','0-','O-','O+','A+','O+','A+','A-','O+','B-','AB+','AB-','O+','0-','O-','O+','A+','B+','O+','A+','A-','O+','B-','AB+','AB-','O+','0-','O-','O+','A+']

bins=[5,10,15,20]

plt.hist(blood_grps,bins,histtype='bar',rwidth=0.5,colour='red')

plt.xlabel('number of patients')

plt.ylabel('blood groups')

plt.title('blood samples')

plt.show()
import matplotlib.pyplot as plt



subjects = ['English','Bengali','Hindi','Maths','History','Geography']

marks = [100,75,69,92,85,80]



plt.pie(marks, labels = subjects, startangle=90, shadow=True, explode=(0,0,0.2,0,0,0), autopct='%1.1f%%')



plt.title('Marks Pie Chart')

plt.show()
#3. Store height of 50 students in inches. 

#Now while the data was beign recorded manually there has been some typing mistake and therefore height of 2 students have been recorded as 172 inch and 2 students have been recorded as 12 inch. 

#Graphically plot and show how you can seggregate correct data from abnormal data.



import random 

import matplotlib.pyplot as plt



height = []

   

for i in range(50):

    height.append(random.randint(60, 80))

height[10] = 172

height[20] = 172

height[30] = 12

height[40] = 12

   

plt.boxplot(height)

plt.show()