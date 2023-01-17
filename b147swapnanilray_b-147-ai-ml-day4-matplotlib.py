#1. Store blood group of 50 different patients (O+,A+,B+,AB+,O-,A-,B-,AB-). 

#Graphically show how mnay patients have O- blood group.



import matplotlib.pyplot as plt

 

bldgrps = ['O+','A+','B+','AB+','O-','A-','B-','AB-']    

count = [12,10,10,5,7,2,2,2]



plt.bar(bldgrps, count,width = 0.8, color = ['green', 'green','green','green','red','green','green','green']) 



plt.xlabel('BloodGroups') 

plt.ylabel('Count')  

plt.title('Blood Group Counter')

plt.show()  

#2. For a certain student Store data of marks obtained by the student in English, Bengali, Hindi, Maths, History and Geography. 

#Graphically represent the data in a pie chart and bring out the slice having least marks. ( No subject should have same marks).



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