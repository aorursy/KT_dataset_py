import matplotlib.pyplot as plt  
activities = ['eng', 'bengali', 'hindi', 'maths','hist','geog'] 
slices = [99, 79, 88, 62,92,74] 
colors = ['r', 'y', 'g', 'b','r','c']
plt.pie(slices, labels = activities, colors=colors,  
        startangle=90, shadow = True, explode = (0, 0, 0, 0.3,0,0), 
        radius = 1.2, autopct = '%1.1f%%')  
plt.show() 
import matplotlib.pyplot as plt
plt.bar(["O+","A+","B+","AB+","O-","A-","B-","AB-"],[10,5,6,4,9,9,5,2],color="r")
plt.xlabel("BLOOD BANK")
plt.xlabel('Blood group') 
plt.ylabel('no of people') 
plt.title('BLOOD BANK')  
plt.show() 
import matplotlib.pyplot as plt
data=[62,66,68,59,43,172,78,49,12,43,79,67,172,65,87,56,
      76,61.5,66,62,61,87,47,64.5,12,62,66,68,59,43,65,87,99,
      45,76,67.2,89,56,87,76,54,43,89,78.3,76,34,76,56,34,82]
plt.boxplot(data)
plt.show()
