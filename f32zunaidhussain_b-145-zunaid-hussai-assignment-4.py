import matplotlib.pyplot as plt
bg=['O+','A+','B+','AB+','O-','A-','B-','AB-']
no=[12,11,2,15,22,14,34,21]
plt.bar(bg,no,color=['red','red','red','red','black','red','red','red'])
plt.title('BLOOD GROUP')
plt.xlabel('X-axis')
plt.ylabel("Y-axis")
slc=[ 'English', 'Bengali', 'Hindi', 'Maths', 'History','Geography']
mrks=[100,85,95,99,98,75]
plt.pie(mrks,labels=slc,explode=(0,0,0,0,0,0.1))
import numpy as np
import random as rn
ht=np.array([])
for i in range(50):
    ht=np.append(ht,[rn.randint(50,70)])
    
ht
ht[10]=172
ht[11]=172
ht[22]=2
ht[49]=2
plt.boxplot(ht)
plt.title('Height')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

