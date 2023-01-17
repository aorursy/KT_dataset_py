import matplotlib.pyplot as plt

slices=[7,2,4,5,6,8]

subj=['eng','beng','math','hindi','hist','geo']

cols=['c','m','r','b','y','g']

plt.pie(slices,labels=subj,colors=cols,startangle=90,shadow=True,explode=(0,0.5,0,0,0,0),autopct='%1.1f%%')

plt.title('ML Lab pie plot')

plt.show()
import matplotlib.pyplot as plt

data=[50,66,60,172,45,69,58,50,60,72,172,12,45,66,65,61,63,12,60,61,58,59,64,67,63,61,69,68,49,56,58,57,71,66,64,61,62,58,45,47,39,46,48,38,44,65,55,51,53,50]

plt.boxplot(data)

plt.show()
import matplotlib.pyplot as plt

blood_grps=['O+','A+','A-','O+','B-','AB+','AB-','O+','0-','O-','O+','A+','B+','O+','A+','A-','O+','B-','AB+','AB-','O+','0-','O-','O+','A+','O+','A+','A-','O+','B-','AB+','AB-','O+','0-','O-','O+','A+','B+','O+','A+','A-','O+','B-','AB+','AB-','O+','0-','O-','O+','A+']

bins=[5,10,15,20]

plt.hist(blood_grps,bins,histtype='bar',rwidth=0.5,colour='red')

plt.xlabel('number of patients')

plt.ylabel('blood groups')

plt.title('blood samples')

plt.show()