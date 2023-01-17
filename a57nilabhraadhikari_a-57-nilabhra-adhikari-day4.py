from matplotlib import pyplot as plt

x=['O+','A+','B+','AB+','O-','A-','B-','AB-']

y=[9,12,2,10,7,1,4,5]

plt.plot(x,y,'r--',x,y,'g^')

plt.title('Blood group distribution of 50 patients')

plt.ylabel('No of Patients')

plt.xlabel('Blood group')

plt.show()
from matplotlib import pyplot as plt

slices=[85,87,92,98,80,83]

Subject=['English','Bengali','Hindi','Maths','History','Geography']

plt.pie(slices,labels=Subject,startangle=90,shadow=True,explode=(0.08,0.08,0.08,0.08,0.5,0.08),autopct='%1.1f%%')

from matplotlib import pyplot as plt

heights=[161,150,154,165,168,161,154,162,150,121,162,164,171,165,158,154,156,172,160,170,153,159,161,170,162,165,166,168,165,164,154,152,153,156,158,172,172,161,12,166,161,12,162,167,168,159,158,153,154,159]

bins=[150,155,160,165,170]

plt.hist(heights,bins,histtype='bar',rwidth=0.5,color='blue')

plt.xlabel('Height range')

plt.ylabel('No of persons')

plt.title("Heights Histogram")

plt.show()