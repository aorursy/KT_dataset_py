import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt



# Data to plot

labels = ['Max Heart Rate','Age','Resting Blood Pressure','Cholestrol','Gender','Fasting Blood Pressure','Resting Ecg Anomaly']



sizes = [0.269317961,0.219836343821885, 0.177982438918357,0.175863062209091,0.0906731804595421,0.0334835490380651,0.0328434641093557]

colors = ['red', 'yellowgreen', 'lightcoral', 'lightskyblue','yellowgreen', 'lightcoral', 'lightskyblue']

explode = (0.1, 0, 0, 0,0,0,0)  # explode 1st slice



# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)





plt.title('Risk Factors')

plt.axis('equal')

plt.savefig('page1.png')

plt.show()
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt



objects = ('Jul30','Aug15','Aug31', 'sep15', 'Sep30', 'Oct15', 'Oct31')

y_pos = np.arange(len(objects))

performance = [76.2,77.5,78,79,82,83.4,85]



plt.stackplot(y_pos, performance) #align='center', alpha=0.5)



plt.axhline(y=79,color = 'r')

plt.ylim(75,85)

plt.xticks(y_pos, objects)

plt.ylabel('Rate')

plt.title('Mean Heart Rate')

plt.savefig('page2.png')

plt.show()
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt



objects = ('Fri','Sat','Sun', 'Mon', 'Tue', 'Wed', 'Thurs')

y_pos = np.arange(len(objects))

performance = [6020,7200,8000,6453,4500,5000,3000]



plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.axhline(y=3500,color = 'r')

plt.xticks(y_pos, objects)

plt.ylabel('Steps')

plt.title('Steps Last Week')

plt.savefig('page3.png')

plt.show()