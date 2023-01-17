import matplotlib.pyplot as plt

import numpy as np
plt.ion() 

x = np.random.randn(10000)  

plt.hist(x, bins = 15)

plt.show()
import datetime as dt  

prima = 600 + np.random.randn(5) * 10  # invented values

fechas = (dt.date.today() - dt.timedelta(5)) + dt.timedelta(1) * np.arange(5) #Generating dates

plt.axes((0.1, 0.3, 0.8, 0.6))  # We define the position of the axes

plt.bar(np.arange(5), prima)  # We draw the bar graph

plt.ylim(550,650)  # We limit the values of the y axis to the defined range [450, 550]

plt.title('prima de riesgo')  # We put the title

plt.xticks(np.arange(5), fechas, rotation = 45)  # Colocamos las etiquetas del eje x, en este caso, las fechas

plt.show()

plt.axes((0.2,0.1,0.7,0.8))  # We create the axes in the position that we want

plt.title('Evolution')  # We put

plt.broken_barh([(0,1),(3,3), (10,5), (21,3)], (9500, 1000))  # We Draw moments when there were high clouds 

plt.broken_barh([(0,24)], (4500, 1000))  # We Draw moments when there were middle clouds 

plt.broken_barh([(0,9), (12,5), (20,2)], (1500, 1000))  # We Draw moments when there were low clouds 

plt.xlim(-1,25)  # We limit the range of values of the x axis

plt.yticks([2000, 5000, 10000], ['Low Clouds', 'Middle Clousds','High Clouds'])  # We put tags in axis y

plt.xlabel('t(h)')  # We put the title in axis x

plt.show()
x = np.arange(10) + 1

y = np.random.rand(10)

plt.step(x, y, where = 'mid', color = 'r', linewidth = 3)

plt.title('stair graph')

plt.xlim(0,11)

plt.show()