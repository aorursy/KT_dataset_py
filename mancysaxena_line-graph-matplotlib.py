from matplotlib import pyplot as plt

import random
#Create Two Lists

dev_x = [1,2,3,4,5,6,7,8,9,10]

dev_y = [100,200,300,400,500,900,1100,1220,1900,2000]

dev_y1 = [190,200,390,487,568,678,789,809,923,1003]
dev_x

dev_y
##plt will have both the graphs



##If you remove label and add the label Values as list in legends like 



plt.plot(dev_x,dev_y)

plt.xlabel('Numbers')

plt.ylabel('Scores')

plt.title("Add Line Graph")

plt.plot(dev_x,dev_y1)

plt.legend(['Plot1', 'Plot2'])



plt.plot(dev_x,dev_y,'b--', label='Plot1')

plt.xlabel('Numbers')

plt.ylabel('Scores')

plt.title("Add Line Graph")

plt.plot(dev_x,dev_y1,'r',label='Plot2')

plt.legend()

##plt.legend('Line Chart2')
## Pass marker, line and color as arguments

plt.plot(dev_x,dev_y, color='b', marker='o', label='Plot1')

plt.xlabel('Numbers')

plt.ylabel('Scores')

plt.title("Add Line Graph")

plt.plot(dev_x,dev_y1,'r',label='Plot2')

plt.legend()

##plt.legend('Line Chart2')
## colors can also be parsed as hex values and add linewidth

plt.plot(dev_x,dev_y, color='#8921cf', marker='o', linewidth=5, label='Plot1')

plt.xlabel('Numbers')

plt.ylabel('Scores')

plt.title("Add Line Graph")

plt.plot(dev_x,dev_y1,'r',label='Plot2')

plt.legend()

##Add Tight Layout to add padding to the graph



plt.tight_layout()
## Add Grid to check which values are actually high in the end plt.grid(True)

plt.plot(dev_x,dev_y, color='#8921cf',  label='Plot1')

plt.xlabel('Numbers')

plt.ylabel('Scores')

plt.title("Add Line Graph")

plt.plot(dev_x,dev_y1,'r',label='Plot2')

plt.legend()

plt.grid(True)
## ggplot, seaborn, fivethrityeight are some of the available style

print(plt.style.available)
## using plt style, SEE GRID IS AVAILABLE BY DEFAULT



plt.style.use('fivethirtyeight')



plt.plot(dev_x,dev_y,  label='Plot1')

plt.xlabel('Numbers')

plt.ylabel('Scores')

plt.title("Add Line Graph")

plt.plot(dev_x,dev_y1,'r',label='Plot2')

plt.legend()

##plt.grid(True)
## comic like graphic use ?

## use plt.xkcd(),

plt.xkcd()

plt.plot(dev_x,dev_y,  label='Plot1')

plt.xlabel('Numbers')

plt.ylabel('Scores')

plt.title("Add Line Graph")

plt.plot(dev_x,dev_y1,'r',label='Plot2')

plt.legend()
## save in a path as png



plt.plot(dev_x,dev_y,  label='Plot1')

plt.xlabel('Numbers')

plt.ylabel('Scores')

plt.title("Add Line Graph")

plt.plot(dev_x,dev_y1,'r',label='Plot2')

plt.legend()

plt.savefig('plot1.png')