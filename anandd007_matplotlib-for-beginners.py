import matplotlib.pyplot as plt # import Matplotlib submodule for plotting

plt.rcParams['figure.figsize'] = (12, 8) # to show plots centered and large

plt.plot([1,2,3,4])            # To plot a simple graph on matplotlib

plt.show()                     # To Display the plotted graph in the output
# Another Simple graph :

plt.plot([1,2,3,4,10])

plt.show()
x = range(5)                      # Sequences of values for the x-axis

plt.plot(x, [x1**1 for x1 in x])  # Vertical co-ordinates of the points plotted: y = x^2

plt.show()



# # Another Example :



import numpy as np

x2 = np.arange(0,5,0.1)

plt.plot(x2,[i**2 for i in x2])

plt.show()
x = range(5)

plt.plot(x,[x1 for x1 in x])        # Three lines are plotted

plt.plot(x,[x1*x1 for x1 in x])

plt.plot(x,[x1*x1*x1 for x1 in x])

plt.show()



# # Another Example :

x = range(5)

plt.plot(x,[x1 for x1 in x], x,

         [x1*x1 for x1 in x], x,[x1*x1*x1 for x1 in x])

plt.show()
x = range(5)

plt.plot(x,[x1 for x1 in x],x,[x1*2 for x1 in x],x,[x1*4 for x1 in x])

plt.grid(True)

plt.axis([-1,5,-1,10])           # Sets new axes limits

plt.show()
y = range(5)

plt.plot(y,[x1 for x1 in y],y,[x1*2 for x1 in y],y,[x1*4 for x1 in y])

plt.grid(True)

plt.xlim(-1, 6)

plt.ylim(-1,12)

plt.show()
x_values = list(range(1000))

squares = [x ** 2 for x in x_values]

plt.scatter(x_values, squares, s=8)  # 's' is size of the scatter points or plot scattering size

plt.title("Square Numbers", fontsize=20)

plt.xlabel("Value", fontsize=14)

plt.ylabel("Square of Value", fontsize=14)

# plt.tick_params(axis='both', which='major', labelsize=14) # 'labelsize' is the size of the label

plt.axis([0, 1100, 0, 1100000])

plt.show()
plt.scatter(x_values, squares, c=squares,

#             cmap=plt.cm.Blues,

            cmap=plt.get_cmap(), # cmap is color mapping

            edgecolor='none',s=10)

plt.title("Scatter Numbers Line", fontsize=14)

plt.ylabel("A", fontsize=12)

plt.xlabel("Value", fontsize=14)

plt.show()
x_values = list(range(1000))

squares = [x ** 2 for x in x_values]



plt.scatter(x_values, squares, s=10)

plt.title("Scatter Plot Line")

plt.show()
x_values = list(range(1000))

squares = [x ** 2 for x in x_values]

plt.scatter(x_values, squares, c=squares, cmap=plt.get_cmap(), edgecolor='none', s=30,alpha=0.8)

plt.colorbar(label='Squares')

plt.scatter(x_values[0], squares[0], c='green', edgecolor='none', s=100)

plt.scatter(x_values[-1], squares[-1], c='red', edgecolor='none', s=100)

plt.title("Square Numbers", fontsize=24)

plt.show()
plt.title('Historigram',fontsize=24)

plt.ylabel('Y Values',fontsize=14)

plt.xlabel('X Values',fontsize =14)

y = np.random.randn(10,10)

plt.hist(y,rwidth=30)

plt.savefig('Matplotlib.png')

plt.show()
plt.title('Simple Bar Chart',fontsize =20)

plt.ylabel('Y Values',fontsize =14)

plt.xlabel('X Values',fontsize=14)

plt.bar([1.5,2.5,3.5,4.5,5.5,6.5],[25,45,89,78,56,49])

plt.savefig('Matplotlib.png')

plt.show()
plt.bar([1,3,5,7,9],[5,2,7,8,2],label="Example One",color="g")

plt.bar([2,4,6,8,10],[8,6,4,5,2],label="Example Two",color="c")

plt.legend()

plt.xlabel("Bar Number")

plt.ylabel("Bar Height")

plt.title('Style Bar Chart')

plt.show()
import numpy



d = {'a':25,'b':45,'c':52}

plt.title('Dictionary Using Bar Chart',fontsize=20)

plt.ylabel('Y')

plt.xlabel('x')

for i,key in enumerate(d):

    print(i,key)

    plt.bar(i,d[key])

plt.show()
dictionary = {'A':25,'B':70,'C':80,'D':90}

for i,key in enumerate(dictionary):

    print(i,key)

    plt.bar(i,dictionary[key])

plt.xticks(numpy.arange(len(dictionary)),dictionary.keys())

plt.ylabel('Y values',fontsize =10)

plt.xlabel('Adds the keys as labels on the x-axis',fontsize =10)

plt.show()
dictionary = {'A':25,'B':70,'C':80,'D':90}

for i,key in enumerate(dictionary):

    print(i,key)

    plt.bar(i,dictionary[key])

plt.xticks([0,1,2,3],['A-BAR','B-BAR','C-BAR','D-BAR',])

plt.show()
# plt.figure(figsize=(5,5))

plt.title('Simple Pie Chart')

plt.pie([20,30,40,50,],labels=['agri','health','weath','work'])

plt.show()
x = numpy.random.rand(10)    # x & y values should be the same size.

y = numpy.random.rand(10)

plt.title('Scattering Plots')

print(x,y)

print("---------------------------------------------------------------------------------")

plt.scatter(x,y)

plt.show()
# y = numpy.arange(1,3)

plt.title('Colouring Plots')

plt.plot(y,'y')

plt.plot(y+5,'m')

plt.plot(y+10,'c')

plt.plot(y*2,'k')

plt.plot(y+15,'g')

plt.plot(y+20,'b')

plt.show()
y = numpy.arange(1,100)

plt.title('Line Styling')

plt.plot(y, '--', y*5, '-.', y*10, ':') # Specifying line styling

plt.show()
y = numpy.arange(1,3,0.2)

plt.title('Control Marker Styling')

plt.plot(y,'*',y+0.5,'o',y+1,'D',y+2,'^',y+3,'s')

plt.show()
plt.figure(figsize=(8,8))

plt.pie([70,50,80,70,25,30],labels=['Python','C','Html','SQL','CSS','Others'],startangle=90,shadow=True,autopct='%1.2f%%')

plt.title('Pie Chart Styling',color='darkred',fontsize=20)

plt.show()
from matplotlib import style

style.use('ggplot')



x = [5,8,10]

y = [12,16,6]



x2 = [6,9,11]

y2 = [6,9,11]



plt.plot(x,y,'g',label='line one',linewidth = 5)

plt.plot(x2,y2,'c',label='line two',linewidth=5)



plt.title('Epic Info')

plt.ylabel('Y-axis')

plt.xlabel('X-axis')



plt.legend()

plt.grid(True,color='g')

plt.show()
plt.bar([1,3,5,7,9,11],[5,2,7,8,2,4],label="Example One",color="g")

plt.bar([2,4,6,8,10,12],[8,6,4,5,2,11],label="Example Two",color="c")

plt.legend()

plt.xlabel("Bar Number")

plt.ylabel("Bar Height")

plt.title('Style Bar Graph')

plt.show()
x = [1,2.3,3.7,4,5.2,6.4,7,8]

y = [5,2.4,3.6,4.5,1.8,2.6,5,3]



plt.scatter(x,y,label='Scatter Graph',c=y,cmap=plt.get_cmap(),s=40)

plt.legend()

plt.show()
days = [1,2,3,4,5]

sleeping = [7,8,10,9,6]

eating = [2,3,4,1,2]

working = [7,8,9,6,4]

playing = [3,6,7,4,8]



plt.plot([],[],color='m', label='Sleeping',linewidth=5)

plt.plot([],[],color='c', label='Eating',linewidth=5)

plt.plot([],[],color='g',label='Working',linewidth=5)

plt.plot([],[],color='r',label='Playing',linewidth=5)



plt.stackplot(days,sleeping,eating,working,playing,edgecolor="k",colors=['m','c','g','r'])



plt.xlabel("Days")

plt.ylabel("Efficiency")

plt.title("Stack Plot")

plt.legend()

plt.show()
