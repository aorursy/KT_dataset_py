from matplotlib import pyplot as plt



# Lets add title and labels

x=[1,5,10]

y=[11,15,6]

plt.plot(x,y)



plt.title("Graph")

plt.ylabel("Y axis")

plt.xlabel("X axis")

plt.show()
# Adding Style in our graph

from matplotlib import style

style.use("ggplot")

x=[1,5,10]

y=[11,16,8]



x1=[7,12,13]

y1=[7,15,8]



plt.plot(x,y,'g',label="one", linewidth=3)

plt.plot(x1,y1,'c',label="two",linewidth=3)



plt.title("Info")

plt.ylabel("Y axis")

plt.xlabel("X axis")

plt.legend()

plt.grid(True, color="k")

plt.show()
plt.bar([2,4,6,8,10,12],[1,3,5,7,9,11],label="1")

plt.bar([1,5,6,9,12,15],[2,6,8,10,1,16],label="2",color='orange')

plt.legend()

plt.xlabel("number")

plt.ylabel("height")

plt.title("info")

plt.show()
import warnings

warnings.filterwarnings("ignore")

population=[20,25,26,28,18,16,36,46,28,27,23,45,46,55,35,36]

bins=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]

plt.hist(population,bins,histtype="bar",rwidth=0.5)

plt.xlabel("X")

plt.ylabel("Y")

plt.title("Histogram")

plt.legend()

plt.show()
x=[2,4,6,8,10,12,14,16]

y=[2,3,5,7,9,11,13,19]

plt.scatter(x,y,color="orange", s=35,marker="o")

plt.xlabel("X")

plt.ylabel("Y")

plt.title("Scatter Plot")

plt.legend()

plt.show()
slices=[5,3,6,13]

activities=[ "eating", "dancing","singing","driving"]

cols=["m","r","b","c"]

plt.pie(slices, labels=activities, colors=cols, startangle=90, shadow=True, explode=(0,0.1,0,0),autopct="%1.1f%%")

plt.title("Pie Chart")

plt.show()
import numpy as np

def f(n):

    return np.exp(-n)*np.cos(2*np.pi*n)

n1= np.arange(0.0,4.5,0.2)

n2= np.arange(0.0,4.5, 0.05)



plt.subplot(211)

plt.plot(n1, f(n1),"bo",n2, f(n2))



plt.subplot(212)

plt.plot(n2, np.cos(2*np.pi*n2))

plt.show()