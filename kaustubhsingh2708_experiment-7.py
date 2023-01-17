#Plotting using points
from matplotlib import pyplot as plt
plt.plot([1,2,3],[4,5,1])
plt.show()

#Plotting using variables
x=[5,8,10]
y=[12,16,6]
plt.plot(x,y)
plt.title("info")
plt.ylabel("Y axis")
plt.xlabel("X axis")
plt.show()
#Adding Style
from matplotlib import style
style.use("ggplot")
x1=[5,8,10]
y1=[12,16,6]
x2=[6,9,11]
y2=[6,15,7]
plt.plot(x1,y1,'g',label="line one",linewidth=5)
plt.plot(x2,y2,'c',label="line two",linewidth=5)

plt.title("Epic info")
plt.ylabel("Y axis")
plt.xlabel("X axis")
plt.legend()
plt.grid(True,color='k')
plt.show()
#Plotting Bar Graph
plt.bar([1,3,5,7,9],[5,2,7,8,2],label="Example 1",color="c")
plt.bar([2,4,6,8,10],[8,6,2,5,6],label="Example two",color="g")
plt.legend()
plt.xlabel("bar number")
plt.ylabel("bar height")
plt.title("Bar graph")
plt.show()
#Plotting Histogram
population_ages=[22,55,62,34,21,22,43,53,52,4,99,102,120,111,121,130,111,75,85,63,72,62,75,23,52,64,12,42,54,48]
bins=[0,10,20,30,40,50,60,70,80,90,100,110,120,130]
plt.hist(population_ages,bins,histtype="bar",rwidth=0.8,color="c")
plt.title("Histogram")
plt.legend()
plt.show()
#Scatter Plot
x=[1,2,3,4,5,6,7,8]
y=[5,2,4,2,1,4,5,2]
plt.scatter(x,y,label="skitscat",color="k")
plt.title("Scatter plot")
plt.legend()
plt.show()

#Stack plot
days=[1,2,3,4,5]
sleeping=[7,8,6,11,7]
eating=[2,3,4,3,2]
working=[7,8,7,2,2]
playing=[8,5,7,8,13]

plt.plot([],[],color="m",label="Sleeping",linewidth=5)
plt.plot([],[],color="c",label="Eating",linewidth=5)
plt.plot([],[],color="r",label="Working",linewidth=5)
plt.plot([],[],color="k",label="Playing",linewidth=5)
plt.stackplot(days,sleeping,eating,working,playing,colors=["m","c","r","k"])

plt.title("Stack Plot")
plt.legend()
plt.show()
#Pie chart
slices=[7,2,8,3]
act=["sleeping","eating","working","playing"]
cols=["c","m","g","r"]
plt.pie(slices,
       labels=act,
       colors=cols,
       startangle=90,
       shadow=True,
        explode=(0,0.1,0,0),
       autopct="%1.1f%%")
plt.title("Pie Chart")
plt.show()
#Working with multiple plots
import numpy as np
def f(t):
    return np.exp(-t)*np.cos(2*np.pi*t)
t1=np.arange(0.0,5.0,0.1)
t2=np.arange(0.0,5.0,0.02)
plt.subplot(211)
plt.plot(t1,f(t1),'bo',t2,f(t2))
plt.subplot(212)
plt.plot(t2,np.cos(2*np.pi*t2))
plt.show()