def logistic(r, x):

    return r * x * (1 - x)



r=2.5

a=0.023

aValList=[]

aValList.append(logistic(r,a))

for i in range(1,11,1):

    x=logistic(r,aValList[i-1])

    aValList.append(x)

   

print(aValList)
#linegraph

import matplotlib.pyplot as plt

import numpy as np

x=[1,2,3]

y=[5,7,4]

x2=[1,2,3]

y2=[1,1,1]

plt.plot(x,y,label="First Line")

plt.plot(x2,y2,label="Second Line")

plt.xlim(0.1,10,1)

plt.xticks(np.arange(min(x), max(x)+1, 1))

plt.xlabel("X-axis")

plt.ylabel("Y-axis")

plt.title("First Graph\n")

plt.legend()

plt.ylim(0.1,10,1)

plt.show()
#Bar Charts and Historgram

import matplotlib.pyplot as plt

import random

i=random.Random()

x=[2,4,6,8]

y=[1,2,3,4]

x2=[4,6,8,10,12]

y2=[5,6,7,8,9]

#plt.bar(x,y,label="Bar Graph 2",color="c")

#plt.bar(x2,y2,label="Bar Graph 2",color="r")

plt.hist(x2,y2,histtype="bar",rwidth=0.8)

plt.xlabel("X")

plt.xlabel("Y")

#plt.legend()

plt.show()
#scatter plot shows correlation between two variables 3D scatter plot can show correlation for 3 variables

x=[1,2,3,4,5]

y=[1,2,3,4,5]

plt.scatter(x,y,s=100,marker="*")

plt.show()
#stack plot

days=[11,20,32,42,50]

working=[11,20,30,41,52]

playing=[10,26,37,48,52]

eating=[10,20,31,40,51]

boring=[10,22,32,14,56]

plt.plot([],[],color='r',label="working",linewidth=5)

plt.plot([],[],color='g',label="playing",linewidth=5)

plt.plot([],[],color='c',label="eating",linewidth=5)

plt.plot([],[],color='k',label="boring",linewidth=5)

plt.stackplot(days,working,playing,eating,boring,colors=['r','g','c','k'])

plt.legend()

plt.show()
#pie chart

days=[11,20,32,42,50]

working=[11,20,30,41,52]

playing=[10,26,37,48,52]

eating=[10,20,31,40,51]

boring=[10,22,32,14,56]



slices=[7,2,2,13]

activities=["Sleeping","Playing","Working","Eating"]

cols=['m','c','b','r']

plt.pie(slices,labels=activities,colors=cols,startangle=90,shadow=True,explode=(0,0.1,0,0),autopct='%1.1f%%')



plt.title("Piechart")

plt.legend()

plt.show()
