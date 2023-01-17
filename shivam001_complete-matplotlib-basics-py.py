# import all dependencies
# import style  also 
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
# simple example using matplotlib
#list of numbers in x1
x1=[6,9,11]
#list of numbers in y1
y1=[13,17,8]

#list of numbers in x2
x2=[8,11,13]

#list of numbers in y2
y2=[8,18,9]


plt.plot(x1,y1,'g',label="line1",linewidth=5)
plt.plot(x2,y2,'c',label="line2",linewidth=5)
plt.title("GRAPH")
plt.xlabel("No of tiger in the jungle")
plt.ylabel("No of people who care tiger")
# to make grid in our graph
plt.grid(True,color='y')
plt.legend()
plt.show()
#Now we will plot bar graph
# it use categorical variables
x=[1,2,3,5,6,7]
y=[4,5,6,1,2,7]

a=[2,4,6,8,10]
b=[8,6,2,5,6]

plt.bar(x,y,label='example_1',color='y')
plt.bar(a,b,label='example_2',color='c')
plt.title('Information')
plt.xlabel('Height')
plt.ylabel('Weight')
#because we wants to show the effect of style(ggplot)
plt.legend()
plt.show()
# Now we will plot histogram graph
# it use Quantitative variables
age=[10,20,25,30,35,40,21,4,56,50,31,34,45,63,90,85,110]
bins=[0,10,20,30,40,50,60,70,80,90,100,110]
plt.hist(age,bins,label="Contributes in GDP",histtype='bar',color='g',rwidth=0.8)
plt.title("View from Histogram")
plt.xlabel("Age groups")
plt.ylabel("Contributes in GDP")
plt.legend()
plt.show()
# Now Scatter plot
# Use scatter plot to relate 2-3 or groups to each other
x=[7,6,5,9,3,4]
y=[3,5,2,3,2,4]

plt.scatter(x,y,label="Example",color='r',marker="*")
plt.title("Scatter Plot")
plt.xlabel("No of person on a family")
plt.ylabel("How many they have job")
plt.grid(True,color='y')
plt.legend()
plt.show()