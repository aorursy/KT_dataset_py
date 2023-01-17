# Import the necessary packages and modules

import matplotlib.pyplot as plt

import numpy as np



# Prepare the data

x = [1,2,3,4]

y= [1,4,9,10]



# Plot the data

plt.plot(x, y, label='legend')



#Add Title

plt.title("First Graph")

plt.ylabel("Y Label")

plt.xlabel("X Label")



# Add a legend

plt.legend(loc="upper left")



# Show the plot

plt.show()





# Prepare the data

x = [1,2,3,4]

y= [1,4,9,10]



#FigureSize

plt.figure(figsize=(15,6))



# Plot the data

plt.plot(x, y, label='legend')



#Add Title

plt.title("Title")

plt.ylabel("Y Label")

plt.xlabel("X Label")



# Add a legend

plt.legend()



# Show the plot

plt.show()

#FigureSize

plt.figure(figsize=(15,6))



# Plot the data

plt.plot(x, x, "go") #Go is the additional arugument , it will print graph using the dots or circle.



#Add Title

plt.title("Title")



# Add a legend

plt.legend()



# Show the plot

plt.show()
#FigureSize

plt.figure(figsize=(15,6))



# Plot the data

plt.plot([3,6,9,13],[2,5,9,16], "go", [4,5,8,12],[1,4,8,15],'r^') #Go is the additional arugument , it will print graph using the dots or circle.



#Add Title

plt.title("Title")



# Add a legend

plt.legend()



# Show the plot

plt.show()
#FigureSize

plt.figure(figsize=(15,6))



# Plot the data

plt.subplot(1,2,1)

plt.plot([3,6,9,13],[2,5,9,16], "go")

plt.title("Subplot-1")



plt.subplot(1,2,2)

plt.plot([4,5,8,12],[1,4,8,15],'r^')

plt.title("Subplot-2")



# Add a legend

plt.legend()



# Show the plot

plt.show()
#FigureSize

plt.figure(figsize=(15,6))



# Plot the data

plt.subplot(2,2,1)

plt.plot([3,6,9,13],[2,5,9,16], "go")

plt.title("Subplot-1")



plt.subplot(2,2,2)

plt.plot([4,5,8,12],[1,4,8,15],'r^')

plt.title("Subplot-2")



# Add a legend

plt.legend()



# Show the plot

plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2)

fig.suptitle('Subplots')

ax1.plot([3,6,9,13],[2,5,9,16], "go")

ax2.plot([4,5,8,12],[1,4,8,15],'r^')
cls_name = ['A','B','C','D','E']

total_std = [20,30,40,50,60]

plt.bar(cls_name,total_std,color="Blue")

plt.title("Bar Graph")

plt.ylabel(total_std)

plt.xlabel(cls_name)

plt.show()
cls_name = ['A','B','C','D','E']

total_std = [20,30,40,50,60] 

variance = [7,8,9,10,11]

plt.barh(cls_name,total_std, xerr=variance,color="Blue")

plt.title("Bar Graph")

plt.ylabel(total_std)

plt.xlabel(cls_name)

plt.show()
cls_name = ['A','B','C','D','E']

total_std = [20,30,40,50,60] 

total_boys = [7,8,9,10,11]



index = np.arange(5)

width=0.30



plt.bar(index,total_std, width,color="Green", label = "Total Students")

plt.bar(index+width,total_boys, width,color="Yellow", label = "Total Boys")



plt.title("Horizontally Stacked Bar Graph")

plt.ylabel("Total Students")

plt.xlabel("Total Boys")

plt.xticks(index+width/2,cls_name)

plt.legend(loc="best")

plt.show()
cls_name = ['A','B','C','D','E']

total_std = [20,30,40,50,60] 

total_boys = [7,8,9,10,11]



index = np.arange(5)

width=0.30



plt.bar(index,total_std, width,color="Green", label = "Total Students")

plt.bar(index,total_boys, width,color="Yellow", label = "Total Boys", bottom =total_std)



plt.title("Horizontally Stacked Bar Graph")

plt.ylabel("Total Students")

plt.xlabel("Total Boys")

plt.xticks(index,cls_name)

plt.legend(loc="best")

plt.show()
x = np.random.randn(1000)

plt.title("Histogram")



plt.ylabel("Frequency")

plt.xlabel("Random Data")



plt.hist(x,bins=30)

plt.show()
Technologies = ["Tech-1","Tech-2","Tech-3","Tech-4","Tech-5"]

tech_pop = [30,25,40,60,80]

Explode = [0,0.1,0,0,0.2]

plt.pie(tech_pop,explode=Explode,labels=Technologies,shadow=True,startangle=45)

plt.axis('equal')

plt.legend(loc="best")

plt.show()

Height = np.array([110,120,130,160,180,190])

Weight = np.array([50,60,70,85,95,100])

# Plot

plt.scatter(Height,Weight)

plt.title('Scatter plot')

plt.xlabel('Height')

plt.ylabel('Weight')

plt.show()
from mpl_toolkits import mplot3d

ax = plt.axes(projection='3d')

ax.scatter3D(Height,Weight,color="green")

ax.set_xlabel("x")

ax.set_ylabel("y")