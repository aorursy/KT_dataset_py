import matplotlib.pyplot as plt

%matplotlib inline
import numpy as np
x=np.arange(0,10)

y=np.arange(11,21)
plt.scatter(x,y)
#plotting using matplotlib 



#plt scatter

plt.scatter(x,y , c='r')

plt.xlabel("X-Aix")

plt.ylabel("Y-Axis")

plt.title("Graph ")

# saving picture of 2d graph 

plt.savefig("2dGraph.png")
y=x*x
y
#plt plot

plt.plot(x,y, 'g*', linestyle='solid', linewidth=2 , markersize =12)

#ploting sub_graph

plt.subplot(2,2,1)

plt.plot(x,y ,'r*')

plt.subplot(2,2,2)

plt.plot(x,y ,'g--')

plt.subplot(2,2,3)

plt.plot(x,y ,'bo')

plt.subplot(2,2,4)

plt.plot(x,y ,'go')

x=np.arange(1,11)

y=5*x+5

plt.title("2d Graph")

plt.plot(x,y)
np.pi


# Compute the x and y coordinates for points on a sine curve 



x=np.arange(0,4*np.pi,0.1)

y=np.sin(x)

plt.title("sine wave form") 



# Plot the points using matplotlib 

plt.plot(x, y) 

plt.show()


#Subplot()

# Compute the x and y coordinates for points on sine and cosine curves 

x = np.arange(0, 5 * np.pi, 0.1) 

y_sin = np.sin(x) 

y_cos = np.cos(x)  

   

# Set up a subplot grid that has height 2 and width 1, 

# and set the first such subplot as active. 

plt.subplot(2, 1, 1)

   

# Make the first plot 

plt.plot(x, y_sin,'r--') 

plt.title('Sine')  

   

# Set the second subplot as active, and make the second plot. 

plt.subplot(2, 1, 2) 

plt.plot(x, y_cos,'g--') 

plt.title('Cosine')  

   

# Show the figure. 

plt.show()
#subplot()

# compute the x and y coordinates for points on sine and cosine curves

x=np.arange(0,5*np.pi,0.1)

y_sin=np.sin(x)

y_cos=np.cos(x)

# Set up a subplot grid that has height 2 and width 1, 

# and set the first such subplot as active.

plt.subplot(2, 1, 1)

# Make the first plot 

plt.plot(x, y_sin,'r--') 

# Set the second subplot as active, and make the second plot. 

plt.subplot(2, 1, 2) 

plt.plot(x, y_cos,'g--') 

plt.title('Cosine')  

   

# Show the figure. 

plt.show()



#Bar plot

x = [2,8,10] 

y = [11,16,9]  



x2 = [3,9,11] 

y2 = [6,15,7] 

plt.bar(x, y) 

plt.bar(x2, y2, color = 'g') 


a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27]) 

plt.hist(a) 

plt.title("histogram") 

plt.show()
#Box plot using matplot



data = [np.random.normal(0, std, 100) for std in range(1, 4)]



# rectangular box plot

plt.boxplot(data,vert=True,patch_artist=False);
 #Data to plot

labels = 'Python', 'C++', 'Ruby', 'Java'

sizes = [215, 130, 245, 210]

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

explode = (0.4, 0, 0, 0)  # explode 1st slice



# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=False)



plt.axis('equal')

plt.show()