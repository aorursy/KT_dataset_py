# Installing the matplotlib package



!pip install matplotlib
# Import the matplotlib package 



import matplotlib.pyplot as plt



%matplotlib inline
# Check all methods in plt library 



dir(plt)
# Plot 



List_var = [1,2,3,4,10]



plt.plot(List_var)
# 'go' stands for green dots



X_vals = [1,2,3,4,5]



Y_vals = [1,2,3,4,10]



plt.plot(X_vals,

         Y_vals, 

         'go')



plt.show()
# Create a fully Complete Plot with Title , Legend and Names of Axis .



X_vals = [1,2,3,4,5]



Y_vals = [1,2,3,4,10]



plt.plot(X_vals,

         Y_vals, 

         'go',

         label='GreenDots')



plt.title('A Simple Scatterplot')



plt.xlabel('X Data ')

plt.ylabel('Y Data')



plt.legend(loc='best')  # legend text comes from the plot's label parameter.



plt.show()
# Increase / change the default Plot Canvas Size 



plt.figure(figsize=(12,7)) # 12 is width, 7 is height



X_vals = [1,2,3,4,5]



Y_vals = [1,2,3,4,10]



plt.plot(X_vals,

         Y_vals, 

         'go',

         label='GreenDots')



plt.title('A Simple Scatterplot')



plt.xlabel('X Data ')

plt.ylabel('Y Data')



plt.legend(loc='best')  # legend text comes from the plot's label parameter.



plt.show()
# Create Figure and Subplots

fig, (ax1, ax2) = plt.subplots(1,2,                   # 1 row & 2 colums 

                               figsize=(12,5),        # 12 width and 5 as height

                               sharey=True,           # Y axis is to be common for subplots

                               dpi=120)               # Dots per inch - Resolution 



# Plot

ax1.plot([1,2,3,4,5], [1,2,3,4,10], 'go')  # greendots

ax2.plot([1,2,3,4,5], [2,3,4,5,11], 'b*')  # bluestart



# Title, X and Y labels, X and Y Lim

ax1.set_title('Scatterplot Greendots') 

ax2.set_title('Scatterplot Bluestars')



ax1.set_xlabel('X');  ax2.set_xlabel('X')  # x label

ax1.set_ylabel('Y');  ax2.set_ylabel('Y')  # y label



ax1.set_xlim(0, 6) ;  ax2.set_xlim(0, 6)   # x axis limits

ax1.set_ylim(0, 12);  ax2.set_ylim(0, 12)  # y axis limits





plt.tight_layout()

plt.show()
# Draw a Scatter plot 



import numpy as np



X_Data = np.linspace(0, 1., 100)



# Scatterplot

plt.scatter(X_Data, X_Data + np.random.randn(len(X_Data)))



plt.title("Scatter Plot")

# Draw Bar Plot



# Input Arrays

n = np.array([0,1,2,3,4,5])



# Bar Chart

plt.bar(n, n**2, 

        align="center", 

        width=0.5, 

        alpha=0.5)



plt.title("Bar Chart")

# Draw a Box plot 



X_Data = np.linspace(0, 1., 100)



# Box Plot

plt.boxplot(np.random.randn(len(X_Data)))



plt.title("Box Plot")
# Draw a Histogram plot 



X_Data = np.linspace(0, 1., 100)



# Box Plot

plt.hist(np.random.randn(len(X_Data)))



plt.title("Histogram")