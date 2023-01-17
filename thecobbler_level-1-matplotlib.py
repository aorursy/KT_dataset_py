import matplotlib.pyplot as plt 
fig = plt.figure(figsize=(8,6)) 

ax = fig.add_subplot(111) 

ax.set_title("My First Plot") 

ax.set_xlabel("X-Axis"); 

ax.set_ylabel('Y-Axis') 

ax.set_xlim([0,5]); 

ax.set_ylim([0,10]) 

plt.show()



fig = plt.figure(figsize=(8,6)) 

ax = fig.add_subplot(111) 

ax.set(title='My First Plot', xlabel='X-Axis', ylabel='Y-Axis', xlim=(0, 5), ylim=(0,10)) 

x = [1, 2, 3, 4]; y = [2, 4, 6, 8] 

plt.plot(x, y) 

plt.show()
fig = plt.figure(figsize=(8,6))

x = [1, 2, 3, 4];

y = [2, 4, 6, 8] 

plt.plot(x, y) 

plt.title('My First Plot') 

plt.xlabel('X-Axis'); 

plt.ylabel('Y-Axis') 

plt.xlim(0,5); 

plt.ylim(0,10) 

plt.plot(x, y) 

plt.show()
fig = plt.figure(figsize=(8,6)) 

ax = fig.add_subplot(111) 

ax.set(title='My First Plot', xlabel='X-Axis', ylabel='Y-Axis', xlim=(0, 5), ylim=(0,10)) 

x = [1, 2, 3, 4]; y = [2, 4, 6, 8] 

plt.plot(x, y, label='linear-growth') 

plt.legend()

plt.show()
fig = plt.figure(figsize=(8,6)) 

ax = fig.add_subplot(111) 

ax.set(title='Avg. Daily Temperature in Jan 2018', xlabel='Day', ylabel='Temperature (in deg)',

xlim=(0, 30), ylim=(25, 35)) 

days = [1, 5, 8, 12, 15, 19, 22, 26, 29] 

temp = [29.3, 30.1, 30.4, 31.5, 32.3, 32.6, 31.8, 32.4, 32.7] 

ax.plot(days, temp) 

plt.show()
fig = plt.figure(figsize=(8,6)) 

ax = fig.add_subplot(111) 

ax.set(title='Avg. Daily Temperature of Jan 2018', xlabel='Day', ylabel='Temperature (in deg)',

xlim=(0, 30), ylim=(25, 35)) 

days = [1, 5, 8, 12, 15, 19, 22, 26, 29] 

location1_temp = [29.3, 30.1, 30.4, 31.5, 32.3, 32.6, 31.8, 32.4, 32.7] 

location2_temp = [26.4, 26.8, 26.1, 26.4, 27.5, 27.3, 26.9, 26.8, 27.0] 

ax.plot(days, location1_temp, color='green', marker='o', linewidth=3) 

ax.plot(days, location2_temp, color='red', marker='o', linewidth=3) 

plt.show()
fig = plt.figure(figsize=(8,6)) 

ax = fig.add_subplot(111) 

ax.set(title='Avg. Daily Temperature of Jan 2018', xlabel='Day', ylabel='Temperature (in deg)',

xlim=(0, 30), ylim=(25, 35)) 

days = [1, 5, 8, 12, 15, 19, 22, 26, 29] 

temp = [29.3, 30.1, 30.4, 31.5, 32.3, 32.6, 31.8, 32.4, 32.7] 

ax.scatter(days, temp) 

plt.show()
#plot function can also create a scatter plot when linestyle is set to none, 

#and a marker is chosen, as shown in below code. 

fig = plt.figure(figsize=(8,6)) 

ax = fig.add_subplot(111) 

ax.set(title='Avg. Daily Temperature of Jan 2018', xlabel='Day', ylabel='Temperature (in deg)',

xlim=(0, 30), ylim=(25, 35)) 

days = [1, 5, 8, 12, 15, 19, 22, 26, 29] 

temp = [29.3, 30.1, 30.4, 31.5, 32.3, 32.6, 31.8, 32.4, 32.7] 

ax.plot(days, temp, marker='o', linestyle='none') 

plt.show()
fig = plt.figure(figsize=(8,6)) 

ax = fig.add_subplot(111) 

ax.set(title='Avg. Quarterly Sales', xlabel='Quarter', ylabel='Sales (in millions)') 

quarters = [1, 2, 3] 

sales_2017 = [25782, 35783, 36133] 

ax.bar(quarters, sales_2017) 

ax.set_xticks(quarters) 

ax.set_xticklabels(['Q1-2017', 'Q2-2017', 'Q3-2017']) 

plt.show()
fig = plt.figure(figsize=(8,6)) 

ax = fig.add_subplot(111) 

ax.set(title='Avg. Quarterly Sales', xlabel='Quarter', ylabel='Sales (in millions)') 

quarters = [1, 2, 3] 

sales_2017 = [25782, 35783, 36133] 

ax.bar(quarters, sales_2017, color='red', width=0.6, edgecolor='black')

ax.set_xticks(quarters) 

ax.set_xticklabels(['Q1-2017', 'Q2-2017', 'Q3-2017']) 

plt.show()
fig = plt.figure(figsize=(8,6)) 

ax = fig.add_subplot(111) 

ax.set(title='Avg. Quarterly Sales', xlabel='Quarter', ylabel='Sales (in millions)') 

quarters = [1, 2, 3] 

x1_index = [0.8, 1.8, 2.8]; 

x2_index = [1.2, 2.2, 3.2] 

sales_2016 = [28831, 30762, 32178]; sales_2017 = [25782, 35783, 36133] 

ax.bar(x1_index, sales_2016, color='yellow', width=0.4, edgecolor='black', label='2016') 

ax.bar(x2_index, sales_2017, color='red', width=0.4, edgecolor='black', label='2017') 

ax.set_xticks(quarters) 

ax.set_xticklabels(['Q1', 'Q2', 'Q3']) 

ax.legend() 

plt.show()
fig = plt.figure(figsize=(8,6)) 

ax = fig.add_subplot(111) 

ax.set(title='Avg. Quarterly Sales', xlabel='Sales (in millions)', ylabel='Quarter') 

quarters = [1, 2, 3] 

sales_2017 = [25782, 35783, 36133] 

ax.barh(quarters, sales_2017, 

height=0.6, color='red') 

ax.set_yticks(quarters) 

ax.set_yticklabels(['Q1-2017', 'Q2-2017', 'Q3-2017']) 

plt.show()
fig = plt.figure(figsize=(6,6)) 

ax = fig.add_subplot(111) 

ax.set(title='Avg. Quarterly Sales') 

sales_2017 = [25782, 35783, 36133] 

ax.pie(sales_2017) 

plt.show()
fig = plt.figure(figsize=(6,6)) 

ax = fig.add_subplot(111) 

ax.set(title='Avg. Quarterly Sales') 

sales_2017 = [25782, 35783, 36133] 

quarters = ['Q1-2017', 'Q2-2017', 'Q3-2017'] 

ax.pie(sales_2017, labels=quarters, startangle=90, autopct='%1.1f%%') 

plt.show()
import numpy as np 

np.random.seed(100) 

x = 60 + 10*np.random.randn(1000) 

fig = plt.figure(figsize=(8,6)) 

ax = fig.add_subplot(111) 

ax.set(title="Distribution of Student's Percentage", ylabel='Count', xlabel='Percentage') 

ax.hist(x) 

plt.show()
import numpy as np 

np.random.seed(100) 

x = 60 + 10*np.random.randn(1000) 

fig = plt.figure(figsize=(8,6)) 

ax = fig.add_subplot(111) 

ax.set(title="Distribution of Student's Percentage", ylabel='Proportion', xlabel='Percentage') 

ax.hist(x, color='blue', bins=30, normed=True) 

plt.show()
import numpy as np 

np.random.seed(100) 

x = 50 + 10*np.random.randn(1000) 

fig = plt.figure(figsize=(8,6)) 

ax = fig.add_subplot(111) 

ax.set(title="Box plot of Student's Percentage", xlabel='Class', ylabel='Percentage') 

ax.boxplot(x) 

plt.show()
import numpy as np 

np.random.seed(100) 

x = 50 + 10*np.random.randn(1000) 

y = 70 + 25*np.random.randn(1000) 

z = 30 + 5*np.random.randn(1000) 

fig = plt.figure(figsize=(8,6)) 

ax = fig.add_subplot(111) 

ax.set(title="Box plot of Student's Percentage", xlabel='Class', ylabel='Percentage') 

ax.boxplot([x, y, z], labels=['A', 'B', 'C'], notch=True, bootstrap=10000) 

plt.show()
import numpy as np 

np.random.seed(100) 

x = 50 + 10*np.random.randn(1000) 

y = 70 + 25*np.random.randn(1000) 

z = 30 + 5*np.random.randn(1000) 

fig = plt.figure(figsize=(8,6)) 

ax = fig.add_subplot(111) 

ax.set(title="Box plot of Student's Percentage", xlabel='Percentage', ylabel='Class') 

ax.boxplot([x, y, z], labels=['A', 'B', 'C'], vert=False, notch=True, bootstrap=10000) 

plt.show()
print(plt.style.available)
#plt.style.use('ggplot') or

#plt.style.context('ggplot') 

plt.style.context('ggplot')

fig = plt.figure(figsize=(8,6)) 

ax = fig.add_subplot(111) 

ax.set(title='Avg. Daily Temperature of Jan 2018', xlabel='Day', ylabel='Temperature (in deg)', xlim=(0, 30), ylim=(25, 35)) 

days = [1, 5, 8, 12, 15, 19, 22, 26, 29] 

temp = [29.3, 30.1, 30.4, 31.5, 32.3, 32.6, 31.8, 32.4, 32.7] 

ax.plot(days, temp, color='green', linestyle='--', linewidth=3) 

plt.show()
#The location of active matplotlibrc file used by matplotlib can be found with below expression

#import matplotlib matplotlib.matplotlib_fname()



#Matplotlib rcParams All rc settings



#present in matplotlibrc file are stored in a dictionary named matplotlib.rcParams. 

#Any settings can be changed by editing values of this dictionary. 
#For example, if you want to change linewidth and color, the following expressions can be used.

import matplotlib as mpl 

mpl.rcParams['lines.linewidth'] = 2 

mpl.rcParams['lines.color'] = 'r'
#Syntax subplot(nrows, ncols, index)



#'index' is the position in a virtual grid with 'nrows' and 'ncols'

#'index' number varies from 1 to nrows*ncols.

#subplot creates the Axes object at index position and returns it.
fig = plt.figure(figsize=(10,8)) 

axes1 = plt.subplot(2, 2, 1, title='Plot1') 

axes2 = plt.subplot(2, 2, 2, title='Plot2') 

axes3 = plt.subplot(2, 2, 3, title='Plot3') 

axes4 = plt.subplot(2, 2, 4, title='Plot4') 

plt.show() 
#The above shown code creates a figure with four subplots, having two rows and two columns.

#The third argument, index value varied from 1 to 4, and respective subplots are drawn in row-major order.
fig = plt.figure(figsize=(10,8)) 

axes1 = plt.subplot(2, 2, (1,2), title='Plot1') 

axes1.set_xticks([]); 

axes1.set_yticks([]) 

axes2 = plt.subplot(2, 2, 3, title='Plot2') 

axes2.set_xticks([]); 

axes2.set_yticks([]) 

axes3 = plt.subplot(2, 2, 4, title='Plot3') 

axes3.set_xticks([]); 

axes3.set_yticks([]) 

#The above code also removes all ticks of x and y axes.

plt.show() 
import matplotlib.gridspec as gridspec 

import matplotlib.pyplot as plt 

fig = plt.figure(figsize=(10,8)) 

#Initially,a grid with given number of rows and columns is set up. 

#Later while creating a subplot, the number of rows and columns of grid, spanned by the subplot are provided as inputs to subplot function.

gd = gridspec.GridSpec(2,2) 

#A GridSpec object, gd is created with two rows and two columns. 

#axes1 = plt.subplot(gd[0,:],title='Plot1') 

#Then a selected grid portion is passed as an argument to subplot.

axes1.set_xticks([]); axes1.set_yticks([]) 

axes2 = plt.subplot(gd[1,0]) 

axes2.set_xticks([]); 

axes2.set_yticks([]) 

axes3 = plt.subplot(gd[1,-1]) 

axes3.set_xticks([]); 

axes3.set_yticks([]) 

plt.show()
import matplotlib.gridspec as gridspec 

import matplotlib.pyplot as plt 

fig = plt.figure(figsize=(12,10)) 

gd = gridspec.GridSpec(3,3) 

axes1 = plt.subplot(gd[0,:],title='Plot1') 

axes1.set_xticks([]); axes1.set_yticks([]) 

axes2 = plt.subplot(gd[1,:-1], title='Plot2') 

axes2.set_xticks([]); axes2.set_yticks([]) 

axes3 = plt.subplot(gd[1:, 2], title='Plot3') 

axes3.set_xticks([]); axes3.set_yticks([]) 

axes4 = plt.subplot(gd[2, :-1], title='Plot4') 

axes4.set_xticks([]); axes4.set_yticks([]) 

plt.show()