from matplotlib import pyplot as plt
# plotting

plt.plot([1,2,3],[4,5,1])



# Showing the plot

plt.show()
# Adding Title and Labels to our graph.

x = [5,8,10]

y = [12,16,6]



# Plotting

plt.plot(x,y)



plt.title('Just plotting a Graph with Title & Labels')

plt.ylabel('Y-Axis')

plt.xlabel('X-Axis')



# Showing the graph

plt.show()
# styling the graph

from matplotlib import style

style.use('ggplot')
# Plotting multiple lines in a single graph.

x = [5,8,10]

y = [12,16,6]



x2 = [6,9,11]

y2 = [6,15,7]



plt.plot(x,y,'g',label = 'Line One', linewidth = 5)

plt.plot(x2,y2,'c',label = 'Line Two', linewidth = 5)



plt.title('Plotting multiple lines in a Single Graph')

plt.xlabel('X-Axis')

plt.ylabel('Y-Axis')



plt.legend()



plt.grid(True, color = 'k')



plt.show()
plt.bar([1,3,5,7,9], [5,2,7,8,2], color = 'g', label = 'Group 1')

plt.bar([2,4,6,8,10], [8,6,2,5,6], color = 'b', label = 'Group 2')



plt.legend(loc = "best")

'''

Other options are

	best

	upper right

	upper left

	lower left

	lower right

	right

	center left

	center right

	lower center

	upper center

	center

'''



plt.xlabel('X-Axis')

plt.ylabel('Y-Axis')



plt.title("My Bar Graph")



plt.show()
population_age = [1,4,7,8,9,44,12,25,48,90,63,41,44,34,46,26,37]



plt.hist(population_age, bins = 10, histtype = 'bar', rwidth = 0.5)

# Check the graph by changing the bins to 5;20 etc

#plt.hist(population_age, bins = 5, histtype = 'bar', rwidth = 0.5)

#plt.hist(population_age, bins = 20, histtype = 'bar', rwidth = 0.5)



plt.xlabel('X-Axis-Population Age')

plt.ylabel('Y-Axis')



plt.title("My Histogram Graph")



plt.show()
x = [1,2,3,4,5,6,7,8]

y = [5,2,4,2,1,4,5,2]



plt.scatter(x,y,label = 'Numbers', color = 'r')



plt.xlabel('X-Axis')

plt.ylabel('Y-Axis')



plt.title("My Scatter Plot Graph")

plt.legend(loc = 'best')

plt.show() 
# First Plot

plt.subplot(211)



# Plotting multiple lines in a single graph.



x = [5,8,10]

y = [12,16,6]



x2 = [6,9,11]

y2 = [6,15,7]



plt.plot(x,y,'g',label = 'Line One', linewidth = 5)

plt.plot(x2,y2,'c',label = 'Line Two', linewidth = 5)



plt.title('Plotting multiple lines in a Single Graph')

plt.xlabel('X-Axis')

plt.ylabel('Y-Axis')



plt.legend()



plt.grid(True, color = 'k')



# Second Plot

plt.subplot(212)



#Scatter Plot

x3 = [1,2,3,4,5,6,7,8]

y3 = [5,2,4,2,1,4,5,2]



plt.scatter(x3,y3,label = 'Numbers', color = 'r')



plt.xlabel('X-Axis')

plt.ylabel('Y-Axis')



plt.title("My Scatter Plot Graph")

plt.legend(loc = 'best')



plt.show()
# First Plot

plt.subplot(221)



# Plotting multiple lines in a single graph.



x = [5,8,10]

y = [12,16,6]



x2 = [6,9,11]

y2 = [6,15,7]



plt.plot(x,y,'g',label = 'Line One', linewidth = 5)

plt.plot(x2,y2,'c',label = 'Line Two', linewidth = 5)



plt.title('Plotting multiple lines in a Single Graph')

plt.xlabel('X-Axis')

plt.ylabel('Y-Axis')



plt.legend()



plt.grid(True, color = 'k')



# Second Plot

plt.subplot(222)



#Scatter Plot

x3 = [1,2,3,4,5,6,7,8]

y3 = [5,2,4,2,1,4,5,2]



plt.scatter(x3,y3,label = 'Numbers', color = 'r')



plt.xlabel('X-Axis')

plt.ylabel('Y-Axis')



plt.title("My Scatter Plot Graph")

plt.legend(loc = 'best')



plt.show()
data = {'apples': 10, 'oranges': 15, 'lemons': 5, 'limes': 20}

names = list(data.keys())

values = list(data.values())



fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

axs[0].bar(names, values)

axs[1].scatter(names, values)

axs[2].plot(names, values)

fig.suptitle('Categorical Plotting')
plt.figure(figsize=(20,5))



plt.subplot(131)

plt.bar(names, values, label = 'Bar Graph')

plt.legend(loc = "best")

plt.title('Bar Plot')



plt.subplot(132)

plt.scatter(names, values, label = 'Scatter Graph')

plt.legend(loc = "best")

plt.title('Scatter Plot')



plt.subplot(133)

plt.plot(names, values, label = 'Line / Plot Graph')

plt.legend(loc = "best")

plt.title('Line Plot')



plt.suptitle('Categorical Plotting')

plt.show()