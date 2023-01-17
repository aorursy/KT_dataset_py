import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.datasets import make_moons



X,y=make_moons(n_samples=1000, noise=0.05)
fig, axs= plt.subplots(3,3)
from matplotlib.gridspec import GridSpec

fig = plt.figure()

fig.suptitle("Subplots without  width_ratios and height_ratios")



gs = GridSpec(2, 2)

ax1= gs[0,:]

ax2= gs[1,:]



fig.add_subplot(ax1)

fig.add_subplot(ax2)

#annotate_axes(fig)



plt.show()
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec







fig = plt.figure()

fig.suptitle("Controlling subplot sizes with width_ratios and height_ratios")



gs = GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[4, 1])



ax1= gs[0,:]

ax2= gs[1,:]

fig.add_subplot(ax1)

fig.add_subplot(ax2)

#annotate_axes(fig)



plt.show()
fig, ax = plt.subplots()

ax.plot(X[50:80,:])

axin1 = ax.inset_axes([0.8, 0.1, 0.15, 0.15])

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable



fig, ax = plt.subplots()

d= make_axes_locatable(ax)

ax= d.new_horizontal('1%')

ax= d.append_axes("top", size="7%", pad="2%") #we can also use append_axes

style_list= ['default'   , 'classic',

             'grayscale' , 'ggplot',

             'seaborn'   , 'fast',

             'bmh'       , 'Solarize_Light2',

             'seaborn-notebook']



for style_type in style_list:

      plt.figure(figsize=(5,2))      

      plt.style.use(style_type)

      plt.title(style_type)

      plt.plot(X[50:80,1]) 

      plt.show()
plt.style.use('default')

plt.plot(X[50:80,:])
plt.xlabel('X')

plt.ylabel('Y')

plt.scatter(X[70:100,0], y[70:100], c=X[70:100,1], cmap='viridis' ) #70:100 is to select only some data points

                                                                    
plt.bar(x= y, data=X[:,0], height= 0.5, width= 0.1)
from imageio import imread

image = imread('https://cdn.sstatic.net/Sites/stackoverflow/img/logo.png')

plt.imshow(image)
import numpy as np



feature_x = np.arange(0, 50, 2) 

feature_y = np.arange(0, 50, 3) 

  

# Creating 2-D grid of features 

[X, Y] = np.meshgrid(feature_x, feature_y) 

  

fig, ax = plt.subplots(1, 1) 

  

Z = np.cos(X / 2) + np.sin(Y / 4) 

  

# plots contour lines 

ax.contour(X, Y, Z) 

  

ax.set_title('Contour Plot') 

ax.set_xlabel('feature_x') 

ax.set_ylabel('feature_y') 

  

plt.show() 
feature_x = np.linspace(-5.0, 3.0, 70) 

feature_y = np.linspace(-5.0, 3.0, 70) 

  

# Creating 2-D grid of features 

[X, Y] = np.meshgrid(feature_x, feature_y) 

  

fig, ax = plt.subplots(1, 1) 

  

Z = X ** 2 + Y ** 2        #circle equations

  

# plots filled contour plot 

ax.contourf(X, Y, Z) 

  

ax.set_title('Filled Contour Plot') 

ax.set_xlabel('feature_x') 

ax.set_ylabel('feature_y') 

  

plt.show() 
X = np.arange(-10, 10, 1)

Y = np.arange(-10, 10, 1)

U, V = np.meshgrid(X, Y)



fig, ax = plt.subplots()

q = ax.quiver(X, Y, U, V)

ax.quiverkey(q, X=0.3, Y=1.1, U=10,

             label='Quiver key, length = 10', labelpos='E')



plt.show()
# Pie chart, where the slices will be ordered and plotted counter-clockwise:

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'

sizes = [15, 30, 45, 10]

explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
#using previous charts

feature_x = np.linspace(-5.0, 3.0, 70) 

feature_y = np.linspace(-5.0, 3.0, 70) 

  

# Creating 2-D grid of features 

[X, Y] = np.meshgrid(feature_x, feature_y) 

  

fig, ax = plt.subplots(1, 1) 

  

Z = X ** 2 + Y ** 2        #circle equations



ax.text(-2, 2, 'This is the text i was talking about', style='italic')

  

# plots filled contour plot 

ax.contourf(X, Y, Z) 

  

ax.set_title('Filled Contour Plot') 

ax.set_xlabel('feature_x') 

ax.set_ylabel('feature_y') 

  

plt.show() 

a = np.linspace(0,2*3.14,50) 

b = np.sin(a) 

  

plt.fill_between(a, b, 0, 

                 where = (a > 2) & (a <= 3), 

                 color = 'green') 

plt.plot(a,b) 