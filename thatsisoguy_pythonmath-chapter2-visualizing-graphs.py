emptyList= []
#Append elements into the empty list using .append()



emptyList.append(1)

emptyList.append(2)

emptyList.append(3)
emptyList
simpleTuple = (1,2,3)
simpleTuple[2]
for item in simpleTuple:

    print(item)



for item in emptyList:

    print(item)
for index, item in enumerate(simpleTuple):

    print(index, item)



#This also works for list.
X_num = [0,1,2,3,4,5,6,7,8]

y_num = [0,1,2,3,4,1,2,3,4]



from pylab import plot, show

plot(X_num, y_num,marker = 'o')

show()
nyc_temp = [53.9, 56.3, 56.4, 53.4, 54.5, 55.8, 56.8, 55.0, 55.3, 54.0, 56.7, 56.4, 57.3]

years = range(2000,2013)

plot(years,nyc_temp, marker='o')

show()
from pylab import legend, xlabel, ylabel, title, axis



nyc_temp_2000 = [31.3, 37.3, 47.2, 51.0, 63.5, 71.3, 72.3, 72.7, 66.0, 57.0, 45.3, 31.1]

nyc_temp_2006 = [40.9, 35.7, 43.1, 55.7, 63.1, 71.0, 77.9, 75.8, 66.6, 56.2, 51.9, 43.6]

nyc_temp_2012 = [37.3, 40.9, 50.9, 54.8, 65.1, 71.0, 78.8, 76.7, 68.8, 58.0, 43.9, 41.5]



months = range(1,13)

plot(months, nyc_temp_2000, months, nyc_temp_2006, months, nyc_temp_2012)

legend([2000,2006,2012])

title('Average Monthly temperature for three years')

xlabel('Month')

ylabel('Temprature in Fehrenhite')

axis(ymax = 90,ymin = 20, xmin = 1)

show()
from matplotlib import pyplot as plt



nyc_temp_2000 = [31.3, 37.3, 47.2, 51.0, 63.5, 71.3, 72.3, 72.7, 66.0, 57.0, 45.3, 31.1]

nyc_temp_2006 = [40.9, 35.7, 43.1, 55.7, 63.1, 71.0, 77.9, 75.8, 66.6, 56.2, 51.9, 43.6]

nyc_temp_2012 = [37.3, 40.9, 50.9, 54.8, 65.1, 71.0, 78.8, 76.7, 68.8, 58.0, 43.9, 41.5]



months = range(1,13)



plt.plot(months, nyc_temp_2000)

plt.plot(months, nyc_temp_2006)

plt.plot(months, nyc_temp_2012)

plt.legend([2000,2006,2012])

plt.title('Average Monthly temperature for three years')

plt.xlabel('Month')

plt.ylabel('Temprature in Fehrenhite')

plt.axis(ymax = 90,ymin = 20, xmin = 1)

plt.show()

from matplotlib import pyplot as plt



def graph_plot(X,y):

    plt.plot(X,y,marker = 'o')

    plt.title('Variation in Gravity with Distance')

    plt.xlabel('Distance in Meters')

    plt.ylabel('Gravitational force in newtons')

    plt.show()



def getData():

    r = range(100, 1001, 50)

    # Empty list to store the calculated values of F

    F = []

    G = 6.674*(10**-11)

    m1 = 0.5

    m2 = 1.5

    for dist in r:

        force = G*(m1*m2)/(dist**2)

        F.append(force)

    graph_plot(r,F)

    

getData()

    
from matplotlib import pyplot as plt



def frange(start, end, increment):

    numbers = []

    while start<end:

        numbers.append(start)

        start = start + increment

    

    return numbers

### We’ve defined a function frange() (“floating point” range)
import math



def draw_graph(x,y):

    plt.plot(x,y)

    plt.xlabel('x-coordinate')

    plt.ylabel('y-coordinate')

    plt.title('Projectile motion of a ball')

    

def draw_trajectory(u,theta):

    theta = math.radians(theta)

    

    g = 9.81

    

    t_flight  = (2 * u * math.sin(theta))/g

    

    intervals = frange(0,t_flight,0.001)

    x = []

    y = []

    for t in intervals:

        x.append(u*math.cos(theta)*t)

        y.append(u*math.sin(theta)*t - 0.5*g*t*t)

    draw_graph(x,y)

    

if __name__ == '__main__':

    try:

        u =[20,30,40]

        theta = 45

        

    except ValueError:

        print('You entered an invalid input')

        

    else:

        for i in range(0,3):

                draw_trajectory(u[i],theta)
def drawEq(x,y):

    plt.plot(x,y)

    plt.xlabel('X')

    plt.ylabel('y')

    plt.title('Quadratic equation plot')

    plt.xlim(0,20)

    

def quadFunc(x):

    length = len(x)

    y = []

    for i in range(0,length):

        y.append(x[i] ** 2 + x[i]**4 + 1)

    

    return y



if __name__ == '__main__':

    x_values = range(1,20)

    y = []

    y = quadFunc(x_values)

    drawEq(x_values,y)

    plt.show()



def fibo(x):

    if x == 1:

        return [1]

    if x == 2:

        return [1,1]

    

    a = 1

    b = 1

    series = [a, b]

    

    for i in range(x):

        c = a + b 

        series.append(c)

        a = b

        b = c

    

    return series



def drawFibo(x,y):

    x1 = []

    for i in range(0,x):

        x1.append(i)

    plt.plot(x1,y)

    plt.xlabel('No.')

    plt.ylabel('Ratio')

    plt.title('Ratio between consecutive Fibonacci numbers')





if __name__ == '__main__':

    

    x = 100

    series = fibo(x)

    y  = []

    for i in range(x):

        y.append(series[i+1]/series[i])

    

    drawFibo(x,y)

    