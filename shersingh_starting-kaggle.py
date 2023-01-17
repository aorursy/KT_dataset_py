import random as rand
randList = []
for i in range(0,1000):
    randList.append(rand.randint(0,10))
    
    def gaussianPDF(x):
        y = 1/((((2*3.14)**0.5)*2)*(2.71**((-(x-0.5)/2)**2)))
        return y
    for i in range(0,len(randList)):
        print(gaussianPDF(randList[i]))
        
import matplotlib.pyplot as plt
import random as rand
def lineEqn(m,x,c):
    return(m*x + c)
randomX = []
randomY = []
for i in range(0,1000):
    randomX.append(rand.randint(0,1000))
for i in range(0,len(randomX)):
    randomY.append(lineEqn(0.1,randomX[i],2))
plt.plot(randomX,randomY)
plt.show
print(5/2)

hat_height_cm = 25
person_height_cm = 165
total_height_meters = (hat_height_cm + person_height_cm)/100
print("height in meters=", total_height_meters,"?")

print(abs(32))
print(abs(-32))
def least_difference(a, b, c):
    d1 = abs(a-b)
    d2 = abs(b-c)
    d3 = abs(c-a)
    return min(d1, d2, d3)
print(least_difference(1, 10, 100),
     least_difference(5, 7, 10))

    

def greet(who="collin"):
 print("hello,",who)
greet()
greet(who="world")
greet(who="kaggle")
import matplotlib.pyplot as plt
import random as rand
def parabolaEqn(x):
    return((4*5*x)**0.5)
randomX = []
randomY = []
for i in range(0,1000):
    randomX.append(rand.randint(0,1000))
for i in range(0,len(randomX)):
    randomY.append(parabolaEqn(randomX[i],))
plt.plot(randomX,randomY)
plt.show
import matplotlib.pyplot as plt
import random as rand
def hyperbolaEqn(x):
    return((3/2)*((2**2+x**2)**0.5))
randomX = []
randomY = []
for i in range(-500,500):
    randomX.append(rand.randint(-500,500))
for i in range(0,len(randomX)):
    randomY.append(hyperbolaEqn(randomX[i],))
plt.plot(randomX,randomY)
plt.show
import matplotlib.pyplot as plt
import random as rand
def ellipseEqn(x):
    return((3/2)*((2**2-x**2)**0.5))
randomX = []
randomY = []
for i in range(-100,100):
    randomX.append(rand.randint(-100,100))
for i in range(0,len(randomX)):
    randomY.append(ellipseEqn(randomX[i],))
plt.plot(randomX,randomY)
plt.show

