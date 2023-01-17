# Math module in python 
import math 
math.pi
math.sqrt(2)
math.atan(45)
math.sin(30)
math.ceil(4.6) # ceil will give us the next integer after the decimal
math.floor(4.6) # floor will remove decimal part 
math.copysign(12,-13) # math.copysign is written as  copysign(x,y)# copysign will copy the sign of y in x and return x 
math.degrees(30)
import math
math.radians(30)
math.sin(math.radians(30))
math.e
math.factorial(5)
math.log(2)
math.log(2,10)
math.log(2,2)
math.hypot(3,4)
math.pow(3,5)
math.pow(9,2)
math.sqrt(math.pow(3,2)+ math.pow(4,2))
import math as m
m.pi
for x in range(1,100):
    print(m.log(x,10))
import math as m
ml = list()
for x in range(1,100):
    ml.append(m.log(x))
print(ml)
len(ml)
ml[1]
from matplotlib import pyplot as plt
plt.plot(nl,ml)
nl = list(range(1,100))
print(nl)
plt.plot(nl,ml)
ml[1]
from matplotlib import pyplot as plt
nl = list(range(1,100))
print(nl)
a = list(range(1,11))
print(a)
d =list()
for num in a:
    d.append(num*5)
print(d)
plt.plot(a,d)