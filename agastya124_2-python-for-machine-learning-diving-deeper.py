# import pandas as pd 

import pandas as pd 

  

# Creating empty series 

ser = pd.Series() 

  

print(ser) 





# import pandas and numpy  

import pandas as pd  

import numpy as np  

    

# series with numpy linspace()   

series = pd.Series(np.linspace(1950, 2049, 100))

print(series)



    
years = [ i for i in range(1900,2100,4)]

print(years)
# generate random integer values

from random import seed

from random import randint

# seed random number generator

seed(1)

# generate some integers

for _ in range(10):

	value = randint(0, 10)

	print(value)
from random import seed

from random import randint

seed(2)

crime = [ randint(134587,9857630) for x in range(0,len(years),1)]



print(crime)
print(len(years))

print(len(crime))
len(years)==len(crime)



# Print the last item from year and pop

print(years[-1])

print(crime[-1])



%matplotlib inline

# Import matplotlib.pyplot as plt

import matplotlib.pyplot as plt



# Make a line plot: year on the x-axis, pop on the y-axis

plt.plot(years,pop)

plt.xlabel("Years")

plt.ylabel("Crime")

plt.title("Crime over Years")



# Display the plot with plt.show()

plt.show()
%matplotlib inline



# evenly sampled time at 200ms intervals

t = np.arange(0., 5., 0.2)



# red dashes, blue squares and green triangles

plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

plt.show()
data = {'a': np.arange(50),

        'c': np.random.randint(0, 50, 50),

        'd': np.random.randn(50)}

data['b'] = data['a'] + 10 * np.random.randn(50)

data['d'] = np.abs(data['d']) * 100



plt.scatter('a', 'b', c='c', s='d', data=data)

plt.xlabel('Entry a')

plt.ylabel('Entry b')

plt.show()
names = ['Group A', 'Group B', 'Group C']

values = [1, 10, 100]



plt.figure(figsize=(18, 3))



plt.subplot(131)

plt.bar(names, values)

plt.subplot(132)

plt.scatter(names, values)

plt.subplot(133)

plt.plot(names, values)

plt.suptitle('Categorical Plotting')

plt.show()
import matplotlib.pyplot as plt

from numpy.random import rand

from numpy.random import randint

from numpy.random import seed



seed(3)





gdp_cap = [randint(223,4510) for i in range(0,100,1)]





life_exp = [randint(60,80) for i in range(0,100,1)]

len(gdp_cap)
len(life_exp)
# Print the last item of gdp_cap and life_exp



print(gdp_cap[-1])



print(life_exp[-1])





# Make a line plot, gdp_cap on the x-axis, life_exp on the y-axis

plt.plot(gdp_cap,life_exp)



# Display the plot

plt.show()