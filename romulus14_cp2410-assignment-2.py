import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df_cities = pd.read_csv('../input/cities.csv')

df_cities.head()

cities = dict(pd.read_csv('../input/cities.csv', index_col=['CityId']))

print(type(cities))





def total_distance(dfcity,path):

    prev_city = path[0]

    dict_X = dict(dfcity.X)

    dict_Y = dict(dfcity.Y)

    total_distance = 0

    step_num = 1

    for city_num in path[1:]:

        next_city = city_num

        total_distance = total_distance + np.sqrt(pow((dict_X.get(next_city)-dict_X.get(prev_city)),2) + pow((dict_Y.get(next_city)-dict_Y.get(prev_city)),2)*\

                                 (1+ 0.1*((step_num % 10 == 0)*int(not(prime_cities[prev_city])))))

        prev_city = next_city

        step_num = step_num + 1

    return total_distance



#https://matplotlib.org/users/pyplot_tutorial.html

import numpy as np

import matplotlib.pyplot as plt



#http://courses.csail.mit.edu/6.867/wiki/images/3/3f/Plot-python.pdf

n = np.arange(-5, 1000, 1)

On = (4*(n**2))

plt.plot(n, On) # Create line plot with yvals against xvals

plt.title('O(n) = 4n^2')

plt.ylabel('O(n)')

plt.xlabel('values of n')

plt.show()
import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import time

import matplotlib.pyplot as plt

df_cities = pd.read_csv('../input/cities.csv')

df_cities.head()





def total_distance(dfcity,path):

    prev_city = path[0]

    dict_X = dict(dfcity.X)

    dict_Y = dict(dfcity.Y)

    total_distance = 0

    step_num = 1

    for city_num in path[1:]:

        next_city = city_num

        total_distance = total_distance+ np.sqrt(pow((dict_X.get(next_city)-dict_X.get(prev_city)),2) + pow((dict_Y.get(next_city)-dict_Y.get(prev_city)),2)*\

                                 (1+ 0.1*((step_num % 10 == 0)*int(not(prime_cities[prev_city])))))

        prev_city = next_city

        step_num = step_num + 1

    return total_distance



def sieve_of_eratosthenes(n):

    primes = [True for i in range(n+1)] # Start assuming all numbers are primes

    primes[0] = False # 0 is not a prime

    primes[1] = False # 1 is not a prime

    for i in range(2,int(np.sqrt(n)) + 1):

        if primes[i]:

            k = 2

            while i*k <= n:

                primes[i*k] = False

                k += 1

    return(primes)



prime_cities = sieve_of_eratosthenes(max(df_cities.CityId))







increment = [ i for i in range(0, 100,10)]

times = []

for i in increment:

    start = time.time()

    percent_rows = int(len(df_cities)*(i/100))

    percent_data = df_cities.head(percent_rows)

    percent_path = list(percent_data.CityId[:].append(pd.Series([0])))

    total_distance(percent_data,percent_path)

    end = time.time()

    times.append(end-start)

    

    

n = increment

T = times

plt.plot(n, T) # Create line plot with yvals against xvals

plt.title('Running time of total_distance(percent_data, percent_path)')

plt.ylabel('T (secs)')

plt.xlabel('values of n')

plt.show()
#https://matplotlib.org/users/pyplot_tutorial.html

import numpy as np

import matplotlib.pyplot as plt



#http://courses.csail.mit.edu/6.867/wiki/images/3/3f/Plot-python.pdf

n = np.arange(-5, 1000, 1)

On = ((21*n) + 10)

plt.plot(n, On) # Create line plot with yvals against xvals

plt.title('O(n) = 21n + 10')

plt.ylabel('O(n)')

plt.xlabel('values of n')

plt.show()
import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import time

import matplotlib.pyplot as plt

df_cities = pd.read_csv('../input/cities.csv')

df_cities.head()



cities = df_cities.CityId



cities_path = list(df_cities.CityId[:])



def sieve_of_eratosthenes(n):

    primes = [True for i in range(n+1)] # Start assuming all numbers are primes

    primes[0] = False # 0 is not a prime

    primes[1] = False # 1 is not a prime

    for i in range(2,int(np.sqrt(n)) + 1):

        if primes[i]:

            k = 2

            while i*k <= n:

                primes[i*k] = False

                k += 1

    return(primes)



prime_cities = sieve_of_eratosthenes(max(df_cities.CityId))





def graph_structure(path):

    graph = {}

    index = 0

    for city in path:

        path = list(path)

        next_index = index + 1

        prev_index = index-1

        prev_city = path[prev_index]

        if index < len(path)-1:

            next_city = path[next_index]

        else:

            next_city = path[0]

        graph[city]= (prev_city, next_city)

        index = index+1

    return graph







graph = graph_structure(cities_path)





def distance_of_graph_path(graph):

    cities = pd.read_csv("../input/cities.csv")

    dict_X= cities.X

    dict_Y = cities.Y

    distance = 0

    step_num = 1

    for city in graph:

        next_city = graph.get(city)[1]

        currentX,currentY = dict_X.get(city), dict_Y.get(city)

        nextX,nextY = dict_X.get(next_city),dict_Y.get(next_city)

        distance = distance + np.sqrt(pow((nextX - currentX),2) + pow((nextY - currentY),2)*\

                                      (1+ 0.1*((step_num % 10 == 0)*int(not(prime_cities[city])))))

        step_num = step_num + 1  

    return distance



print(distance_of_graph_path(graph))

#https://matplotlib.org/users/pyplot_tutorial.html

import numpy as np

import matplotlib.pyplot as plt



#http://courses.csail.mit.edu/6.867/wiki/images/3/3f/Plot-python.pdf

n = np.arange(-5, 1000, 1)

On = (25*n + 10)

plt.plot(n, On) # Create line plot with yvals against xvals

plt.title('O(n) = 25n+10')

plt.ylabel('O(n)')

plt.xlabel('values of n')

plt.show()
import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import time

import matplotlib.pyplot as plt

df_cities = pd.read_csv('../input/cities.csv')

df_cities.head()



cities = df_cities.CityId



cities_path = list(df_cities.CityId[:])



def sieve_of_eratosthenes(n):

    primes = [True for i in range(n+1)] # Start assuming all numbers are primes

    primes[0] = False # 0 is not a prime

    primes[1] = False # 1 is not a prime

    for i in range(2,int(np.sqrt(n)) + 1):

        if primes[i]:

            k = 2

            while i*k <= n:

                primes[i*k] = False

                k += 1

    return(primes)



prime_cities = sieve_of_eratosthenes(max(df_cities.CityId))





def graph_structure(path):

    graph = {}

    index = 0

    for city in path:

        path = list(path)

        next_index = index + 1

        prev_index = index-1

        prev_city = path[prev_index]

        if index < len(path)-1:

            next_city = path[next_index]

        else:

            next_city = path[0]

        graph[city]= (prev_city, next_city)

        index = index+1

    return graph





def distance_of_graph_path(graph):

    cities = pd.read_csv("../input/cities.csv")

    dict_X= cities.X

    dict_Y = cities.Y

    distance = 0

    step_num = 1

    for city in graph:

        next_city = graph.get(city)[1]

        currentX,currentY = dict_X.get(city), dict_Y.get(city)

        nextX,nextY = dict_X.get(next_city),dict_Y.get(next_city)

        distance = distance + np.sqrt(pow((nextX - currentX),2) + pow((nextY - currentY),2)*\

                                      (1+ 0.1*((step_num % 10 == 0)*int(not(prime_cities[city])))))

        step_num = step_num + 1  

    return distance





increment = [ i for i in range(0, 100,10)]

times = []

for i in increment:

    start = time.time()

    percent_rows = int(len(df_cities)*(i/100))

    percent_data = df_cities.head(percent_rows)

    percent_path = list(percent_data.CityId[:].append(pd.Series([0])))

    graph=graph_structure(percent_path)

    distance_of_graph_path(graph)

    end = time.time()

    times.append(end-start)

    

    

n = increment

T = times

plt.plot(n, T) # Create line plot with yvals against xvals

plt.title('Running time of distance_of_graph_path(graph)')

plt.ylabel('T (secs)')

plt.xlabel('values of n')

plt.show()
import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import time

import matplotlib.pyplot as plt

df_cities = pd.read_csv('../input/cities.csv')

df_cities.head()



cities = df_cities.CityId



cities_path = list(df_cities.CityId[:])



def sieve_of_eratosthenes(n):

    primes = [True for i in range(n+1)] # Start assuming all numbers are primes

    primes[0] = False # 0 is not a prime

    primes[1] = False # 1 is not a prime

    for i in range(2,int(np.sqrt(n)) + 1):

        if primes[i]:

            k = 2

            while i*k <= n:

                primes[i*k] = False

                k += 1

    return(primes)



prime_cities = sieve_of_eratosthenes(max(df_cities.CityId))





def graph_structure(path):

    graph = {}

    index = 0

    for city in path:

        path = list(path)

        next_index = index + 1

        prev_index = index-1

        prev_city = path[prev_index]

        if index < len(path)-1:

            next_city = path[next_index]

        else:

            next_city = path[0]

        graph[city]= (prev_city, next_city)

        index = index+1

    return graph





def distance_of_graph_path(graph):

    cities = pd.read_csv("../input/cities.csv")

    dict_X= cities.X

    dict_Y = cities.Y

    distance = 0

    step_num = 1

    for city in graph:

        next_city = graph.get(city)[1]

        currentX,currentY = dict_X.get(city), dict_Y.get(city)

        nextX,nextY = dict_X.get(next_city),dict_Y.get(next_city)

        distance = distance + np.sqrt(pow((nextX - currentX),2) + pow((nextY - currentY),2)*\

                                      (1+ 0.1*((step_num % 10 == 0)*int(not(prime_cities[city])))))

        step_num = step_num + 1  

    return distance





increment = [ i for i in range(0, 100,10)]

graphs = []

times = []



for i in increment:

    percent_rows = int(len(df_cities)*(i/100))

    percent_data = df_cities.head(percent_rows)

    percent_path = list(percent_data.CityId[:].append(pd.Series([0])))

    graph=graph_structure(percent_path)

    graphs.append(graph)

    

for graph in graphs:

    start = time.time()

    distance_of_graph_path(graph)

    end = time.time()

    times.append(end-start)

    

    

n = increment

T = times

plt.plot(n, T) # Create line plot with yvals against xvals

plt.title('Running time of distance_of_graph_path(graph)')

plt.ylabel('T (secs)')

plt.xlabel('values of n')

plt.show()