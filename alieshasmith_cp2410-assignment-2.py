import math

import os

import time



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt





data = pd.read_csv("../input/cities.csv")



data_use = 1



# Remove unwanted rows of data

data_cutoff = int(data.count(0)['X']*data_use)

data = data.drop(data.index[data_cutoff:])

origin = data[data.CityId == 0]



data['Distance'] = np.sqrt(pow((data['X'] - float(origin.X)), 2) + pow((data['Y'] - float(origin.Y)), 2))





# print(data.head())



# Put data into dictionary

data_dict = {}

index_list = []





for index, row in data.iterrows():

        data_dict[row['CityId'].astype(int)] = (

            row['X'].astype(float), row['Y'].astype(float), row['Distance'].astype(float))

        index_list.append(row['CityId'].astype(int))





def calcTotalDist(arr):

        total_distance = 0



        for i in range(0, len(arr) - 1):

                first_point = data_dict[arr[i]]

                second_point = data_dict[arr[i + 1]]



                total_distance += math.sqrt(

                        pow((second_point[0] - first_point[0]), 2) + pow((second_point[1] - first_point[1]), 2))

        return total_distance





def mergeSort(arr):

    t1 = time.time()

    time_array.append(t1 - start_time)



    if len(arr) > 1:

        mid = len(arr) // 2

        left_split = arr[:mid]

        right_split = arr[mid:]



        mergeSort(left_split)

        mergeSort(right_split)



        i = j = k = 0



        # Copy data to temp arrays left_split[] and right_split[]

        while i < len(left_split) and j < len(right_split):



            if data_dict[left_split[i]][2] < data_dict[right_split[j]][2]:

                arr[k] = left_split[i]

                i += 1

            else:

                arr[k] = right_split[j]

                j += 1

            k += 1



        # Checking if any element was left

        while i < len(left_split):

            arr[k] = left_split[i]

            i += 1

            k += 1



        while j < len(right_split):

            arr[k] = right_split[j]

            j += 1

            k += 1

            



# Run Function

time_array = []



start_dist = calcTotalDist(index_list)

print("Start Distance: ", start_dist)



start_time = time.time()

mergeSort(index_list)

finish_time = time.time()



total_time = finish_time-start_time

time_array.append(total_time)



sorted_dist = calcTotalDist(index_list)

print("Sorted Distance: ", sorted_dist)



print("Improvement: ", 100-(sorted_dist/start_dist)*100, "%")

print("Total Time: ", finish_time-start_time, "s")



# Time vs Recursion Count Graph

bars = range(1, len(time_array) + 1)

y_pos = np.arange(len(bars))

plt.plot(y_pos, time_array)



plt.title('Merge Sort Time Graph')

plt.xlabel('No. of Iterations')

plt.ylabel('Time (Sec)')



plt.show()

import math

import os

import time



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt





data = pd.read_csv("../input/cities.csv")

# scriptPath = os.path.dirname(os.path.realpath("_file_"))

# print(scriptPath)

# file_name = scriptPath + "\cities.csv"

# data = pd.read_csv(file_name, index_col=False)



data_use = 1



# Remove unwanted rows of data

data_cutoff = int(data.count(0)['X']*data_use)

data = data.drop(data.index[data_cutoff:])

origin = data[data.CityId == 0]



data['Distance'] = np.sqrt(pow((data['X'] - float(origin.X)), 2) + pow((data['Y'] - float(origin.Y)), 2))





# print(data.head())



# Put data into dictionary

data_dict = {}

index_list = []





for index, row in data.iterrows():

        data_dict[row['CityId'].astype(int)] = (

            row['X'].astype(float), row['Y'].astype(float), row['Distance'].astype(float))

        index_list.append(row['CityId'].astype(int))





def calcTotalDist(arr):

        total_distance = 0



        for i in range(0, len(arr) - 1):

                first_point = data_dict[arr[i]]

                second_point = data_dict[arr[i + 1]]



                total_distance += math.sqrt(

                        pow((second_point[0] - first_point[0]), 2) + pow((second_point[1] - first_point[1]), 2))

        return total_distance





def insertionSort(arr):



    # Iterate through the array

    for i in range(1, len(arr)):



        t1 = time.time()

        time_array.append(t1 - start_time)



        key = arr[i]



        j = i - 1

        while j >= 0 and data_dict[key][2] < data_dict[arr[j]][2]:

            arr[j + 1] = arr[j]

            j -= 1

        arr[j + 1] = key





# Run Function

time_array = []



start_dist = calcTotalDist(index_list)

print("Start Distance: ", start_dist)



start_time = time.time()

insertionSort(index_list)

finish_time = time.time()



sorted_dist = calcTotalDist(index_list)

print("Sorted Distance: ", sorted_dist)



print("Improvement: ", 100-(sorted_dist/start_dist)*100, "%")



print("Total time: ", finish_time - start_time, "s")

# Time vs Recursion Count Graph

bars = range(1, len(time_array) + 1)

y_pos = np.arange(len(bars))

plt.plot(y_pos, time_array)



plt.title('Merge Sort Time Graph')

plt.xlabel('No. of Iterations')

plt.ylabel('Time (Sec)')



plt.show()