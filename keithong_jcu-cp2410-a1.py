import matplotlib.pyplot as plt

import time

import math

import csv

import collections
city_list =[]



with open("../input/10p_city_data2.csv", "r") as file:

    cities = csv.reader(file)  

    for row in cities:

        city_list.append([int(row[0]), float(row[1]), float(row[2])])

 

# print(city_list)
plt.subplots(figsize = (15,10))



for i in range(len(city_list)):

    x = city_list[i][1]

    y = city_list[i][2]

    

    if i % 1000 == 0:

        print (i, "Cities Processed")

    

    plt.scatter(x,y, s = 0.5)

    

plt.show()
start_time = time.time()

total_dist = 0

for i in range(len(city_list)-1):

    dist = math.sqrt((((city_list[i][1])- (city_list[i+1][1]))**2) + (((city_list[i][2])-(city_list[i+1][2]))**2))

    

    if i % 10 == 0:

        if ((city_list[i][0]) % 2 == 0) or ((city_list[i][0]) % 3 == 0):

            total_dist += dist

        

        else:

            total_dist += (dist * 1.1)

            

    else:

        total_dist += dist

        

end_time = time.time()



#print(Q[0])

#print(Q[19777])



print("Total time taken:", end_time - start_time)

print("Total distance: ", total_dist)
class Stack:

    

    def __init__(self):

        self._data=[] #empty list

    

    #PUSH operation

    def push(self,item):

        self._data.append(item)



    #POP operation

    def pop(self):



        if not self.is_empty():

            item = self._data.pop()

            return item

        else:

            print("Error: Stack empty")    

            

    #Is Stack empty?

    def is_empty(self):

        if len(self._data) == 0:

            return True

        else:

            return False

        

    #display block

    def display(self):

        print("S contains:", self._data)
D1 = collections.deque(city_list)

S = Stack()

plot_s = Stack()



north_pole = D1[0]

D1.popleft()

S.push(north_pole)

plot_s.push(north_pole)



start_time = time.time()



for k in range(len(D1)):

    lowest_number = D1[0]

#     print(lowest_number[1])

    for j in range(len(D1)):

        if (lowest_number[1] > D1[j][1]):

            lowest_number = D1[j]

            lowest_location = j+1



#     print(lowest_number)

#     print(lowest_location)



    for i in range(lowest_location):

        item = D1.popleft()

        D1.append(item)



    item = D1.pop()

    S.push(item)

    plot_s.push(item)



    if k % 1000 == 0:

        print(k, "Cities processed")

        

S.push(north_pole)

plot_s.push(north_pole)



end_time = time.time()

print("Total time taken:", end_time - start_time)

print("Done! All cities processed")

# S.display()

    
total_dist = 0

start_time = time.time()



f_dist = S.pop()

# print(f_dist)

# print(f_dist[1])

for i in range(len(S._data)-1):

    s_dist = S.pop()

#     print(f_dist)

#     print(s_dist)

    

    dist = math.sqrt((((f_dist[1])- (s_dist[1]))**2) + (((f_dist[2])-(s_dist[2]))**2))



    f_dist = s_dist

    

    total_dist += dist

# print(len((S._data)))

# print(dist)

#print(Q[19777])



end_time = time.time() 

print("Total distance: ", total_dist)
fig, ax = plt.subplots(figsize = (15,10))





previous_point = plot_s.pop()



for i in range(len(plot_s._data)):

    point = plot_s.pop()

    

    xs = point[1]

    ys = point[2]

    ax.plot([previous_point[1], xs], [previous_point[2], ys], c="blue", linewidth = 0.2)

    

    previous_point = point

    

plt.show()
D = collections.deque(city_list)

Q = collections.deque()

def euclidean_dist():

    sum = 0

    start_time = time.time()

    north_pole = D[0]

    

    for k in range(len(D)):

        short_dist = 100000

        for j in range(len(D)-1):



            dist = math.sqrt((math.pow(((D[0][1])- (D[(j+1)][1])),2)) + (math.pow(((D[0][2])-(D[(j+1)][2])),2)))

            





            if (dist < short_dist):

                short_dist = dist

                location = j+1

        

        sum += short_dist

            

#         print("short", short_dist)

#         print("location ", location)



        if len(D) != 0:

            item = D.popleft()

        for l in range(location):

            if len(D) != 0:

                item = D.popleft()

                D.append(item)



        if len(D) != 0:

            item = D.pop()

            D.appendleft(item)

            Q.append(item)



    

        if (len(Q) % 1000 == 0):

            print(len(Q), " Cities processed")

    

    Q.append(north_pole)

    Q.appendleft(north_pole)

    end_time = time.time()

    print("Total distance: ", sum -100000)

    print("All Cities processed")

    print("Total time taken:", end_time - start_time)

#     print(Q)

    

euclidean_dist()
total_dist = 0

start_time = time.time()

for i in range(len(Q)-1):

    ecu_dist = math.sqrt((math.pow(((Q[i][1])- (Q[i+1][1])),2)) + ( math.pow(((Q[i][2])-(Q[i+1][2])),2)))

    

#     print(ecu_dist)

#     print(Q[i])

#     print(i)

#     print()

    

    if i % 10 == 0:

        if ((Q[i][0]) % 2 == 0) or ((Q[i][0]) % 3 == 0):

            total_dist += ecu_dist

        

        else:

            total_dist += (ecu_dist * 1.1)

            

    else:

        total_dist += ecu_dist





#print(Q[0])



end_time = time.time()

print("Total distance: ", total_dist)
fig, ax = plt.subplots(figsize = (15,10))





previous_point = Q.popleft()



for i in range(len(Q)):

    point = Q.popleft()

    

    xs = point[1]

    ys = point[2]

    ax.plot([previous_point[1], xs], [previous_point[2], ys], c="blue", linewidth = 0.5)

    

    previous_point = point

    

plt.show()