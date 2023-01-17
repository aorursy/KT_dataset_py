import csv

import math

import matplotlib.pyplot as plt



class LinkedList:

    class Node:

        def __init__(self, val, nxt):

            self._data = val

            self._next = nxt





    def __init__(self):

        #create empty head and tail here

        #cuz it's empty, so value and next are both empty

        self._head = self.Node(None, None)

        self._tail = self.Node(None, None)

        self._size = 0





    def insert_head(self,val):

        new_node = self.Node(val, None)

        if self._size == 0: #i.e. first node

            self._head = new_node

            self._tail = new_node

        else:

            new_node._next = self._head #new node now is pointing to old head

            self._head = new_node   #change the head to the beginning of the list



        self._size += 1





    # insert node at end of list

    def insert_tail(self, val):

        new_node = self.Node(val, None)

        if self._size == 0: #i.e. first node

            self._head = new_node

            self._tail = new_node

        else:

            self._tail._next = new_node

            self._tail = new_node



        self._size += 1





    def delete_head(self):

        if self._size == 0:

            print("Deletion error: List empty")

            return

        #at least one node in the list

        cur = self._head    #take backup of head

        self._head = self._head._next #Change head to point to its next neighbour

        cur._next = None #break the connection of the old head from the rest of the list

        self._size -= 1







    def delete_tail(self):

        if self._size == 0:

            print("Deletion error: List empty")





    def display(self):

        #Traverse the list here

        print("\nLinked list num of nodes", self._size)

        if self._size == 0:

            print("\nList empty")

            return

        cur = self._head

        while cur!= None:

            print("Node data:", cur._data)

            cur = cur._next





    def delete_node(self, key):

        prev = None

        cur = self._head

        while True:

            if cur._data[0] == key:

                

                if cur._next != None:

                    if prev == None:

                        # first

                        self.delete_head()

                    else:

                        # middle

                        prev._next = cur._next

                        cur = None

                        self._size -= 1

                else:

                    # last

                    prev._next = None

                    self._size -= 1

                break

            prev = cur

            cur = cur._next

    





if __name__ == "__main__":



    def store_data():

        skip = True

        with open("../input/cities10/cities10.csv", 'r') as cities_file:

            cities_data = csv.reader(cities_file)



            cities_list = LinkedList()

            c_split_list = []

            for i in cities_data:

                if skip == True:

                    skip = False

                else:

                    cities_list.insert_tail(i)

            



        cur = cities_list._head       #cur should be fst_city._next

        distance_total = 0

        while cur._next!= None:

           distance = math.sqrt( math.pow( float(cur._next._data[1]) - float(cur._data[1]) , 2 ) + math.pow( float(cur._next._data[2]) - float(cur._data[2]) ,2))

           distance_total = distance + distance_total

            

           cur = cur._next

           cities_list.delete_head()

        print(distance_total)







store_data()



    
import csv

import math

import matplotlib.pyplot as plt
import collections



class Queue:

    def __init__(self, *args, **kwargs):

        self._data = [] #empty list





    #PUSH operation

    def push(self, item):

        self._data.append(item)





    #POP operation

    def pop(self):

        if not self.is_empty():

            item = self._data.pop(0)

            return item

        else:

            print("Error: Queue empty")





    def is_empty(self):

        if len(self._data) == 0:

            return True

        else:

            return False



    def display(self):

        print("Order of cities: ", self._data)





    def top(self):

        if not self.is_empty():

            return self._data[-1]

        else:

            print("Error: Queue empty")

class LinkedList:

    class Node:

        def __init__(self, val, nxt):

            self._data = val

            self._next = nxt





    def __init__(self):

        #create empty head and tail here

        #cuz it's empty, so value and next are both empty

        self._head = self.Node(None, None)

        self._tail = self.Node(None, None)

        self._size = 0





    def insert_head(self,val):

        new_node = self.Node(val, None)

        if self._size == 0: #i.e. first node

            self._head = new_node

            self._tail = new_node

        else:

            new_node._next = self._head #new node now is pointing to old head

            self._head = new_node   #change the head to the beginning of the list



        self._size += 1





    # insert node at end of list

    def insert_tail(self, val):

        new_node = self.Node(val, None)

        if self._size == 0: #i.e. first node

            self._head = new_node

            self._tail = new_node

        else:

            self._tail._next = new_node

            self._tail = new_node



        self._size += 1





    def delete_head(self):

        if self._size == 0:

            print("Deletion error: List empty")

            return

        #at least one node in the list

        cur = self._head    #take backup of head

        self._head = self._head._next #Change head to point to its next neighbour

        cur._next = None #break the connection of the old head from the rest of the list

        self._size -= 1







    def delete_tail(self):

        if self._size == 0:

            print("Deletion error: List empty")





    def display(self):

        #Traverse the list here

        print("\nLinked list num of nodes", self._size)

        if self._size == 0:

            print("\nList empty")

            return

        cur = self._head

        while cur!= None:

            print("Node data:", cur._data)

            cur = cur._next





    #To delete a specific node

    def delete_node(self, key):

        prev = None

        cur = self._head

        while True:

            if cur._data[0] == key:

                

                if cur._next != None:

                    if prev == None:

                        # first

                        self.delete_head()

                    else:

                        # middle

                        prev._next = cur._next

                        cur = None

                        self._size -= 1

                else:

                    # last

                    prev._next = None

                    self._size -= 1

                break

            prev = cur

            cur = cur._next


    def main():

        queue = Queue()

        queue_x = Queue()

        queue_y = Queue()

        #read csv file

        skip = True

        with open("../input/cities10/cities10.csv", 'r') as cities_file:

            cities_data = csv.reader(cities_file)



            cities_list = LinkedList()

            c_split_list = []

            for i in cities_data:

                if skip == True:     #to skip the header

                    skip = False

                else:

                    cities_list.insert_tail(i)

            



        cur = cities_list._head

        distance_total = 0

        

        fixed_point_x = cur._data[1]

        fix_point_y = cur._data[2]

        first = True



        distance_total = 0

        cur_distance_list = []

        while cities_list._size > 1:

            if first == True:

                first = False

            else:

                fixed_point_x = min_fixed_point_x

                fix_point_y = min_fixed_point_y

                

            min = 99999999

            cur2 = cities_list._head

            city_num = 0

            while cur2 != None:

                cur_distance = math.sqrt( math.pow( float(cur2._data[1]) - float(fixed_point_x) , 2 ) + math.pow( float(cur2._data[2]) - float(fix_point_y) ,2))



                

                #calculate the prime condition

                if city_num%10 == 0:

                    if (cur_distance%2 == 0) or (cur_distance%3 == 0):

                        cur_distance = cur_distance + cur_distance *0.1

                    

                    

                #In case the condition when it's first point

                #which the distance will always be 0

                if cur_distance > 0:

                    if cur_distance < min:

                        min = cur_distance

                        

                        #next fixed point

                        min_fixed_point_x = cur2._data[1]

                        min_fixed_point_y = cur2._data[2]

                        min_value = cur2

                        

                cur2 = cur2._next



            distance_total = distance_total + min

            cities_list.delete_node(min_value._data[0])

            city_num = city_num + 1

            

            queue.push(min_value._data[0])

            queue_x.push(float(min_value._data[1]))

            queue_x.push(float(cities_list._head._data[1]))

            queue_y.push(float(min_value._data[2]))

            queue_y.push(float(cities_list._tail._data[1]))

            

            

            

            cur = min_value

            if cities_list._size %1000 ==0:

                print("Num of citites", cities_list._size, "Total distance", distance_total)

        

        cities_list.insert_head(['0', '316.8367391', '2202.340707'])

        last_to_first_d = math.sqrt( math.pow(float(cities_list._head._data[1]) - float(cities_list._tail._data[1]),2) + math.pow( float(cities_list._head._data[2]) - float(cities_list._tail._data[2]),2))    

        distance_total = distance_total + last_to_first_d

        print("total", distance_total)

#         stk.display()

        



        plt.figure()

        plt.plot(queue_x._data,queue_y._data, linewidth=0.2)

        plt.ylabel('some numbers')

        plt.show()

main()