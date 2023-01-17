def isPrime(n):

    for i in range(2,4):

        if n%i == 0:

            return False

    return True
import csv

import math



class CircularQueue:

    class _Node:

        __slots__ = '_element_CityID', '_element_PosX', '_element_PosY', '_next'

        

        def __init__(self, cityID, PosX, PosY, next):

            self._element_CityID = int(cityID)

            self._element_PosX = float(PosX)

            self._element_PosY = float(PosY)

            self._next = next

            

    def __init__(self):

        self._tail = None

        self._size = 0

        

    def __len__(self):

        return self._size

    

    def is_empty(self):

        return self._size == 0

    

    def first(self):

        if self.is_empty():

            raise Empty('Queue is empty')

        head = self._tail._next

        return head._element_CityID, head._element_PosX, head._element_PosY

    

    def dequeue(self):

        if self.is_empty():

            raise Empty('Queue is empty')

        oldhead = self._tail._next

        if self._size == 1: 

            self._tail = None

        else:

            self._tail._next = oldhead._next

        self._size -= 1

        return oldhead._element_CityID, oldhead._element_PosX, oldhead._element_PosY



    def enqueue(self, e1, e2, e3):

        newest = self._Node(e1, e2, e3, None)

        if self.is_empty():

            newest._next = newest

        else:

            newest._next = self._tail._next

            self._tail._next = newest

        self._tail = newest

        self._size += 1

        

    def rotate(self):

        if self._size > 0:

            self._tail = self._tail._next

            

    def readCSV(self, csv_name, ignore_head):

        # Initialise

        file = open(csv_name, "r")

        reader = csv.reader(file)

        # Check if ignore_head is used

        if ignore_head:

            Temp = 1

        else:

            Temp = 0

       # Do the loop, and enqueue

        for row in reader:

            if Temp == 1:

                Temp = Temp -1

            else:

                self.enqueue(row[0], row[1], row[2])

                

    def showElementsAt(self, position):

        '''

        This is return the city details from the node position.

        '''

        if self.is_empty():

            raise Empty('Queue is empty')

        head = self._tail._next

        while position > 0:

            head = head._next

            position = position -1

        return head._element_CityID, head._element_PosX, head._element_PosY

    

    def showElementsAtCityID(self, CityID):

        '''

        This is return the city details from the city ID.

        Usually to convert city ID to node position in the circular linked list

        '''

        if self.is_empty():

            raise Empty('Queue is empty')

        head = self._tail._next

        position = 0

        found = False

        while found == False:

            head = head._next

            if head._element_CityID == CityID:

                return position, head._element_PosX, head._element_PosY

            position = position + 1

        raise Empty('CityID not found')

    

    def removeThisPosition(self, position):

        '''

        This is to delete the specified node from the circular linked list with the given node position as parameter

        '''

        if self.is_empty():

            raise Empty('Queue is empty')

        while position > -1:

            self.rotate()

            position = position -1

        return self.dequeue()

        

    

    def showSmallestPythogorasDistanceFrom(self, CityID, PointX, PointY, is10thCity):

        '''

        This is top calculate the nearest node from the specified coordinates

        '''

        # Initialise

        if self.is_empty():

            raise Empty('Queue is empty')

        head = self._tail._next

        position = 0

        distance = 99999999

        minCityID = 0

        minPosX = 0

        minPosY = 0

        # The loop

        while position < self._size + 1:

            # Calculate Distance

            DistanceX = abs(PointX - head._element_PosX)

            DistanceY = abs(PointY - head._element_PosY)

            Temp_distance = math.sqrt(math.pow(DistanceX, 2) + math.pow(DistanceY, 2))

            # Check 10th City

            if (is10thCity):

                if (isPrime(CityID)):

                    pass

                else:

                    Temp_distance = Temp_distance * 1.1

            # Compare Distance

            if Temp_distance < distance:

                if Temp_distance != 0:

                    distance = Temp_distance

                    minCityID = head._element_CityID

                    minPosX = head._element_PosX

                    minPosY = head._element_PosY

            # Next iteration of the loop

            head = head._next

            position = position + 1

        return minCityID, minPosX, minPosY, distance
class CityQueue:

    __slots__ = '_cityID', '_posX', '_posY', '_size'



    def __init__(self):

        self._cityID = []

        self._posX = []

        self._posY = []

        self._size = 0

        

    def add(self, cityID, posX, posY):

        self._cityID.append(cityID)

        self._posX.append(posX)

        self._posY.append(posY)

        self._size = self._size + 1

        

    def delete(self):

        cityID = self._cityID.pop(0)

        posX = self._posX.pop(0)

        poxY = self._posY.pop(0)

        self._size = self._size - 1

        return cityID, posY, posX

    

    def getTheIDcomponent(self):

        return self._cityID

    

    def getTheXcomponent(self):

        return self._posX

    

    def getTheYcomponent(self):

        return self._posY

    

    def __len__(self):

        return self._size
MainData = CircularQueue()

MainData.readCSV("../input/cities10.csv", True)

len(MainData)
# Initialise

distance = 0

i = 0

cityQueue = CityQueue()



# First City

City0Position, City0PositionX, City0PositionY = MainData.showElementsAtCityID(0)

CityPosition, CityPositionX, CityPositionY = MainData.showElementsAtCityID(0)

CurCityID = 0

ID1, X1, Y1 = MainData.removeThisPosition(CityPosition)

cityQueue.add(ID1, X1, Y1)



# Find shortest

while len(MainData) > 1:

    if i%10 == 0:

        NextCity, NextCityPositionX, NextCityPositionY, TempDistance = MainData.showSmallestPythogorasDistanceFrom(CurCityID, CityPositionX, CityPositionY, True)

    else:

        NextCity, NextCityPositionX, NextCityPositionY, TempDistance = MainData.showSmallestPythogorasDistanceFrom(CurCityID, CityPositionX, CityPositionY, False)

    NextCityPos, NextCityPosX, NextCityPosY = MainData.showElementsAtCityID(NextCity)

    ID1, X1, Y1 = MainData.removeThisPosition(NextCityPos)

    cityQueue.add(ID1, X1, Y1)

    distance = TempDistance + distance

    CurCityID, CityPosition, CityPositionX, CityPositionY = NextCity, NextCityPos, NextCityPosX, NextCityPosY

    

    if i%1000 == 0:

        print("Iteration:"+str(i)+" len:"+str(len(MainData))+" Distance:"+str(distance))

    i = i + 1



# Add the final destination (Back to origin city)

MainData.enqueue(City0Position, City0PositionX, City0PositionY)

NextCity, NextCityPositionX, NextCityPositionY, TempDistance = MainData.showSmallestPythogorasDistanceFrom(CurCityID, CityPositionX, CityPositionY, False)

cityQueue.add(NextCity, NextCityPositionX, NextCityPositionY)

distance = TempDistance + distance



# Add the last node (which is the starting city) of to the cityQueue

cityQueue.add(City0Position, City0PositionX, City0PositionY)



# display Distance

print("Iteration:"+str(i)+" len:"+str(len(MainData))+" Distance:"+str(distance))
len(MainData)
len(cityQueue)
import matplotlib.pyplot as plt

plt.figure(figsize=(15,15))

plt.plot(cityQueue.getTheXcomponent(), cityQueue.getTheYcomponent(), linewidth=0.2)

plt.grid(False)

plt.show()