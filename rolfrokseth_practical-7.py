#https://interactivepython.org/courselib/static/pythonds/SortSearch/TheInsertionSort.html

def insertionSort(alist):

   for index in range(1,len(alist)):



     currentvalue = alist[index]

     position = index



     while position>0 and alist[position-1]>currentvalue:

         alist[position]=alist[position-1]

         position = position-1



     alist[position]=currentvalue



alist = [5,6,3,1,2,7,9,8]

insertionSort(alist)

print(alist)

# https://interactivepython.org/lpomz/courselib/static/pythonds/SortSearch/TheSelectionSort.html

def selectionSort(alist):

   for fillslot in range(len(alist)-1,0,-1):

       positionOfMax=0

       for location in range(1,fillslot+1):

           if alist[location]>alist[positionOfMax]:

               positionOfMax = location



       temp = alist[fillslot]

       alist[fillslot] = alist[positionOfMax]

       alist[positionOfMax] = temp





alist = [5,6,3,1,2,7,9,8]

selectionSort(alist)

print(alist)
class BinHeap:

    def __init__(self):

        self.heapList = [0]

        self.currentSize = 0





    def percUp(self,i):

        while i // 2 > 0:

          if self.heapList[i] < self.heapList[i // 2]:

             tmp = self.heapList[i // 2]

             self.heapList[i // 2] = self.heapList[i]

             self.heapList[i] = tmp

          i = i // 2



    def insert(self,k):

      self.heapList.append(k)

      self.currentSize = self.currentSize + 1

      self.percUp(self.currentSize)



    def percDown(self,i):

      while (i * 2) <= self.currentSize:

          mc = self.minChild(i)

          if self.heapList[i] > self.heapList[mc]:

              tmp = self.heapList[i]

              self.heapList[i] = self.heapList[mc]

              self.heapList[mc] = tmp

          i = mc



    def minChild(self,i):

      if i * 2 + 1 > self.currentSize:

          return i * 2

      else:

          if self.heapList[i*2] < self.heapList[i*2+1]:

              return i * 2

          else:

              return i * 2 + 1



    def delMin(self):

      retval = self.heapList[1]

      self.heapList[1] = self.heapList[self.currentSize]

      self.currentSize = self.currentSize - 1

      self.heapList.pop()

      self.percDown(1)

      return retval



    def buildHeap(self,alist):

      i = len(alist) // 2

      self.currentSize = len(alist)

      self.heapList = [0] + alist[:]

      while (i > 0):

          self.percDown(i)

          i = i - 1



bh = BinHeap()

bh.buildHeap([5,1,4,7,3,9,0,2,8])



print(bh.delMin())

print(bh.delMin())

print(bh.delMin())

print(bh.delMin())

print(bh.delMin())

print(bh.delMin())

print(bh.delMin())

print(bh.delMin())

print(bh.delMin())
