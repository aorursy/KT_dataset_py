# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/million-song-data-set-subset/song_data.csv")

df = df.drop_duplicates(subset = "title")

df = df.dropna()

df.head()
appName = df["title"]

appName.head()
appName = appName.sample(frac=1)

appName.describe()
# Set size of hash table to prime number

CAPACITY = 700199

numItems = round(CAPACITY / 4)

selectedDF = appName.iloc[:numItems]
testDF = appName.iloc[:600000]
hashTable = [None] * CAPACITY
def calculateHash(string):

    return hash(string) % CAPACITY
def storeLinearProbing(magazine):

    for element in magazine:

        # print("STORED " + element)

        position = calculateHash(element)

        # print(position)

        if hashTable[position] == None:

            hashTable[position] = element

            # print("Stored into " + str(position))

        else:

            i = 1

            # print("collided into " + str((position) % CAPACITY))

            while hashTable[(position + i) % CAPACITY] != None:

                # print("collided into " + str((position + i) % CAPACITY))

                i += 1

            hashTable[(position + i) % CAPACITY] = element
def searchLinearProbing(note):

    sumCount = 0

    count = 0

    for item in note:

        position = calculateHash(item)

        sumCount += 1

        if hashTable[position] == None:

            count += 1

            continue

        if hashTable[position] != item:

            i = 1

            sumCount+= 1



            # Linear Probing

            while hashTable[(position + i ) % CAPACITY] != item:        

                if hashTable[(position + i ) % CAPACITY] == None:

                    count += 1

                    break

                else:

                    i += 1

                    sumCount += 1

            hashTable[(position + i ) % CAPACITY] = "FILLED"

        else:

            hashTable[position] = "FILLED"

    return count, sumCount
# def checkMagazine(magazine, note):

#     # Store Magazine into hashtable

#     storeLinearProbing(magazine)



#     # Check if all items in note is in hashtable

#     result = searchLinearProbing(note)

#     print("Not Found = " + str(result))
start = time.process_time()

storeLinearProbing(selectedDF)

print("Time taken for Store = "),

print(time.process_time() - start)
start = time.process_time()

result = searchLinearProbing(testDF)

print("Not Found = " + str(result))

print("Time Taken for Search = " + str(time.process_time() - start))
for i in hashTable:

    if i != "FILLED" and i != None:

        print(i)
CAPACITY = 700199

LOADFACTOR = 0.75

searchNum = 1

numItems = round(CAPACITY * LOADFACTOR)

selectedDF = appName.iloc[:numItems]

testDF = appName.iloc[numItems-searchNum:numItems]

hashTable = [None] * CAPACITY

loopSearchNum = 20



# PRINT STATS

print("DEMO IMPLEMENTATION (Successful) - Load Factor: " + str(LOADFACTOR))



# STORE AND SEARCH 20 times

timeSum = 0

hashTable = [None] * CAPACITY

storeLinearProbing(selectedDF)

start = time.process_time()

result, searches = searchLinearProbing(testDF)

timeSum += time.process_time() - start

print("Trial: " + ": " + str(time.process_time() - start))

print("Num of searches: " + str(((searches + 1))))
CAPACITY = 700199

LOADFACTOR = 0.75

searchNum = 1

numItems = round(CAPACITY * LOADFACTOR)

selectedDF = appName.iloc[:numItems]

testDF = appName.iloc[numItems+1:numItems + 2]

hashTable = [None] * CAPACITY

loopSearchNum = 20



# PRINT STATS

print("DEMO IMPLEMENTATION (Unsuccessful) - Load Factor: " + str(LOADFACTOR))



# STORE AND SEARCH 20 times

timeSum = 0

hashTable = [None] * CAPACITY

storeLinearProbing(selectedDF)

start = time.process_time()

result, searches = searchLinearProbing(testDF)

timeSum += time.process_time() - start

print("Trial: " + ": " + str(time.process_time() - start))

print("Num of searches: " + str(((searches + 1))))
CAPACITY = 700199

LOADFACTOR = 0.25

searchNum = 100000

numItems = round(CAPACITY * LOADFACTOR)

selectedDF = appName.iloc[:numItems]

testDF = appName.iloc[numItems-searchNum:numItems]

hashTable = [None] * CAPACITY

loopSearchNum = 20



# PRINT STATS

print("TESTING SUCCESSFUL CASES - Load Factor: " + str(LOADFACTOR))









# STORE AND SEARCH 20 times

timeSum = 0

for i in range(loopSearchNum):

    hashTable = [None] * CAPACITY

    storeLinearProbing(selectedDF)

    start = time.process_time()

    result, searches = searchLinearProbing(testDF)

    timeSum += time.process_time() - start

    print("Trial " + str(i+1) + ": " + str(time.process_time() - start))



print("Average Time Taken for Search = " + str(timeSum/loopSearchNum))

print("\nElements in hashtable: " + str(searchNum - result))

print("Elements not in hashtable: " + str(result))

print("Average num of searches: " + str(((searches)/searchNum)))
CAPACITY = 700199

LOADFACTOR = 0.25

searchNum = 100000

numItems = round(CAPACITY * LOADFACTOR)

selectedDF = appName.iloc[:numItems]

testDF = appName.iloc[numItems+1:numItems+1+searchNum]

hashTable = [None] * CAPACITY

loopSearchNum = 20



# PRINT STATS

print("TESTING UNSUCCESSFUL CASES - Load Factor: " + str(LOADFACTOR))









# STORE AND SEARCH 20 times

timeSum = 0

for i in range(loopSearchNum):

    hashTable = [None] * CAPACITY

    storeLinearProbing(selectedDF)

    start = time.process_time()

    result, searches = searchLinearProbing(testDF)

    timeSum += time.process_time() - start

    print("Trial " + str(i+1) + ": " + str(time.process_time() - start))



print("Average Time Taken for Search = " + str(timeSum/loopSearchNum))

print("\nElements in hashtable: " + str(searchNum - result))

print("Elements not in hashtable: " + str(result))

print("Average num of searches: " + str(((searches)/searchNum)))
CAPACITY = 700199

LOADFACTOR = 0.5

searchNum = 100000

numItems = round(CAPACITY * LOADFACTOR)

selectedDF = appName.iloc[:numItems]

testDF = appName.iloc[numItems-searchNum:numItems]

hashTable = [None] * CAPACITY

loopSearchNum = 20



# PRINT STATS

print("TESTING SUCCESSFUL CASES - Load Factor: " + str(LOADFACTOR))









# STORE AND SEARCH 20 times

timeSum = 0

for i in range(loopSearchNum):

    hashTable = [None] * CAPACITY

    storeLinearProbing(selectedDF)

    start = time.process_time()

    result, searches = searchLinearProbing(testDF)

    timeSum += time.process_time() - start

    print("Trial " + str(i+1) + ": " + str(time.process_time() - start))



print("Average Time Taken for Search = " + str(timeSum/loopSearchNum))

print("\nElements in hashtable: " + str(searchNum - result))

print("Elements not in hashtable: " + str(result))

print("Average num of searches: " + str(((searches)/searchNum)))
CAPACITY = 700199

LOADFACTOR = 0.5

searchNum = 100000

numItems = round(CAPACITY * LOADFACTOR)

selectedDF = appName.iloc[:numItems]

testDF = appName.iloc[numItems+1:numItems+1+searchNum]

hashTable = [None] * CAPACITY

loopSearchNum = 20



# PRINT STATS

print("TESTING UNSUCCESSFUL CASES - Load Factor: " + str(LOADFACTOR))









# STORE AND SEARCH 20 times

timeSum = 0

for i in range(loopSearchNum):

    hashTable = [None] * CAPACITY

    storeLinearProbing(selectedDF)

    start = time.process_time()

    result, searches = searchLinearProbing(testDF)

    timeSum += time.process_time() - start

    print("Trial " + str(i+1) + ": " + str(time.process_time() - start))



print("Average Time Taken for Search = " + str(timeSum/loopSearchNum))

print("\nElements in hashtable: " + str(searchNum - result))

print("Elements not in hashtable: " + str(result))

print("Average num of searches: " + str(((searches)/searchNum)))
CAPACITY = 700199

LOADFACTOR = 0.75

searchNum = 100000

numItems = round(CAPACITY * LOADFACTOR)

selectedDF = appName.iloc[:numItems]

testDF = appName.iloc[numItems-searchNum:numItems]

hashTable = [None] * CAPACITY

loopSearchNum = 20



# PRINT STATS

print("TESTING SUCCESSFUL CASES - Load Factor: " + str(LOADFACTOR))









# STORE AND SEARCH 20 times

timeSum = 0

for i in range(loopSearchNum):

    hashTable = [None] * CAPACITY

    storeLinearProbing(selectedDF)

    start = time.process_time()

    result, searches = searchLinearProbing(testDF)

    timeSum += time.process_time() - start

    print("Trial " + str(i+1) + ": " + str(time.process_time() - start))



print("Average Time Taken for Search = " + str(timeSum/loopSearchNum))

print("\nElements in hashtable: " + str(searchNum - result))

print("Elements not in hashtable: " + str(result))

print("Average num of searches: " + str(((searches)/searchNum)))
CAPACITY = 700199

LOADFACTOR = 0.75

searchNum = 100000

numItems = round(CAPACITY * LOADFACTOR)

selectedDF = appName.iloc[:numItems]

testDF = appName.iloc[numItems+1:numItems+1+searchNum]

hashTable = [None] * CAPACITY

loopSearchNum = 20



# PRINT STATS

print("TESTING UNSUCCESSFUL CASES - Load Factor: " + str(LOADFACTOR))









# STORE AND SEARCH 20 times

timeSum = 0

for i in range(loopSearchNum):

    hashTable = [None] * CAPACITY

    storeLinearProbing(selectedDF)

    start = time.process_time()

    result, searches = searchLinearProbing(testDF)

    timeSum += time.process_time() - start

    print("Trial " + str(i+1) + ": " + str(time.process_time() - start))



print("Average Time Taken for Search = " + str(timeSum/loopSearchNum))

print("\nElements in hashtable: " + str(searchNum - result))

print("Elements not in hashtable: " + str(result))

print("Average num of searches: " + str(((searches)/searchNum)))