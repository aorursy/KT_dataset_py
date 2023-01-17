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
PRIME = 573757 # Less than hash table size

appName = appName.sample(frac=1)

appName.describe()

def calculateHash(string):

    return hash(string) % CAPACITY
def store(magazine):

    count = 0

    for element in magazine:

        position = calculateHash(element)

        if hashTable[position] == None:

            hashTable[position] = element

            count += 1

        else:

            # DOUBLE HASHING FUNCTION

            d = PRIME - (hash(element) % PRIME)

            while hashTable[(position + d) % CAPACITY] != None:

                position = (position + d) % CAPACITY

            else:

                hashTable[(position + d) % CAPACITY] = element

            count += 1

    return count

            
def search(note):

    count = 0

    sumCount = 0

    for item in note:

        hashedValue = hash(item)

        position = hashedValue % CAPACITY

        sumCount += 1

        if hashTable[position] == None:

            count += 1

            continue

        if hashTable[position] != item:

            sumCount += 1

            # DOUBLE HASHING FUNCTION

            d = PRIME - (hashedValue % PRIME)

            while hashTable[(position + d ) % CAPACITY] != item:     

                if hashTable[(position + d ) % CAPACITY] == None:

                    count += 1

                    break

                else:

                    position = (position + d) % CAPACITY

                    sumCount += 1

                    

            hashTable[(position + d ) % CAPACITY] = "FILLED"

        else:

            hashTable[position] = "FILLED"

    return count, sumCount
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

store(selectedDF)

start = time.process_time()

result, searches = search(testDF)

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

store(selectedDF)

start = time.process_time()

result, searches = search(testDF)

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

    store(selectedDF)

    start = time.process_time()

    result, searches = search(testDF)

    timeSum += time.process_time() - start

    print("Trial " + str(i+1) + ": " + str(time.process_time() - start))



print("Average Time Taken for Search = " + str(timeSum/loopSearchNum))

print("\nElements in hashtable: " + str(searchNum - result))

print("Elements not in hashtable: " + str(result))

print("Average num of searches: " + str(((searches )/searchNum)))
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

    store(selectedDF)

    start = time.process_time()

    result, searches = search(testDF)

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

    store(selectedDF)

    start = time.process_time()

    result, searches = search(testDF)

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

    store(selectedDF)

    start = time.process_time()

    result, searches = search(testDF)

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

    store(selectedDF)

    start = time.process_time()

    result, searches = search(testDF)

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

    store(selectedDF)

    start = time.process_time()

    result, searches = search(testDF)

    timeSum += time.process_time() - start

    print("Trial " + str(i+1) + ": " + str(time.process_time() - start))



print("Average Time Taken for Search = " + str(timeSum/loopSearchNum))

print("\nElements in hashtable: " + str(searchNum - result))

print("Elements not in hashtable: " + str(result))

print("Average num of searches: " + str(((searches)/searchNum)))