def build_stats():

    print("build_stats")

    #sort

    candidates.sort()

    #reverse sort so I can get the last index

    revCandidates = candidates[::-1]

    for candidate in candidates:

        min = candidates.index(candidate)

        max = len(candidates) - revCandidates.index(candidate)-1

        stats[candidate] = {"min": min, "max": max}

    

def less(item):

    return stats[item]["min"]

    

def between(minItem, maxItem):

    return len(candidates)-greater(maxItem)-less(minItem)

    

def greater(item):

    return len(candidates) - stats[item]["max"]-1



def addCandidate(entry):

    return candidates.append(entry)



#initialize my test data

candidates = []

stats = {}

addCandidate(3)

addCandidate(3)

addCandidate(4)

addCandidate(6)

addCandidate(9)





#build the stats

build_stats()

#log

print(candidates)

print(less(4))

print(greater(4))

print(between(3,6))
import collections



#let's make sure I know how to use this thing first

testDeque = collections.deque([2])



#add something to the end

testDeque.append(3)

print(testDeque)



#add something to the start

testDeque.appendleft(4)

print(testDeque)



#peek the end

print(testDeque[-1])



#peek start

print(testDeque[0])
