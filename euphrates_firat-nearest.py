import math

def distance(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

def findPossibleDistances(locations, current):
    distances = {}
    for index, loc in enumerate(locations):
        distances[index] = distance(loc, current)
    return distances

def sortList(distances):
    return {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}

def initMatrix():
    rows, cols = (10, 10) 
    worldMap = [["."]*10 for _ in range(10)]
    printMatrix(worldMap)
    return worldMap

def calculateMatrix(sortedDistances, locations, current):
    worldMap = initMatrix()
    for index, (key, value) in enumerate(sortedDistances.items()):
        if (index == 0):
            worldMap[locations[key][0]][locations[key][1]] = str(key) + "v"
        else:
            worldMap[locations[key][0]][locations[key][1]] = str(key) + "*"
    worldMap[current[0]][current[1]] = "x"
    return worldMap
    
def printMatrix(matrix):
    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in matrix]))

def main():
    locations = [
        [5, 5],
        [2, 5],
        [2, 6],
        [1, 4],
        [7, 7]
    ]
    
    current = [3,2]
    
    distances = findPossibleDistances(locations, current)
    sortedList = sortList(distances)
    print(distances)
    print(sortedList)
    matrix = calculateMatrix(sortedList, locations, current)
    printMatrix(matrix)
    
main()    