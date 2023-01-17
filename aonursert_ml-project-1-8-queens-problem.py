import pandas as pd
import random
import time

queens = [[4, 0], [5, 1], [6, 2], [3, 3], [4, 4], [5, 5], [6, 6], [5, 7]]
chess = []
hMatrix = pd.DataFrame(0, ["A", "B", "C", "D", "E", "F", "G", "H"], [0, 1, 2, 3, 4, 5, 6, 7])
lastList = []

finalH = -1
replaceCount = 0
randomStartCount = -1

for i in range(8):
    for j in range(8):
        chess.append([i, j])


def init():
    for i in range(8):
        randRow = random.randrange(8)
        randCol = random.randrange(8)
        if [randRow, randCol] not in queens:
            queens[i] = [randRow, randCol]


def findH():
    h = 0
    for x in queens:
        for y in queens[queens.index(x) + 1:]:
            # If queens' row equal, it's a conflict. Horizontal conflict
            if x[0] == y[0]:
                h += 1
            # If queens' column equal, it's a conflict. Vertical conflict
            elif x[1] == y[1]:
                h += 1
            # If queens' difference of rows equal to difference of columns, it's a conflict. Diagonal conflict
            elif abs(x[0] - y[0]) == abs(x[1] - y[1]):
                h += 1
    return h


def findNextMoves(currentH):
    global replaceCount
    replaceCount += 1

    triedDotsPosition = []
    triedDotsH = []

    minDotOriginalPosition = [-1, -1]
    minDotPosition = [-1, -1]
    minDotsH = currentH

    for q in queens:
        originalX = q
        for cell in chess:
            if cell not in queens:
                if q[0] == cell[0]:
                    if cell not in triedDotsPosition:
                        queens[queens.index(q)] = cell
                        triedDotsPosition.append(cell)
                        hValue = findH()
                        triedDotsH.append(hValue)
                        queens[queens.index(cell)] = originalX

                        if hValue < minDotsH:
                            minDotOriginalPosition = queens[queens.index(q)]
                            minDotPosition = cell
                            minDotsH = hValue
                elif q[1] == cell[1]:
                    if cell not in triedDotsPosition:
                        queens[queens.index(q)] = cell
                        triedDotsPosition.append(cell)
                        hValue = findH()
                        triedDotsH.append(hValue)
                        queens[queens.index(cell)] = originalX

                        if hValue < minDotsH:
                            minDotOriginalPosition = queens[queens.index(q)]
                            minDotPosition = cell
                            minDotsH = hValue
                elif abs(q[0] - cell[0]) == abs(q[1] - cell[1]):
                    if cell not in triedDotsPosition:
                        queens[queens.index(q)] = cell
                        triedDotsPosition.append(cell)
                        hValue = findH()
                        triedDotsH.append(hValue)
                        queens[queens.index(cell)] = originalX

                        if hValue < minDotsH:
                            minDotOriginalPosition = queens[queens.index(q)]
                            minDotPosition = cell
                            minDotsH = hValue

    for i in triedDotsPosition:
        hMatrix.iloc[i[0]][i[1]] = triedDotsH[triedDotsPosition.index(i)]

    if minDotsH < currentH:
        queens[queens.index(minDotOriginalPosition)] = minDotPosition
        findNextMoves(minDotsH)
    else:
        for i in queens:
            hMatrix.iloc[i[0]][i[1]] = currentH
        global finalH
        finalH = currentH


for i in range(20):
    startTime = time.time()

    finalH = -1
    replaceCount = 0
    randomStartCount = -1
    while finalH != 0:
        randomStartCount += 1
        init()
        findNextMoves(findH())
        # print("Final H =", finalH)

    endTime = time.time()
    elapsedTime = endTime - startTime

    print()
    print(hMatrix)
    print("Replace Count:", replaceCount)
    print("Random Start Count:", randomStartCount)
    print("Elapsed Time: " + str(elapsedTime))
    lastList.append([replaceCount, randomStartCount, elapsedTime])

print()
lastList = pd.DataFrame(lastList)
lastList.columns = ["Replace Count", "Random Start Count", "Elapsed Time"]
print("Zeros represents queens in final position")
lastList.loc["Mean"]=lastList[["Replace Count", "Random Start Count", "Elapsed Time"]].mean(axis=0)
print(lastList)