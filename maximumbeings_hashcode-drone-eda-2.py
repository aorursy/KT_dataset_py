# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pprint
import copy

def parseInputData(path):
    f = open(path)
    lines=f.readlines()
    activeInputLine = 0
    inputFileLength = len(lines)
    print()
    print("Input File:")
    print("-----------------------")
    print()
    for l in lines:
        print(l.rstrip())

    globalSimulation = []
    simulations = {}

    while activeInputLine < inputFileLength-1:
      rows,columns,drones,turns,maxpayload = lines[activeInputLine].split()
      simulations["rows"] = rows
      simulations["columns"] = columns
      simulations["drones"] = drones
      simulations["turns"] = turns
      simulations["maxpayload"] = maxpayload
      activeInputLine = activeInputLine + 1
      products = dict()
      for x in range(int(lines[activeInputLine])):
        products[str(x)] = lines[activeInputLine+1].split()[x]
      activeInputLine = activeInputLine + 2
      simulations["products"]=products
      warehouses = dict()
      lineToUse=activeInputLine
      for x in range(int(lines[lineToUse])):
        activeInputLine = activeInputLine + 1
        key = lines[activeInputLine].rstrip()
        activeInputLine = activeInputLine + 1
        val = lines[activeInputLine].split()
        warehouses[key] = val
      activeInputLine = activeInputLine + 1
      simulations["warehouses"]=warehouses
      orders = dict()
      lineToUse = activeInputLine
      for x in range(int(lines[lineToUse])):
        activeInputLine = activeInputLine + 1
        orderLoc = lines[activeInputLine].rstrip()
        activeInputLine = activeInputLine + 1
        quantity = int(lines[activeInputLine].rstrip())
        activeInputLine = activeInputLine + 1
        productTypes = lines[activeInputLine].split()
        orders[orderLoc] = [quantity,productTypes]
      simulations["orders"]=orders
      globalSimulation.append(simulations)
      simulations = {}
    return globalSimulation

globalSimulation = parseInputData('/kaggle/input/hashcodein/hashcode.in')
print()
print("Parsed Small Sized Input File:")
print("------------------------------")
print()
pprint.pprint(globalSimulation[0])


globalSimulation = parseInputData('/kaggle/input/hashcode-drone-delivery/busy_day.in')


print()
print("Parsed Competition Input File - Sample Orders for First Simulation:")
print("-------------------------------------------------------------------")
print()
print("Orders  Qty Products:")
print("-------------------------------------------------------------------")
print()
firstSimulationOrders = globalSimulation[0]["orders"]
for x in firstSimulationOrders:
  print(x, firstSimulationOrders[x])
print()
print("Parsed Competition Input File - Sample Products & Weights for First Simulation:")
print("-------------------------------------------------------------------------------")
print()
print("Products & Weights:")
print("-------------------------------------------------------------------------------")
print()
firstSimulationProducts = globalSimulation[0]["products"]
pprint.pprint(firstSimulationProducts)

print()
print("Parsed Competition Input File - Sample Warehouse Coordinates & Inventory for First Simulation:")
print("----------------------------------------------------------------------------------------------")
print()
print("Coordinates & Inventory:")
print("----------------------------------------------------------------------------------------------")
print()
firstSimulationWarehouses = globalSimulation[0]["warehouses"]
for x in firstSimulationWarehouses:
  print(x, firstSimulationWarehouses[x])
  print()
print()
print("Total Number of Drones:")
print("----------------------------------------------------------------------------------------------")
firstSimulationDrones = globalSimulation[0]["drones"]
print(firstSimulationDrones)
print()
print("Total Number of Turns:")
print("----------------------------------------------------------------------------------------------")
firstSimulationTurns = globalSimulation[0]["turns"]
print(firstSimulationTurns)

print()
print("Maximum Payloads:")
print("----------------------------------------------------------------------------------------------")
firstSimulationmaxPayLoads = globalSimulation[0]["maxpayload"]
print(firstSimulationmaxPayLoads)

print()
print("Number of Columns:")
print("----------------------------------------------------------------------------------------------")
firstSimulationColumns = globalSimulation[0]["columns"]
print(firstSimulationColumns)
print()
print("Number of Rows:")
print("----------------------------------------------------------------------------------------------")
firstSimulationRows = globalSimulation[0]["rows"]
print(firstSimulationRows)
"""
We don't really need a grid or a row by column matrix - but assuming we do it looks like below:

res = []

def state():
    for x in range(5):
        for y in range(5):
            res.append((x,y))
    return res

print(state())

[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), 
 (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), 
 (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), 
 (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), 
 (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
"""
"""
Helper function to calculate the distance between
two coordinates. Refactored from an old code from 
the archives. Given two coordinates on a 2D plane. 
Find the distance between two coordinates.

                         * [0,4]
                    *    *
                *   *    *
            *   *   *    *
  [0,0] *   *   *   *    * [4,4]

Distance along horizontal lines = y_2 - y_1 e.g. abs(4 - 0)
Distance along vertical lines = x_2 - x_1 = e.g. abs(0 - 4)

Otherwise use Pythogoras theorem as follows;
distance = sqrt((4-0)**2 + (4-0)**2)

"""
import math
firstCoordinate =  [1,3]
secondCoordinate = [[1,5], [1,-5], [3,2],[3,1],[2,3]]

minDistance = float("-inf")
def distanceBtwLocations(firstCoordinate, secondCoordinate):
    distances = []
    if firstCoordinate[0] == secondCoordinate[0]:
        temp = abs(firstCoordinate[1] - secondCoordinate[1])
        distances.append(round(temp,4))
    if firstCoordinate[1] == secondCoordinate[1]:
        temp = abs(firstCoordinate[0] - secondCoordinate[0])
        distances.append(round(temp,4))
    elif firstCoordinate[0] != secondCoordinate[0] \
    and firstCoordinate[1] != secondCoordinate[1]:
        horizontalDist = abs(firstCoordinate[1] - secondCoordinate[1])
        verticalDist = abs(firstCoordinate[0] - secondCoordinate[0])
        pythogoras = math.sqrt(horizontalDist**2 + verticalDist**2)
        distances.append(round(pythogoras,4))
    return distances[0]

for x in secondCoordinate:
    print(distanceBtwLocations(firstCoordinate, x))

globalSimulation = parseInputData('/kaggle/input/hashcodein/hashcode.in')
print()
print("Parsed Small Sized Input File:")
print("------------------------------")
print()
pprint.pprint(globalSimulation[0])
"""
At the beginning of the simulation, all drones are at the first warehouse (warehouse with id 0).
So get the key of the first warehouse with id 0
and then initialize the addresses of all the available drones to the address of the first
warehouse.
"""
firstWarehouseId = list(globalSimulation[0]['warehouses'].keys())[0]
droneLocs = dict()
for x in range(int(globalSimulation[0]['drones'])):
    droneLocs[str(x)] = firstWarehouseId

print("Initial Location of Drones")
print("--------------------------")
print(droneLocs)
"""
Building our prototypes little by little

(a) create a copy of the orders using deepcopy so that we do not unintentionally mutate it
"""

workingOrders = copy.deepcopy(globalSimulation[0]['orders'])
workingOrders
firstOrder = list(globalSimulation[0]['orders'].keys())[0]
firstOrderLoc = [int(x) for x in firstOrder.split()]
firstOrderLoc
"""
Get the address and coordinates of the nearest warehouse with the supplies that can fulfil this order.

"""
firstOrderItems = globalSimulation[0]['orders'].get(firstOrder)
firstOrderItems
"""
'warehouses': {'0 0': ['5', '1', '0'], '5 5': ['0', '10', '2']}
"""