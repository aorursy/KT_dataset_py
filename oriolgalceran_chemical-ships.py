# BOILERPLATE IMPORTS

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import random
from scipy import optimize as opt
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#OPTIMIZATION MODE: Due to technical issues, the program must be run ONCE on optimization = False at the beginning
optimization = True
#CONSTANTS

ATankCapacity = 58488; #tons
BTankCapacity = 27546;

AProduction = 10437; #tons/day
BProduction = 10437;

ATransferFlow = 40000; #tons/day
BTransferFlow = 20000;

mooringTime = 0.075; #days
journeyTime = 4; #days

shipCapacity = 25000; #tons

amountOfShips = 7; #ships
shipSpawnDelay = 4456;

simulationLength = 40; #days

ALoadingPos = 1;
AWaitingPos = 2;
BWaitingPos = 3;
BLoadingPos = 4;
"""
POSITIONS:
0 = free
1 = loading at A
2 = waiting at A
3 = waiting at B
4 = loading at B
"""

#VARIABLES

currentTime = 0; #minutes

#CONVERSION TO MINUTES

AProduction /= 1440; #tons/minute
BProduction /= 1440;

ATransferFlow /= 1440; #tons/minute
BTransferFlow /= 1440;

mooringTime *= 1440; #minutes
journeyTime *= 1440; #minutes

simulationLength *= 1440; #minutes
#CLASSES

class Ship:  
    
    """
    SHIP STATUSES:
    0 = free
    1 = mooring
    2 = loading
    3 = cruising
    4 = waiting
    5 = unloading
    """
    
    def __init__(self, number, journeyTime, mooringTime, spawnTime):  
        self.number = str(number)
        self.spawnTime = spawnTime
        self.mooringSpeed = 1/mooringTime
        self.cruisingSpeed = 1/journeyTime
        self.speed = 0
        self.destination = 1
        self.stopped = False
        self.stoppeds = []
        self.functionOnArrival = None
        self.level = 0
        self.maxLevel = 25000
        self.levels = []
        self.capacity = shipCapacity
        self.positions = []
        self.position = 2.1
        self.mooring = False
        self.sleeping = True
        self.status = 0
        self.statuses = []
        
    def GoTo(self, destination, functionOnArrival):
        self.stopped = False
        self.destination = destination
        self.functionOnArrival = functionOnArrival
    
    def Stop(self):
        self.stopped = True
        self.speed = 0

    
    def Update(self):
        #print(self.goingToB)
        
        if currentTime>=self.spawnTime and self.sleeping:
            self.status = 3
            self.sleeping = False
            self.GoTo(AWaitingPos, getattr(aTank, "requestQueue"))

        if not self.sleeping:
            if not self.stopped:
                if self.destination < self.position:
                    self.direction = -1
                if self.destination > self.position:
                    self.direction = 1

                self.speed = self.mooringSpeed
                if self.position > AWaitingPos and self.position < BWaitingPos:
                    self.speed = self.cruisingSpeed

                if self.direction == -1:
                    self.position -= self.speed
                    if self.position < self.destination:
                        self.functionOnArrival(self)
                if self.direction == 1:
                    self.position += self.speed
                    if self.position > self.destination:
                        self.functionOnArrival(self)

        self.positions.append(self.position)
        self.levels.append(self.level)
        self.statuses.append(self.status)    
        self.stoppeds.append(self.stopped)

class ATank:
    """
    TANK STATUSES:
    0: free
    1: ship incoming
    2: loading ship
    3: ship leaving
    """
    
    def __init__(self, capacity, transferFlow):
        self.maxLevel = capacity
        self.transferFlow = transferFlow
        self.status = 0
        self.full = False
        self.fulls = []
        self.empty = False
        self.level = capacity
        self.levels = []
        self.queue = []
        self.queues = []
        self.loading = None
        
    def requestQueue(self, ship):
        ship.Stop()
        ship.status = 4
        self.queue.append(ship)
        if not optimization:
            print("-----------")
            print(str(currentTime) + ": A QUEUE:")
            for ship in self.queue:
                print("Barco N." + str(ship.number))
    
    def requestLoad(self, ship):
        if not optimization:
            print("-----------")
            print(str(currentTime) + ": LOADING: Barco N." + str(ship.number))
        self.loading.status = 2
        self.loading.Stop()
        self.status = 2
    
    def requestDeparture(self, ship):
        self.loading.GoTo(BWaitingPos, getattr(bTank, "requestQueue"))
        self.loading.status = 3
        self.loading = None
        self.status = 0      
        
    
    def Update(self):
        self.empty = False
        self.full = False
        
        if self.level > self.maxLevel:
            self.level = self.maxLevel
            self.full = True
        if self.level < 0:
            self.level = 0
            self.empty = True
            
        if not self.full:
            self.level += AProduction
        
        if self.status == 0 and len(self.queue)>0:
            self.loading = self.queue.pop(0)
            self.loading.GoTo(ALoadingPos, getattr(self, "requestLoad"))
            self.loading.status = 1
            self.loading.waiting = False
            ship.mooring = True
            self.status = 1
        
        if self.status == 2 and not self.empty:
            self.loading.level += self.transferFlow
            self.level -= self.transferFlow
            
            if self.level < 0:
                self.loading.level += self.level
                self.level = 0
                
            if self.loading.level>self.loading.maxLevel:
                self.loading.level -= self.transferFlow
                self.level += self.transferFlow
                if not optimization:
                    print("-----------")
                    print(str(currentTime)+ ": Barco N."+ self.loading.number + " full")
                self.diff = self.loading.maxLevel - self.loading.level
                self.loading.level += self.diff
                self.level += self.transferFlow - self.diff
                self.loading.GoTo(AWaitingPos, getattr(aTank, "requestDeparture"))
                self.status = 3
                self.loading.status = 1
        
        self.levels.append(self.level)
        self.queues.append(len(self.queue))
        self.fulls.append(self.full)

        

    
class BTank:
    """
    TANK STATUSES:
    0: free
    1: ship incoming
    2: loading ship
    3: ship leaving
    """
    
    def __init__(self, capacity, transferFlow):
        self.maxLevel = capacity
        self.transferFlow = transferFlow
        self.status = 0
        self.full = False
        self.empty = False
        self.empties = []
        self.level = 0
        self.levels = []
        self.queue = []
        self.queues = []
        self.loading = None
        
    def requestQueue(self, ship):
        ship.Stop()
        ship.status = 4
        self.queue.append(ship)
        if not optimization:
            print("-----------")
            print(str(currentTime) + ": B QUEUE:")
            for ship in self.queue:
                print("Barco N." + str(ship.number))
        
    
    def requestLoad(self, ship):
        if not optimization:
            print("-----------")
            print(str(currentTime) + ": UNLOADING: Barco N." + str(ship.number))
        self.loading.Stop()
        self.status = 2
        self.loading.status = 5

    
    def requestDeparture(self, ship):
        self.loading.GoTo(AWaitingPos, getattr(aTank, "requestQueue"))
        self.loading.status = 3
        self.loading = None
        self.status = 0      
        
    
    def Update(self):
        self.empty = False
        self.full = False
        
        if self.level > self.maxLevel:
            self.level = self.maxLevel
            self.full = True
        if self.level < 0:
            self.level = 0
            self.empty = True
            
        if not self.empty:
            self.level -= BProduction
        
        if self.status == 0 and len(self.queue)>0:
            self.loading = self.queue.pop(0)
            self.loading.GoTo(BLoadingPos, getattr(self, "requestLoad"))
            self.loading.status = 1
            self.loading.waiting = False
            ship.mooring = True
            self.status = 1
        
        if self.status == 2 and not self.full:
            self.loading.level -= self.transferFlow
            self.level += self.transferFlow
            
            if self.level > self.maxLevel:
                self.loading.level += self.level-self.maxLevel
                self.level = self.maxLevel
            
            if self.loading.level<0:
                self.loading.level += self.transferFlow
                self.level -= self.transferFlow
                if not optimization:
                    print("-----------")
                    print(str(currentTime)+ ": Barco N."+ self.loading.number + " empty")
                self.level += self.loading.level
                self.loading.level = 0
                self.loading.GoTo(BWaitingPos, getattr(bTank, "requestDeparture"))
                self.status = 3
                self.loading.status = 1

        
        self.levels.append(self.level)
        self.queues.append(len(self.queue))
        self.empties.append(self.empty)




ships = []

aTank = ATank(ATankCapacity, ATransferFlow)
bTank = BTank(BTankCapacity, BTransferFlow)
if optimization:
    
    def runSimulation(shipSpawnDelay,amountOfShips):
        global aTank
        global bTank
        global ships
        global currentTime
        ships = []

        aTank.__init__(ATankCapacity, ATransferFlow)
        bTank.__init__(BTankCapacity, BTransferFlow)
        
        for i in range(amountOfShips):
            ships.append(Ship(i+1, journeyTime, mooringTime, (shipSpawnDelay*i)))#((mooringTime*2+journeyTime)/amountOfShips)*i))

        currentTime = 0
        for minute in range(simulationLength):
            for ship in ships:
                ship.Update()
            aTank.Update()
            bTank.Update()
            currentTime += 1
        return [aTank, bTank, ships]

if not optimization:
    aTank = ATank(ATankCapacity, ATransferFlow)
    bTank = BTank(BTankCapacity, BTransferFlow)
    for i in range(amountOfShips):
        ships.append(Ship(i+1, journeyTime, mooringTime, (shipSpawnDelay*i)))#((mooringTime*2+journeyTime)/amountOfShips)*i))

    currentTime = 0
    for minute in range(simulationLength):
        for ship in ships:
            ship.Update()
        aTank.Update()
        bTank.Update()
        currentTime += 1
    
    totalCost = 0
    accum = 0
    for i in aTank.fulls:
        if i:
            accum+=1.
            
            
    totalCost += accum*APlantStoppedCost
    
    accum = 0
    for i in bTank.empties:
        if i:
            accum+=1
            
    totalCost += accum*BPlantStoppedCost
    
    accum = 0
    for ship in ships:
        for i in ship.stoppeds: 
            if i:
                accum+=1
            
    totalCost += accum*shipStoppedCost
    print(totalCost)
"""# Uncomment for ship by ship graphs
for ship in ships:
    data1 = ship.positions
    data2 = ship.levels

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (m)')
    ax1.set_ylabel('Ship position', color=color)

    ax1.plot(data1, color=color)

    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Ship load', color=color)  # we already handled the x-label with ax1
    ax2.plot(data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()"""

def makeLevelPlot(data1, data2, title):

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_title(title)

    color = 'tab:red'
    ax1.set_xlabel('time (m)')
    ax1.set_ylabel('A Tank Level', color=color)
    ax1.plot(data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('B Tank Level', color=color)  # we already handled the x-label with ax1
    ax2.plot(data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    return plt


def makeQueuePlot(data1, data2, title):

    data1 = aTank.queues
    data2 = bTank.queues
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.set_title(title)

    color = 'tab:red'
    ax1.set_xlabel('time (m)')
    ax1.set_ylabel('A Port Queue', color=color)
    ax1.plot(data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('B Port Queue', color=color)  # we already handled the x-label with ax1
    ax2.plot(data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
def makePositionPlot(ships, title):

    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_title(title)

    axes = []
    data_series = []
    tags = []
    colors = []

    #fig.subplots_adjust(right=0.75)

    for ship in ships:
        data_series.append(ship.positions)
        tags.append("Ship Nº" + ship.number)
        colors.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
        axes.append(ax)
    plt.yticks(np.arange(5), ('', 'A Loading', 'A Waiting', 'B Waiting', 'B Loading'))

    for ax, data, color, tag in zip(axes, data_series, colors, tags):
        ax.plot(data,  color=color, label=tag)

    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    plt.show()
def makeTimescalePlot(ships, title):


    data = []

    for ship in ships:
        newStatuses = []
        for i in range(len(ship.statuses)):
            if i%100 == 0:
                newStatuses.append(ship.statuses[i])

        data.append(newStatuses)

    fig = plt.figure(figsize=(300, 30))

    ax = fig.add_subplot(111)
    ax.set_title(title)

    ax.axes.get_yaxis().set_visible(False)
    ax.set_aspect(5)

    def avg(a, b):
        return (a + b) / 2.0

    for y, row in enumerate(data):
        for x, col in enumerate(row):
            x1 = [x, x+1]
            y1 = np.array([y, y])
            y2 = y1+1

            if col == 0:
                plt.fill_between(x1, y1, y2=y2, color='green')

            if col == 1:
                plt.fill_between(x1, y1, y2=y2, color='yellow')

            if col == 2:
                plt.fill_between(x1, y1, y2=y2, color='grey')

            if col == 3:
                plt.fill_between(x1, y1, y2=y2, color='blue')

            if col == 4:
                plt.fill_between(x1, y1, y2=y2, color='red')

            if col == 5:
                plt.fill_between(x1, y1, y2=y2, color='black')


    plt.ylim(len(ships), 0)
    plt.show()
#shipRunningCost = 1
shipStoppedCost = 1000 #euro/minute
APlantStoppedCost = 10000 #euro/minute
BPlantStoppedCost = 10000 #euro/minute

"""
[aTank, bTank, ships]

"""
def runAndAddCosts(spacing, amount, verbose):
    data = runSimulation(spacing, amount)
    totalCost = 0
    accum = 0
    output = []
    for i in data[0].fulls:
        if i:
            accum+=1.
            
    output.append(accum*APlantStoppedCost)        
    totalCost += accum*APlantStoppedCost
    
    accum = 0
    for i in data[1].empties:
        if i:
            accum+=1
    output.append(accum*BPlantStoppedCost)                 
    totalCost += accum*BPlantStoppedCost
    
    accum = 0
    for ship in data[2]:
        for i in ship.stoppeds: 
            if i:
                accum+=1
    output.append(accum*shipStoppedCost)         
    totalCost += accum*shipStoppedCost

    """for ship in data[2]:
        for i in ship.stoppeds: 
            if not i:
                accum+=1
            
    totalCost += accum*shipRunningCost"""
    
    if verbose:
        output.append(totalCost)
        return output
    if not verbose:
        return totalCost

if optimization:
    for i in range(2,8):
        x0 = np.array([10000])
        res = opt.minimize_scalar(runAndAddCosts, args=(i, False), method='brent', tol=10)
        result = runAndAddCosts(res.x, i, True)
        #print(str(i) + " ships: Total Cost: "+ str(result[3]) + "€ A Cost: "+ str(result[0]) + "€, B Cost: "+ str(result[1]) + "€, Ship Cost: "+ str(result[2]) + "€, spacing "+str(np.round(res.x, 2))+ " minutes.")
        makeLevelPlot(aTank.levels, bTank.levels, str(i) + " ships, at an optimized spacing of "+ str(np.round(res.x, 2)) + " minutes").show()
        """    
for sp in range (5000,20000):
        if sp%1000==0:
            print(str(sp)+": "+str(runAndAddCosts(sp, 5)))
            """
    

        
        
    
res = opt.minimize_scalar(runAndAddCosts, args=(5, False), method='brent', tol=10)
result = runAndAddCosts(res.x, 5, True)
title = (str(5) + " ships, at an optimized spacing of "+ str(np.round(res.x, 2)) + " minutes")

makeLevelPlot(aTank.levels, bTank.levels, "Tank levels for " + title)
makeQueuePlot(aTank.queue, bTank.queue, "Tank queues for " + title)
makePositionPlot(ships, "Ship positions for " + title)
makeTimescalePlot(ships, "Timescale for " + title)