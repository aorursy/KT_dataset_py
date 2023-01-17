%%writefile submission.py



# Imports helper functions

from kaggle_environments.envs.halite.helpers import *

import numpy as np

from math import sqrt

import random

#import logging



# Returns best direction to move from one position (fromPos) to another (toPos)

# Example: If I'm at pos 0 and want to get to pos 55, which direction should I choose?

def getDirTo(fromPos, toPos, size, possList=[True, True, True, True]):

    fromX, fromY = fromPos[0], fromPos[1]

    toX, toY = toPos[0], toPos[1]

    if abs(fromX-toX) > size / 2:

        fromX += size

    if abs(fromY-toY) > size / 2:

        fromY += size

    if fromY < toY and possList[0]: return ShipAction.NORTH

    if fromY > toY and possList[1]: return ShipAction.SOUTH

    if fromX < toX and possList[2]: return ShipAction.EAST

    if fromX > toX and possList[3]: return ShipAction.WEST

    return random.choice([ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST])



def getSafeDirs(board, cell, me):

    neighbors = [cell.north, cell.south, cell.east, cell.west]

    return_val = []

    for neighbor in neighbors:

        noShip = neighbor.ship is None

        goodShipyard = neighbor.shipyard is None or neighbor.shipyard.player == me

        if noShip and goodShipyard:

            isSafe = True

        else:

            isSafe = False

        return_val.append(isSafe)

    return return_val



def getListOfPoints(start, end):

    if start.x == end.x:

        return_val = []

        for i in range(start.y, end.y+1):

            return_val.append(Point(start.x, i))

        return return_val

    if start.y == end.y:

        return_val = []

        for i in range(start.x, end.x+1):

            return_val.append(Point(i, start.y))

        return return_val

    return_val = []

    for i in range(start.x, end.x+1):

        return_val.append(getListOfPoints(Point(i, start.y), Point(i, end.y)))

    return return_val



def scanForHalite(board, center, radius, size, badPoints=[], dontcarenum=0):

    start = Point(center.x-radius, center.y-radius)

    end = Point(center.x+radius, center.y+radius)

    coordsToScan = np.reshape(getListOfPoints(start, end),((end.x-start.x+1)*(end.y-start.y+1),2))

    pointsToScan = []

    for coord in coordsToScan:

        newx = coord[0]

        newy = coord[1]

        if newx < 0:

            newx += size

        if newx > size-1:

            newx -= size

        if newy < 0:

            newy += size

        if newy > size-1:

            newy -= size

        pointsToScan.append(Point(newx, newy))

    max_so_far = pointsToScan[0]

    for point in pointsToScan:

        if board.cells[point].halite-dontcarenum > board.cells[max_so_far].halite and not point in badPoints:

            max_so_far = point

    return max_so_far

        

def distBetween(pointA, pointB):

    x = abs(pointA.x-pointB.x)

    y = abs(pointA.y-pointB.y)

    return sqrt(x**2 + y**2)

# Directions a ship can move

directions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST, None]



# Will keep track of whether a ship is collecting halite or carrying cargo to a shipyard

ship_states = {}

ship_targets = {}



# Returns the commands we send to our ships and shipyards

def agent(obs, config):

    #logging.basicConfig(level=logging.DEBUG)

    size = config.size

    board = Board(obs, config)

    me = board.current_player

    #logging.debug('starting!')

    

    #logging.debug(f'Round {board.step}')

    # If there are no ships, use first shipyard to spawn a ship.

    if len(me.ships) <= 2*len(me.shipyards) and len(me.shipyards) > 0 and me.halite >= 1000:

        #logging.debug('making new ship')

        me.shipyards[0].next_action = ShipyardAction.SPAWN



    # If there are no shipyards, convert first ship into shipyard.

    

        #logging.debug('shipyarding')

    if len(me.ships) > 0 and len(me.shipyards) < 1:

        me.ships[0].next_action = ShipAction.CONVERT

    sideNeighbors = []

    canConvert = False if me.halite > 2000 else False

    for ship in me.ships:

        smallestDist = None

        for shipyard in me.shipyards:

            if smallestDist is None or distBetween(ship.position, shipyard.position) < smallestDist:

                smallestDist = distBetween(ship.position, shipyard.position)

        if smallestDist is not None and smallestDist >=10 and canConvert:

            ship.next_action = ShipAction.CONVERT

            canConvert = False

        if ship.next_action == None:

            sideNeighbors = [ship.cell.north, ship.cell.east, ship.cell.south, ship.cell.west]

            try:

                if ship.halite > 500:

                    ship_targets[ship.id] = me.shipyards[0].position

                    ship_states[ship.id] = "GOTARGET"

                elif ship.cell.halite < 50 and ship_states[ship.id] == "COLLECT":

                    ship_states[ship.id] = "FINDNEW"

            except KeyError:

                ship_states[ship.id] = "FINDNEW"

            except IndexError:

                ship_states[ship.id] = None

                

            if ship_states[ship.id] == "FINDNEW":

                # move to the adjacent square containing the most halite

                targets = list(ship_targets.values())

                pointWithMost = scanForHalite(board, ship.position, 2, size, targets, 50)

                ship_targets[ship.id] = pointWithMost

                if pointWithMost == ship.position:

                    ship_states[ship.id] = "COLLECT"

                else:

                    ship_states[ship.id] = "GOTARGET"

            if ship_states[ship.id] == 'GOTARGET':

                ship.next_action = getDirTo(ship.position, ship_targets[ship.id], size, getSafeDirs(board, ship.cell, me))

                if ship.position == ship_targets[ship.id]:

                    ship_states[ship.id] = 'COLLECT'

            if ship_states[ship.id] == "COLLECT":

                ship.next_action == None

                

    return me.next_actions
from kaggle_environments import make

env = make("halite", debug=True)

env.run(["submission.py", "random", "random", "random"])

env.render(mode="ipython", width=800, height=600)