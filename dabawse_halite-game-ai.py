import os, random, numpy, pandas, matplotlib.pyplot as plt

import numpy as np

from kaggle_environments.envs.halite.helpers import *

from kaggle_environments import make, evaluate

from random import randint
def agent(obs, config):

    board = Board(obs, config)

    me = board.current_player

    directions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]

    

    if me.halite >= 500 and len(me.shipyards) == 1 and len(me.ships) == 0:

        for shipyard in me.shipyards:

            if me.halite >= 500 and shipyard.cell.ship == None:

                me.shipyards[0].next_action = ShipyardAction.SPAWN

                return me.next_actions

        

    for ship in me.ships:

        

        if len(me.shipyards)<1 and ship.cell.shipyard == None:

            me.ships[0].next_action = ShipAction.CONVERT

        else:

            h_avail = [ship.cell.north.halite, ship.cell.east.halite, ship.cell.south.halite, ship.cell.west.halite]

            avail = [ship.cell.north.ship, ship.cell.east.ship, ship.cell.south.ship, ship.cell.west.ship]



            i=0

            for a in avail:

                if a!=None:

                    h_avail[i] = -1000

                i+=1

            best = max(h_avail)

            

            if ship.halite >= 500:

                shipyard_poses = np.array([])

                for shipyard in me.shipyards:

                     shipyard_poses = np.append(shipyard_poses, shipyard.position)

                shipyard_poses = shipyard_poses.reshape(-1, 2)

                

                for i in shipyard_poses:

                    poses = np.array([])

                    for pos in abs(shipyard_poses - ship.position):

                        poses = np.append(poses, np.sum(pos))    

                shipyard_pos = shipyard_poses[list(poses).index(min(poses))]

                    

                if shipyard_pos[1] > ship.position[1]:

                    ship.next_action = ShipAction.NORTH

                elif shipyard_pos[1] < ship.position[1]:

                    ship.next_action = ShipAction.SOUTH

                else:

                    if shipyard_pos[0] > ship.position[0]:

                        ship.next_action = ShipAction.EAST

                    elif shipyard_pos[0] < ship.position[0]:

                        ship.next_action = ShipAction.WEST

                return me.next_actions   

                

            elif best > ship.cell.halite and ship.cell.halite < 100 and ship.cell.shipyard == None:

                hal_pos = h_avail.index(best)

                next_dir = directions[hal_pos]

                ship.next_action = next_dir

                return me.next_actions

            elif ship.cell.halite >= 10:

                return

            elif best == ship.cell.halite and best != 0 and ship.cell.halite != 0:

                return

            else:

                if None in avail:

                    while True:

                        rand = randint(0, 3)

                        if avail[rand] == None:

                            ship.next_action = directions[rand]

                            return me.next_actions

                else:

                    print('cannot move')

                    return

            

    return me.next_actions
env = make('halite', debug=True)

env.run([agent, 'random', 'random', 'random'])

env.render(mode='ipython', width=800, height=600)