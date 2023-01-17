from collections import deque

import numpy as np


MAP_SIZE = 21
CONVERT_COST = 500
SPAWN_COST = 500


class Helper:
    """This class is a template for agent helpers.
    """

    def __init__(self):
        self.state_shape = None
        self.action_shape = None
        self.memory = {}

    def convert_obs(self, uid, obs):
        return np.array([0])

    def convert_action(self, uid, obs, action):
        return 'NONE'

    def reset(self):
        pass

    def step(self):
        pass


class BasicShipyardHelper(Helper):
    """This class provides basic provisions for helping a shipyard agent.
    """

    def __init__(self):
        # steps, phalite, occupied, num ships
        self.state_shape = (4,)
        # None, spawn
        self.action_shape = (2,)

        self.reset()

    def convert_obs(self, uid, obs):
        phalite, shipyards, ships = obs.players[obs.player]
        occupied = shipyards[uid] in [x[0] for x in ships.values()]
        occupied = 1 if occupied else 0
        return np.array([obs.step, phalite, occupied, len(ships)])

    def convert_action(self, uid, obs, action):
        phalite, shipyards, ships = obs.players[obs.player]
        occupied = shipyards[uid] in [x[0] for x in ships.values()]
        if action == 1 and phalite >= SPAWN_COST and not occupied:
            return 'SPAWN'
        return 'NONE'


class BasicShipHelper(Helper):
    """This class provides basic provisions for helping a ship agent.
    """

    def __init__(self, stack=2, radius=5):
        self.stack = stack
        self.radius = radius
        self.frame_state_shape = (self.radius*2+1, self.radius*2+1, 2)
        self.state_shape = ((self.radius*2+1)**2 * 2 * self.stack,)
        self.action_shape = (6,)
        self.reset()

    def convert_obs(self, uid, obs):
        phalite, shipyards, ships = obs.players[obs.player]
        if uid not in self.state:
            self.state[uid] = deque(maxlen=self.stack)
            for _ in range(self.stack-1):
                self.state[uid].append(np.zeros(self.frame_state_shape))
        if uid in ships:
            entities_map = np.zeros((MAP_SIZE, MAP_SIZE))
            for player, (_, sy, s) in enumerate(obs.players):
                for shipyard in sy.values():
                    if obs.player == player:
                        entities_map[shipyard // MAP_SIZE,
                                     shipyard % MAP_SIZE] += 2
                    else:
                        entities_map[shipyard // MAP_SIZE,
                                     shipyard % MAP_SIZE] -= 2
                for ship in s.values():
                    if obs.player == player:
                        entities_map[ship[0] // MAP_SIZE,
                                     ship[0] % MAP_SIZE] += 1
                    else:
                        entities_map[ship[0] // MAP_SIZE,
                                     ship[0] % MAP_SIZE] -= 1
            halite_map = np.reshape(obs.halite, (MAP_SIZE, MAP_SIZE))
        
            state_map = np.stack([halite_map, entities_map], axis=2)
            state_map = np.tile(state_map, (3, 3, 1))

            spos, shalite = ships[uid]
            y = spos // MAP_SIZE + MAP_SIZE
            x = spos % MAP_SIZE + MAP_SIZE
            r = self.radius
            self.state[uid].append(state_map[y-r:y+r+1, x-r:x+r+1])
            return np.dstack(self.state[uid]).flatten()
        self.state[uid].append(np.zeros(self.frame_state_shape))        
        return np.dstack(self.state[uid]).flatten()

    def convert_action(self, uid, obs, action):
        phalite, shipyards, ships = obs.players[obs.player]
        pos, shalite = ships[uid]
        sy, sx = pos // MAP_SIZE, pos % MAP_SIZE

        if action == 0:
            return 'NONE'
        if action == 1:
            if phalite > CONVERT_COST and pos not in shipyards.values():
                return 'CONVERT'
            return 'NONE'
        if action == 2:
            return 'NORTH'
        if action == 3:
            return 'SOUTH'
        if action == 4:
            return 'WEST'
        if action == 5:
            return 'EAST'

    def reset(self):
        self.state = {}
# Code below is incomplete
import os
from copy import deepcopy

from kaggle_environments import make


class HaliteEnv:
    def __init__(self, opponents, shipyard_helper, ship_helper,
                 replay_save_dir='', **kwargs):
        self.shipyard_helper = shipyard_helper
        self.ship_helper = ship_helper
        self.env = make('halite', **kwargs)
        self.trainer = self.env.train([None, *opponents])
        self.replay_save_dir = replay_save_dir

    def update_obs(self, uid, action):
        """Simulate environment step forward and update observation"""
        pass
    
    def reset(self):
        """Reset trainner environment"""
        self.obs = deepcopy(self.trainer.reset())
        return self.obs
    
    def step(self, actions):
        """Step forward in actual environment"""
        self.obs, reward, terminal, info = self.trainer.step(actions)
        self.obs = deepcopy(self.obs)
        return self.obs, reward, terminal
    
    def play_episdoe(self, shipyard_agent, ship_agent, max_steps=400):
        """Play one episode"""
        self.shipyard_helper.reset()
        self.ship_helper.reset()
        obs = self.reset()
        states = {}
        total_reward = 0
        for step in range(1, max_steps + 1):
            actions = {}
            phalite, shipyards, ships = obs.players[obs.player]
            init_shipyards = deepcopy(shipyards)
            init_ships = deepcopy(ships)
            
            for uid in init_shipyards.keys():
                # implementations vary
                # convert observation
                # select action
                # convert action
                # update simulated environment
                # Add experience/memory
                pass
            
            for uid in init_ships.keys():
                # implementations vary
                # convert observation
                # select action
                # convert action
                # update simulated environment
                # Add experience/memory
                pass
                
            obs, ep_reward, ep_terminal = self.step(actions)
            self.shipyard_helper.step()
            self.ship_helper.step()
            
            if ep_terminal:
                break

        # For sparse rewards we revise all ships and
        # shipyards last memory to be terminal and have the same reward
        # Important to note that training ships and shipyards at the same
        # time may produce poor results as they will get a reward that may
        # or may not represent that agents effectiveness.
        phalite, shipyards, ships = obs.players[obs.player]
        if obs.step < 399:
            if len(shipyards) == len(ships) == 0:
                total_reward = -1.0
            else:
                total_reward = 1.0
        else:
            halites = [halite for (halite, _, _) in obs.players]
            max_halite = np.max(halites)
            if max_halite == 0:
                max_halite = 1
            del halites[obs.player]
            total_reward = (phalite - np.max(halites)) / max_halite
        
        # Revise the last memory of each ship/shipyard
        # reward to be total_reward
        pass
    
        with open(os.path.join(self.replay_save_dir, 'out.html'), 'w') as file:
            file.write(self.env.render(mode='html'))
        
        