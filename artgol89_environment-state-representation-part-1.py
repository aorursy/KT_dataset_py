import math
import numpy as np
import matplotlib.pyplot as plt

class BoardImageRepresentation:

    def __init__(self):
        self.__cmap = {'empty': np.array([0,0,0]), # empty cells

                       'hlt_75_100': np.array([0,0,255]), # halite
                       'hlt_50_75': np.array([0,0,212]),
                       'hlt_25_50': np.array([0,0,170]),
                       'hlt_0_25': np.array([0,0,128]),

                       'player_ship': np.array([255,0,0]), # player units
                       'player_crt_ship_cargo_0_25': np.array([255,0,128]),
                       'player_crt_ship_cargo_25_50': np.array([255,0,170]),
                       'player_crt_ship_cargo_50_75': np.array([255,0,212]),
                       'player_crt_ship_cargo_75_100': np.array([255,0,255]),
                       'player_yard': np.array([128,0,0]),
                       'player_crt_yard': np.array([128,0,128]),

                       'enemy_ship': np.array([0,255,0]), # enemy units
                       'enemy_yard': np.array([0,128,0])}

        
    def represent(self, board):
        gen_view = self.__get_general_view(board)
        highlighted_ships = self.__get_highlighted_ships(board, gen_view)
        highlighted_shipyards = self.__get_highlighted_shipyards(board, gen_view)
        
        board_img = {'general_view': gen_view,
                     'highlighted_ships': highlighted_ships,
                     'highlighted_shipyards': highlighted_shipyards}
        
        board_img = self.__rotate_board_img(board_img)
        board_img = self.__normalize_board_img(board_img)
        
        return board_img
    
    
    def render(self, board_img):
        plt.figure(figsize=(5,5))
        plt.subplot(1,1,1)
        plt.imshow(board_img['general_view'])
        plt.axis('off')
        plt.title(f"General view", fontsize=20)
        plt.show()

        ships_count = len(board_img['highlighted_ships'])
        if ships_count > 0:
            row_count = math.ceil(ships_count / 3)
            plt.figure(figsize=(6*3,5*row_count))
            for i, (ship_id, mtx) in enumerate(board_img['highlighted_ships'].items()):
                ax = plt.subplot(row_count,3,i+1)
                ax.imshow(mtx)
                plt.axis('off')
                plt.title(f"Ship ID: {ship_id}", fontsize=20)        
            plt.show()
            
        shipyards_count = len(board_img['highlighted_shipyards'])
        if shipyards_count > 0:
            row_count = math.ceil(shipyards_count / 3)
            plt.figure(figsize=(6*3,5*row_count))
            for i, (shipyard_id, mtx) in enumerate(board_img['highlighted_shipyards'].items()):
                ax = plt.subplot(row_count,3,i+1)
                ax.imshow(mtx)
                plt.axis('off')
                plt.title(f"Shipyard ID: {shipyard_id}", fontsize=20)        
            plt.show()
    
    
    def __get_general_view(self, board):
        board_size = board.configuration.size
        max_cell_halite = board.configuration.max_cell_halite
        player_id = board.current_player_id

        gen_view = np.zeros((board_size,board_size,3))

        for coords, cell in board.cells.items():    
            if cell.ship is not None:
                role = 'player' if cell.ship.player_id == player_id else 'enemy'
                gen_view[coords] = self.__cmap[f'{role}_ship']

            elif cell.shipyard is not None:
                role = 'player' if cell.shipyard.player_id == player_id else 'enemy'
                gen_view[coords] = self.__cmap[f'{role}_yard']

            elif cell.halite > 0:
                hlt_percent = cell.halite / max_cell_halite * 100
                hlt_interval = self.__get_hlt_percent_interval(hlt_percent)
                gen_view[coords] = self.__cmap[f'hlt_{hlt_interval}']    

        return gen_view

    
    def __get_highlighted_ships(self, board, general_view):
        highlighted_ships = dict()
        
        for ship in board.current_player.ships:
            cargo_interval = self.__get_cargo_percent_interval(ship.halite)
            gen_view_cp = general_view.copy()
            gen_view_cp[ship.position] = self.__cmap[f'player_crt_ship_cargo_{cargo_interval}']
            highlighted_ships[ship.id] = gen_view_cp
            
        return highlighted_ships
    
    
    def __get_highlighted_shipyards(self, board, general_view):
        highlighted_shipyards = dict()
        
        for shipyard in board.current_player.shipyards: 
            gen_view_cp = general_view.copy()
            gen_view_cp[shipyard.position] = self.__cmap['player_crt_yard']
            highlighted_shipyards[shipyard.id] = gen_view_cp
    
        return highlighted_shipyards
    
    
    def __get_hlt_percent_interval(self, hlt_percent):
        interval_dict = {(0,25):'0_25', (25,50):'25_50', (50,75):'50_75', (75,np.inf):'75_100'}
        for interval in interval_dict.keys():
            if interval[0] < hlt_percent <= interval[1]:
                return interval_dict[interval]
    
    
    def __get_cargo_percent_interval(self, cargo_amount):
        interval_dict = {(0,250):'0_25', (250,500):'25_50', (500,1000):'50_75', (1000,np.inf):'75_100'}
        for interval in interval_dict.keys():
            if interval[0] <= cargo_amount < interval[1]:
                return interval_dict[interval]
    
    
    def __apply_func_to_board_img(self, board_img, func):
        board_img['general_view'] = func(board_img['general_view'])
        
        for ship_id, mtx in board_img['highlighted_ships'].items():
            board_img['highlighted_ships'][ship_id] = func(mtx)
            
        for shipyard_id, mtx in board_img['highlighted_shipyards'].items():
            board_img['highlighted_shipyards'][shipyard_id] = func(mtx)
        
        return board_img
         
        
    def __normalize_board_img(self, board_img):
        func = lambda x: np.round(x / 255.0, 3)
        return self.__apply_func_to_board_img(board_img, func)
    
    
    def __rotate_board_img(self, board_img):
        func = lambda x: np.rot90(x)
        return self.__apply_func_to_board_img(board_img, func)
from numpy.random import choice

def test_agent(obs, config):    
    halite, shipyards, ships = obs['players'][obs['player']]
    
    actions = {}

    if len(ships) == 0 and len(shipyards) > 0:
        shipyard_id = list(shipyards.keys())[0]
        actions[shipyard_id] = 'SPAWN'   
    elif len(ships) > 0 and len(shipyards) == 0:
        ship_id = list(ships.keys())[0]
        actions[ship_id] = 'CONVERT'
    else:    
        for ship_id in ships.keys():
            ship_action = choice(["NORTH", "SOUTH", "EAST", "WEST", "CONVERT", None], 1, 
                                 p=[0.2, 0.2, 0.2, 0.2, 0.05, 0.15])[0]
            if ship_action is not None:
                actions[ship_id] = ship_action
                
        for shipyard_id in shipyards.keys():
            shipyard_action = choice(["SPAWN", None], 1, 
                                     p=[0.1, 0.9])[0]
            if shipyard_action is not None:
                actions[shipyard_id] = shipyard_action

    return actions
!pip install kaggle-environments --upgrade
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
env = make("halite", debug=True)
env_config = env.configuration

training_env = env.train([None, "random", "random", "random"])
obs = training_env.reset()
board = Board(obs, env_config)
print(board)
env.render(mode="ipython", width=400, height=400)
board_img_repr = BoardImageRepresentation()
board_img = board_img_repr.represent(board)
board_img_repr.render(board_img)
game_step_count = 25
for i in range(game_step_count):
    actions = test_agent(obs, env_config)  
    obs, reward, done, info = training_env.step(actions)

env.render(mode="ipython", width=400, height=400)
board = Board(obs, env_config)

board_img_repr = BoardImageRepresentation()
board_img = board_img_repr.represent(board)
board_img_repr.render(board_img)
