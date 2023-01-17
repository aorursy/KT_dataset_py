# Make sure we have the latest kaggle-environments installed
!pip install kaggle-environments --upgrade
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *

# Create a test environment for use later
board_size = 5
environment = make("halite", configuration={"size": board_size, "startingHalite": 1000})
agent_count = 2
environment.reset(agent_count)
state = environment.state[0]
board = Board(state.observation, environment.configuration)
print(board)
print(f"Player Ids: {[player.id for player in board.players.values()]}")
print(f"Ship Ids: {[ship.id for ship in board.ships.values()]}")
# Note there are currently no shipyards because our Board just initialized
print(f"Shiyard Ids: {[shipyard.id for shipyard in board.shipyards.values()]}")
assert len(board.cells) == board_size * board_size
point = Point(3, -4)

# Points are tuples
assert isinstance(point, tuple)
# Points have x and y getters (no setters)
assert point.x == 3
assert point.y == -4
# Points implement several common operators
assert point == Point(3, -4)
assert abs(point) == Point(3, 4)
assert -point == Point(-3, 4)
assert point + point == (6, -8)
assert point - point == Point(0, 0)
assert point * 3 == Point(9, -12)
# Point supports floordiv but not div since x and y are ints not floats
assert point // 2 == Point(1, -2)
assert point % 3 == Point(0, 2)
# Prints like a tuple
print(point)
print(board[point])
print([action.name for action in ShipAction])
print([action.name for action in ShipyardAction])

# Grab a ship to test with
ship = next(iter(board.ships.values()))
print(f"Initial action: {ship.next_action}")
ship.next_action = ShipAction.NORTH
print(f"New action: {ship.next_action}")
print(board)
board = board.next()
print(board)

# Let's make sure we moved north!
next_ship = board.ships[ship.id]
assert next_ship.position - ship.position == ShipAction.NORTH.to_point()
# We'll use this in the next cell
ship = next_ship

# What happens if we call board.next()?
print(board.next())
ship.next_action = ShipAction.CONVERT
board = board.next()
print(board)
shipyard = board[ship.position].shipyard
shipyard.next_action = ShipyardAction.SPAWN
board = board.next()
print(board)
for ship in board.ships.values():
    ship.next_action = ShipAction.SOUTH
board = board.next()
print(board)
current_player = board.current_player
for ship in current_player.ships:
    ship.next_action = ShipAction.SOUTH
print(current_player.next_actions)
def move_ships_north_agent(observation, configuration):
    board = Board(observation, configuration)
    current_player = board.current_player
    for ship in current_player.ships:
        ship.next_action = ShipAction.NORTH
    return current_player.next_actions

environment.reset(agent_count)
environment.run([move_ships_north_agent, "random"])
environment.render(mode="ipython", width=500, height=450)
@board_agent
def move_ships_north_agent(board):
    for ship in board.current_player.ships:
        ship.next_action = ShipAction.NORTH

environment.reset(agent_count)
environment.run([move_ships_north_agent, "random"])
environment.render(mode="ipython", width=500, height=450)
first_player_actions = {'0-1': 'CONVERT'}
second_player_actions = {'0-2': 'NORTH'}

actions = [first_player_actions, second_player_actions]
board = Board(state.observation, environment.configuration, actions)
print(board)
print(board.next())