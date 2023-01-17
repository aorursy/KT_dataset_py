import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# 1D implementation
class TicTacToe:
    def __init__(self, board=None, board_size=3, players=None, first_player=None):
        self.board_size = board_size
        if board is None:
            self.board = np.zeros((self.board_size ** 2), dtype=int)
        else:
            # Make sure the board is of 1D array
            self.board = np.array(board)

        self.EMPTY = 0
        if players is None:
            self.players = {self.EMPTY: '?', 1: 'X', 2: 'O'}
        else:
            self.players = players

        self.playerCount = len(self.players) - 1
        self.history = list()
        if first_player is None:
            self.player = 1
        else:
            self.player = first_player

    def visualize(self):
        print(f'Round {len(self.history) + 1}:', end='')
        for i, c in enumerate(self.board):
            if i % self.board_size == 0:
                print()
            print(self.players[c], end='\t')
        print()

    # The pos is 0-based
    def placeMove(self, pos):
        if self.isAvailable(pos):
            self.board[pos] = self.player
            self.history.append([pos, self.player])
            self.player = self.getNextPlayer()
        return self.checkGameEnds()

    def isAvailable(self, pos):
        if 0 <= pos < self.board_size ** 2:
            return self.board[pos] == self.EMPTY
        return False

    def getCurrentPlayer(self):
        return self.player

    def getNextPlayer(self):
        if self.player == self.playerCount:
            self.player = 1
        else:
            self.player += 1
        return self.player

    def checkGameEnds(self):
        # -1 means no one wins and all spaces has used up
        # 0 means the game hasn't ended yet
        # If a player won, it'll return the corresponding integer that represents the player
        reshaped_board = self.board.reshape(self.board_size, self.board_size)
        for row in reshaped_board:
            if len(set(row)) == 1 and row[0] != 0:
                return row[0]

        for column in reshaped_board.transpose():
            if len(set(column)) == 1 and column[0] != 0:
                return column[0]

        diagonal = [reshaped_board[i][i] for i in range(self.board_size)]
        if len(set(diagonal)) == 1 and diagonal[0] != 0:
            return diagonal[0]

        inverted_diagonal = [np.flip(reshaped_board, 0)[i][i] for i in range(self.board_size)]
        if len(set(inverted_diagonal)) == 1 and inverted_diagonal[0] != 0:
            return inverted_diagonal[0]

        if self.EMPTY in set(reshaped_board.flatten().tolist()):
            return 0
        else:
            return -1

    def explain(self):
        state = self.checkGameEnds()
        if state == -1:
            print('No one wins and all spaces has used up.')
        elif state == 0:
            print('The game hasn\'t ended yet.')
        else:
            print(f'Player {state} ({self.players[state]}) won.')
class Environment(TicTacToe):
    def __init__(self, first_player=1, debug=False, **kwargs):
        '''
        Initialize the environment.
        Input:
        first_player <int>: Representing the starting player.
        debug <bool>: Default to False. If it was set to True, it will display debugging information.
        kwargs <dict>: A dictionary for holding keyword arguments being passed.
        '''
        self.first_player = first_player
        self.debug = debug
        self.kwargs = kwargs
        if self.kwargs == {}:
            super().__init__()             # Initializes the game
        else:
            super().__init__(self.kwargs)  # Initializes the game with keyword arguments

    @staticmethod
    def getHash(board):
        '''
        Get hash of the current board.
        Input:
        board <list>: A board which is of list type. Eg: [1, 2, 1, 2, 1, 2, 1, 2, 1].
        
        Return:
        <str>: The hash of the current state of the game.
        '''
        return str(board.tolist())

    def reset(self, first_player=None):
        '''
        For resetting the environment after the game has finished. Keyword arguments being passed during initialization of the environment will be passed again for re-initialization.
        Input:
        first_player <int>: Representing the starting player.
        
        Return:
        <str>: Hash of a board with only empty spaces.
        '''
        if self.kwargs == {}:
            super().__init__()             # Initializes the game
        else:
            super().__init__(self.kwargs)  # Initializes the game with keyword arguments
        if first_player is not None:
            self.player = first_player
        return self.getHash(self.board)

    def render(self):
        ''' Renders the current environment. '''
        self.visualize()

    def getAvailableMoves(self):
        ''' Get available moves of the current board. '''
        return np.where(self.board == self.EMPTY)[0]

    def action_space(self):
        ''' The action space of the current environment. A fancy wrap for consistency with Gym API. '''
        return self.getAvailableMoves()

    @staticmethod
    def getReward(state, maximized_player):
        '''
        Get the reward given the game state and the player to be maximized for his reward.
        Input:
        state <int>: The game state of the game. Possible values: -1: Tie game; 0: The game hasn't ended yet; 1: Player 1 won; 2: Player 2 won, ...
        maximized_player <int>: The player to be maximized for his reward.

        Return:
        <int>: Representing the reward in different situations.
        '''
        if state == -1:                  # No one wins
            return 0
        elif state == 0:                 # The game hasn't ended yet
            return 0
        elif state == maximized_player:  # The player which is to be maximized won
            return 1
        else:                            # Other players won
            return -1

    def step(self, action, player, maximized_player):
        '''
        Input:
        action <int>: The position of the move to be placed. It is 0-based, with range [0.. <board_size>].
        player <int>: The integer that represents the player. Eg 1 for player 1, 2 for player 2, ...
        maximized_player <int>: The player to be maximized for his reward.

        Return:
        observation <str>: The hash of the board before the move
        reward <int>: The reward due to the move
        game_state <int>: The game state of the game. Possible values: -1: Tie game; 0: The game hasn't ended yet; 1: Player 1 won; 2: Player 2 won, ...
        info <func>: For explaining the current situation of the game.
        '''
        if self.debug:
            print('Player:', player, 'Action:', action, 'Action space:', self.action_space())
        game_state = self.placeMove(action)
        observation = self.getHash(self.board)
        reward = self.getReward(game_state, maximized_player)
        if self.debug:
            print('Maximized Player:', maximized_player, 'Game State:', game_state, 'Reward:', reward, 'Observation:', observation)
            self.visualize()
            self.explain()
            input()
        info = self.explain
        return observation, reward, game_state, info

    def close(self):
        ''' For shutting down the environment. A fancy wrap for consistency with Gym API. '''
        self.game = None


class Agent:
    def __init__(self, debug=False):
        self.debug = debug
        pass

    def getAction(self, observation, action_space):
        '''
        To get action given the observation of the environment and the action space. As this is the most basic form of the agent, it will just return random actions.
        Input:
        observation <str>: The hash of the current state of the game.
        action_space <list>: A list of integers indicating available moves of the current state of the game.
        
        Return:
        action <int>: The move to be placed on the board.
        '''
        action = np.random.choice(action_space)
        return action

    def train(self, experience, f):
        ''' To train the agent. As this is the most basic form of the agent, it could not be trained. '''
        if self.debug:
            print('The agent is unable to learn.')
            input()
        pass


class QAgent(Agent):
    def __init__(self, discount_rate=0.9, learning_rate=1, epsilon=1, decay_rate=0.9999, debug=False):
        '''
        discount_rate <float>: To discount the reward if it takes more turns to get the reward. Eg if the reward is available after 2 turns, reward for the current move will be <reward> * discount_rate ** 2.
        learning_rate <float>: To specify the speed of learning from moves given rewards. Since this is a deterministic game, learning_rate could be set to 1.
        epsilon <float>: Representing the decay of trade-offs between exploration and exploitation.
        debug <bool>: Default to False. If it was set to True, it will display debugging information.
        '''
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.q_table = {}
        self.debug = debug

    @staticmethod
    def select_argmax(dictionary):
        '''
        Returns the move that maximizes the reward. If there are several moves with the maximum reward, it will randomly choose one in these moves.
        Input:
        dictionary <dict>: A dictionary that contains available moves and their corresponding rewards.
        
        Return:
        <int>: The action to be taken by the QAgent.
        '''
        if len(dictionary) == 0:
            raise ValueError('The length of the dictionary should not be zero!')
        max_q_value = max(dictionary.values())
        choices = []
        for key in dictionary:
            if dictionary[key] == max_q_value:
                choices.append(key)
        return np.random.choice(choices)

    @staticmethod
    def select_argmin(dictionary):
        '''
        Returns the move that minimizes the reward. If there are several moves with the minimum reward, it will randomly choose one in these moves.
        Input:
        dictionary <dict>: A dictionary that contains available moves and their corresponding rewards.
        
        Return:
        <int>: The exploitation (greedy) action to be taken by the QAgent.
        '''
        if len(dictionary) == 0:
            raise ValueError('The length of the dictionary should not be zero!')
        min_q_value = min(dictionary.values())
        choices = []
        for key in dictionary:
            if dictionary[key] == min_q_value:
                choices.append(key)
        return np.random.choice(choices)

    def getActionMax(self, observation, action_space):
        '''
        To get the action that maximizes the reward. It will be either an exploration or exploitation action, subject to the epsilon.
        Input:
        observation <str>: The hash of the current state of the game.
        action_space <list>: A list of integers indicating available moves of the current state of the game.
        
        Return:
        <int>: An action to be taken by the QAgent.
        '''
        action_random = super().getAction(observation, action_space)
        q_value_dict = {}
        for action in action_space:
            q_value_dict[action] = self.q_table.get(observation, {}).get(action, 0)
        action_greedy = self.select_argmax(q_value_dict)
        if self.debug:
            print('Action random:', action_random, 'Action greedy:', action_greedy)
        return action_random if np.random.random() < self.epsilon else action_greedy

    def getActionMin(self, observation, action_space):
        '''
        To get the action that minimizes the reward. It will be either an exploration or exploitation action, subject to the epsilon.
        Input:
        observation <str>: The hash of the current state of the game.
        action_space <list>: A list of integers indicating available moves of the current state of the game.
        
        Return:
        <int>: An action to be taken by the QAgent.
        '''
        action_random = super().getAction(observation, action_space)
        q_value_dict = {}
        for action in action_space:
            q_value_dict[action] = self.q_table.get(observation, {}).get(action, 0)
        action_greedy = self.select_argmin(q_value_dict)
        if self.debug:
            print('Action random:', action_random, 'action greedy:', action_greedy)
        return action_random if np.random.random() < self.epsilon else action_greedy

    def train(self, experience, f):
        '''
        To train the QAgent, given experience replay and a function representing whether to maximize or minimize the action.
        Input:
        experience <tuple>: The experiene replay of the game. It could be decomposed into five variables: observation, action, next_observation, reward, game_state.
        f <func>: A function representing whether to maximize or minimize the action of the player.
        
        Important variables:
        observation <str>: The hash of the board before the move
        action <int>: The position of the move to be placed. It is 0-based, with range [0.. <board_size>].
        next_observation <str>: The hash of the board after the move
        reward <int>: The reward due to the move
        game_state <int>: The game state of the game. Possible values: -1: Tie game; 0: The game hasn't ended yet; 1: Player 1 won; 2: Player 2 won, ...
        current_state_q_value: The Q-value of the current state. It is retrieved from the QAgent's Q-table.
        next_state_best_q_value: The predicted best Q-value of the next state. Since this is a minimax game, its objective is to minimize other players' rewards while maximizing self reward. It is retrieved from the QAgent's Q-table.
        '''
        observation, action, next_observation, reward, game_state = experience

        if self.q_table.get(observation) is None:         # Create a new entry for the q_table with the newly observed state
            self.q_table[observation] = {action: 0}       # Initialize the q_value of the action to 0

        if self.q_table[observation].get(action) is None: # What if such observation exists, but such action does not exist?
            self.q_table[observation][action] = 0         # Initialize the q_value of the action to 0

        if self.debug:
            print('observation:', observation)
            print('action:', action)
            print('self.q_table[observation]:', self.q_table[observation])
            print('current_state_q_value:', self.q_table[observation][action])
            print('next_state_best_q_value:', f(self.q_table.get(next_observation, {0: 0}).values()))

        next_state_best_q_value = f(self.q_table.get(next_observation, {0: 0}).values())
        current_state_q_value = self.q_table[observation][action]
        if current_state_q_value != 1:
            self.q_table[observation][action] = current_state_q_value + self.learning_rate * (reward + self.discount_rate * next_state_best_q_value - current_state_q_value)

        if self.debug:
            print(f'self.q_table[observation][action]: {self.q_table[observation][action]} = {current_state_q_value} + {self.learning_rate} * ({reward} + {self.discount_rate} * {next_state_best_q_value} - {current_state_q_value})')
            input()

        if game_state:
            self.epsilon *= self.decay_rate
def visualize(board, board_size=3, players={0: '?', 1: 'X', 2: 'O'}):
    '''
    For fast visualization during debugging.
    Input:
    board <str>: Expected to be the board hash. Eg: '[1, 2, 1, 2, 1, 2, 1, 2, 1]'.
    board_size <int>: Default to be 3. If you are using a board with different size, remember to specify this.
    players <dict>: A dictionary for mapping numbers to symbols for better readability of the game board.
    '''
    for i, c in enumerate([int(x) for x in board[1:-1].split(',')]):
        if i % board_size == 0:
            print()
        print(players[c], end='\t')
    print()

def collect_statistics(stat, result):
    '''
    stat <list>: a list that stores the result of every game play.
    result <int>: the game state that indicates the result of the game.
    '''
    stat.append(result)
    return stat
env = Environment(debug=False)
random_agent = Agent(debug=False)
episode = 50000
stat = []

start_time = time.time()
for i_episode in range(episode):
    observation = env.reset()
    i = 0
    for t in range(100):
        player = env.getCurrentPlayer()
        if player == 1:
            action = random_agent.getAction(observation, env.getAvailableMoves())
            next_observation, reward, game_state, info = env.step(action, player, 1)
            random_agent.train(None, None)
        elif player == 2:
            action = random_agent.getAction(observation, env.getAvailableMoves())
            next_observation, reward, game_state, info = env.step(action, player, 1)
            random_agent.train(None, None)
        observation = next_observation
        if game_state != 0:
            stat = collect_statistics(stat, env.checkGameEnds())
            if (i_episode + 1) % 10000 == 0:
                print(f'Episode {i_episode + 1}, {time.time() - start_time:.2f}s elapsed')
                start_time = time.time()
            break
    env.close()
# First decode to three columns
summary = pd.get_dummies(stat)
draw = summary[-1]
agent_1 = summary[1]
agent_2 = summary[2]

# Then apply rolling(100).mean() to each of them
# Finally plot them out
interval = 100  # Take samples to calculate game outcomes per <interval>
plt.plot(draw.rolling(interval).mean()[::interval], label='Draw')
plt.plot(agent_1.rolling(interval).mean()[::interval], label='Agent 1 Win')
plt.plot(agent_2.rolling(interval).mean()[::interval], label='Agent 2 Win')
plt.legend()
plt.title(f'{episode} Games Played by Random Agent vs Random Agent')
plt.xlabel('Number of Games Played')
plt.ylabel('Rate')
plt.show()
# Now we set epsilon to 0, to stop exploration and only retain expolitation.
# Let it play 10000 games we check the result of the game.
env.debug = False
random_agent.debug = False
episode = 10000
test_stat = []

start_time = time.time()
for i_episode in range(episode):
    first_player = 1
    observation = env.reset(first_player=first_player)
    i = 0
    for t in range(100):
        player = env.getCurrentPlayer()
        if player == 1:
            action = random_agent.getAction(observation, env.getAvailableMoves())
            next_observation, reward, game_state, info = env.step(action, player, 1)
        elif player == 2:
            action = random_agent.getAction(observation, env.getAvailableMoves())
            next_observation, reward, game_state, info = env.step(action, player, 1)
        observation = next_observation
        if game_state != 0:
            test_stat = collect_statistics(test_stat, env.checkGameEnds())
            if (i_episode + 1) % 10000 == 0:
                print(f'Episode {i_episode + 1}, {time.time() - start_time:.2f}s elapsed')
                start_time = time.time()
            break
    env.close()

# First decode to three columns
test_summary = pd.get_dummies(test_stat)
draw = test_summary.get(-1, pd.Series(index=test_stat, name=-1))
agent_1 = test_summary.get(1, pd.Series([0] * len(test_stat), name=1))
agent_2 = test_summary.get(2, pd.Series([0] * len(test_stat), name=2))

# Then apply rolling(100).mean() to each of them
# Finally plot them out
interval = 100  # Take samples to calculate game outcomes per <interval>
plt.plot(draw.rolling(interval).mean()[::interval], label='Draw')
plt.plot(agent_1.rolling(interval).mean()[::interval], label='Agent 1 Win')
plt.plot(agent_2.rolling(interval).mean()[::interval], label='Agent 2 Win')
plt.legend()
plt.title('Games Played by Random Agent vs Random Agent')
plt.xlabel('Number of Games Played')
plt.ylabel('Rate')
plt.show()

print(f'QAgent vs QAgent, for the past {episode} games:')
print(f'Draw Rate        : {draw.mean()} ({pd.Series(test_stat).value_counts().get(-1, 0)} Games)')
print(f'Agent 1 Win Rate : {agent_1.mean()} ({pd.Series(test_stat).value_counts().get(1, 0)} Games)')
print(f'Agent 2 Win Rate : {agent_2.mean()} ({pd.Series(test_stat).value_counts().get(2, 0)} Games)')
episode = 100000
env = Environment(debug=False)
q_agent = QAgent(debug=False, decay_rate=0.001 ** (1 / episode))
stat = []

start_time = time.time()
for i_episode in range(episode):
    observation = env.reset()
    i = 0
    for t in range(100):
        player = env.getCurrentPlayer()
        if player == 1:
            action = q_agent.getActionMax(observation, env.getAvailableMoves())
            next_observation, reward, game_state, info = env.step(action, player, 1)
            q_agent.train((observation, action, next_observation, reward, game_state), min)
        elif player == 2:
            action = q_agent.getActionMin(observation, env.getAvailableMoves())
            next_observation, reward, game_state, info = env.step(action, player, 1)
            q_agent.train((observation, action, next_observation, reward, game_state), max)
        observation = next_observation
        if game_state != 0:
            stat = collect_statistics(stat, env.checkGameEnds())
            if (i_episode + 1) % 10000 == 0:
                print(f'Episode {i_episode + 1}, {time.time() - start_time:.2f}s elapsed')
                start_time = time.time()
            break
    env.close()
# First decode to three columns
summary = pd.get_dummies(stat)
draw = summary[-1]
agent_1 = summary[1]
agent_2 = summary[2]

# Then apply rolling(100).mean() to each of them
# Finally plot them out
interval = 100  # Take samples to calculate game outcomes per <interval>
plt.plot(draw.rolling(interval).mean()[::interval], label='Draw')
plt.plot(agent_1.rolling(interval).mean()[::interval], label='Agent 1 Win')
plt.plot(agent_2.rolling(interval).mean()[::interval], label='Agent 2 Win')
plt.legend()
plt.title(f'{episode} Games Played by QAgent vs QAgent')
plt.xlabel('Number of Games Played')
plt.ylabel('Rate')
plt.show()
# Now we set epsilon to 0, to stop exploration and only retain expolitation.
# Let it play 10000 games we check the result of the game.

q_agent.epsilon = 0
episode = 50000
test_stat = []

start_time = time.time()
for i_episode in range(episode):
    observation = env.reset()
    i = 0
    for t in range(100):
        player = env.getCurrentPlayer()
        if player == first_player:
            action = q_agent.getActionMax(observation, env.getAvailableMoves())
            next_observation, reward, game_state, info = env.step(action, player, 1)
        else:
            action = q_agent.getActionMin(observation, env.getAvailableMoves())
            next_observation, reward, game_state, info = env.step(action, player, 1)
        observation = next_observation
        if game_state != 0:
            test_stat = collect_statistics(test_stat, env.checkGameEnds())
            if (i_episode + 1) % 10000 == 0:
                print(f'Episode {i_episode + 1}, {time.time() - start_time:.2f}s elapsed')
                start_time = time.time()
            break
    env.close()

# First decode gameplay results to three columns
test_summary = pd.get_dummies(test_stat)
draw = test_summary.get(-1, pd.Series(index=test_stat, name=-1))
agent_1 = test_summary.get(1, pd.Series([0] * len(test_stat), name=1))
agent_2 = test_summary.get(2, pd.Series([0] * len(test_stat), name=2))

# Then apply rolling(100).mean() to each of them to calculate average draw or win rate per 100 intervals
# Finally plot them out
interval = 100  # Take samples to calculate game outcomes per <interval>
plt.plot(draw.rolling(interval).mean()[::interval], label='Draw')
plt.plot(agent_1.rolling(interval).mean()[::interval], label='Agent 1 Win')
plt.plot(agent_2.rolling(interval).mean()[::interval], label='Agent 2 Win')
plt.legend()
plt.title('Games Played by QAgent vs QAgent')
plt.xlabel('Number of Games Played')
plt.ylabel('Rate')
plt.show()

print(f'QAgent vs QAgent, for the past {episode} games:')
print(f'Draw Rate        : {draw.mean()} ({pd.Series(test_stat).value_counts().get(-1, 0)} Games)')
print(f'Agent 1 Win Rate : {agent_1.mean()} ({pd.Series(test_stat).value_counts().get(1, 0)} Games)')
print(f'Agent 2 Win Rate : {agent_2.mean()} ({pd.Series(test_stat).value_counts().get(2, 0)} Games)')
len(q_agent.q_table)
q_agent.q_table
board = '[1, 1, 0, 2, 2, 0, 1, 2, 1]'
visualize(board)
print(q_agent.q_table[board])
if q_agent.q_table[board][2] > q_agent.q_table[board][5]:
    print('OK')
else:
    print('Failed, position 2 should have greater q value than position 5.')
board = '[1, 0, 0, 2, 2, 1, 1, 2, 1]'
visualize(board)
print(q_agent.q_table[board])

if q_agent.q_table[board][1] < q_agent.q_table[board][2]:
    print('OK')
else:
    print('Failed, position 1 should have smaller q value than position 2.')
board = '[1, 0, 0, 2, 2, 0, 1, 2, 1]'
visualize(board)
print(q_agent.q_table[board])
if q_agent.q_table[board][1] == q_agent.q_table[board][5]:
    print('OK')
else:
    print('Failed, position 1 should have the same q value with position 5.')

if q_agent.q_table[board][1] != q_agent.q_table[board][2] and q_agent.q_table[board][5] != q_agent.q_table[board][2]:
    print('OK')
else:
    print('Failed, position 1 and position 5 should have more negative q value when compared with position 2.')
board = '[1, 0, 0, 0, 1, 0, 2, 0, 0]'
visualize(board)
print(q_agent.q_table[board])
if q_agent.q_table[board][8] == min(q_agent.q_table[board].values()):
    print('OK')
else:
    print('Failed, position 8 should have the smallest q value.')
board = '[2, 0, 2, 0, 1, 0, 1, 1, 0]'
visualize(board)
print(q_agent.q_table[board])
if q_agent.q_table[board][1] == min(q_agent.q_table[board].values()):
    print('OK')
else:
    print('Failed, position 1 should have the smallest q value.')
board = '[0, 0, 2, 1, 0, 2, 0, 1, 0]'
visualize(board)
print(q_agent.q_table[board])
if q_agent.q_table[board][8] == max(q_agent.q_table[board].values()):
    print('OK')
else:
    print('Failed, position 8 should have either the largest q value.')
# Now we set epsilon to 0, to stop exploration and only retain expolitation.
# Let it play 10000 games we check the result of the game.

q_agent.epsilon = 0
episode = 10000
test_stat = []

start_time = time.time()
for i_episode in range(episode):
    observation = env.reset()
    history = []
    history.append(observation)
    i = 0
    for t in range(100):
        player = env.getCurrentPlayer()
        if player == 1:
            if first_player == 1:
                action = q_agent.getActionMax(observation, env.getAvailableMoves())
            else:
                action = q_agent.getActionMin(observation, env.getAvailableMoves())
            next_observation, reward, game_state, info = env.step(action, player, 1)
        elif player == 2:
            action = random_agent.getAction(observation, env.getAvailableMoves())
            next_observation, reward, game_state, info = env.step(action, player, 1)
        observation = next_observation
        history.append(observation)
        if game_state != 0:
            test_stat = collect_statistics(test_stat, env.checkGameEnds())
            if (i_episode + 1) % 10000 == 0:
                print(f'Episode {i_episode + 1}, {time.time() - start_time:.2f}s elapsed')
                start_time = time.time()
            break
    env.close()

# First decode to three columns
test_summary = pd.get_dummies(test_stat)
draw = test_summary.get(-1, pd.Series(index=test_stat, name=-1))
agent_1 = test_summary.get(1, pd.Series([0] * len(test_stat), name=1))
agent_2 = test_summary.get(2, pd.Series([0] * len(test_stat), name=2))

# Then apply rolling(100).mean() to each of them
# Finally plot them out
interval = 100  # Take samples to calculate game outcomes per <interval>
plt.plot(draw.rolling(interval).mean()[::interval], label='Draw')
plt.plot(agent_1.rolling(interval).mean()[::interval], label='Agent 1 Win')
plt.plot(agent_2.rolling(interval).mean()[::interval], label='Agent 2 Win')
plt.legend()
plt.title('Games played by QAgent vs Random Agent')
plt.xlabel('Number of games played')
plt.ylabel('Rate')
plt.show()

print(f'QAgent vs QAgent, for the past {episode} games:')
print(f'Draw Rate        : {draw.mean()} ({pd.Series(test_stat).value_counts().get(-1, 0)} Games)')
print(f'Agent 1 Win Rate : {agent_1.mean()} ({pd.Series(test_stat).value_counts().get(1, 0)} Games)')
print(f'Agent 2 Win Rate : {agent_2.mean()} ({pd.Series(test_stat).value_counts().get(2, 0)} Games)')
answer = input('Would you like to start first? Enter anything to start as the first player, or just press <Enter> to become the second player: ')
if answer != '':
    HUMAN_PLAYER = 1
    COMPUTER_PLAYER = 2
    first_player = HUMAN_PLAYER
else:
    COMPUTER_PLAYER = 1
    HUMAN_PLAYER = 2
    first_player = COMPUTER_PLAYER

observation = env.reset(first_player=first_player)
q_agent.epsilon = 0
env.render()
i = 0
for t in range(100):
    player = env.getCurrentPlayer()
    if player == HUMAN_PLAYER:
        action = int(input(f'Human\'s move (0 to {env.board_size ** 2 - 1}): '))
    elif player == COMPUTER_PLAYER:
        if first_player == COMPUTER_PLAYER:
            action = q_agent.getActionMax(observation, env.getAvailableMoves())
        else:
            action = q_agent.getActionMin(observation, env.getAvailableMoves())
        print(f'QAgent\'s move (0 to {env.board_size ** 2 - 1}): {action}')
    next_observation, reward, done, info = env.step(action, player, True)
    observation = next_observation
    reward += reward
    env.render()
    if done:
        break
#         sleep(2)
#     clear_output(wait=True)
env.explain()
env.close()
