summary_dir = 'summary'

num_episodes = 500000

display = False # boolean for diplaying/printing the Tic-Tac-Toe board on the terminal. It is suggested to set it to False for training purposes  



# exploration-exploitation trade-off factor

epsilon = 0.4  # must be a real number between (0,1)



# learning-rate

alpha   = 0.3  # must be a real number between (0,1)



# discount-factor

gamma   = 0.95 # must be a real number between (0,1)



playerX_QLearningAgent_name = 'QLearningAgent_X'

player0_QLearningAgent_name = 'QLearningAgent_0'



def display_board(board, action, playerID, player1, player2, reward, done, possible_actions, training = True, episode_reward_player1=None, episode_reward_player2=None):

    '''

    prints out the Tic-Tac-Toe board in the terminal.

    prints the action taken by the players, the reward they recieved and the status of the game (Done ->  True or False)

    prints if either of the players have won or lost the game or if it is a tied between the players

    prints all the possible next actions if the training argument is set to True

    '''

    print('\n')

    for i in range(3):

        print('  '.join(board[i*3:(i+1)*3]))

    

    player = player1.name if playerID else player2.name

    print(f'{player} takes action {action}, gets reward {reward}. Done -> {done}')

    if episode_reward_player1 is not None:

        print(f'episode_reward_player1 -> {episode_reward_player1}')

        print(f'episode_reward_player2 -> {episode_reward_player2}')

    if reward < 0:

        if playerID:

            print(f'\n{player2.name} wins !')

        else:

            print(f'\n{player1.name} wins !')

    elif reward == 1:

        if playerID:

            print(f'\n{player1.name} wins !')

        else:

            print(f'\n{player2.name} wins !')

    elif reward == 0.5:

        print(f'\nit is a draw !')

    else:

        if training:

            print(f'\npossible_actions: {possible_actions}\n')
class TicTacToe:

    def __init__(self):

        '''

        the environment starts with 9 empty spaces representing a board of Tic-Tac-Toe 

        '''

        self.board = ['_']*9   # the initial blank board

        self.done  = False     # done = True means the game has ended        



    def reset(self):

        '''

        resets the board for a new game

        '''

        self.board = ['_']*9



    def possible_actions(self):

        '''

        returns a list of possible actions that can be taken on the current board

        '''

        return [moves + 1 for moves, v in enumerate(self.board) if v == '_']



    def step(self, playerID, action):

        '''

        takes the action for the player with the given playerID resulting in a change in the board configuration.  

        it finally returns the reward and done status by evaluating the board congiguration. 

        '''

        if playerID:

            ch = 'X'

        else:

            ch = '0'



        if self.board[action - 1] != '_': # negative reward for picking the action which corresponds to marking the square which is already marked

            return -5, True



        self.board[action - 1] = ch



        # reward, done = self.evaluate(playerID)

        return self.evaluate(playerID)



    def evaluate(self, playerID):

        '''

        returns reward and done status for the player with the given playerID by evaluating the board congiguration.

        '''  

        if playerID:

            ch = 'X'

        else:

            ch = '0'



        # WIN CONDITIONS

        # rows checking        

        for i in range(3):

            if (ch == self.board[(i*3)+0] == self.board[(i*3)+1] == self.board[(i*3)+2]):

                # print(f'---row num: {i}')

                return 1, True

        

        # cols checking        

        for i in range(3):

            if (ch == self.board[i+0] == self.board[i+3] == self.board[i+6]):

                # print(f'---col num: {i}')

                return 1, True



        # diagonal checking

        if (ch == self.board[0] == self.board[4] == self.board[8]):

            # print('---diagonal 1')

            return 1.0, True



        if (ch == self.board[2] == self.board[4] == self.board[6]):

            # print('---diagonal 2')

            return 1.0, True



        # DRAW CONDITION

        # if all positions are filled

        if not any(c == '_' for c in self.board):

            # print('---all positions filled')

            return 0.5, True

        

        # GAME-ON CONDITION

        return 0, False
import random, pickle



class Hoooman():

    def __init__(self):

        self.name = 'Hoooman'



class RandomActionAgent():

    def __init__(self, name = 'RandomActionAgent'):

        self.name = name



    def choose_action(self, possible_actions):

        # random action selection

        action = random.choice(possible_actions)

        return action



class QLearningAgent:

    def __init__(self, name, epsilon = epsilon, alpha = alpha, gamma = gamma):

        self.name = name       

        self.epsilon = epsilon # exploration-exploiataion trade-off factor

        self.alpha   = alpha   # learning-rate

        self.gamma   = gamma   # discount-factor

        self.Q       = {}      # Q-Table

        self.last_board = None

        self.state_action_last = None

        self.q_last = 0.0



    def reset(self):

        '''

        resets the last_board, state_action_last and q_last 

        '''

        self.last_board = None

        self.state_action_last = None

        self.q_last = 0.0



    def epsilon_greedy(self, state, possible_actions):

        '''

        returns action by using epsilon-greedy algorithm

        '''

        # print('in epsilon-greedy, possible_actions: ', possible_actions)



        state = tuple(state) # because state is going to be a part of the key of Q-dict and dictionaries can not have lists as the keys

        self.last_board = state # 

        if random.random() < self.epsilon:

            # random action selection

            action = random.choice(possible_actions)

            

        else:

            # greedy action selection

            q_list = [] 

            # we will store q-values for all the possiblle actions available for the current state

            for action in possible_actions:

                q_list.append(self.getQ(state, action))

            maxQ = max(q_list)



            # print('q_list: ', q_list)



            # we need to handle the cases where more than 1 action has the same maxQ

            if q_list.count(maxQ) > 1:

                # in case when we have more than 1 action having the same maxQ, we randomly pick one of those actions

                maxQ_actions = [i for i in range(len(possible_actions)) if q_list[i] == maxQ]

                action_idx = random.choice(maxQ_actions)

            else:

                # in case when we have only 1 action having the same maxQ, simply pick that action

                action_idx = q_list.index(maxQ)



            action = possible_actions[action_idx]



        # update state_action_last and q_last        

        self.state_action_last = (state, action) 

        self.q_last = self.getQ(state, action)



        return action



    def getQ(self, state, action):

        '''

        return q-value for a given state-action pair

        '''

        return self.Q.get((state, action), 1.0) 



    def updateQ(self, reward, state, possible_actions):

        '''

        performs Q-Learning update

        '''

        q_list = []

        for action in possible_actions:

            q_list.append(self.getQ(tuple(state), action))

        

        if q_list:

            maxQ_next = max(q_list)

        else:

            maxQ_next = 0



        # Q-Table update

        self.Q[self.state_action_last] = self.q_last + self.alpha*((reward + self.gamma*maxQ_next) - self.q_last)



    def saveQtable(self):

        '''

        saves the Q-Table as a pickle file

        '''

        save_name = self.name + '_QTable'

        with open(save_name, 'wb') as handle:

            pickle.dump(self.Q, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'\nQ-Table for {self.name} saved as >{save_name}<\n')



    def loadQtable(self): # load table

        '''

        loads the Q-Table from a pickle file

        '''

        load_name = self.name + '_QTable'

        with open(load_name, 'rb') as handle:

            self.Q = pickle.load(handle)

        print(f'\nQ-Table for {self.name} loaded as >{load_name}< B)\n')
import sys, os, shutil, pdb, random

from tqdm import tqdm



def q(text = ''):

    '''

    a function that exits the code after printing a message. used for dubugging purposes

    '''

    print(f'>{text}<') # f-strings work only with python3

    sys.exit()



# from environment import TicTacToe

# from agent import QLearningAgent, RandomActionAgent 

# import config as cfg

# from config import display_board



# it is a little tricky on run SummaryWriter by installing a suitable version of pytorch. so if you are able to import SummaryWriter from torch.utils.tensorboard, this script will record summaries. Otherwise it would not.

try:

    from torch.utils.tensorboard import SummaryWriter

    write_summary = not True # This is set to False only for this kaggle notebook. We are not going to use tensorboard to plst the rewards here.

except:

    print('--------------------------------------------------------')

    print('You do not have SummaryWriter in torch.utils.tensorboard')

    print('This code will run anyway. It is just that the summaries would not be recorded')

    print('For the summaries to get recorded, install a suitable version of pytorch which has SummaryWriter in torch.utils.tensorboard')

    print('I had torch: 1.5.0+cu101 installed on my machine and it worked fine for me')

    print('--------------------------------------------------------')

    write_summary = False



# cfg.summary_dir is the path of the directory where the tensorboard SummaryWriter files are written

# the directory is removed, if it already exists

if write_summary:

    if os.path.exists(cfg.summary_dir):

        shutil.rmtree(cfg.summary_dir)



    writer = SummaryWriter(cfg.summary_dir) # this command automatically creates the directory at cfg.summary_dir



# initializing the TicTacToe environment and 2 QLearningAgent

env = TicTacToe()

player1 = QLearningAgent(name = playerX_QLearningAgent_name)

player2 = QLearningAgent(name = player0_QLearningAgent_name)

# player2 = RandomActionAgent()



episodes = num_episodes

for i in tqdm(range(episodes)):

    

    if display:

        print(f'TRAINING {str(i+1).zfill(len(str(episodes)))}/{episodes}')



    # initializing the episode reward for both the players to 0 

    episode_reward_player1 = 0

    episode_reward_player2 = 0

    

    # resetting the environemnt and both the agents 

    env.reset()

    player1.reset()

    player2.reset()



    done = False # the episode goes on as long as done is False



    # deciding which player makes a move first

    playerID = random.choice([True, False]) # True means player1

    

    while not done:

        # select player action by using epsilon-greedy algorithm, depending on the environment's board configuration and the possible actions available to the player

        if playerID:

            action = player1.epsilon_greedy(env.board, env.possible_actions())

        else:

            action = player2.epsilon_greedy(env.board, env.possible_actions())

            # action = player2.choose_action(env.possible_actions()) # action selection for RandomActionAgent

            

        # take selected action

        reward, done = env.step(playerID, action)



        if playerID:

            episode_reward_player1 += reward

        else:

            episode_reward_player2 += reward



        # display board

        if display:

            display_board(env.board, action, playerID, player1, player2, reward, done, env.possible_actions(), episode_reward_player1, episode_reward_player2, training = True)



        # Q-Table update based on the reward

        if reward == 1: # WIN n LOSE

            if playerID:

                # player1 wins and player2 loses

                player1.updateQ(reward   , env.board, env.possible_actions())                

                player2.updateQ(-1*reward, env.board, env.possible_actions())

            else:

                # player2 wins and player1 loses                

                player1.updateQ(-1*reward, env.board, env.possible_actions())

                player2.updateQ(reward, env.board, env.possible_actions())



        elif reward == 0.5: # DRAW

            player1.updateQ(reward, env.board, env.possible_actions())                

            player2.updateQ(reward, env.board, env.possible_actions())                



        elif reward == -5: # ILLEGAL ACTION

            if playerID:    

                player1.updateQ(reward, env.board, env.possible_actions())                

            else:

                player2.updateQ(reward, env.board, env.possible_actions())

        

        elif reward == 0: 

            if not playerID:

                player1.updateQ(reward, env.board, env.possible_actions())

            else:

                player2.updateQ(reward, env.board, env.possible_actions())



        # switch turns

        playerID = not playerID 



    if write_summary:

        # write tensorboard summaries

        writer.add_scalar(f'episode_reward/{player1.name}', episode_reward_player1, i)

        writer.add_scalar(f'episode_reward/{player2.name}', episode_reward_player2, i)



# save Q-Tables for both the players. either of these could be used as an opponent by a user.

player1.saveQtable()

player2.saveQtable()
os.listdir('')
# initializing the TicTacToe environment and a QLearningAgent (the master Tic-Toc-Toc player, your opponent!)

env = TicTacToe()

player1 = QLearningAgent(name = playerX_QLearningAgent_name)

player1.loadQtable() # load the learnt Q-Table

player1.epsilon = 0.0 # greedy actions only, 0 exploration



# initializing the agent class that let's you, the human user take the actions in the game 

player2 = Hoooman() 



# replay decides whether to rematch or not, at the end of a game 

replay = True

while replay:



    done = False   # the episode goes on as long as done is False

    

    # deciding which player makes a move first    

    playerID = random.choice([True, False]) # True means player1

    

    while not done:

        # select player action by using epsilon-greedy algorithm, depending on the environment's board configuration and the possible actions available to the player

        if playerID:

            action = player1.epsilon_greedy(env.board, env.possible_actions())

        else:

            # human user takes an action my entering one of the possible inputs in the terminal 

            print(f'\nPossible Actions: {env.possible_actions()}')

            action = int(input('Select an action ! '))

            

        # take selected action

        reward, done = env.step(playerID, action)



        # display board

        display_board(env.board, action, playerID, player1, player2, reward, done, env.possible_actions(), training=False)



        playerID = not playerID # switch turns



    # asks the human user if she/he wants to play another match

    replay = input('\nPlay Again ? [y/n] ')

    if replay.lower() == 'y':

        # resetting the environemnt and the Q-Learning agent if the human user choses to play another match 

        env.reset()     

        player1.reset() 

        print('\n-----------------------------NEW GAME-----------------------------')



    elif replay.lower() == 'n':

        # setting replay to False if the human user choses not to play another match

        replay = False  

        print('Thank you for wasting your time :)') 