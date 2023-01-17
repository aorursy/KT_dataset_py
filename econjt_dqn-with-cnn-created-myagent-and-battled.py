# !pip install 'kaggle-environments>=0.1.6' > /dev/null 2>&1

import numpy as np

import gym

import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F



import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

from kaggle_environments import evaluate, make
class ConnectX(gym.Env):

    def __init__(self, switch_prob=0.5, pair=[None, 'random']):

        self.env = make('connectx', debug=False)

        self.pair = [None, 'random']

        self.trainer = self.env.train(self.pair)

        self.switch_prob = switch_prob



        # Define required gym fields (examples):

        config = self.env.configuration

        self.action_space = gym.spaces.Discrete(config.columns)

        self.observation_space = gym.spaces.Discrete(config.columns * config.rows)



    def switch_trainer(self):

        self.pair = self.pair[::-1]

        self.trainer = self.env.train(self.pair)



    def step(self, action):

        return self.trainer.step(action)

    

    def reset(self):

        if np.random.random() < self.switch_prob:

            self.switch_trainer()

        return self.trainer.reset()

    

    def render(self, **kwargs):

        return self.env.render(**kwargs)

    

    

class CNN_model(nn.Module):



    def __init__(self, h, w, outputs):



        super(CNN_model, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1)

#         self.bn1 = nn.BatchNorm2d(16)

#         self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)

#         self.bn2 = nn.BatchNorm2d(32)

#         self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1)

#         self.bn3 = nn.BatchNorm2d(32)



        # Number of Linear input connections depends on output of conv2d layers

        # and therefore the input image size, so compute it.

        def conv2d_size_out(size, kernel_size = 2, stride = 1):

            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(w)

        convh = conv2d_size_out(h)

        linear_input_size = convw * convh * 16

        self.head = nn.Linear(linear_input_size, outputs)



    # Called with either one element to determine next action, or a batch

    # during optimization. Returns tensor([[left0exp,right0exp]...]).

    def forward(self, x):

        # import pdb; pdb.set_trace()

        x = F.relu(self.conv1(x))

#         x = F.relu(self.bn2(self.conv2(x)))

#         x = F.relu(self.bn3(self.conv3(x)))

        return self.head(x.view(x.size(0), -1))





class DQN:

    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):

        self.num_actions = num_actions

        self.batch_size = batch_size

        self.gamma = gamma

        self.model = CNN_model(6,7, num_actions)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.criterion = nn.MSELoss()

        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []} # The buffer

        self.max_experiences = max_experiences

        self.min_experiences = min_experiences



    def predict(self, inputs):

        return self.model(torch.from_numpy(inputs).float().view(-1, 1, 6, 7))



    def train(self, TargetNet):

        if len(self.experience['s']) < self.min_experiences:

            # Only start the training process when we have enough experiences in the buffer

            return 0



        # Randomly select n experience in the buffer, n is batch-size

        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)

        states = np.asarray([self.preprocess(self.experience['s'][i]) for i in ids])

        actions = np.asarray([self.experience['a'][i] for i in ids])

        rewards = np.asarray([self.experience['r'][i] for i in ids])



        # Prepare labels for training process

        states_next = np.asarray([self.preprocess(self.experience['s2'][i]) for i in ids])

        dones = np.asarray([self.experience['done'][i] for i in ids])

        value_next = np.max(TargetNet.predict(states_next).detach().numpy(), axis=1)

        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)



        actions = np.expand_dims(actions, axis=1)

        actions_one_hot = torch.FloatTensor(self.batch_size, self.num_actions).zero_()

        actions_one_hot = actions_one_hot.scatter_(1, torch.LongTensor(actions), 1)

        selected_action_values = torch.sum(self.predict(states) * actions_one_hot, dim=1)

        actual_values = torch.FloatTensor(actual_values)



        self.optimizer.zero_grad()

        loss = self.criterion(selected_action_values, actual_values)

        loss.backward()

        self.optimizer.step()



    # Get an action by using epsilon-greedy

    def get_action(self, state, epsilon):

        if np.random.random() < epsilon:

            return int(np.random.choice([c for c in range(self.num_actions) if state.board[c] == 0]))

        else:

            prediction = self.predict(np.atleast_2d(self.preprocess(state)))[0].detach().numpy()

            for i in range(self.num_actions):

                # もうすでに埋まっているcellは対象外

                if state.board[i] != 0:

                    prediction[i] = -1e7

            return int(np.argmax(prediction))



    # Method used to manage the buffer

    def add_experience(self, exp):

        if len(self.experience['s']) >= self.max_experiences:

            for key in self.experience.keys():

                self.experience[key].pop(0)

        for key, value in exp.items():

            self.experience[key].append(value)



    def copy_weights(self, TrainNet):

        self.model.load_state_dict(TrainNet.state_dict())



    def save_weights(self, path):

        torch.save(self.model.state_dict(), path)



    def load_weights(self, path):

        self.model.load_state_dict(torch.load(path))

    

    # Each state will consist of the board and the mark

    # in the observations

    def preprocess(self, state):

        result = state.board[:]

        

        # I omitted state.mark from input. I'm guessing this feat is redundant

#         result.append(state.mark)



        return result


def set_direction(d, direction):

    if direction == "row":

        dx, dy = d, 0

    elif direction == "col":

        dx, dy = 0, d

    elif direction == "diag":

        dx, dy = d, d

    elif direction == "anti-diag":

        dx, dy = d, -1*d

    return dx, dy 



def count_seq(new_stone_loc, state,mark):

    """change state for each direction"""

    ans = 0

    i, j = new_stone_loc

    for direction in ["row", "col", "diag", "anti-diag"]:

        count_sequences = 0

        for dir_ in [1, -1]:

            for d in range(4):

                try:

                    dx, dy = set_direction(dir_*d,direction)

                    if dx == 0 and dy == 0:

                        continue

                    elif state[i + dx, j + dy] == mark:

                        count_sequences += 1

                    else:

                        break

                except IndexError:

                    break

        ans = max(count_sequences, ans)

    return ans
# np.zeros((6,7))
def reward_coordination(obs, prev_obs):

    # prev_observationとobservationを比較して

    # 自分のstoneが連結しているかいなかでrewardを変更する。

    # 連結確認メソッド

    # import pdb; pdb.set_trace()



    obs_mat = np.array(obs.board).reshape(-1,7)

    prev_obs_mat = np.array(prev_obs.board).reshape(-1,7)

    new_stone_loc = np.where(obs_mat - prev_obs_mat == obs.mark)

    out = count_seq(new_stone_loc, obs_mat, obs.mark)



    return out
def play_game(env, TrainNet, TargetNet, epsilon, copy_step):

    rewards = 0

    iter_ = 0

    done = False

    observations = env.reset()

    while not done:

        # Using epsilon-greedy to get an action

        action = TrainNet.get_action(observations, epsilon)



        # Caching the information of current state

        prev_observations = observations



        # Take action

        observations, reward, done, _ = env.step(action)



        # Apply new rules

        if done:

            if reward == 1: # Won

                reward = 20

            elif reward == 0: # Lost

                reward = -20

            else: # Draw

                reward = 10

        else:

#             reward = -0.05 # Try to prevent the agent from taking a long move



            # Try to promote the agent to "struggle" when playing against negamax agent

            # as Magolor's (@magolor) idea

            reward = 0.5

#             reward = reward_coordination(observations, prev_observations) * 0.5



        rewards += reward



        # Adding experience into buffer

        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}

        TrainNet.add_experience(exp)



        # Train the training model by using experiences in buffer and the target model

        TrainNet.train(TargetNet)

        iter_ += 1

        if iter_ % copy_step == 0:

            # Update the weights of the target model when reaching enough "copy step"

            TargetNet.copy_weights(TrainNet)

    return rewards
def rule_based(obs, conf):

    def get_results(x, y, mark, multiplier):

        """ get list of points, lowest cells and "in air" cells of a board[x][y] cell considering mark """

        # set board[x][y] as mark

        board[x][y] = mark

        results = []

        # if some points in axis already found - axis blocked

        blocked = [False, False, False, False]

        # i is amount of marks required to add points

        for i in range(conf.inarow, 2, -1):

            # points

            p = 0

            # lowest cell

            lc = 0

            # "in air" points

            ap = 0

            # axis S -> N, only if one mark required for victory

            if i == conf.inarow and blocked[0] is False:

                (p, lc, ap, blocked[0]) = process_results(p, lc, ap,

                              check_axis(mark, i, x, lambda z : z, y + inarow_m1, lambda z : z - 1))

            # axis SW -> NE

            if blocked[1] is False:

                (p, lc, ap, blocked[1]) = process_results(p, lc, ap,

                    check_axis(mark, i, x - inarow_m1, lambda z : z + 1, y + inarow_m1, lambda z : z - 1))

            # axis E -> W

            if blocked[2] is False:

                (p, lc, ap, blocked[2]) = process_results(p, lc, ap,

                    check_axis(mark, i, x + inarow_m1, lambda z : z - 1, y, lambda z : z))

            # axis SE -> NW

            if blocked[3] is False:

                (p, lc, ap, blocked[3]) = process_results(p, lc, ap, 

                    check_axis(mark, i, x + inarow_m1, lambda z : z - 1, y + inarow_m1, lambda z : z - 1))

            results.append((p * multiplier, lc, ap))

        # restore board[x][y] original value

        board[x][y] = 0

        return results

    

    def check_axis(mark, inarow, x, x_fun, y, y_fun):

        """ check axis (NE -> SW etc.) for lowest cell and amounts of points and "in air" cells """

        (x, y, axis_max_range) = get_x_y_and_axis_max_range(x, x_fun, y, y_fun)

        zeros_allowed = conf.inarow - inarow

        #lowest_cell = y

        # lowest_cell calculation turned off

        lowest_cell = 0

        for i in range(axis_max_range):

            x_temp = x

            y_temp = y

            zeros_remained = zeros_allowed

            marks = 0

            # amount of empty cells that are "in air" (don't have board bottom or mark under them)

            in_air = 0

            for j in range(conf.inarow):

                if board[x_temp][y_temp] != mark and board[x_temp][y_temp] != 0:

                    break

                elif board[x_temp][y_temp] == mark:

                    marks += 1

                # board[x_temp][y_temp] is 0

                else:

                    zeros_remained -= 1

                    if (y_temp + 1) < conf.rows and board[x_temp][y_temp + 1] == 0:

                        in_air -= 1

#                 if y_temp > lowest_cell:

#                     lowest_cell = y_temp

                if marks == inarow and zeros_remained == 0:

                    return (sp, lowest_cell, in_air, True)

                x_temp = x_fun(x_temp)

                y_temp = y_fun(y_temp)

                if y_temp < 0 or y_temp >= conf.rows or x_temp < 0 or x_temp >= conf.columns:

                    return (0, 0, 0, False)

            x = x_fun(x)

            y = y_fun(y)

        return (0, 0, 0, False)

        

    def get_x_y_and_axis_max_range(x, x_fun, y, y_fun):

        """ set x and y inside board boundaries and get max range of axis """

        axis_max_range = conf.inarow

        while y < 0 or y >= conf.rows or x < 0 or x >= conf.columns:

            x = x_fun(x)

            y = y_fun(y)

            axis_max_range -= 1

        return (x, y, axis_max_range)

    

    def process_results(p, lc, ap, axis_check_results):

        """ process results of check_axis function, return lowest cell and sums of points and "in air" cells """

        (points, lowest_cell, in_air, blocked) = axis_check_results

        if points > 0:

            if lc < lowest_cell:

                lc = lowest_cell

            ap += in_air

            p += points

        return (p, lc, ap, blocked)

    

    def get_best_cell(best_cell, current_cell):

        """ get best cell by comparing factors of cells """

        for i in range(len(current_cell["factors"])):

            # index 0 = points, 1 = lowest cell, 2 = "in air" cells

            for j in range(3):

                # if value of best cell factor is smaller than value of

                # the same factor in the current cell

                # best cell = current cell and break the loop,

                # don't compare lower priority factors

                if best_cell["factors"][i][j] < current_cell["factors"][i][j]:

                    return current_cell

                # if value of best cell factor is bigger than value of

                # the same factor in the current cell

                # break loop and don't compare lower priority factors

                if best_cell["factors"][i][j] > current_cell["factors"][i][j]:

                    return best_cell

        return best_cell

    

    def get_factors(results):

        """ get list of factors represented by results and ordered by priority from highest to lowest """

        factors = []

        for i in range(conf.inarow - 2):

            if i == 1:

                # my checker in this cell means my victory two times

                factors.append(results[0][0][i] if results[0][0][i][0] > st else (0, 0, 0))

                # opponent's checker in this cell means my defeat two times

                factors.append(results[0][1][i] if results[0][1][i][0] > st else (0, 0, 0))

                # if there are results of a cell one row above current

                if len(results) > 1:

                    # opponent's checker in cell one row above current means my defeat two times

                    factors.append(results[1][1][i] if -results[1][1][i][0] > st else (0, 0, 0))

                    # my checker in cell one row above current means my victory two times

                    factors.append(results[1][0][i] if -results[1][0][i][0] > st else (0, 0, 0))

                else:

                    for j in range(2):

                        factors.append((0, 0, 0))

            else:

                for j in range(2):

                    factors.append((0, 0, 0))

                for j in range(2):

                    factors.append((0, 0, 0))

            # consider only if there is no "in air" cells

            if results[0][1][i][2] == 0:

                # placing opponent's checker in this cell means opponent's victory

                factors.append(results[0][1][i])

            else:

                factors.append((0, 0, 0))

            # placing my checker in this cell means my victory

            factors.append(results[0][0][i])

            # central column priority

            factors.append((1 if i == 1 and shift == 0 else 0, 0, 0))

            # if there are results of a cell one row above current

            if len(results) > 1:

                # opponent's checker in cell one row above current means my defeat

                factors.append(results[1][1][i])

                # my checker in cell one row above current means my victory

                factors.append(results[1][0][i])

            else:

                for j in range(2):

                    factors.append((0, 0, 0))

        # if there are results of a cell two rows above current

        if len(results) > 2:

            for i in range(conf.inarow - 2):

                # my checker in cell two rows above current means my victory

                factors.append(results[2][0][i])

                # opponent's checker in cell two rows above current means my defeat

                factors.append(results[2][1][i])

        else:

            for i in range(conf.inarow - 2):

                for j in range(2):

                    factors.append((0, 0, 0))

        return factors





    # define my mark and opponent's mark

    my_mark = obs.mark

    opp_mark = 2 if my_mark == 1 else 1

    

    # define board as two dimensional array

    board = []

    for column in range(conf.columns):

        board.append([])

        for row in range(conf.rows):

            board[column].append(obs.board[conf.columns * row + column])

    

    best_cell = None

    board_center = conf.columns // 2

    inarow_m1 = conf.inarow - 1

    

    # standard amount of points

    sp = 1

    # "seven" pattern threshold points

    st = 1

    

    # start searching for best_cell from board center

    x = board_center

    

    # shift to right or left from board center

    shift = 0

    

    # searching for best_cell

    while x >= 0 and x < conf.columns:

        # find first empty cell starting from bottom of the column

        y = conf.rows - 1

        while y >= 0 and board[x][y] != 0:

            y -= 1

        # if column is not full

        if y >= 0:

            # results of current cell and cells above it

            results = []

            results.append((get_results(x, y, my_mark, 1), get_results(x, y, opp_mark, 1)))

            # if possible, get results of a cell one row above current

            if (y - 1) >= 0:

                results.append((get_results(x, y - 1, my_mark, -1), get_results(x, y - 1, opp_mark, -1)))

            # if possible, get results of a cell two rows above current

            if (y - 2) >= 0:

                results.append((get_results(x, y - 2, my_mark, 1), get_results(x, y - 2, opp_mark, 1)))

            

            # list of factors represented by results

            # ordered by priority from highest to lowest

            factors = get_factors(results)



            # if best_cell is not yet found

            if best_cell is None:

                best_cell = {

                    "column": x,

                    "factors": factors

                }

            # compare values of factors in best cell and current cell

            else:

                current_cell = {

                    "column": x,

                    "factors": factors

                }

                best_cell = get_best_cell(best_cell, current_cell)

                        

        # shift x to right or left from board center

        if shift >= 0: shift += 1

        shift *= -1

        x = board_center + shift



    # return index of the best cell column

    return best_cell["column"]
env = ConnectX(pair=[rule_based,"negamax"])
gamma = 0.99

copy_step = 25

hidden_units = [128, 128, 128, 128, 128]

max_experiences = 50000

min_experiences = 100

batch_size = 32

lr = 1e-2

epsilon = 0.95

decay = 0.999

min_epsilon = 0.05

episodes = 60000



precision = 7
num_states = env.observation_space.n + 1

num_actions = env.action_space.n



all_total_rewards = np.empty(episodes)

all_avg_rewards = np.empty(episodes) # Last 100 steps

all_epsilons = np.empty(episodes)



# Initialize models

TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)

TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
import copy





pbar = tqdm(range(episodes))

for n in pbar:

    epsilon = max(min_epsilon, epsilon * decay)

    total_reward = play_game(env, TrainNet, TargetNet, epsilon, copy_step)

    all_total_rewards[n] = total_reward

    avg_reward = all_total_rewards[max(0, n - 100):(n + 1)].mean()

    all_avg_rewards[n] = avg_reward

    all_epsilons[n] = epsilon



    pbar.set_postfix({

        'episode reward': total_reward,

        'avg (100 last) reward': avg_reward,

        'epsilon': epsilon

    })



    if n % 10000 == 0:

        epsilon = 0.999

        TrainNet_adversarial = copy.deepcopy(TrainNet)

        env = ConnectX(switch_prob=0.5, pair=["negamax", TrainNet_adversarial])

        range_st = n//5000

        range_ed = range_st + 5000

        plt.plot(all_avg_rewards[range_st:range_ed])

        plt.xlabel('Episode')

        plt.ylabel('Avg rewards (100)')

        plt.show()

    if n % 10000 == 0:

        epsilon = 0.999

        TrainNet_adversarial = copy.deepcopy(TrainNet)

        env = ConnectX(switch_prob=0.5, pair=["negamax", rule_based])

        range_st = n//5000

        range_ed = range_st + 5000

        plt.plot(all_avg_rewards[range_st:range_ed])

        plt.xlabel('Episode')

        plt.ylabel('Avg rewards (100)')

        plt.show()
#　上記繰り返す。
all_avg_rewards
plt.plot(all_avg_rewards)

plt.xlabel('Episode')

plt.ylabel('Avg rewards (100)')

plt.show()
TrainNet
plt.plot(all_epsilons)

plt.xlabel('Episode')

plt.ylabel('Epsilon')

plt.show()
TrainNet.save_weights('./weights.pth')
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):

    """

    Parameters

    ----------

    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ

    filter_h : フィルターの高さ

    filter_w : フィルターの幅

    stride : ストライド

    pad : パディング

    Returns

    -------

    col : 2次元配列

    """

    N, C, H, W = input_data.shape

    out_h = (H + 2*pad - filter_h)//stride + 1

    out_w = (W + 2*pad - filter_w)//stride + 1



    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')

    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))



    for y in range(filter_h):

        y_max = y + stride*out_h

        for x in range(filter_w):

            x_max = x + stride*out_w

            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]



    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)

    return col

def relu(tensor):

    return np.where(tensor>0, tensor,0 )
# TrainNet.model.conv1.weight.shape
in_h = 6; in_w = 7



stride = 1;pad = 0; filter_h = 2;filter_w = 2

board = np.arange(42).reshape((1, 1, in_h, in_w))

out_C = 16

N, C, H, W = board.shape

out_h = (H + 2*pad - filter_h)//stride + 1

out_w = (W + 2*pad - filter_w)//stride + 1

output_cnn = np.zeros((N, out_C, out_h, out_w), dtype=np.float)
# vectorized_board = im2col(board.reshape((1,1, in_h, in_w)),

#           filter_h=filter_h, filter_w=filter_w, stride=stride, pad=pad) 



# conv_weight = TrainNet.model.conv1.weight[i].flatten().detach().numpy()

# conv_bias = TrainNet.model.conv1.bias.flatten().detach().numpy()



# for i in range(out_C):

#     filter_ = conv_weight

#     # 'filter_{} = np.array({}, dtype=np.float32)\n'.format(i, filter_)

#     "out_elems_n = reshaped_board@filter_0.T"

#     # print(vectorized_board.shape, ":", filter_0.detach().numpy().shape)

#     vectorized_out = vectorized_board@filter_.reshape((-1, 1))

#     vectorized_out = vectorized_out.reshape((out_h, out_w))

#     output_cnn[:, i, :, :] = vectorized_out

# output_cnn = output_cnn + conv_bias.reshape(1, out_C, 1, 1)
# tensor_board = torch.FloatTensor(board.reshape((1, 1, 6, -1)))



# out = TrainNet.model.head.weight.detach().numpy()@output_cnn.reshape(-1, 1)
# out.argmax()

# out # myagent関数の最終的なリターン
# TrainNet.model.conv1(tensor_board).shape
# 再現

# out = TrainNet.model.head.weight.detach().numpy()@output_cnn.reshape((-1, 1)) #+ out_bias.reshape(-1, 1)

# out
# # Write hidden layers

# for i, (w, b) in enumerate(fc_layers[:-1]):

#     my_agent += '    hl{}_w = np.array({}, dtype=np.float32)\n'.format(i+1, w)

#     my_agent += '    hl{}_b = np.array({}, dtype=np.float32)\n'.format(i+1, b)

# # Write output layer

# my_agent += '    ol_w = np.array({}, dtype=np.float32)\n'.format(fc_layers[-1][0])

# my_agent += '    ol_b = np.array({}, dtype=np.float32)\n'.format(fc_layers[-1][1])

# print(my_agent)
# Create the agent

my_agent = '''def my_agent(observation, configuration):

    import numpy as np



'''

my_agent += f'''    in_h = {in_h}; in_w = {in_w}; stride = {stride};pad = {pad}; filter_h = {filter_h};filter_w = {filter_w}

    board = np.array(observation.board[:])

    board = board.reshape((1, 1, in_h, in_w))

    out_C = {out_C}

    N, C, H, W = board.shape

    out_h = (H + 2*pad - filter_h)//stride + 1

    out_w = (W + 2*pad - filter_w)//stride + 1

    output_cnn = np.zeros((N, out_C, out_h, out_w), dtype=np.float)\n'''



import inspect 

inner_method = ""

# print('    ' + inspect.getsource(im2col))

method_list = [im2col, relu]

for method in method_list:

    for line in inspect.getsource(method).split('\n'):

        inner_method += "    " + line + "\n"

my_agent += inner_method



#f"np.array({TrainNet.model.conv1.weight[i].flatten().detach().tolist()}, dtype=np.float32)\n"



my_agent += '''    vectorized_board = im2col(board.reshape((1,1, in_h, in_w)),

          filter_h=filter_h, filter_w=filter_w, stride=stride, pad=pad)\n'''







# Ongoing development 6/13

#ここを numpy.arrayとして変数conv_weightに渡してあげる

#　複数CNN層対応できるようにする...TBD

my_agent += f"    conv_weight = np.array({TrainNet.model.conv1.weight.detach().tolist()}, dtype=np.float32)\n"

my_agent += f"    conv_bias = np.array({TrainNet.model.conv1.bias.flatten().detach().tolist()}, dtype=np.float32)\n"





my_agent += """    for i in range(out_C):

        filter_ = conv_weight[i].flatten()

        vectorized_out = vectorized_board@filter_.reshape((-1, 1))

        vectorized_out = vectorized_out.reshape((out_h, out_w))

        output_cnn[:, i, :, :] = vectorized_out



    output_cnn = output_cnn + conv_bias.reshape((1, out_C, 1, 1))

    output_cnn = relu(output_cnn)

"""







my_agent += f"    out_weight = np.array({TrainNet.model.head.weight.detach().tolist()}, dtype=np.float32)\n"

my_agent += f"    out_bias = np.array({TrainNet.model.head.bias.detach().tolist()}, dtype=np.float32)\n"



my_agent += """    out = out_weight@output_cnn.reshape((-1, 1)) + out_bias.reshape((-1, 1))\n"""

# 各列一番上のセルが空いていなかった場合、そこは選ばないようにする。

my_agent += '''

    for i in range(configuration.columns):

        if observation.board[i] != 0:

            out[i] = -1e7

    return int(np.argmax(out))

    '''

with open('submission.py', 'w') as f:

    f.write(my_agent)
from submission import my_agent
def mean_reward(rewards):

    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)



# Run multiple episodes to estimate agent's performance.

print("My Agent vs. Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))

print("My Agent vs. Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))

print("Random Agent vs. My Agent:", mean_reward(evaluate("connectx", ["random", my_agent], num_episodes=10)))

print("Negamax Agent vs. My Agent:", mean_reward(evaluate("connectx", ["negamax", my_agent], num_episodes=10)))
# evaluate("connectx", ["negamax", "random"], num_episodes=10)

# print(inspect.getsource(evaluate))
# env.run(["random", "negamax"])