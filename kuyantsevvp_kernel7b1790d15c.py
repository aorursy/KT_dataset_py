!pip install 'kaggle-environments>=0.1.6'
# !pip install torch
# !pip install tensorflow
# import numpy as np
# import gym
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
# from kaggle_environments import evaluate, make

# class ColumnFilled(RuntimeError):
#     pass

# cluster_prior = [0, 0, 1, 4, 16]

# num_cols = 7
# num_rows = 6
# dims = (num_rows, num_cols)

# def is_valid_ind(ind):
#     row, col = ind
#     return 0 <= row < num_rows and 0 <= col < num_cols

# def get_neighbor_indices(ind):
#     row, col = ind
#     row = int(row)
#     col = int(col)
#     n_inds = []
#     for i in range(row - 1, row + 2):
#         for j in range(col - 1, col + 2):
#             if not is_valid_ind((i,j)):
#                 continue
#             if i == row and j == col:
#                 continue
#             n_inds.append((i,j))
#     return n_inds

# def all_neighbors_empty(board, ind):
#     neighb = get_neighbor_indices(ind)
#     return sum(board[n] == 0 for n in neighb) == len(neighb)

# def all_neighbors_nonempty(board, ind):
#     neighb = get_neighbor_indices(ind)
#     non_em = sum(board[n] != 0 for n in neighb) 
#     return non_em == len(neighb)

# def has_same_neighbors(board, ind, mark):
#     neighb = get_neighbor_indices(ind)
#     return sum((board[n] == mark for n in neighb)) > 0

# def has_neighbors(board, ind, mark):
#     neighb = get_neighbor_indices(ind)
#     return sum((board[n] != 0 for n in neighb)) > 0

# def check_array_win(arr, mark):
#     for i in range(len(arr) - 3):
#         to_check = arr[i:i+4]
#         check = sum(el == mark for el in to_check)
#         # print(to_check)
#         if check == 4:
#             return True
#     return False


# board = [[0,0,0,0,0,0,0],
#          [0,0,0,0,0,0,0],
#          [0,0,2,0,0,0,0],
#          [0,0,1,0,1,0,0],
#          [0,0,1,0,1,0,0],
#          [0,0,1,0,1,0,0]]
# board = np.array(board)



# # board = np.array(board).ravel()
# print(get_neighbor_indices((0,0)))
# print()
# print(get_neighbor_indices((5,2)))

# print(is_valid_ind((4,3)))
# def check_win(board,place,mark):
#     reshaped_b = np.reshape(board, dims)
#     row, col = np.unravel_index(place, dims)
#     hor = reshaped_b[row, :]
#     ver = reshaped_b[:, col]
#     # print(ver)
#     # print(check_array_win(ver, mark))
#     # print('end')
#     offset = col - row
#     diag1 = np.diagonal(reshaped_b, offset)
#     diag2 = np.flipud(reshaped_b).diagonal(offset)
    
#     return check_array_win(hor, mark) \
#            or check_array_win(ver, mark) \
#            or check_array_win(diag1, mark) \
#            or check_array_win(diag2, mark)

# board = [[2,2,2,2,0,0,0],
#          [0,0,0,0,0,0,0],
#          [0,0,2,0,0,1,0],
#          [0,0,1,0,1,0,0],
#          [0,0,1,1,1,0,0],
#          [0,0,1,0,1,0,0]]
# board = np.array(board).ravel()

# for i in range(len(board)):
#     if check_win(board, i, 1):
#         print(1, np.unravel_index(i, dims))
#     if check_win(board, i, 2):
#         print(2, np.unravel_index(i, dims))

# # ind = np.ravel_multi_index((2,4), dims)
# # print(check_win(board,ind, 1))


# def greedy_strategy(board, mark):
#     choose_from = []
#     for i in range(num_cols):
#         try:
#             new_board, ind = apply_action(board, mark, i)
#             new_board = new_board.reshape(dims)
#             ind = np.unravel_index(ind, dims)
#             if has_same_neighbors(new_board, ind, mark):
#                 choose_from.append(i)
#         except ColumnFilled as e:
#             pass
#     if len(choose_from) > 0:
#         return np.random.choice(choose_from)
#     return np.random.choice([c for c in range(num_cols) if board[c] == 0])


# def greedy_cluster_strategy(board, mark):
#     choose_from = []
#     for i in range(num_cols):
#         try:
#             new_board, ind = apply_action(board, mark, i)
#             new_board = new_board.reshape(dims)
#             ind = np.unravel_index(ind, dims)
#             if has_neighbors(new_board, ind, mark):
#                 choose_from.append(i)
#         except ColumnFilled as e:
#             pass
#     if len(choose_from) > 0:
#         return np.random.choice(choose_from)
#     return np.random.choice([c for c in range(num_cols) if board[c] == 0])


# def random_strategy(board, mark):
#     return np.random.choice([c for c in range(num_cols) if board[c] == 0])


# def check_win(board,place,mark):
#     reshaped_b = np.reshape(board, dims)
#     row, col = np.unravel_index(place, dims)
#     hor = reshaped_b[row, :]
#     ver = reshaped_b[:, col]
#     offset = col - row
#     diag1 = np.diagonal(reshaped_b, offset)
#     diag2 = np.flipud(reshaped_b).diagonal(offset)
    
#     return check_array_win(hor, mark) \
#            or check_array_win(ver, mark) \
#            or check_array_win(diag1, mark) \
#            or check_array_win(diag2, mark)

    
# def evaluate_position(board, ind, mark):
#     neighbors = get_neighbor_indices(ind)
#     counter = 1
#     num_clusters = 0
#     for n in neighbors:
#         if board[n] == mark:
#             num_clusters += 1
#             cluster_counter = 1
#             d = (n[0] - ind[0], n[1] - ind[1])  # direction
#             next_ind = (ind[0] + d[0], ind[1] + d[1])
#             while is_valid_ind(next_ind) and board[next_ind] == mark:
#                 cluster_counter += 1
#                 next_ind = (next_ind[0] + d[0], next_ind[1] + d[1])
#             # if d != (0, 1) or d != (1, 0):
#             #     counter *= 2
#             # print(mark, ind, cluster_counter)
#             counter += cluster_prior[cluster_counter] * cluster_counter
#     point_score = counter * 0.5 * num_clusters
#     return point_score


# def inverse_mark(mark):
#     return 2 - mark // 2


# def sigmoid(x):
#     return 1/(1 + np.exp(-x))


# def get_expected_rewards(raveled_board, mark):
#     board = raveled_board.reshape(dims)
#     rewards = [1] * num_cols
#     for i in range(num_cols):
#         try:
#             new_board, ind = apply_action(raveled_board, mark, i)
#             # my_win = check_win(new_board, ind, mark)
#             # opp_win = check_win(new_board, ind, inverse_mark(mark))
#             # if my_win or opp_win:
#             #     rewards[i] = 10000
#             #     continue
#         except ColumnFilled as e:
#             rewards[i] = 0
#             continue
#         ind = np.unravel_index(ind, dims)
#         gained_score = evaluate_position(board, ind, mark)
#         blocked_opponent_score = evaluate_position(board, ind, inverse_mark(mark))
        
#         rewards[i] += gained_score + blocked_opponent_score
#     return np.array(rewards)


# def get_expected_state_rewards(state):
#     return get_expected_rewards(state[:-1], state[-1])


# def get_expected_batch_rewards(batch):
#     return np.apply_along_axis(get_expected_state_rewards, 1, batch)


# def apply_action(board, mark, action):
#     # find lowest empty cell in column
#     board = np.array(board)
#     # if bottoms is empty, place checker there
#     bottom_index = np.ravel_multi_index((num_rows - 1, action), dims)
#     if board[bottom_index] == 0:
#         board[bottom_index] = mark
#         return board, bottom_index
#     for i in range(num_rows):
#         if board[np.ravel_multi_index((i, action), dims)]:
#             break
#     if i == 0:
#         raise ColumnFilled("Column is filled")
#     place_index = np.ravel_multi_index((i-1, action), dims)
#     board[place_index] = mark
#     return board, place_index
   
    
# board = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# board = [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
# # print(np.reshape(board, dims))
# board = np.array(board)
# obs = {"board": board, "mark": 2}

# board = [[0,0,1,0,0,0,0],
#          [0,0,2,0,0,0,0],
#          [0,0,2,0,0,0,0],
#          [0,0,1,0,1,0,0],
#          [0,0,1,0,1,0,0],
#          [0,0,1,0,1,0,0]]
# board = np.array(board).ravel()
# # board = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# # print(greedy_strategy(board, 2))
# # print(get_expected_rewards(board, 1))
# print(get_expected_rewards(board, 2))
# # states = [
# #     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1],
# #     [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2]
# # ]
# # print(get_expected_batch_rewards(states))
# class ConnectX(gym.Env):
#     def __init__(self, switch_prob=0.5):
#         self.env = make('connectx', debug=False)
#         self.pair = [None, 'random']
#         self.trainer = self.env.train(self.pair)
#         self.switch_prob = switch_prob

#         # Define required gym fields (examples):
#         config = self.env.configuration
#         self.action_space = gym.spaces.Discrete(config.columns)
#         self.observation_space = gym.spaces.Discrete(config.columns * config.rows)

#     def switch_trainer(self):
#         self.pair = self.pair[::-1]
#         self.trainer = self.env.train(self.pair)

#     def step(self, action):
#         return self.trainer.step(action)
    
#     def reset(self):
#         if np.random.random() < self.switch_prob:
#             self.switch_trainer()
#         return self.trainer.reset()
    
#     def render(self, **kwargs):
#         return self.env.render(**kwargs)
    
    
# class DeepModel(tf.keras.Model):
#     def __init__(self, num_states, hidden_units, num_actions):
#         super(DeepModel, self).__init__()
#         self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
#         self.hidden_layers = []
#         for i in hidden_units:
#             self.hidden_layers.append(tf.keras.layers.Dense(
#                 i, activation='relu', kernel_initializer='RandomNormal'))
#         self.output_layer = tf.keras.layers.Dense(
#             num_actions, activation='sigmoid', kernel_initializer='RandomNormal')

# #     @tf.function
#     def call(self, inputs):
#         z = self.input_layer(inputs)
#         print(z)
#         for layer in self.hidden_layers:
#             z = layer(z)
#             print(z)
#         output = self.output_layer(z)
#         return output


# class DQN:
#     def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
#         self.num_actions = num_actions
#         self.batch_size = batch_size
#         self.optimizer = tf.optimizers.Adam(lr)
#         self.gamma = gamma
#         self.model = DeepModel(num_states, hidden_units, num_actions)
#         self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []} # The buffer
#         self.max_experiences = max_experiences
#         self.min_experiences = min_experiences

#     def predict(self, inputs):
#         return self.model(np.atleast_2d(inputs.astype('float32')))

# #     @tf.function
#     def train(self, TargetNet):
#         if len(self.experience['s']) < self.min_experiences:
#         # if len(self.experience['s']) < 1:
#             # Only start the training process when we have enough experiences in the buffer
#             return 0

#         # Randomly select n experience in the buffer, n is batch-size
#         ## TODO maybe choose them not randomly ? 
#         ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
#         # ids = np.random.randint(low=0, high=len(self.experience['s']), size=1)
#         # state before action
#         states = np.asarray([self.preprocess(self.experience['s'][i]) for i in ids])

#         with tf.GradientTape() as tape:
#             pred = self.predict(states)
#             # print('preds', pred)
#             rewards = get_expected_batch_rewards(states)
#             # print('rewards', rewards)
#             loss = -tf.math.reduce_sum(pred * rewards)
#         variables = self.model.trainable_variables
#         gradients = tape.gradient(loss, variables)
#         self.optimizer.apply_gradients(zip(gradients, variables))

#     # Get an action by using epsilon-greedy
#     def get_action(self, state, epsilon, verbose=False):
#         if np.random.random() < epsilon:
#             board = state['board']
#             mark = state['mark']
#             return int(greedy_cluster_strategy(board, mark))
#             # return int(np.random.choice([c for c in range(self.num_actions) if state['board'][c] == 0]))
#         else:
#             prediction = self.predict(np.atleast_2d(self.preprocess(state)))[0].numpy()
#             if verbose:
#                 print('rew', get_expected_state_rewards(state))
#                 print('pred',prediction)
#             for i in range(self.num_actions):
#                 if state['board'][i] != 0:
#                     prediction[i] = -1e7
#             return int(np.argmax(prediction))

#     # Method used to manage the buffer
#     def add_experience(self, exp):
#         if len(self.experience['s']) >= self.max_experiences:
#             for key in self.experience.keys():
#                 self.experience[key].pop(0)
#         for key, value in exp.items():
#             self.experience[key].append(value)

#     def copy_weights(self, TrainNet):
#         variables1 = self.model.trainable_variables
#         variables2 = TrainNet.model.trainable_variables
#         for v1, v2 in zip(variables1, variables2):
#             v1.assign(v2.numpy())

#     def save_weights(self, path):
#         self.model.save_weights(path)

#     def load_weights(self, path):
#         ref_model = tf.keras.Sequential()

#         ref_model.add(self.model.input_layer)
#         for layer in self.model.hidden_layers:
#             ref_model.add(layer)
#         ref_model.add(self.model.output_layer)

#         ref_model.load_weights(path)
    
#     # Each state will consist of the board and the mark
#     # in the observations
#     def preprocess(self, state):
#         result = state['board'][:]
#         result.append(state.mark)
#         return result
    
# def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
#     rewards = 0
#     iter = 0
#     done = False
#     observations = env.reset()
#     while not done:
#         # end.render()
#         # Using epsilon-greedy to get an action
#         action = TrainNet.get_action(observations, epsilon)

#         # Caching the information of current state
#         prev_observations = observations

#         # Take action
#         observations, reward, done, _ = env.step(action)

#         rewards += reward

#         # Adding experience into buffer
#         exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
#         TrainNet.add_experience(exp)

#         # Train the training model by using experiences in buffer and the target model
#         TrainNet.train(TargetNet)
#         iter += 1
#         if iter > min_experiences and iter % copy_step == 0:
#             # Update the weights of the target model when reaching enough "copy step"
#             TargetNet.copy_weights(TrainNet)
#     return rewards


# env = ConnectX()

# gamma = 0.99
# copy_step = 25
# hidden_units = [100, 200, 200, 100]
# max_experiences = 10000
# min_experiences = 100
# batch_size = 32
# lr = 1e-2
# epsilon = 0.99
# decay = 0.99995
# min_epsilon = 0.1

# episodes = 30000
# # episodes = 1

# precision = 7

# # log_dir = 'logs/'
# # summary_writer = tf.summary.create_file_writer(log_dir)


# num_states = env.observation_space.n + 1
# num_actions = env.action_space.n

# all_total_rewards = np.empty(episodes)
# all_avg_rewards = np.empty(episodes) # Last 100 steps
# all_epsilons = np.empty(episodes)

# # Initialize models
# TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
# TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)


# pbar = tqdm(range(episodes))
# for n in pbar:
#     epsilon = max(min_epsilon, epsilon * decay)
#     total_reward = play_game(env, TrainNet, TargetNet, epsilon, copy_step)
#     all_total_rewards[n] = total_reward
#     avg_reward = all_total_rewards[max(0, n - 100):(n + 1)].mean()
#     all_avg_rewards[n] = avg_reward
#     all_epsilons[n] = epsilon

#     pbar.set_postfix({
#         'episode reward': total_reward,
#         'avg (100 last) reward': avg_reward,
#         'epsilon': epsilon
#     })

# #     with summary_writer.as_default():
# #         tf.summary.scalar('episode reward', total_reward, step=n)
# #         tf.summary.scalar('running avg reward (100)', avg_reward, step=n)
# #         tf.summary.scalar('epsilon', epsilon, step=n)


# plt.plot(all_avg_rewards)
# plt.xlabel('Episode')
# plt.ylabel('Avg rewards (100)')
# plt.show()
# plt.plot(all_epsilons)
# plt.xlabel('Episode')
# plt.ylabel('Epsilon')
# plt.show()

# fc_layers = []

# # fc_layers.extend([
# #     TrainNet.model.input_layer.weights[0].numpy().tolist(), # weights
# #     TrainNet.model.input_layer.weights[1].numpy().tolist() # bias
# # ])

# # Get all hidden layers' weights
# for i in range(len(hidden_units)):
#     fc_layers.extend([
#         TrainNet.model.hidden_layers[i].weights[0].numpy().tolist(), # weights
#         TrainNet.model.hidden_layers[i].weights[1].numpy().tolist() # bias
#     ])

# # Get output layer's weights
# fc_layers.extend([
#     TrainNet.model.output_layer.weights[0].numpy().tolist(), # weights
#     TrainNet.model.output_layer.weights[1].numpy().tolist() # bias
# ])

# # Convert all layers into usable form before integrating to final agent
# fc_layers = list(map(
#     lambda x: str(list(np.round(x, precision))) \
#         .replace('array(', '').replace(')', '') \
#         .replace(' ', '') \
#         .replace('\n', ''),
#     fc_layers
# ))
# fc_layers = np.reshape(fc_layers, (-1, 2))

# # Create the agent
# my_agent = '''def my_agent(observation, configuration):
#     import numpy as np

# '''

# # my_agent += '    il_w = np.array({}, dtype=np.float32)\n'.format(fc_layers[0][0])
# # my_agent += '    il_b = np.array({}, dtype=np.float32)\n'.format(fc_layers[0][1])

# # Write hidden layers
# for i, (w, b) in enumerate(fc_layers[0:-1]):
#     my_agent += '    hl{}_w = np.array({}, dtype=np.float32)\n'.format(i+1, w)
#     my_agent += '    hl{}_b = np.array({}, dtype=np.float32)\n'.format(i+1, b)
# # Write output layer
# my_agent += '    ol_w = np.array({}, dtype=np.float32)\n'.format(fc_layers[-1][0])
# my_agent += '    ol_b = np.array({}, dtype=np.float32)\n'.format(fc_layers[-1][1])

# my_agent += '''
#     state = observation['board'][:]
#     state.append(observation['mark'])
#     out = np.array(state, dtype=np.float32)

# '''

# # Calculate hidden layers
# # my_agent += '    out = np.matmul(out, il_w) + il_b\n'
# for i in range(len(fc_layers[:-1])):
#     my_agent += '    out = np.matmul(out, hl{0}_w) + hl{0}_b\n'.format(i+1)
    
#     my_agent += '    out = np.maximum(0,out)\n' # Relu function
#     # my_agent += '    out = 1/(1 + np.exp(-out))\n' # Sigmoid function
# # Calculate output layer
# my_agent += '    out = np.matmul(out, ol_w) + ol_b\n'
# my_agent += '    out = 1/(1 + np.exp(-out))\n' # Sigmoid function

# my_agent += '''
#     for i in range(configuration.columns):
#         if observation['board'][i] != 0:
#             out[i] = -1e7

#     return int(np.argmax(out))
#     '''
# print(my_agent)

# with open('agent_expect_loss.py', 'w') as f:
#     f.write(my_agent)
    
    

# with open('test_submission.py', 'w') as f:
#     f.write(my_agent)
    