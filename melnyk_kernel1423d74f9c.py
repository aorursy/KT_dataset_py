# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import gym
!pip install atari-py
env = gym.make('Pong-v4')
import cv2
import numpy as np


# This function can resize to any shape, but was built to resize to 84x84
def process_frame(frame, shape=(84, 84)):
    """Preprocesses a 210x160x3 frame to 84x84x1 grayscale
    Arguments:
        frame: The frame to process.  Must have values ranging from 0-255
    Returns:
        The processed frame
    """
    frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34:34 + 160, :160]  # crop image
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))

    return frame.transpose(2, 1, 0)
import gym
import torch

class GameWrapper:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state = None
        self.last_lives = 0

    def reset(self):
        self.last_lives = 0
        state = self.env.reset()
        processed_state = process_frame(state)

        self.state = np.repeat(processed_state, 4, axis=0)
        return self.state

    def step(self, action):

        state, reward, done, info = self.env.step(action)
        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = done
        self.last_lives = info['ale.lives']
        processed_state = process_frame(state)
        self.state = np.append(self.state[1:, :, :], processed_state, axis=0)
        assert (self.state[-1, :, :] - process_frame(state)).mean() == 0
        return processed_state, reward, done, terminal_life_lost, info

    def render(self):
        return self.env.render()
game = GameWrapper('Pong-v4')
def epsilon_select(q_values, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(game.env.action_space.n)
    return np.argmax(q_values)
from torch.optim import Adam
import torch.nn.functional as F
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size=(8, 8), stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1)
        self.conv4 = torch.nn.Conv2d(64, 1024, kernel_size=(7, 7), stride=1)
        self.fc1 = torch.nn.Linear(1024, game.env.action_space.n)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x

torch.cuda.is_available()
class ReplayMemory(object):
    """Replay Memory that stores the last size=1,000,000 transitions"""

    def __init__(self, size=500000, frame_height=84, frame_width=84,
                 agent_history_length=4, batch_size=32):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number if transitions returned in a minibatch
        """
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.float)
        self.new_states = np.empty((self.batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.float)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward, terminal):
        """
        Args:
            action: An integer between 0 and env.action_space.n - 1 
                determining the action the agent perfomed
            frame: A (84, 84, 1) frame of an Atari game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index - self.agent_history_length + 1:index + 1, ...]

    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = np.random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current >= index - self.agent_history_length:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):
        """
        Returns states, actions, rewards, new_states, terminal_flags
        """
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)

        return self.states, self.actions[self.indices], self.rewards[self.indices], self.new_states, \
               self.terminal_flags[self.indices]

model = SimpleCNN()
loss_fn = torch.nn.SmoothL1Loss()
learning_rate = 2.5e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

episodes = 500
epsilon = 1.0
gamma = 0.99
losses = []
replay_memory = ReplayMemory()
LEARN_FREQ = 4

model.cuda()
import copy
def process_state_to_pytorch(state):
    return torch.from_numpy(state).float().cuda() / 255.
def learn(memory: ReplayMemory, model: SimpleCNN, model2: SimpleCNN, loss_fn):
    print("Here 10")
    states, actions, rewards, new_states, terminal_flags = memory.get_minibatch()
    print("Here1")
    rewards = torch.Tensor(rewards).cuda()
    print("Here2")
    terminal_flags = torch.Tensor(rewards).cuda()
    print("Here3")
    actions = torch.Tensor(actions)
    print("Here4")
    
    q_vals = model(process_state_to_pytorch(states))
    print("Here5")
    with torch.no_grad():
        # next_q_vals = model(process_state_to_pytorch(new_states))
        model2_vals = model2(process_state_to_pytorch(new_states))
        print("Here6")

    # best_actions = torch.argmax(next_q_vals, dim=1) # we select actions using model network
    max_q = model2_vals.max(1)[0]
    print("Here7")
    Y = rewards + gamma * max_q * (1 - terminal_flags)
    print("Here8")
    X = q_vals.gather(dim=1, index=actions.cuda().long().unsqueeze(dim=1))
    print("Here9")
    loss = loss_fn(X.squeeze(1), Y)
    print("Here10")
    optimizer.zero_grad()
    print("Here11")
    loss.backward()
    print("Here12")
    optimizer.step()
    print("Here13")
    return loss.item()

def copy_model(model: SimpleCNN, model2: SimpleCNN = None):
    if not model2:
        model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())
    model2.eval()
    return model2
learn(replay_memory, model, model2, loss_fn)
model2 = copy_model(model)

model2 = copy_model(model)
UPDATE_MODEL = 5_000
MAX_EPISODE_LENGTH = 18_000
MIN_REPLAY_SIZE = 50_000
frames = 0
max_episode = 0
for i in range(2):
    game.reset()
    done = False
    episode_length = 0
    while not done:
        episode_length += 1
        frames += 1
        q_val = model(process_state_to_pytorch(game.state).unsqueeze(0))
      
        q_val_ = q_val.data.cpu().numpy()
        action = epsilon_select(q_val_, epsilon)
        processed_state, reward, done, terminal, info = game.step(action)

        replay_memory.add_experience(action, processed_state.squeeze(), reward, terminal)

        if reward > 0:
            print("received positive reward", reward)

        if frames % LEARN_FREQ == 0 and frames > MIN_REPLAY_SIZE:
            
            loss = learn(replay_memory, model, model2, loss_fn)
            losses.append(loss)
        if frames % UPDATE_MODEL == 0 and frames > MIN_REPLAY_SIZE:
            print("Updating model")
            model2 = copy_model(model, model2)
        if frames % 200_000 == 0:
            import pickle
            with open('replay.pickle', 'wb') as f:
                pickle.dump(replay_memory, f)
        if frames % 10000 == 0:
            print('10_000 frames')
    max_episode = max(max_episode, episode_length)
    print("Max episode: ", max_episode)
    if i % 10 == 0:
        torch.save(model.state_dict(), 'my_model')
    if losses:
        print(np.mean(losses))
    if epsilon > 0.1 and frames > 50_000:
        epsilon -= (1 / (i + 1))

