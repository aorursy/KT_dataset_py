import random
import numpy as np
from collections import namedtuple
from typing import List


class ReplayMemory(object):
  def __init__(self, capacity: int) -> None:
    self.capacity = capacity
    self.memory = []
    self.position = 0
    self.transition = namedtuple("Transition",
                                 field_names=["prev_state", "action",
                                              "reward", "curr_state",
                                              "done"])

  def push(self, prev_state: np.ndarray, action: int,
           reward: int, curr_state: np.ndarray, done: bool) -> None:
    if self.position < self.capacity:
      self.memory.append(self.transition(
          prev_state, action, reward, curr_state, done))
    else:
      self.memory[self.position] = self.transition(
          prev_state, action, reward, curr_state, done)

    self.position = (self.position+1) % self.capacity

  def sample(self, batch_size: int) -> List:
    return random.sample(self.memory, batch_size)

  def __len__(self) -> int:
    return len(self.memory)

import torch


class Model(torch.nn.Module):
  def __init__(self, num_states: int, num_actions: int, hidden_layer_size: int) -> None:
    super(Model, self).__init__()

    self.layer1 = torch.nn.Sequential(
        torch.nn.Linear(num_states, hidden_layer_size),
        torch.nn.BatchNorm1d(hidden_layer_size),
        torch.nn.PReLU()
    )

    self.layer2 = torch.nn.Sequential(
        torch.nn.Linear(hidden_layer_size, hidden_layer_size),
        torch.nn.BatchNorm1d(hidden_layer_size),
        torch.nn.PReLU()
    )

    self.final_layer = torch.nn.Linear(hidden_layer_size, num_actions)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.final_layer(x)
    return x

import inspect
import os
import sys
import numpy as np
from typing import Dict, List, Tuple
from kaggle_environments import evaluate, make, utils, environments, Environment
import gym


class ConnectXEnvironment(gym.Env):
  def __init__(self, vs_agent: str = 'random', debug: bool = True) -> None:
    self._env = make(environment="connectx", debug=debug)
    self._vs_agent = vs_agent

    # gym fields
    self._config = self._env.configuration
    self._actions = gym.spaces.Discrete(self._config.columns)
    self._states = gym.spaces.Discrete(self._config.columns * self._config.rows)

    # agent
    self._pair = [None, vs_agent]
    self._trainer = self._env.train(self._pair)

  @property
  def vs_agent(self) -> str:
    return self._vs_agent

  @vs_agent.setter
  def vs_agent(self, new_agent: str) -> None:
    self._vs_agent = new_agent
    self._pair = [None, new_agent]
    self._trainer = self._env.train(self._pair)

  @property
  def env(self) -> Environment:
    return self._env

  @property
  def config(self) -> Tuple:
    return self._config

  @property
  def actions(self) -> int:
    return self._actions.n

  @property
  def states(self) -> int:
    # return self._states.n+1
    return self._states.n

  def switch_trainer(self) -> None:
    self._pair = self._pair[::-1]
    self._trainer = self._env.train(self._pair)

  def step(self, action: int) -> List:
    """
      Returns List[Dict[str, List[int]], int, bool, Dict]
    """
    observation, reward, done, info = self._trainer.step(action)
    observation = self.__preprocess_observation(observation)
    return observation, reward, done, info

  def reset(self) -> Dict[str, List[int]]:
    return self.__preprocess_observation(self._trainer.reset())

  def render(self, **kwargs) -> str:
    return self._env.render(**kwargs)

  def __preprocess_observation(self, observation: Dict[str, int]) -> np.ndarray:
    return np.array(observation.board)

  def _mean_reward(self, rewards: int) -> float:
    return sum(r[0] for r in rewards if r[0] == 1) / len(rewards)

  def evaluate(self, my_agent: 'Agent', num_episodes_eval: int = 10) -> None:
    if self._vs_agent == 'negamax':
      print("My Agent vs " + self._vs_agent +
            " Agent: " + str(self._evaluate_negamax(my_agent)))
    else:
      print("My Agent vs " + self._vs_agent +
            " Agent: " + str(self._evaluate_random(my_agent)))

  def _evaluate_random(self, my_agent: 'Agent', num_episodes_eval: int = 10) -> float:
    return self._mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=num_episodes_eval))

  def _evaluate_negamax(self, my_agent: 'Agent', num_episodes_eval: int = 10) -> float:
    return self._mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=num_episodes_eval))

import os
import torch
import numpy as np
from typing import List, Dict, Tuple


class Agent():
  def __init__(self, env: ConnectXEnvironment, debug: bool, checkpoint_path: str,
               hidden_layer_size: int, replay_memory_cap: int, batch_size: int,
               learning_rate: float, learning_rate_decay: float, discount_factor: float) -> None:
    """
      Agent class that is responsible for training our neural network and overall 
      managing the DQN.
    """
    self.env = env
    self.num_states = env.states
    self.num_actions = env.actions

    self.debug = debug

    # if gpu is to be used
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.device(self.device)
    # if debug:
    print("Using device: %s" % self.device)

    self.__model = Model(num_states=self.num_states, num_actions=self.num_actions,
                         hidden_layer_size=hidden_layer_size).to(self.device)

    self.learning_rate_decay = learning_rate_decay
    self.optimizer = torch.optim.Adam(
        self.__model.parameters(), lr=learning_rate)
    self.__scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=self.optimizer, gamma=learning_rate_decay)
    self.loss_function = torch.nn.MSELoss()
    self.checkpoint_path = checkpoint_path
    self.discount_factor = discount_factor

    self.replay_memory = ReplayMemory(replay_memory_cap)
    self.batch_size = batch_size

  @property
  def model(self) -> Model:
    return self.__model

  def preprocess_observation(self, observation: np.ndarray) -> torch.Tensor:
    return torch.autograd.Variable(torch.Tensor(observation).to(self.device))

  def predict(self, input_data: np.ndarray) -> torch.Tensor:
    processed_data = self.preprocess_observation(
        input_data.reshape(-1, self.num_states))
    self.__model.train(mode=False)
    return self.__model(processed_data)

  def get_action(self, observation: np.ndarray, epsilon: float) -> int:
    """
      Actions that we can take is any of the columns that are not full.

      args:
        observation: Numpy array representation of the board.
    """
    regular_observation = observation.tolist()
    if np.random.rand() < epsilon:
      final_action = int(np.random.choice(
          [c for c in range(self.num_actions) if regular_observation[c] == 0]))
    else:
      self.__model.train(mode=False)
      scores = self.predict(observation)[0].cpu().detach().numpy()
      for i in range(self.num_actions):
        if regular_observation[i] != 0:
          scores[i] = -1e7
      return int(np.argmax(scores))

    return final_action

  def decay_learning_rate(self) -> None:
    if len(self.replay_memory) >= self.batch_size and self.learning_rate_decay > 0.00:
      self.__scheduler.step()

  def get_last_lr(self) -> float:
    return self.__scheduler.get_last_lr()[0]

  def save_weights(self) -> None:
    # make the file if not exists, torch.save doesn't work without existing file
    try:
      if not os.path.exists(self.checkpoint_path):
        if not os.path.isdir(os.path.dirname(self.checkpoint_path)):
          os.mkdir(os.path.dirname(self.checkpoint_path))
        with open(self.checkpoint_path, 'w+'):
          pass

      if self.debug:
        print("Saving weights to: " + str(self.checkpoint_path))

      torch.save(self.__model.state_dict(), self.checkpoint_path)
    except Exception as e:
      print("Could not save weights to: " + str(self.checkpoint_path))
      print("ERROR: %s" % e)

  def load_weights(self, name: str = None) -> None:
    try:
      self.__model.load_state_dict(torch.load(self.checkpoint_path))
      if self.debug:
        print("Loaded weights for " + name +
              ", from: " + str(self.checkpoint_path))
    except Exception as e:
      print("Could not load weights for " + name +
            ", from: " + str(self.checkpoint_path))
      print("ERROR: %s" % e)

  def copy_weights(self, agent_to_copy: 'Agent') -> None:
    self.__model.load_state_dict(agent_to_copy.model.state_dict())

  def add_experience(self, prev_state: np.ndarray, action: int,
                     reward: int, curr_state: np.ndarray, done: bool) -> None:
    self.replay_memory.push(prev_state, action, reward, curr_state, done)

  def train(self, target_agent: 'Agent') -> Tuple[float, float]:
    """
      Train on a single game. Only train if our replay memory has enough saved memory, 
      which should be >= batch size.

      We take a minibatch (of size batch_size) from our replay memory. We use our 
      train_agent (policy network) to predict the Q values for the previous states.
      We use our target_agent (target network) to predict the Q values for the 
      current states, but we use these Q values from target_agent in our bellman
      equation to get the real Q values. Finally, we compare the Q values from
      the policy network with the Q values we get from the bellman equation.
    """
    # only start training process when we have enough experiences in the replay
    if len(self.replay_memory) < self.batch_size:
      return 0.00, 0.00

    # sample random batch from replay memory
    minibatch = self.replay_memory.sample(self.batch_size)
    prev_states = np.vstack([x.prev_state for x in minibatch])
    actions = np.array([x.action for x in minibatch])
    rewards = np.array([x.reward for x in minibatch])
    curr_states = np.vstack([x.curr_state for x in minibatch])
    dones = np.array([x.done for x in minibatch])

    # use train network to predict q values of prior states (before actual states)
    q_predict = self.predict(prev_states)

    # use bellman equation to get expected q-value of actual states
    q_target = q_predict.cpu().clone().data.numpy()
    q_curr_state_values = np.max(target_agent.predict(curr_states).cpu().data.numpy(),
                                 axis=1)
    bellman_eq = rewards + self.discount_factor * q_curr_state_values * ~dones
    q_target[np.arange(len(q_target)), actions] = bellman_eq
    q_target = self.preprocess_observation(q_target)

    # train our network based on the results from its
    # q_predict to expected values given by our target network (q_target)
    self.__model.train(mode=True)
    self.optimizer.zero_grad()
    loss = self.loss_function(q_predict, q_target)
    loss.backward()
    self.optimizer.step()
    return loss, np.mean(bellman_eq)

import os
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from collections import deque


def train_agent(env: ConnectXEnvironment, train_agent: Agent, target_agent: Agent,
                progress_print_per_iter: int, total_episodes: int, episode_epsilon: float,
                min_epsilon: float, max_epsilon_episodes: int, epsilon_decay: float,
                copy_max_count: int, saved_results_path: str, saved_results_name: str,
                hyperparams_dict: Dict) -> None:
  """
    Train the agent on a number of games/episodes.
    Decay epsilon by epsilon_decay if given, otherwise decay using max_epsilon_episodes.
    Decay learning rate by learning_rate_decay if present, otherwise don't decay.
    Record and plot data on matplotlib and also save the figues/numbers.
  """
  total_rewards = 0
  total_loss = 0
  total_steps = 0
  total_bellman_eq = 0
  avg_reward = deque(maxlen=100)
  progress_bar = tqdm(total=total_episodes)
  plotting_data = {
      'avg_rewards[last_%s]' % avg_reward.maxlen: np.empty(total_episodes),
      'total_rewards': np.empty(total_episodes),
      'epsilon': np.empty(total_episodes),
      'loss': np.empty(total_episodes),
      'bellman_eq': np.empty(total_episodes),
      'learning_rate': np.empty(total_episodes),
  }

  for episode in range(total_episodes):
    # epsilon decay
    episode_epsilon = epsilon_decay_formula(episode=episode, max_episode=max_epsilon_episodes,
                                            min_epsilon=min_epsilon, epsilon=episode_epsilon,
                                            epsilon_decay=epsilon_decay)

    # train game/episode, save weights, decay learning rate
    (total_rewards, total_loss,
     total_steps, total_bellman_eq) = train_single_game(env=env,
                                                        train_agent=train_agent,
                                                        target_agent=target_agent,
                                                        epsilon=episode_epsilon,
                                                        copy_max_count=copy_max_count,
                                                        total_steps=total_steps)
    train_agent.decay_learning_rate()
    train_agent.save_weights()

    # update matplotlib data
    avg_reward.append(total_rewards)
    plotting_data['avg_rewards[last_%s]' %
                  avg_reward.maxlen][episode] = np.mean(avg_reward)
    plotting_data['total_rewards'][episode] = total_rewards
    plotting_data['epsilon'][episode] = episode_epsilon
    plotting_data['loss'][episode] = total_loss
    plotting_data['bellman_eq'][episode] = total_bellman_eq
    plotting_data['learning_rate'][episode] = train_agent.get_last_lr()

    # update progress bar
    if ((episode+1) % progress_print_per_iter) == 0:
      progress_bar.update(progress_print_per_iter)
      progress_bar.set_postfix({
          'episode reward': total_rewards,
          'avg reward (last %s)' % avg_reward.maxlen: np.mean(avg_reward),
          'epsilon': episode_epsilon,
      })

  env.close()
  save_results(plotting_data=plotting_data, progress_bar=progress_bar,
               name=saved_results_name, directory_name=saved_results_path,
               hyperparams_dict=hyperparams_dict)


def train_single_game(env: ConnectXEnvironment, train_agent: Agent,
                      target_agent: Agent, epsilon: float,
                      copy_max_count: int, total_steps: int) -> Tuple[int, int, int, float]:
  """
    Train the agent on one game/episode.
    Update target agent model weights to the same as train agent's after some number of steps.
    Return the total rewards given by the environment, and the loss given by the agent/model.
  """
  prev_observation = env.reset()
  observation = None
  total_rewards = 0
  total_loss = 0
  avg_bellman_eq = 0
  total_bellman_eq = 0
  reward, game_done = None, False

  while not game_done:
    # Get our agent's action and record the environment
    action = train_agent.get_action(
        observation=prev_observation, epsilon=epsilon)
    observation, reward, game_done, _ = env.step(action)

    if game_done:
      # win
      if reward == 1:
        reward = 20
      # lost
      elif reward == -1:
        reward = -20
      # draw
      else:
        reward = 10
    else:
      # prevent agent from taking a long move
      reward = -0.05

    total_rewards += reward
    total_steps += 1

    # Add the observations we got from the environment
    train_agent.add_experience(prev_observation, action, reward,
                               observation, game_done)
    # Get the loss and bellman equation values from training
    total_loss, avg_bellman_eq = train_agent.train(target_agent)
    total_bellman_eq = avg_bellman_eq if avg_bellman_eq > 0 else total_bellman_eq
    # adjust prev state to curr state for next iteration
    prev_observation = observation

    # copy weights of policy net to our target net after a certain amount of steps
    if total_steps % copy_max_count == 0:
      target_agent.copy_weights(train_agent)

  return total_rewards, total_loss, total_steps, total_bellman_eq


def save_results(plotting_data: Dict[str, np.ndarray], progress_bar: tqdm,
                 name: str, directory_name: str, hyperparams_dict: Dict) -> None:
  """
    Save the progress bar and the matplotlib figures to a directory.
  """
  # create all the necessary directories
  parent_dir = os.path.abspath(os.path.join(directory_name, os.pardir))
  if not os.path.exists(parent_dir):
    os.mkdir(parent_dir)
  if not os.path.exists(directory_name):
    os.mkdir(directory_name)
  directory_name = directory_name + "/" + name

  # plot all our data
  def plot_figure(data: np.ndarray, xlabel: str, ylabel: str, save_path: str) -> None:
    plt.clf()
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.show()

  for name, plot_data in plotting_data.items():
    plot_figure(data=plot_data, xlabel='Episode', ylabel=name,
                save_path=directory_name + name + '.png')

  # Save the hyperparams and progress bar in a text file
  with open(directory_name+'pbar.txt', 'w', encoding="utf-8") as filetowrite:
    filetowrite.write("==== Hyperparams: ====\n")
    for key, val in hyperparams_dict.items():
      filetowrite.write(key + ": %s" % val)
      filetowrite.write("\n")
    filetowrite.write("\n")
    filetowrite.write(str(progress_bar))


def play_game(env: ConnectXEnvironment, agent: Agent, epsilon: float,
              game_render: bool = False) -> None:
  """
    Play a single game.
  """
  observation = env.reset()
  done = False
  reward = 0

  while not done:
    if game_render:
      env.render()
    action = agent.get_action(observation=observation, epsilon=epsilon)
    observation, reward, done, _ = env.step(action)

  print()
  if reward == 1:
    print("Agent won!")
  elif reward == -1:
    print("Agent lost.")
  else:
    print("Draw.")

  print("Final board: ")
  print(observation.reshape(env.config.rows, env.config.columns))
  env.close()


def epsilon_decay_formula(episode: int, max_episode: int, min_epsilon: float,
                          epsilon: float, epsilon_decay: float) -> float:
  """
    If there is an epsilon decay value, then we use that.

    Otherwise use max_epsilon_episodes, which will look like the graph below:
    Returns 𝜺-greedy
    1.0---|\
          | \
          |  \
    min_e +---+------->
              |
              max_episode
  """
  if epsilon_decay > 0:
    new_epsilon = max(min_epsilon, epsilon * epsilon_decay)
  else:
    slope = (min_epsilon - 1.0) / max_episode
    new_epsilon = max(min_epsilon, slope * episode + epsilon)

  return new_epsilon

import os

curr_try = 1
hyperparams_dict = {
    # total eps, printing, and memory
    'total_episodes':  10000,
    'replay_memory_cap': 50000,
    'progress_per_iteration': 50,

    # learning rate
    'learning_rate':  0.001,
    'learning_rate_decay': 0.00,
    # 'learning_rate_decay':  0.001,

    # epsilon
    'epsilon': 1.0,
    'max_epsilon_episodes': 100,
    'min_epsilon': 0.01,
    'epsilon_decay': 0.00,
    # 'epsilon_decay': 0.995,

    # other factors
    'discount_factor': 0.99,
    'batch_size': 32,
    'copy_max_step': 30,
    'hidden_layer_size': 24,

    # weights save path
    'checkpoint_path': os.path.join(os.path.join(os.getcwd(), 'nn_saved_weights'), 'training_%s.pth' % curr_try),

    # results save path
    'saved_results_path': os.path.join(os.path.join(os.getcwd(), 'saved_results'), 'training_%s' % curr_try),

    # results save name
    'saved_results_name': '',
}
import os


def print_all_hyperparams() -> None:
  """
    Print all the hyper parameters.
  """
  print()
  print("==== Hyperparams: ====")
  for key, val in hyperparams_dict.items():
    print(key + ": %s" % val)
  print()


def get_agent(env: ConnectXEnvironment, agent_debug: bool) -> Agent:
  """
    Returns Agent class.
  """
  return Agent(env=env,
               debug=agent_debug,
               checkpoint_path=hyperparams_dict['checkpoint_path'],
               hidden_layer_size=hyperparams_dict['hidden_layer_size'],
               batch_size=hyperparams_dict['batch_size'],
               learning_rate=hyperparams_dict['learning_rate'],
               learning_rate_decay=hyperparams_dict['learning_rate_decay'],
               discount_factor=hyperparams_dict['discount_factor'],
               replay_memory_cap=hyperparams_dict['replay_memory_cap'])


def main(vs_agent: str = 'random', agent_debug: bool = False, train_model: bool = False) -> None:
  """
    Train on multiple episodes or play a single game.
  """
  my_env = ConnectXEnvironment(vs_agent=vs_agent, debug=False)
  my_train_agent = get_agent(env=my_env, agent_debug=False)
  my_target_agent = get_agent(env=my_env, agent_debug=False)

  if os.path.exists(hyperparams_dict['checkpoint_path']):
    my_train_agent.load_weights(name="training agent")
    my_target_agent.load_weights(name="target agent")

  if train_model:
    print_all_hyperparams()
    train_agent(env=my_env, train_agent=my_train_agent, target_agent=my_target_agent,
                progress_print_per_iter=hyperparams_dict['progress_per_iteration'],
                total_episodes=hyperparams_dict['total_episodes'],
                episode_epsilon=hyperparams_dict['epsilon'],
                min_epsilon=hyperparams_dict['min_epsilon'],
                max_epsilon_episodes=hyperparams_dict['max_epsilon_episodes'],
                epsilon_decay=hyperparams_dict['epsilon_decay'],
                copy_max_count=hyperparams_dict['copy_max_step'],
                saved_results_path=hyperparams_dict['saved_results_path'],
                saved_results_name=hyperparams_dict['saved_results_name'],
                hyperparams_dict=hyperparams_dict)
  else:
    play_game(env=my_env, agent=my_train_agent, epsilon=0.00, game_render=False)


if __name__ == "__main__":
  main(train_model=True)
  # main()