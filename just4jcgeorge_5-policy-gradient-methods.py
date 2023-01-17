from IPython.display import Image

import os

Image("/kaggle/input/week9dataset/Policy Gradient Methods1.png")
Image("/kaggle/input/week9dataset/Policy Gradient Methods2.png")
Image("/kaggle/input/week9dataset/Policy Gradient Methods2.png")
Image("/kaggle/input/week9dataset/Policy Gradient Methods4.png")
Image("/kaggle/input/week9dataset/Policy Gradient Methods5.png")
Image("/kaggle/input/week9dataset/Policy Gradient Methods6.gif")
Image("/kaggle/input/week9dataset/Policy Gradient Methods7.gif")
import gym
env = gym.make('CartPole-v1')
for _ in range(10):

    t = 0

    env.reset()

    while True:

        action = env.action_space.sample()

        observation, reward, done, _ = env.step(action)

        t += 1

        if done:

            print("Episode finished after {} timesteps".format(t))

            break
def mc_policy_gradient(env, theta, lr, episodes):

    """

    Parameters:

    env -- Environment

    theta -- The policy function parameter

    lr -- Learning rate

    episodes -- The number of iterations

    Returns: 

    episodes -- The cumulative reward value

    """

    for episode in range(episodes):  # Iterations

        episode = []

        start_observation = env.reset()  # Initialize the environment

        t = 0

        while True:

            policy = np.dot(theta, start_observation)  # Calculate the policy value

            # Here action_space is 2, so use Sigmoid

            pi = 1 / (1 + np.exp(-policy))

            if pi >= 0.5:

                action = 1  # Push cart to the right

            else:

                action = 0  # Push cart to the left

            next_observation, reward, done, _ = env.step(action)  # Take actions

            # Add the environment return result to the episode

            episode.append([next_observation, action, pi, reward])

            start_observation = next_observation  # Return observation as the next iteration observation

            t += 1

            if done:

                print("Episode finished after {} timesteps".format(t))

                break

        # Update theta

        for timestep in episode:

            observation, action, pi, reward = timestep

            theta += lr * (1 - pi) * np.transpose(-observation) * reward

    

    return theta
import numpy as np

lr = 0.001

theta = np.random.rand(4)

episodes=10
mc_policy_gradient(env, theta, lr, episodes)
def ac_policy_gradient(env, theta, w, lr, gamma, episodes):

    done = True

    for _ in range(episodes):

        t = 0

        while True:

            if done:  # Determine whether to reset the environment based on the done value

                start_observation = env.reset()  # Initialize the environment

                # Choose actions based on policy

                policy = np.dot(theta, start_observation)

                start_pi = 1 / (1 + np.exp(-policy))

                if start_pi >= 0.5:

                    start_action = 1

                else:

                    start_action = 0

                start_q = np.dot(w, start_observation)  # Calculate Q

            observation, reward, done, _ = env.step(start_action)  # Take actions

            # Select the appropriate action based on the new policy

            policy = np.dot(theta, observation)

            pi = 1 / (1 + np.exp(-policy))

            if pi >= 0.5:

                action = 1

            else:

                action = 0

            q = np.dot(w, observation)

            # Update

            delta = reward + gamma * q - start_q

            theta += lr * (1 - start_pi) * np.transpose(-start_observation) * start_q

            w += lr * delta * np.transpose(start_observation)

            start_pi, start_observation, start_q, start_action = pi, observation, q, action

            t += 1

            if done:

                print("Episode finished after {} timesteps".format(t+1))

                break

    return theta, w
gamma = 1

theta = np.random.rand(4)

w = np.random.rand(4)

lr = 0.001

episodes=10
ac_policy_gradient(env, theta, w, lr, gamma, episodes)