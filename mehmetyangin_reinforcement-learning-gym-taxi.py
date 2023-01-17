import gym
env = gym.make("Taxi-v2").env

env.render()
env.reset()
print(env.observation_space)

print(env.action_space)
state = env.encode(3,1,2,3)

print(state)
env.s = state

env.render()
env.P[331]
env.reset()
total_reward_list = []

# episode

for j in range(5):

    env.reset()

    time_step = 0

    total_reward = 0

    list_visualize = []

    while True:

        time_step += 1

        #choose action

        action = env.action_space.sample()

        #perform action and get reward

        state, reward, done, _ =  env.step(action) # state = next state

        #total reward

        total_reward += reward

        # visualize

        list_visualize.append({"frame": env.render(mode = "ansi"),

                                "state": state, "action": action, "reward":reward,

                                "Total Reward": total_reward})

        if done:

            total_reward_list.append(total_reward)

            break
import time       

for i, frame in enumerate(list_visualize):

    print(frame["frame"])

    print("Timestep: ", i + 1)

    print("State: ", frame["state"])

    print("action: ", frame["action"])

    print("reward: ", frame["reward"])

    print("Total Reward: ", frame["Total Reward"])

    # time.sleep(2)