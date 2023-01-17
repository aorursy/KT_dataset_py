import os

for name in os.listdir("../input"):

    print("Input file: " + name)



for name in os.listdir("/dev/gym/data"):

    print("Gym data file: " + name)
import gym

import numpy as np



env = gym.make()



observation = env.reset()

prediction_column = observation.current_to_predict.columns[-1]



while True:

    # create a random submission

    action = observation.current_to_predict

    last_actuals = observation.last_actuals

    action[prediction_column] = np.random.randn(len(observation.current_to_predict))



    observation, reward, done, info = env.step(action)

    if done:

        print("Public score: {}".format(info["public_score"]))

        break