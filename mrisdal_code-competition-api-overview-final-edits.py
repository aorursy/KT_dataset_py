# Here's an example of loading the CSV using Pandas's built-in HDF5 support:

import pandas as pd



with pd.HDFStore("../input/train.h5", "r") as train:

    # Note that the "train" dataframe is the only dataframe in the file

    df = train.get("train")
# Let's see how many rows are in full training set

len(df)
df.head()
# How many timestamps are in the full training set?

len(df["timestamp"].unique())
import kagglegym
# Create environment

env = kagglegym.make()
# Get first observation

observation = env.reset()
# Look at first few rows of the train dataframe

observation.train.head()
# Get length of the train dataframe

len(observation.train)
# Get number of unique timestamps in train

len(observation.train["timestamp"].unique())
# Note that this is half of all timestamps:

len(df["timestamp"].unique())
# Here's proof that it's the first half:

unique_times = list(observation.train["timestamp"].unique())

(min(unique_times), max(unique_times))
# Look at the first few rows of the features dataframe

observation.features.head()
# Look at the first few rows of the target dataframe

observation.target.head()
# Each step is an "action"

action = observation.target



# Each "step" of the environment returns four things:

observation, reward, done, info = env.step(action)
# Print done

done
# Print info

info
# Look at the first few rows of the observation dataframe for the next timestamp

observation.features.head()
# Note that this timestamp has more id's/rows

len(observation.features)
# Print reward

reward


perfect_action = df[df["timestamp"] == observation.features["timestamp"][0]][["id", "y"]].reset_index(drop=True)
# Look at the first few rows of perfect action

perfect_action.head()
# Submit a perfect action

observation, reward, done, info = env.step(perfect_action)
# Print reward

reward
# Print done ... still more timestamps remaining

done
while True:

    target = observation.target

    timestamp = observation.features["timestamp"][0]

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))



    observation, reward, done, info = env.step(target)

    if done:        

        break
# Print done

done
# Print info

info
# Print "public score" from info

info["public_score"]