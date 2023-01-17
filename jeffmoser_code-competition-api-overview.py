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
# The API is exposed through a "kagglegym" library. Let's import that to get started:

import kagglegym
# Now, we need to create an "environment". This will be our primary interface to the API.

# The kagglegym API has the concept of a default environment name for a competition, so

# just calling "make()" will create the appropriate one for this competition.

env = kagglegym.make()
# To properly initialize things, we need to "reset" it. This will also give us our first "observation":

observation = env.reset()
# Observations are the means by which our code "observes" the world. The very first observation

# has a special property called "train" which is a dataframe which we can use to train our model:

observation.train.head()
# Note that this "train" is about half the size of the full training dataframe.

# This is because we're in an exploratory mode where we simulate the full environment 

# by reserving the first half of timestamps for training and the second half for simulating

# the public leaderboard

len(observation.train)
len(observation.train["timestamp"].unique())
# Note that this is half of all of them:

len(df["timestamp"].unique())
# Here's proof that it's the first half:

unique_times = list(observation.train["timestamp"].unique())

(min(unique_times), max(unique_times))
# Each observation also has a "features" dataframe which contains features for the timestamp 

# you'll be asked to predict in the next "step."



# Note that these features are for timestamp 906 which is just passed the last training timestamp.

# Also, note that the "features" dataframe does *not* have the target "y" column:

observation.features.head()
# The final part of observation is the "target" dataframe which is what we're asking you to fill in.

# It includes the "id"s for the timestamp next step

observation.target.head()
# This target is a valid submission for the step. The OpenAI Gym calls each step an "action"

action = observation.target



# Each "step" of the environment returns four things:

observation, reward, done, info = env.step(action)
# The "done" variable tells us if we're done. In this case, we still have plenty of timestamps to go:

done
# The "info" is just a dictionary used for debugging. In this particular environment,

# we only make use of it at the end (when "done" is True)

info
# "observation" has the same properties as the one we get in "reset"

# However, notice that it's for the next "timestamp":

observation.features.head()
# Note that this timestamp has more id's/rows

len(observation.features)
# Perhaps most interesting is the "reward". This tells you how well you're doing. The goal

# in reinforcement contexts is that you want to maximize the reward. In this competition, we're using

# the R value that ranges from 0 to 1 (higher is better). Note that we submitted all 0's, so we get the

# worst score (if you got worse than 0, it's capped to 0)

reward
# Since we're in exploratory mode, we have access to the ground truth (obviously not available in

# submit mode)

perfect_action = df[df["timestamp"] == observation.features["timestamp"][0]][["id", "y"]].reset_index(drop=True)
perfect_action.head()
# Let's see what happens when we submit a "perfect" action:

observation, reward, done, info = env.step(perfect_action)
# Note that by submitting the perfect value, we get a reward of 1:

reward
# but we're still not done yet:

done
# Now that we've gotten the basics out of the way, we can create a basic loop until we're "done":

while True:

    target = observation.target

    timestamp = observation.features["timestamp"][0]

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))



    observation, reward, done, info = env.step(target)

    if done:        

        break
# Now we're "done":

done
# And since we're done, we have some extra info:

info
# Our score is better than 0 because we had that one submission that was perfect

info["public_score"]