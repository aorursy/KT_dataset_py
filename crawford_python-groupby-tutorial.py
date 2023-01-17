import numpy as np

import pandas as pd

import random



# Random pets column

pet_list = ["cat", "hamster", "alligator", "snake"]

pet = [random.choice(pet_list) for i in range(1,15)]



# Random weight of animal column

weight = [random.choice(range(5,15)) for i in range(1,15)]



# Random length of animals column

length = [random.choice(range(1,10)) for i in range(1,15)]



# random age of the animals column

age = [random.choice(range(1,15)) for i in range(1,15)]



# Put everyhting into a dataframe

df = pd.DataFrame()

df["animal"] = pet

df["age"] = age

df["weight"] = weight

df["length"] = length



# make a groupby object

animal_groups = df.groupby("animal")

df
# Load restaurant ratings and parking lot info

ratings = pd.read_csv("../input/rating_final.csv")

parking = pd.read_csv("../input/chefmozparking.csv")



# Merge the dataframes

df = pd.merge(left=ratings, right=parking, on="placeID", how="left")



# Show the merged data

df.head()
# Group by the parking_lot column

parking_group = df.groupby("parking_lot")



# Calculate the mean ratings

parking_group["service_rating"].mean()
parking_group['service_rating'].describe()