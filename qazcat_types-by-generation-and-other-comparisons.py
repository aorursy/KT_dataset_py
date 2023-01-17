import numpy as np

import pandas as pd



poke = pd.read_csv("../input/Pokemon.csv")

poke = poke[poke["Name"].str.contains("Mega")==False] #Remove mega evolutions



#quick comparison of all generations. Number, generation and stat totals have been dropped.

gen_means = poke.drop(["#","Legendary","Total"], axis = 1).groupby("Generation").mean()



print(gen_means)
#total types for all generations

def type_counter(gen,main_df=poke):

    gen_df = main_df[main_df["Generation"] == gen]

    type_count_1 = gen_df.groupby("Type 1").count()["#"]

    type_count_2 = gen_df.groupby("Type 2").count()["#"]

    for type_name in type_count_2.keys():

        if type_name not in type_count_1:

            type_count_1[type_name] = type_count_2[type_name]

        else:

            type_count_1[type_name] = type_count_1[type_name] + type_count_2[type_name]

    type_count_1["Gen"] = gen

    return type_count_1



gen_dict = {}



for x in range(1,7): #6 generations

    gen_dict[x] = type_counter(x)

    

type_df = pd.DataFrame.from_dict(gen_dict, orient='index',).fillna(0)

type_df.head(6)

#need to normalise by number of pokemon in each generation
print(type_df.drop("Gen", axis=1).sum().sort_values(ascending=False))
import seaborn as sns

import matplotlib.pyplot as plt



sns.barplot(data=type_df,x = "Gen",y = "Water")
sns.barplot(data=type_df,x = "Gen",y = "Normal")
#Get number of pokemon in each generation

pokemon_per_gen = poke.groupby("Generation").count()["#"]



#Drop column with generation, Caluculate percentation of pokemon with that type in generation

norm_type_df = type_df.divide(pokemon_per_gen,axis=0) * 100

norm_type_df["Gen"] = type_df["Gen"]

norm_type_df.head(6)
sns.barplot(data=norm_type_df,x = "Gen",y = "Water")

print(norm_type_df["Water"].sort_values(ascending=False))
sns.barplot(data=norm_type_df,x = "Gen",y = "Normal")

print(norm_type_df["Normal"].sort_values(ascending=False))